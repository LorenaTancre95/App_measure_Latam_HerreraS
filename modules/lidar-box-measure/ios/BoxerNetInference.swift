// BoxerNetInference.swift
// ONNX Runtime wrapper for BoxerNet — CPU-only (no CoreML EP to avoid OOM).
// Predicts full 3D bounding box (center, size, yaw) from image + LiDAR depth.

import Foundation
import Accelerate
import ARKit
import simd
import onnxruntime_objc

// MARK: - Data Types

struct Detection3D {
    let center: simd_float3
    let size: simd_float3   // (width, height, depth) in metres
    let yaw: Float
    let confidence: Float
    let worldTransform: simd_float4x4
    let label: String?
}

struct Box2D {
    let xmin: Float
    let ymin: Float
    let xmax: Float
    let ymax: Float
    var label: String? = nil
    var score: Float = 0
}

// MARK: - BoxerNetInference

final class BoxerNetInference {
    private let session: ORTSession
    private let env: ORTEnv

    static let imageSize: Int  = 960
    static let patchSize: Int  = 16
    static let gridH: Int      = imageSize / patchSize  // 60
    static let gridW: Int      = imageSize / patchSize  // 60
    static let numPatches: Int = gridH * gridW          // 3600

    init(modelPath: String) throws {
        env = try ORTEnv(loggingLevel: .warning)
        let opts = try ORTSessionOptions()
        try opts.setIntraOpNumThreads(2)   // CPU-only — no CoreML EP (avoids OOM)
        session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: opts)
    }

    // MARK: - Predict

    func predict(
        image: [Float],            // CHW float32, len = 3 * 960 * 960
        depthMap: [[Float]],       // HxW depth in metres
        intrinsics: simd_float3x3, // already scaled to 960×960
        cameraTransform: simd_float4x4,
        boxes2D: [Box2D],
        confidenceThreshold: Float = 0.30
    ) throws -> [Detection3D] {
        guard !boxes2D.isEmpty else { return [] }

        let fx = intrinsics[0][0]
        let fy = intrinsics[1][1]
        let cx = intrinsics[2][0]
        let cy = intrinsics[2][1]

        // ARKit (OpenGL: -Z forward) → OpenCV (Z forward) for BoxerNet.
        let flipYZ = simd_float4x4(columns: (
            simd_float4( 1,  0,  0, 0),
            simd_float4( 0, -1,  0, 0),
            simd_float4( 0,  0, -1, 0),
            simd_float4( 0,  0,  0, 1)
        ))
        let T_wc = cameraTransform * flipYZ
        let T_wv = gravityAlign(T_worldCam: T_wc)
        let T_vc = T_wv.inverse * T_wc

        let sdpPatches  = buildSDPPatches(depthMap: depthMap)
        let rayEncoding = buildRayEncoding(T_vc: T_vc, fx: fx, fy: fy, cx: cx, cy: cy)

        let W = Float(Self.imageSize), H = Float(Self.imageSize)
        var bb2dFlat: [Float] = []
        for box in boxes2D {
            bb2dFlat.append((box.xmin + 0.5) / W)
            bb2dFlat.append((box.xmax + 0.5) / W)
            bb2dFlat.append((box.ymin + 0.5) / H)
            bb2dFlat.append((box.ymax + 0.5) / H)
        }

        let M = boxes2D.count
        let (centers, sizes, yaws, confidences) = try runInference(
            image: image, sdpPatches: sdpPatches,
            bb2d: bb2dFlat, rayEncoding: rayEncoding, numBoxes: M
        )

        var detections: [Detection3D] = []
        for i in 0..<M {
            let conf = confidences[i]
            guard conf >= confidenceThreshold else { continue }

            let centerVoxel = simd_float3(centers[i*3], centers[i*3+1], centers[i*3+2])
            let size        = simd_float3(sizes[i*3],   sizes[i*3+1],   sizes[i*3+2])
            let yaw         = yaws[i]

            let centerWorld = (T_wv * simd_float4(centerVoxel, 1.0)).xyz

            let R_wv  = upperLeft3x3(T_wv)
            let R_yaw = rotationZ(angle: yaw)
            let R_world = R_wv * R_yaw

            var transform = simd_float4x4(1.0)
            transform[0] = simd_float4(R_world[0], 0)
            transform[1] = simd_float4(R_world[1], 0)
            transform[2] = simd_float4(R_world[2], 0)
            transform[3] = simd_float4(centerWorld, 1)

            detections.append(Detection3D(
                center: centerWorld, size: size, yaw: yaw,
                confidence: conf, worldTransform: transform,
                label: boxes2D[i].label
            ))
        }
        return detections
    }

    // MARK: - ONNX Inference

    private func runInference(
        image: [Float], sdpPatches: [Float],
        bb2d: [Float], rayEncoding: [Float], numBoxes: Int
    ) throws -> (centers: [Float], sizes: [Float], yaws: [Float], confidences: [Float]) {
        let S  = Self.imageSize
        let gH = Self.gridH, gW = Self.gridW, N = Self.numPatches

        func tensor(_ data: [Float], shape: [NSNumber]) throws -> ORTValue {
            let d = Data(bytes: data, count: data.count * MemoryLayout<Float>.stride)
            return try ORTValue(tensorData: NSMutableData(data: d), elementType: .float, shape: shape)
        }

        let outputs = try session.run(withInputs: [
            "image":        try tensor(image,       shape: [1, 3, NSNumber(value:S), NSNumber(value:S)]),
            "sdp_patches":  try tensor(sdpPatches,  shape: [1, 1, NSNumber(value:gH), NSNumber(value:gW)]),
            "bb2d":         try tensor(bb2d,         shape: [1, NSNumber(value:numBoxes), 4]),
            "ray_encoding": try tensor(rayEncoding,  shape: [1, NSNumber(value:N), 6]),
        ], outputNames: ["center", "size", "yaw", "confidence"], runOptions: nil)

        func floats(_ key: String) throws -> [Float] {
            let d = try outputs[key]!.tensorData() as Data
            return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        return (try floats("center"), try floats("size"), try floats("yaw"), try floats("confidence"))
    }

    // MARK: - SDP Patches

    private func buildSDPPatches(depthMap: [[Float]]) -> [Float] {
        let S  = Self.imageSize
        let P  = Self.patchSize
        let gH = Self.gridH, gW = Self.gridW

        var patchDepths = [[Float]](repeating: [], count: gH * gW)
        let depthH = depthMap.count
        guard depthH > 0 else { return [Float](repeating: -1, count: gH * gW) }
        let depthW  = depthMap[0].count
        let scaleX  = Float(S) / Float(depthW)
        let scaleY  = Float(S) / Float(depthH)
        let step    = max(1, Int(sqrt(Float(depthH * depthW) / 20000.0)))

        for v in stride(from: 0, to: depthH, by: step) {
            for u in stride(from: 0, to: depthW, by: step) {
                let z = depthMap[v][u]
                guard z > 0 else { continue }
                let px = Float(u) * scaleX
                let py = Float(v) * scaleY
                let pi = Int(py) / P, pj = Int(px) / P
                guard pi >= 0, pi < gH, pj >= 0, pj < gW else { continue }
                patchDepths[pi * gW + pj].append(z)
            }
        }

        var result = [Float](repeating: -1, count: gH * gW)
        for idx in 0..<(gH * gW) {
            var depths = patchDepths[idx]
            guard !depths.isEmpty else { continue }
            depths.sort()
            result[idx] = depths[depths.count / 2]
        }
        return result
    }

    // MARK: - Plucker Ray Encoding

    private func buildRayEncoding(
        T_vc: simd_float4x4,
        fx: Float, fy: Float, cx: Float, cy: Float
    ) -> [Float] {
        let P  = Float(Self.patchSize)
        let gH = Self.gridH, gW = Self.gridW
        let R_vc = upperLeft3x3(T_vc)
        let originVoxel = (T_vc * simd_float4(0, 0, 0, 1)).xyz

        var result = [Float](repeating: 0, count: gH * gW * 6)
        for i in 0..<gH {
            for j in 0..<gW {
                let u = Float(j) * P + P / 2
                let v = Float(i) * P + P / 2
                var dirCam = simd_normalize(simd_float3((u-cx)/fx, (v-cy)/fy, 1))
                var dirVox = simd_normalize(R_vc * dirCam)
                let moment = simd_cross(originVoxel, dirVox)
                let idx = (i * gW + j) * 6
                result[idx+0] = dirVox.x; result[idx+1] = dirVox.y; result[idx+2] = dirVox.z
                result[idx+3] = moment.x; result[idx+4] = moment.y; result[idx+5] = moment.z
            }
        }
        return result
    }

    // MARK: - Gravity Alignment

    private func gravityAlign(
        T_worldCam: simd_float4x4,
        gravity_w: simd_float3 = simd_float3(0, -1, 0)
    ) -> simd_float4x4 {
        let R_wc = upperLeft3x3(T_worldCam)
        let t_wc = simd_float3(T_worldCam[3].x, T_worldCam[3].y, T_worldCam[3].z)
        let g_w  = simd_normalize(gravity_w)

        let camZ_w = R_wc * simd_float3(0, 0, 1)
        var d3 = camZ_w - g_w * simd_dot(camZ_w, g_w)
        if simd_length(d3) < 1e-6 { d3 = d3 + simd_float3(0, 0.001, 0) }
        let d2 = simd_cross(d3, g_w)

        var R_wcg = simd_float3x3(columns: (g_w, d2, d3))
        R_wcg[0] = simd_normalize(R_wcg[0])
        R_wcg[1] = simd_normalize(R_wcg[1])
        R_wcg[2] = simd_normalize(R_wcg[2])

        let R_cg_cgz = simd_float3x3(columns: (
            simd_float3( 0,  0, -1),
            simd_float3(-1,  0,  0),
            simd_float3( 0,  1,  0)
        ))
        let R_world_cgz = R_wcg * R_cg_cgz.inverse

        var T_wv = simd_float4x4(1.0)
        T_wv[0] = simd_float4(R_world_cgz[0], 0)
        T_wv[1] = simd_float4(R_world_cgz[1], 0)
        T_wv[2] = simd_float4(R_world_cgz[2], 0)
        T_wv[3] = simd_float4(t_wc, 1)
        return T_wv
    }
}

// MARK: - simd helpers

private func upperLeft3x3(_ m: simd_float4x4) -> simd_float3x3 {
    simd_float3x3(
        simd_float3(m[0].x, m[0].y, m[0].z),
        simd_float3(m[1].x, m[1].y, m[1].z),
        simd_float3(m[2].x, m[2].y, m[2].z)
    )
}

private func rotationZ(angle: Float) -> simd_float3x3 {
    let c = cos(angle), s = sin(angle)
    return simd_float3x3(
        simd_float3(c, s, 0),
        simd_float3(-s, c, 0),
        simd_float3(0, 0, 1)
    )
}

private extension simd_float4x4 {
    var inverse: simd_float4x4 { simd_inverse(self) }
}

private extension simd_float4 {
    var xyz: simd_float3 { simd_float3(x, y, z) }
}

// MARK: - Image helper (center-crop + resize → CHW float)

func pixelBufferToCHW(_ pixelBuffer: CVPixelBuffer, targetSize: Int) -> [Float] {
    var ciImage = CIImage(cvPixelBuffer: pixelBuffer)
    let context = CIContext()
    let w = ciImage.extent.width, h = ciImage.extent.height
    let side = min(w, h)
    ciImage = ciImage.cropped(to: CGRect(x: (w-side)/2, y: (h-side)/2, width: side, height: side))
    let scale = CGFloat(targetSize) / side
    let resized = ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))

    var rgba = [UInt8](repeating: 0, count: targetSize * targetSize * 4)
    context.render(resized, toBitmap: &rgba, rowBytes: targetSize * 4,
                   bounds: CGRect(x: resized.extent.origin.x, y: resized.extent.origin.y,
                                  width: CGFloat(targetSize), height: CGFloat(targetSize)),
                   format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())

    let n = targetSize * targetSize
    var result = [Float](repeating: 0, count: 3 * n)
    for i in 0..<n {
        result[i]         = Float(rgba[i*4])     / 255
        result[n + i]     = Float(rgba[i*4 + 1]) / 255
        result[2*n + i]   = Float(rgba[i*4 + 2]) / 255
    }
    return result
}

// MARK: - Depth map extractor

func extractDepthMap(_ depthBuffer: CVPixelBuffer) -> [[Float]] {
    CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }
    let h = CVPixelBufferGetHeight(depthBuffer)
    let w = CVPixelBufferGetWidth(depthBuffer)
    let bpr = CVPixelBufferGetBytesPerRow(depthBuffer)
    let base = CVPixelBufferGetBaseAddress(depthBuffer)!
    var result = [[Float]](repeating: [Float](repeating: 0, count: w), count: h)
    for y in 0..<h {
        let row = base.advanced(by: y * bpr).assumingMemoryBound(to: Float32.self)
        for x in 0..<w { result[y][x] = row[x] }
    }
    return result
}

// MARK: - Scale intrinsics for center-crop → targetSize

func scaleIntrinsicsForBoxerNet(
    _ intrinsics: simd_float3x3,
    imageResolution: CGSize,
    toSize: Int
) -> simd_float3x3 {
    let w = Float(imageResolution.width), h = Float(imageResolution.height)
    let side  = min(w, h)
    let scale = Float(toSize) / side
    var s = intrinsics
    s[0][0]  = intrinsics[0][0] * scale
    s[1][1]  = intrinsics[1][1] * scale
    s[2][0]  = (intrinsics[2][0] - (w - side) / 2) * scale
    s[2][1]  = (intrinsics[2][1] - (h - side) / 2) * scale
    return s
}
