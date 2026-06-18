import Foundation
import ARKit
import simd

/// Measures a box from the LiDAR depth map using RANSAC plane detection.
///
/// A cardboard box has flat, mutually-perpendicular faces — pure geometry,
/// no trained model needed.
///
/// Pipeline:
///   1. Unproject depth pixels inside YOLO bbox → 3D point cloud.
///   2. RANSAC → dominant plane = front face.
///   3. Project front-face inliers onto plane → width + gravity-aligned height.
///   4. Depth: RANSAC on remaining points for a side plane; fallback to
///      background-depth sampling around the bbox.
///   5. Build a gravity-aligned worldTransform for SceneKit rendering.
struct BoxFitter {

    // MARK: - Public entry point

    static func measure(
        frame: ARFrame,
        yoloBox: YOLOBox,
        minPoints: Int = 40
    ) -> Detection3D? {

        guard let depthBuffer = frame.sceneDepth?.depthMap else { return nil }

        var dFront: Float = 0
        let points = unproject(frame: frame, yoloBox: yoloBox,
                               depthBuffer: depthBuffer, dFront: &dFront)
        guard points.count >= minPoints else { return nil }

        // 1. RANSAC — front face plane.
        guard let front = ransac(points: points, iters: 120, thresh: 0.022) else { return nil }

        let inliers = points.filter { front.dist($0) <= 0.025 }
        guard inliers.count >= 15 else { return nil }

        // 2. Gravity-aligned height from Y extent of front-face inliers.
        let yMin = inliers.map { $0.y }.min()!
        let yMax = inliers.map { $0.y }.max()!
        let height = yMax - yMin
        guard height > 0.02 else { return nil }

        // 3. Width: project inliers onto horizontal axis in the front plane.
        //    hAxis = cross(normal, up) → horizontal direction on the face.
        let up = simd_float3(0, 1, 0)
        let hAxis = simd_normalize(simd_cross(front.normal, up))
        let centroid = inliers.reduce(.zero, +) / Float(inliers.count)
        let projW = inliers.map { simd_dot($0 - centroid, hAxis) }
        let width = projW.max()! - projW.min()!
        guard width > 0.02 else { return nil }

        // 4. Depth estimation.
        let outliers = points.filter { front.dist($0) > 0.03 }
        let depth: Float
        if outliers.count >= 20,
           let side = ransac(points: outliers, iters: 60, thresh: 0.025),
           abs(simd_dot(side.normal, front.normal)) < 0.35 {
            // Side plane found — depth = extent of all box points along the front normal.
            let sideInliers = outliers.filter { side.dist($0) <= 0.025 }
            let allBox = inliers + sideInliers
            let projN = allBox.map { simd_dot($0, front.normal) }
            depth = max(projN.max()! - projN.min()!, 0.03)
        } else {
            // Fallback: background depth around the bbox.
            depth = max(bgDepth(frame: frame, yoloBox: yoloBox, dFront: dFront), 0.03)
        }

        // 5. Ensure front-face normal points toward camera.
        let camPos = simd_float3(frame.camera.transform.columns.3.x,
                                  frame.camera.transform.columns.3.y,
                                  frame.camera.transform.columns.3.z)
        let toCamera = simd_normalize(camPos - centroid)
        let normalTowardCam = simd_dot(front.normal, toCamera) > 0 ? front.normal : -front.normal

        // Box center is depth/2 behind the front face.
        let rawCenter = centroid - normalTowardCam * (depth / 2)
        let center = simd_float3(rawCenter.x, (yMin + yMax) / 2, rawCenter.z)

        // 6. World transform: rotate around Y so local-X aligns with hAxis.
        let yaw = atan2(hAxis.z, hAxis.x)
        let c = cos(yaw), s = sin(yaw)
        let worldTransform = simd_float4x4(
            simd_float4( c, 0, s, 0),
            simd_float4( 0, 1, 0, 0),
            simd_float4(-s, 0, c, 0),
            simd_float4(center.x, center.y, center.z, 1)
        )

        return Detection3D(
            center: center,
            size: simd_float3(width, height, depth),
            yaw: yaw,
            confidence: yoloBox.score,
            worldTransform: worldTransform,
            label: yoloBox.label
        )
    }

    // MARK: - Depth map → 3D point cloud

    private static func unproject(
        frame: ARFrame,
        yoloBox: YOLOBox,
        depthBuffer: CVPixelBuffer,
        dFront: inout Float
    ) -> [simd_float3] {

        let bufW = Float(CVPixelBufferGetWidth(frame.capturedImage))
        let bufH = Float(CVPixelBufferGetHeight(frame.capturedImage))
        let dW = CVPixelBufferGetWidth(depthBuffer)
        let dH = CVPixelBufferGetHeight(depthBuffer)

        // YOLO 640×640 → landscape buffer coords.
        let side = min(bufW, bufH)
        let ox = (bufW - side) / 2, oy = (bufH - side) / 2
        let bx0 = yoloBox.xmin / 640 * side + ox
        let by0 = yoloBox.ymin / 640 * side + oy
        let bx1 = yoloBox.xmax / 640 * side + ox
        let by1 = yoloBox.ymax / 640 * side + oy

        let px0 = max(0, Int(bx0 / bufW * Float(dW)))
        let py0 = max(0, Int(by0 / bufH * Float(dH)))
        let px1 = min(dW - 1, Int(bx1 / bufW * Float(dW)))
        let py1 = min(dH - 1, Int(by1 / bufH * Float(dH)))
        guard px1 > px0, py1 > py0 else { return [] }

        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }
        let base = CVPixelBufferGetBaseAddress(depthBuffer)!
        let rowBytes = CVPixelBufferGetBytesPerRow(depthBuffer)

        // Pass 1: find front surface depth (10th percentile).
        var allD: [Float] = []
        allD.reserveCapacity((px1 - px0) * (py1 - py0))
        for py in py0...py1 {
            let row = base.advanced(by: py * rowBytes).assumingMemoryBound(to: Float32.self)
            for px in px0...px1 {
                let d = row[px]; if d > 0.1 && d < 7 { allD.append(d) }
            }
        }
        guard allD.count >= 20 else { return [] }
        allD.sort()
        dFront = allD[allD.count / 10]
        let dCut  = dFront + 0.40   // include up to 40 cm behind front face

        // Pass 2: unproject box-surface pixels.
        let intr = frame.camera.intrinsics
        let fx = intr[0][0], fy = intr[1][1], cx = intr[2][0], cy = intr[2][1]
        let T = frame.camera.transform

        var pts: [simd_float3] = []
        pts.reserveCapacity(allD.count / 2)
        for py in py0...py1 {
            let row = base.advanced(by: py * rowBytes).assumingMemoryBound(to: Float32.self)
            for px in px0...px1 {
                let d = row[px]
                guard d >= dFront - 0.03, d <= dCut else { continue }
                let ix = Float(px) / Float(dW) * bufW
                let iy = Float(py) / Float(dH) * bufH
                let cam = simd_float4((ix - cx) / fx * d, (iy - cy) / fy * d, -d, 1)
                let w = T * cam
                pts.append(simd_float3(w.x, w.y, w.z) / w.w)
            }
        }
        return pts
    }

    // MARK: - RANSAC plane fitting

    private struct Plane {
        let normal: simd_float3
        let d: Float
        func dist(_ p: simd_float3) -> Float { abs(simd_dot(normal, p) - d) }
    }

    private static func ransac(
        points: [simd_float3],
        iters: Int,
        thresh: Float
    ) -> Plane? {
        guard points.count >= 3 else { return nil }

        var bestNormal = simd_float3(0, 0, 1)
        var bestD: Float = 0
        var bestN = 0

        for _ in 0..<iters {
            let i = Int.random(in: 0..<points.count)
            var j = Int.random(in: 0..<points.count)
            var k = Int.random(in: 0..<points.count)
            while j == i { j = Int.random(in: 0..<points.count) }
            while k == i || k == j { k = Int.random(in: 0..<points.count) }

            let v1 = points[j] - points[i]
            let v2 = points[k] - points[i]
            let nRaw = simd_cross(v1, v2)
            let len = simd_length(nRaw)
            guard len > 1e-6 else { continue }
            let n = nRaw / len
            let d = simd_dot(n, points[i])

            let cnt = points.filter { abs(simd_dot($0, n) - d) <= thresh }.count
            if cnt > bestN { bestN = cnt; bestNormal = n; bestD = d }
        }
        guard bestN >= 10 else { return nil }

        // Refine d as mean projection of inliers.
        let inliers = points.filter { abs(simd_dot($0, bestNormal) - bestD) <= thresh }
        let refinedD = inliers.map { simd_dot($0, bestNormal) }.reduce(0, +) / Float(inliers.count)
        return Plane(normal: bestNormal, d: refinedD)
    }

    // MARK: - Background depth sampling (depth fallback)

    private static func bgDepth(
        frame: ARFrame,
        yoloBox: YOLOBox,
        dFront: Float
    ) -> Float {
        guard let depthBuffer = frame.sceneDepth?.depthMap else { return 0.25 }

        let bufW = Float(CVPixelBufferGetWidth(frame.capturedImage))
        let bufH = Float(CVPixelBufferGetHeight(frame.capturedImage))
        let dW = CVPixelBufferGetWidth(depthBuffer)
        let dH = CVPixelBufferGetHeight(depthBuffer)

        let side = min(bufW, bufH)
        let ox = (bufW - side) / 2, oy = (bufH - side) / 2
        let bx0 = yoloBox.xmin / 640 * side + ox
        let by0 = yoloBox.ymin / 640 * side + oy
        let bx1 = yoloBox.xmax / 640 * side + ox
        let by1 = yoloBox.ymax / 640 * side + oy

        // Expanded region 25% outside the bbox on each side.
        let pw = bx1 - bx0, ph = by1 - by0
        let epx0 = max(0, Int((bx0 - pw * 0.25) / bufW * Float(dW)))
        let epx1 = min(dW - 1, Int((bx1 + pw * 0.25) / bufW * Float(dW)))
        let epy0 = max(0, Int((by0 - ph * 0.25) / bufH * Float(dH)))
        let epy1 = min(dH - 1, Int((by1 + ph * 0.25) / bufH * Float(dH)))
        let ipx0 = Int(bx0 / bufW * Float(dW)), ipx1 = Int(bx1 / bufW * Float(dW))
        let ipy0 = Int(by0 / bufH * Float(dH)), ipy1 = Int(by1 / bufH * Float(dH))

        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }
        let base = CVPixelBufferGetBaseAddress(depthBuffer)!
        let rowBytes = CVPixelBufferGetBytesPerRow(depthBuffer)

        // Collect depths OUTSIDE the bbox that are behind the front face.
        var bgDs: [Float] = []
        for py in epy0...epy1 {
            let row = base.advanced(by: py * rowBytes).assumingMemoryBound(to: Float32.self)
            for px in epx0...epx1 {
                if px >= ipx0 && px <= ipx1 && py >= ipy0 && py <= ipy1 { continue }
                let d = row[px]
                if d > dFront + 0.04 && d < 7 { bgDs.append(d) }
            }
        }
        guard !bgDs.isEmpty else { return 0.25 }
        bgDs.sort()
        return bgDs[bgDs.count / 2] - dFront
    }
}
