import Foundation
import ARKit
import simd

/// Robust box measurement using LiDAR + XZ voxel flood fill.
///
/// Pipeline:
///   1. dRef = median depth at screen crosshair (depth buffer center).
///   2. Unproject points from expanded YOLO region within [dRef-0.05, dRef+0.60].
///   3. Remove floor points.
///   4. Project to XZ voxel grid (2 cm cells).
///   5. BFS from crosshair world-XZ seed → isolated box footprint.
///   6. PCA on flood-filled points → OBB width × depth.
///   7. Y extent → height.
struct BoxMeasurer2 {

    // MARK: - Voxel key

    private struct VoxelKey: Hashable { let x: Int, z: Int }

    // MARK: - Entry point

    static func measure(
        frame: ARFrame,
        yoloBox: YOLOBox,
        minPoints: Int = 30
    ) -> Detection3D? {

        guard let depthBuffer = frame.sceneDepth?.depthMap else { return nil }

        let bufW = Float(CVPixelBufferGetWidth(frame.capturedImage))
        let bufH = Float(CVPixelBufferGetHeight(frame.capturedImage))
        let dW = CVPixelBufferGetWidth(depthBuffer)
        let dH = CVPixelBufferGetHeight(depthBuffer)

        // 1. Reference depth from YOLO bbox center (small 15x15 px region in depth buffer).
        //    Using YOLO center is more reliable than screen center — YOLO already found the box.
        let side = min(bufW, bufH)
        let ox = (bufW - side) / 2, oy = (bufH - side) / 2
        let bCenterX = (yoloBox.xmin + yoloBox.xmax) / 2 / 640 * side + ox
        let bCenterY = (yoloBox.ymin + yoloBox.ymax) / 2 / 640 * side + oy
        let dcx = Int(bCenterX / bufW * Float(dW))
        let dcy = Int(bCenterY / bufH * Float(dH))
        let dRef = bboxCenterDepth(depthBuffer: depthBuffer, dcx: dcx, dcy: dcy, dW: dW, dH: dH)
        guard dRef > 0.15 && dRef < 6.0 else { return nil }

        // 2. Unproject all LiDAR points in depth range around dRef.
        let raw = unprojectRange(
            frame: frame, depthBuffer: depthBuffer, yoloBox: yoloBox,
            dMin: max(dRef - 0.05, 0.10), dMax: dRef + 0.60,
            bufW: bufW, bufH: bufH, dW: dW, dH: dH
        )
        guard raw.count >= minPoints else { return nil }

        // 3. Floor removal — same logic as before.
        let rawYMin = raw.map { $0.y }.min()!
        let anchorFloorY: Float? = frame.anchors
            .compactMap { $0 as? ARPlaneAnchor }
            .filter { $0.alignment == .horizontal }
            .map { Float($0.transform.columns.3.y) }.min()

        let reliableFloorY: Float
        if let anchor = anchorFloorY, anchor <= rawYMin + 0.08 {
            reliableFloorY = anchor
        } else {
            reliableFloorY = rawYMin
        }
        let floorThresh = reliableFloorY + 0.04
        let pts = raw.filter { $0.y > floorThresh }
        guard pts.count >= minPoints else { return nil }

        // 4. Build XZ voxel grid (2 cm resolution).
        let voxelSize: Float = 0.02
        var voxelSet = Set<VoxelKey>()
        var voxelPoints: [VoxelKey: [simd_float3]] = [:]
        for p in pts {
            let key = VoxelKey(x: Int(floor(p.x / voxelSize)), z: Int(floor(p.z / voxelSize)))
            voxelSet.insert(key)
            voxelPoints[key, default: []].append(p)
        }

        // 5. Seed: unproject YOLO bbox center at dRef → world XZ.
        let intr = frame.camera.intrinsics
        let camSeed = simd_float4(
            (bCenterX - intr[2][0]) / intr[0][0] * dRef,
            (bCenterY - intr[2][1]) / intr[1][1] * dRef,
            -dRef, 1
        )
        let wSeed = frame.camera.transform * camSeed
        let seedWorldX = wSeed.x / wSeed.w
        let seedWorldZ = wSeed.z / wSeed.w
        let seedKey = VoxelKey(x: Int(floor(seedWorldX / voxelSize)), z: Int(floor(seedWorldZ / voxelSize)))

        guard voxelSet.contains(seedKey) else { return nil }

        // 6. BFS flood fill — 8-connected in XZ, starting at crosshair seed.
        var visited = Set<VoxelKey>()
        var queue = [seedKey]
        visited.insert(seedKey)
        while !queue.isEmpty {
            let curr = queue.removeFirst()
            for dx in -1...1 {
                for dz in -1...1 {
                    guard dx != 0 || dz != 0 else { continue }
                    let next = VoxelKey(x: curr.x + dx, z: curr.z + dz)
                    if voxelSet.contains(next) && !visited.contains(next) {
                        visited.insert(next)
                        queue.append(next)
                    }
                }
            }
        }
        guard visited.count >= 10 else { return nil }

        // Collect 3D points from flood-filled voxels.
        var connPts: [simd_float3] = []
        for key in visited {
            if let arr = voxelPoints[key] { connPts.append(contentsOf: arr) }
        }
        guard connPts.count >= minPoints else { return nil }

        // 7. Height = Y extent of connected component.
        var ySorted = connPts.map { $0.y }; ySorted.sort()
        let yTop = ySorted[Int(Float(ySorted.count) * 0.97)]
        let yMin = reliableFloorY
        let height = yTop - yMin
        guard height > 0.02 else { return nil }

        // 8. PCA on XZ footprint → OBB orientation + dimensions.
        let n = Float(connPts.count)
        let xMean = connPts.map { $0.x }.reduce(0, +) / n
        let zMean = connPts.map { $0.z }.reduce(0, +) / n

        var cxx: Float = 0, cxz: Float = 0, czz: Float = 0
        for p in connPts {
            let dx = p.x - xMean, dz = p.z - zMean
            cxx += dx*dx; cxz += dx*dz; czz += dz*dz
        }
        cxx /= n; cxz /= n; czz /= n

        // Principal eigenvector of the 2×2 covariance matrix.
        let trace = cxx + czz
        let disc = max(0, trace*trace/4 - (cxx*czz - cxz*cxz))
        let l1 = trace/2 + sqrt(disc)
        let axX: Float, axZ: Float
        if abs(cxz) > 1e-6 {
            let v = simd_normalize(simd_float2(l1 - czz, cxz))
            axX = v.x; axZ = v.y
        } else {
            axX = cxx >= czz ? 1 : 0
            axZ = cxx >= czz ? 0 : 1
        }
        let pxX = -axZ, pxZ = axX   // perpendicular axis

        var pVals: [Float] = [], qVals: [Float] = []
        for p in connPts {
            let dx = p.x - xMean, dz = p.z - zMean
            pVals.append(dx*axX + dz*axZ)
            qVals.append(dx*pxX + dz*pxZ)
        }
        pVals.sort(); qVals.sort()
        let lo = Int(Float(pVals.count) * 0.01), hi = Int(Float(pVals.count) * 0.99)
        let pMin = pVals[lo], pMax = pVals[hi]
        let qMin = qVals[lo], qMax = qVals[hi]
        let dimA = pMax - pMin   // extent along principal axis
        let dimB = qMax - qMin   // extent along secondary axis
        guard dimA > 0.02 && dimB > 0.02 else { return nil }

        // 9. OBB center in world space.
        let pCen = (pMin + pMax) / 2
        let qCen = (qMin + qMax) / 2
        let center = simd_float3(
            xMean + pCen*axX + qCen*pxX,
            (yMin + yTop) / 2,
            zMean + pCen*axZ + qCen*pxZ
        )

        let yaw = atan2(axZ, axX)
        let cosY = cos(yaw), sinY = sin(yaw)
        let worldTransform = simd_float4x4(
            simd_float4( cosY, 0, sinY, 0),
            simd_float4(    0, 1,    0, 0),
            simd_float4(-sinY, 0, cosY, 0),
            simd_float4(center.x, center.y, center.z, 1)
        )

        return Detection3D(
            center: center,
            size: simd_float3(dimA, height, dimB),
            yaw: yaw,
            confidence: yoloBox.score,
            worldTransform: worldTransform,
            label: yoloBox.label
        )
    }

    // MARK: - YOLO bbox center depth (median of 15x15 px region)

    private static func bboxCenterDepth(depthBuffer: CVPixelBuffer, dcx: Int, dcy: Int, dW: Int, dH: Int) -> Float {
        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }
        let base = CVPixelBufferGetBaseAddress(depthBuffer)!
        let rb = CVPixelBufferGetBytesPerRow(depthBuffer)
        let r = 7   // 15x15 px window
        var ds: [Float] = []
        for py in max(0, dcy - r)...min(dH - 1, dcy + r) {
            let row = base.advanced(by: py * rb).assumingMemoryBound(to: Float32.self)
            for px in max(0, dcx - r)...min(dW - 1, dcx + r) {
                let d = row[px]
                if d > 0.1 && d < 7 { ds.append(d) }
            }
        }
        guard !ds.isEmpty else { return 0 }
        ds.sort()
        return ds[ds.count / 2]
    }

    // MARK: - Unproject depth range inside expanded YOLO region

    private static func unprojectRange(
        frame: ARFrame,
        depthBuffer: CVPixelBuffer,
        yoloBox: YOLOBox,
        dMin: Float, dMax: Float,
        bufW: Float, bufH: Float, dW: Int, dH: Int
    ) -> [simd_float3] {

        // Expand YOLO bbox by 50% on each side — flood fill will isolate the box.
        let side = min(bufW, bufH)
        let ox = (bufW - side) / 2, oy = (bufH - side) / 2
        let bx0 = yoloBox.xmin/640*side + ox, by0 = yoloBox.ymin/640*side + oy
        let bx1 = yoloBox.xmax/640*side + ox, by1 = yoloBox.ymax/640*side + oy
        let pw = bx1 - bx0, ph = by1 - by0
        let ex0 = max(0,    Int((bx0 - pw*0.5) / bufW * Float(dW)))
        let ex1 = min(dW-1, Int((bx1 + pw*0.5) / bufW * Float(dW)))
        let ey0 = max(0,    Int((by0 - ph*0.5) / bufH * Float(dH)))
        let ey1 = min(dH-1, Int((by1 + ph*0.5) / bufH * Float(dH)))
        guard ex1 > ex0, ey1 > ey0 else { return [] }

        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }
        let base = CVPixelBufferGetBaseAddress(depthBuffer)!
        let rb = CVPixelBufferGetBytesPerRow(depthBuffer)
        let intr = frame.camera.intrinsics
        let fx = intr[0][0], fy = intr[1][1], cx = intr[2][0], cy = intr[2][1]
        let T = frame.camera.transform
        var pts: [simd_float3] = []

        for py in ey0...ey1 {
            let row = base.advanced(by: py * rb).assumingMemoryBound(to: Float32.self)
            for px in ex0...ex1 {
                let d = row[px]
                guard d >= dMin, d <= dMax else { continue }
                let ix = Float(px) / Float(dW) * bufW
                let iy = Float(py) / Float(dH) * bufH
                let cam = simd_float4((ix - cx)/fx * d, (iy - cy)/fy * d, -d, 1)
                let w = T * cam
                pts.append(simd_float3(w.x, w.y, w.z) / w.w)
            }
        }
        return pts
    }
}
