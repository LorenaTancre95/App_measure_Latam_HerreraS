import Foundation
import ARKit
import simd

/// Geometric box measurement using LiDAR depth + explicit floor removal.
///
/// The key insight missing from previous attempts:
///   Floor points inside the YOLO bbox contaminate PCA/RANSAC.
///   Removing them first leaves only box-surface points, which gives
///   a clean oriented bounding box fit.
///
/// Pipeline:
///   1. Unproject depth pixels inside YOLO bbox → raw 3D cloud.
///   2. Remove floor points (y ≤ floorY + margin).
///   3. Remove background (depth > front + cutoff).
///   4. PCA on XZ footprint → orientation + width + depth.
///   5. Y extent → height.
///   6. If depth from PCA is too thin (nearly flat view), fall back
///      to background-depth subtraction.
struct BoxMeasurer2 {

    // MARK: - Entry point

    static func measure(
        frame: ARFrame,
        yoloBox: YOLOBox,
        minPoints: Int = 30
    ) -> Detection3D? {

        guard let depthBuffer = frame.sceneDepth?.depthMap else { return nil }

        // 1. Unproject raw cloud (includes floor and background — filtered next).
        var dFront: Float = 0
        let raw = unproject(frame: frame, yoloBox: yoloBox,
                            depthBuffer: depthBuffer, dFront: &dFront)
        guard raw.count >= minPoints else { return nil }

        // 2. Floor removal — the critical step.
        let floorY: Float? = frame.anchors
            .compactMap { $0 as? ARPlaneAnchor }
            .filter { $0.alignment == .horizontal }
            .map { Float($0.transform.columns.3.y) }.min()

        let floorThresh: Float
        if let floor = floorY {
            floorThresh = floor + 0.04   // 4 cm above detected floor plane
        } else {
            // No floor plane yet: remove bottom 8% of the Y range as a heuristic.
            let ys = raw.map { $0.y }
            let yMin = ys.min()!, yRange = ys.max()! - yMin
            floorThresh = yMin + yRange * 0.08
        }

        let pts = raw.filter { $0.y > floorThresh }
        guard pts.count >= minPoints else { return nil }

        // 3. Height from Y extent (3rd–97th percentile to reject outliers).
        var ySorted = pts.map { $0.y }; ySorted.sort()
        let yLo = Int(Float(ySorted.count)*0.03), yHi = Int(Float(ySorted.count)*0.97)
        let yMin = ySorted[yLo]
        let yMax = ySorted[yHi]
        let height = yMax - yMin
        guard height > 0.02 else { return nil }

        // 4. PCA on XZ footprint.
        let n = Float(pts.count)
        let xMean = pts.map { $0.x }.reduce(0, +) / n
        let zMean = pts.map { $0.z }.reduce(0, +) / n

        var c00: Float = 0, c01: Float = 0, c11: Float = 0
        for p in pts {
            let dx = p.x - xMean, dz = p.z - zMean
            c00 += dx*dx; c01 += dx*dz; c11 += dz*dz
        }
        c00 /= n; c01 /= n; c11 /= n

        let trace = c00 + c11
        let disc  = sqrt(max(0, trace*trace/4 - (c00*c11 - c01*c01)))
        let lam1  = trace/2 + disc

        var axX: Float, axZ: Float
        if abs(c01) > 1e-6 {
            let e = lam1 - c11; let len = sqrt(e*e + c01*c01)
            axX = e/len; axZ = c01/len
        } else {
            axX = c00 >= c11 ? 1 : 0; axZ = c00 >= c11 ? 0 : 1
        }
        let pxX = -axZ, pxZ = axX   // perpendicular axis in XZ

        // Project all pts onto both XZ axes, then use 3rd–97th percentile to reject outliers.
        var pVals: [Float] = []; var qVals: [Float] = []
        for p in pts {
            let dx = p.x - xMean, dz = p.z - zMean
            pVals.append(dx*axX + dz*axZ)
            qVals.append(dx*pxX + dz*pxZ)
        }
        pVals.sort(); qVals.sort()
        let lo = Int(Float(pVals.count)*0.03), hi = Int(Float(pVals.count)*0.97)
        let pMin = pVals[lo], pMax = pVals[hi]
        let qMin = qVals[lo], qMax = qVals[hi]

        let widthA = pMax - pMin   // larger PCA axis
        let widthB = qMax - qMin   // smaller PCA axis (depth if box at angle)
        guard widthA > 0.02 else { return nil }

        // 5. If box is viewed nearly face-on, widthB is the noise band of the
        //    front face depth — fall back to background sampling for depth.
        let depthEstimate: Float
        let minDepthRatio: Float = 0.20   // < 20% of width → probably face-on
        if widthB < widthA * minDepthRatio {
            depthEstimate = max(backgroundDepth(frame: frame, yoloBox: yoloBox, dFront: dFront),
                                0.05)
        } else {
            depthEstimate = max(widthB, 0.05)
        }

        // 6. Build gravity-aligned world transform.
        let pc = (pMin+pMax)/2, qc = (qMin+qMax)/2
        let cx = xMean + pc*axX + qc*pxX
        let cz = zMean + pc*axZ + qc*pxZ
        let center = simd_float3(cx, (yMin+yMax)/2, cz)

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
            size: simd_float3(widthA, height, depthEstimate),
            yaw: yaw,
            confidence: yoloBox.score,
            worldTransform: worldTransform,
            label: yoloBox.label
        )
    }

    // MARK: - Depth map unprojection

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

        let side = min(bufW, bufH)
        let ox = (bufW-side)/2, oy = (bufH-side)/2
        let bxRaw0 = yoloBox.xmin/640*side+ox, byRaw0 = yoloBox.ymin/640*side+oy
        let bxRaw1 = yoloBox.xmax/640*side+ox, byRaw1 = yoloBox.ymax/640*side+oy
        // Shrink bbox by 8% on each side to avoid edge spill (floor/wall at borders)
        let shrinkX = (bxRaw1-bxRaw0)*0.08, shrinkY = (byRaw1-byRaw0)*0.08
        let bx0 = bxRaw0+shrinkX, by0 = byRaw0+shrinkY
        let bx1 = bxRaw1-shrinkX, by1 = byRaw1-shrinkY

        let px0 = max(0, Int(bx0/bufW*Float(dW))), py0 = max(0, Int(by0/bufH*Float(dH)))
        let px1 = min(dW-1, Int(bx1/bufW*Float(dW))), py1 = min(dH-1, Int(by1/bufH*Float(dH)))
        guard px1 > px0, py1 > py0 else { return [] }

        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }
        let base = CVPixelBufferGetBaseAddress(depthBuffer)!
        let rb = CVPixelBufferGetBytesPerRow(depthBuffer)

        // Pass 1: front depth (10th percentile).
        var allD: [Float] = []
        for py in py0...py1 {
            let row = base.advanced(by: py*rb).assumingMemoryBound(to: Float32.self)
            for px in px0...px1 { let d = row[px]; if d > 0.1 && d < 7 { allD.append(d) } }
        }
        guard allD.count >= 10 else { return [] }
        allD.sort()
        dFront = allD[allD.count/10]
        let dCut = dFront + 0.30   // include up to 30 cm behind front face

        // Pass 2: unproject surface points.
        let intr = frame.camera.intrinsics
        let fx = intr[0][0], fy = intr[1][1], cx = intr[2][0], cy = intr[2][1]
        let T = frame.camera.transform
        var pts: [simd_float3] = []

        for py in py0...py1 {
            let row = base.advanced(by: py*rb).assumingMemoryBound(to: Float32.self)
            for px in px0...px1 {
                let d = row[px]
                guard d >= dFront-0.03, d <= dCut else { continue }
                let ix = Float(px)/Float(dW)*bufW, iy = Float(py)/Float(dH)*bufH
                let cam = simd_float4((ix-cx)/fx*d, (iy-cy)/fy*d, -d, 1)
                let w = T*cam
                pts.append(simd_float3(w.x, w.y, w.z)/w.w)
            }
        }
        return pts
    }

    // MARK: - Background depth (fallback for face-on views)

    private static func backgroundDepth(
        frame: ARFrame, yoloBox: YOLOBox, dFront: Float
    ) -> Float {

        guard let depthBuffer = frame.sceneDepth?.depthMap else { return 0.25 }

        let bufW = Float(CVPixelBufferGetWidth(frame.capturedImage))
        let bufH = Float(CVPixelBufferGetHeight(frame.capturedImage))
        let dW = CVPixelBufferGetWidth(depthBuffer)
        let dH = CVPixelBufferGetHeight(depthBuffer)

        let side = min(bufW, bufH)
        let ox = (bufW-side)/2, oy = (bufH-side)/2
        let bx0 = yoloBox.xmin/640*side+ox, by0 = yoloBox.ymin/640*side+oy
        let bx1 = yoloBox.xmax/640*side+ox, by1 = yoloBox.ymax/640*side+oy
        let pw = bx1-bx0, ph = by1-by0

        let ex0 = max(0, Int((bx0-pw*0.3)/bufW*Float(dW)))
        let ex1 = min(dW-1, Int((bx1+pw*0.3)/bufW*Float(dW)))
        let ey0 = max(0, Int((by0-ph*0.1)/bufH*Float(dH)))
        let ey1 = min(dH-1, Int((by1+ph*0.1)/bufH*Float(dH)))
        let ix0 = Int(bx0/bufW*Float(dW)), ix1 = Int(bx1/bufW*Float(dW))
        let iy0 = Int(by0/bufH*Float(dH)), iy1 = Int(by1/bufH*Float(dH))

        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }
        let base = CVPixelBufferGetBaseAddress(depthBuffer)!
        let rb = CVPixelBufferGetBytesPerRow(depthBuffer)

        var bgDs: [Float] = []
        for py in ey0...ey1 {
            let row = base.advanced(by: py*rb).assumingMemoryBound(to: Float32.self)
            for px in ex0...ex1 {
                if px >= ix0 && px <= ix1 && py >= iy0 && py <= iy1 { continue }
                let d = row[px]
                if d > dFront+0.05 && d < 7 { bgDs.append(d) }
            }
        }
        guard !bgDs.isEmpty else { return 0.25 }
        bgDs.sort()
        return bgDs[bgDs.count/2] - dFront
    }
}
