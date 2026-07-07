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
        // rawYMin ≈ floor level (box sits on floor; lowest visible point in depth slice).
        let rawYMin = raw.map { $0.y }.min()!

        let anchorFloorY: Float? = frame.anchors
            .compactMap { $0 as? ARPlaneAnchor }
            .filter { $0.alignment == .horizontal }
            .map { Float($0.transform.columns.3.y) }.min()

        // Use ARPlaneAnchor only if it's within 8 cm of rawYMin.
        // If the anchor is much higher (e.g., ARKit detected the box top face as a
        // horizontal plane), rawYMin is more reliable.
        let reliableFloorY: Float
        if let anchor = anchorFloorY, anchor <= rawYMin + 0.08 {
            reliableFloorY = anchor
        } else {
            reliableFloorY = rawYMin
        }

        let floorThresh = reliableFloorY + 0.04
        let pts = raw.filter { $0.y > floorThresh }
        guard pts.count >= minPoints else { return nil }

        // 3. Height = top of cloud (97th percentile Y) minus floor level.
        var ySorted = pts.map { $0.y }; ySorted.sort()
        let yTop = ySorted[Int(Float(ySorted.count)*0.97)]
        let yMin = reliableFloorY
        let yMax = yTop
        let height = yMax - yMin
        guard height > 0.02 else { return nil }

        // 4. Orientation from 2D YOLO bbox horizontal direction.
        //    Unproject left/right bbox edges at dFront → width axis in 3D XZ.
        //    This is stable even when box dimensions are similar (PCA flips axes).
        let bufWo = Float(CVPixelBufferGetWidth(frame.capturedImage))
        let bufHo = Float(CVPixelBufferGetHeight(frame.capturedImage))
        let sideo = min(bufWo, bufHo)
        let oxo = (bufWo-sideo)/2, oyo = (bufHo-sideo)/2
        let midVo = (yoloBox.ymin+yoloBox.ymax)/2 / 640 * sideo + oyo
        let leftUo  = yoloBox.xmin / 640 * sideo + oxo
        let rightUo = yoloBox.xmax / 640 * sideo + oxo
        let intro = frame.camera.intrinsics
        let fxo = intro[0][0], fyo = intro[1][1], cxo = intro[2][0], cyo = intro[2][1]
        func edge3D(_ u: Float, _ v: Float) -> simd_float3 {
            let cam = simd_float4((u-cxo)/fxo*dFront, (v-cyo)/fyo*dFront, -dFront, 1)
            let w = frame.camera.transform * cam; return simd_float3(w.x/w.w, 0, w.z/w.w)
        }
        let wL3 = edge3D(leftUo, midVo), wR3 = edge3D(rightUo, midVo)
        guard simd_distance(wL3, wR3) > 0.01 else { return nil }
        let widthVec = simd_normalize(wR3 - wL3)
        let axX = widthVec.x, axZ = widthVec.z
        let pxX = -axZ, pxZ = axX   // depth axis (perpendicular to width)

        // Project floor-removed pts onto 2D-anchored axes to measure sizes.
        let n = Float(pts.count)
        let xMean = pts.map { $0.x }.reduce(0, +) / n
        let zMean = pts.map { $0.z }.reduce(0, +) / n
        var pVals: [Float] = []; var qVals: [Float] = []
        for p in pts {
            let dx = p.x - xMean, dz = p.z - zMean
            pVals.append(dx*axX + dz*axZ)
            qVals.append(dx*pxX + dz*pxZ)
        }
        pVals.sort(); qVals.sort()
        let lo = Int(Float(pVals.count)*0.01), hi = Int(Float(pVals.count)*0.99)
        let pMin = pVals[lo], pMax = pVals[hi]
        let qMin = qVals[lo], qMax = qVals[hi]

        let widthA = pMax - pMin
        let widthB = qMax - qMin
        guard widthA > 0.02 else { return nil }

        // 5. If box is viewed nearly face-on, widthB is the noise band of the
        //    front face depth — fall back to background sampling for depth.
        // Sanity check: widthA > 1.5m means background contamination, bail out.
        guard widthA < 1.5 else { return nil }

        let depthEstimate: Float
        let minDepthRatio: Float = 0.20   // < 20% of width → probably face-on
        if widthB < widthA * minDepthRatio {
            let bg = backgroundDepth(frame: frame, yoloBox: yoloBox, dFront: dFront)
            // Cap background depth consistent with dCut (max 60 cm, max 2× width).
            depthEstimate = max(min(bg, min(widthA * 2.0, 0.60)), 0.05)
        } else {
            depthEstimate = max(widthB, 0.05)
        }

        // 6. Build gravity-aligned world transform.
        // Center XZ: project the YOLO bbox center at mid-depth using camera intrinsics.
        // This avoids PCA centroid bias (near face has more LiDAR points → centroid shifts forward).
        let bufWc = Float(CVPixelBufferGetWidth(frame.capturedImage))
        let bufHc = Float(CVPixelBufferGetHeight(frame.capturedImage))
        let sidec = min(bufWc, bufHc)
        let oxc = (bufWc-sidec)/2, oyc = (bufHc-sidec)/2
        let imgMidX = (yoloBox.xmin+yoloBox.xmax)/2 / 640 * sidec + oxc
        let imgMidY = (yoloBox.ymin+yoloBox.ymax)/2 / 640 * sidec + oyc
        let intr = frame.camera.intrinsics
        let midD = dFront + depthEstimate/2
        let camC = simd_float4(
            (imgMidX - intr[2][0]) / intr[0][0] * midD,
            (imgMidY - intr[2][1]) / intr[1][1] * midD,
            -midD, 1)
        let wc = frame.camera.transform * camC
        let center = simd_float3(wc.x/wc.w, (yMin+yMax)/2, wc.z/wc.w)

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
        // Shrink bbox by 5% on each side to avoid edge spill (floor/wall at borders)
        let shrinkX = (bxRaw1-bxRaw0)*0.05, shrinkY = (byRaw1-byRaw0)*0.05
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

        // Adaptive depth cutoff: at most 2× the YOLO bbox projected width, capped at 60 cm.
        // min(..., 0.60) prevents large/noisy YOLO bboxes from letting wall points in.
        // max(..., 0.25) ensures at least 25 cm depth is always captured.
        let intr = frame.camera.intrinsics
        let fx = intr[0][0], fy = intr[1][1], cx = intr[2][0], cy = intr[2][1]
        let bboxWidthPx = (yoloBox.xmax - yoloBox.xmin) / 640 * side
        let projectedWidth = bboxWidthPx / fx * dFront
        let dCut = dFront + min(max(projectedWidth * 2.0, 0.25), 0.60)

        // Pass 2: unproject surface points.
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
