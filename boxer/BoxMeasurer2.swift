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
        // Sanity: height must be between 3 cm and 1.5 m.
        guard height > 0.03, height < 1.5 else { return nil }

        // 4. Orientation: PCA on XZ footprint, disambiguated by 2D bbox direction.
        //    PCA gives accurate axis sizes from real 3D geometry.
        //    The 2D bbox horizontal direction (unproject left/right edges) tells us
        //    which PCA axis is "width" vs "depth" — stable for any box orientation.
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
        let disc = sqrt(max(0, trace*trace/4 - (c00*c11 - c01*c01)))
        let lam1 = trace/2 + disc

        var ax1X: Float, ax1Z: Float
        if abs(c01) > 1e-6 {
            let e = lam1 - c11; let len = sqrt(e*e + c01*c01)
            ax1X = e/len; ax1Z = c01/len
        } else {
            ax1X = c00 >= c11 ? 1 : 0; ax1Z = c00 >= c11 ? 0 : 1
        }
        // PCA secondary (perpendicular): (-ax1Z, ax1X)

        // 2D bbox LONGER-dimension direction → disambiguate which PCA axis is "width".
        // Use the longer 2D bbox edge (horizontal if wide, vertical if tall) so the
        // disambiguation works regardless of how the box is oriented in the image.
        let bufWo = Float(CVPixelBufferGetWidth(frame.capturedImage))
        let bufHo = Float(CVPixelBufferGetHeight(frame.capturedImage))
        let sideo = min(bufWo, bufHo)
        let oxo = (bufWo-sideo)/2, oyo = (bufHo-sideo)/2
        let midUo = (yoloBox.xmin+yoloBox.xmax)/2 / 640 * sideo + oxo
        let midVo = (yoloBox.ymin+yoloBox.ymax)/2 / 640 * sideo + oyo
        let intro = frame.camera.intrinsics
        let fxo = intro[0][0], fyo = intro[1][1], cxo = intro[2][0], cyo = intro[2][1]
        func edge3D(_ u: Float, _ v: Float) -> simd_float3 {
            let cam = simd_float4((u-cxo)/fxo*dFront, (v-cyo)/fyo*dFront, -dFront, 1)
            let w = frame.camera.transform * cam; return simd_float3(w.x/w.w, 0, w.z/w.w)
        }
        let bbox2DW = yoloBox.xmax - yoloBox.xmin
        let bbox2DH = yoloBox.ymax - yoloBox.ymin
        let bboxVec: simd_float3 = {
            if bbox2DW >= bbox2DH {
                // Wider than tall: left→right edge gives the primary horizontal axis
                let pA = edge3D(yoloBox.xmin/640*sideo+oxo, midVo)
                let pB = edge3D(yoloBox.xmax/640*sideo+oxo, midVo)
                return simd_distance(pA, pB) > 0.01 ? simd_normalize(pB - pA) : simd_float3(1,0,0)
            } else {
                // Taller than wide: top→bottom edge gives the primary vertical axis
                let pA = edge3D(midUo, yoloBox.ymin/640*sideo+oyo)
                let pB = edge3D(midUo, yoloBox.ymax/640*sideo+oyo)
                return simd_distance(pA, pB) > 0.01 ? simd_normalize(pB - pA) : simd_float3(0,0,1)
            }
        }()

        // Dot products: alignment of each PCA axis with the 2D long-dimension direction.
        let dot1 = abs(ax1X * bboxVec.x + ax1Z * bboxVec.z)    // primary vs bbox long dim
        let dot2 = abs(-ax1Z * bboxVec.x + ax1X * bboxVec.z)   // secondary vs bbox long dim
        // Switch to secondary only if it's clearly better (15% margin) — avoids jitter
        // when both are similar (box at 45°), which keeps the stable PCA primary as width.
        let axX: Float = dot2 > dot1 * 1.15 ? -ax1Z : ax1X
        let axZ: Float = dot2 > dot1 * 1.15 ?  ax1X : ax1Z
        let pxX: Float = -axZ
        let pxZ: Float =  axX

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
        // Sanity: each horizontal dimension must be between 3 cm and 2 m.
        guard widthA > 0.03, widthA < 2.0 else { return nil }

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
        guard depthEstimate < 2.0 else { return nil }

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

        // Lock confidence map: only use medium/high confidence depth pixels.
        // Low-confidence pixels (noisy at startup) are skipped to improve first-shot accuracy.
        let confBuffer = frame.sceneDepth?.confidenceMap
        if let c = confBuffer { CVPixelBufferLockBaseAddress(c, .readOnly) }
        defer { if let c = confBuffer { CVPixelBufferUnlockBaseAddress(c, .readOnly) } }
        let confBase = confBuffer.flatMap { CVPixelBufferGetBaseAddress($0) }
        let confRB = confBuffer.map { CVPixelBufferGetBytesPerRow($0) } ?? 0

        // Pass 1: front depth (10th percentile) from medium/high-confidence pixels only.
        var allD: [Float] = []
        for py in py0...py1 {
            let row = base.advanced(by: py*rb).assumingMemoryBound(to: Float32.self)
            for px in px0...px1 {
                if let cb = confBase {
                    let conf = cb.advanced(by: py*confRB + px).assumingMemoryBound(to: UInt8.self).pointee
                    guard conf >= 1 else { continue }  // skip ARConfidenceLevel.low
                }
                let d = row[px]; if d > 0.1 && d < 7 { allD.append(d) }
            }
        }
        guard allD.count >= 10 else { return [] }
        allD.sort()
        dFront = allD[allD.count/10]
        let dCut = dFront + 0.40   // include up to 40 cm behind front face

        // Pass 2: unproject medium/high-confidence surface points.
        let intr = frame.camera.intrinsics
        let fx = intr[0][0], fy = intr[1][1], cx = intr[2][0], cy = intr[2][1]
        let T = frame.camera.transform
        var pts: [simd_float3] = []

        for py in py0...py1 {
            let row = base.advanced(by: py*rb).assumingMemoryBound(to: Float32.self)
            for px in px0...px1 {
                if let cb = confBase {
                    let conf = cb.advanced(by: py*confRB + px).assumingMemoryBound(to: UInt8.self).pointee
                    guard conf >= 1 else { continue }
                }
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
