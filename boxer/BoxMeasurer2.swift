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

        // 4. Minimum Bounding Rectangle (MBR) on XZ footprint → yaw + dimensions.
        // Use only the top 30% of the cloud (highest Y) to focus on the flat top face:
        // it's the cleanest LiDAR surface (no floor/background contamination) and its
        // XZ projection is a clean rectangle aligned with the box edges.
        let topThresh = yMax - height * 0.30
        let topFacePts = pts.filter { $0.y > topThresh }
        let xzSrc = topFacePts.count >= 15 ? topFacePts : pts
        let xzPts = xzSrc.map { SIMD2<Float>($0.x, $0.z) }
        let hull = convexHull2D(xzPts)
        var bestAngle: Float = 0
        if hull.count >= 3 {
            var bestArea: Float = .infinity
            for i in 0..<hull.count {
                let a = hull[i], b = hull[(i+1) % hull.count]
                let angle = atan2(b.y - a.y, b.x - a.x)
                let ca = cos(-angle), sa = sin(-angle)
                var minU = Float.infinity, maxU = -Float.infinity
                var minV = Float.infinity, maxV = -Float.infinity
                for p in xzPts {
                    let u = p.x*ca - p.y*sa
                    let v = p.x*sa + p.y*ca
                    minU = min(minU, u); maxU = max(maxU, u)
                    minV = min(minV, v); maxV = max(maxV, v)
                }
                let area = (maxU - minU) * (maxV - minV)
                if area < bestArea { bestArea = area; bestAngle = angle }
            }
        }

        // Project pts onto MBR axes; use 1%-99% percentiles to suppress edge noise.
        let mbrCa = cos(-bestAngle), mbrSa = sin(-bestAngle)
        var uVals: [Float] = [], vVals: [Float] = []
        for p in pts {
            uVals.append(p.x*mbrCa - p.z*mbrSa)
            vVals.append(p.x*mbrSa + p.z*mbrCa)
        }
        uVals.sort(); vVals.sort()
        let lo = Int(Float(uVals.count)*0.01), hi = Int(Float(uVals.count)*0.99)
        let widthA = uVals[hi] - uVals[lo]
        let widthB = vVals[hi] - vVals[lo]
        guard widthA > 0.03, widthA < 2.0 else { return nil }

        // 5. If nearly face-on, widthB is the noise band of the front face —
        //    fall back to background sampling for the depth dimension.
        let depthEstimate: Float
        let minDepthRatio: Float = 0.20
        if widthB < widthA * minDepthRatio {
            depthEstimate = max(backgroundDepth(frame: frame, yoloBox: yoloBox, dFront: dFront), 0.05)
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

        let yaw = bestAngle
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

    // MARK: - Convex hull (Andrew's monotone chain, O(n log n))

    private static func convexHull2D(_ points: [SIMD2<Float>]) -> [SIMD2<Float>] {
        guard points.count >= 3 else { return points }
        let sorted = points.sorted { $0.x < $1.x || ($0.x == $1.x && $0.y < $1.y) }
        func cross(_ O: SIMD2<Float>, _ A: SIMD2<Float>, _ B: SIMD2<Float>) -> Float {
            (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x)
        }
        var lower: [SIMD2<Float>] = []
        for p in sorted {
            while lower.count >= 2 && cross(lower[lower.count-2], lower[lower.count-1], p) <= 0 {
                lower.removeLast()
            }
            lower.append(p)
        }
        var upper: [SIMD2<Float>] = []
        for p in sorted.reversed() {
            while upper.count >= 2 && cross(upper[upper.count-2], upper[upper.count-1], p) <= 0 {
                upper.removeLast()
            }
            upper.append(p)
        }
        lower.removeLast(); upper.removeLast()
        return lower + upper
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
