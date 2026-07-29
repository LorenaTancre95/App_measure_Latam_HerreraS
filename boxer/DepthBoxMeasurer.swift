import ARKit
import simd

/// Tap-to-select measurement: two-phase LiDAR approach.
///
/// Phase 1 — tight flood fill (±4 cm):
///   Finds the tapped surface in depth-image space without leaking to the floor.
///   At any camera angle where the camera is ≥30 cm above the floor, the floor at
///   the base of a box is ≥5 cm deeper than the box face → fill stops there naturally.
///   Result: 2-D bounding box (depth-map pixels) of the visible box face.
///
/// Phase 2 — full collection within that bounding box:
///   Samples ALL LiDAR pixels inside the Phase-1 bbox (expanded by a few pixels).
///   Depth range: tapD−30 cm (top/side face, closer) to tapD+20 cm (slight box depth).
///   Height filter: excludes pixels whose 3-D Y is at floor level → no floor bleed.
///   Gives the full visible point cloud (front + top + sides).
///
/// OBB: PCA in XZ plane → oriented bounding box handles rotated boxes correctly.
struct DepthBoxMeasurer {

    static func measure(
        frame: ARFrame,
        tapPoint: CGPoint,
        viewportSize: CGSize,
        debugOut: UnsafeMutablePointer<String>? = nil
    ) -> Detection3D? {

        guard let depthBuffer = frame.sceneDepth?.depthMap else { return nil }

        let pixelBuffer = frame.capturedImage
        let bufW = Float(CVPixelBufferGetWidth(pixelBuffer))
        let bufH = Float(CVPixelBufferGetHeight(pixelBuffer))
        let dW   = CVPixelBufferGetWidth(depthBuffer)
        let dH   = CVPixelBufferGetHeight(depthBuffer)

        let intr = frame.camera.intrinsics
        let fx = intr[0][0], fy = intr[1][1], cx = intr[2][0], cy = intr[2][1]
        let T  = frame.camera.transform

        // Portrait tap → depth map pixel
        let displayT  = frame.displayTransform(for: .portrait, viewportSize: viewportSize)
        let invertedT = displayT.inverted()
        let normTap   = CGPoint(x: tapPoint.x / viewportSize.width,
                                y: tapPoint.y / viewportSize.height)
        let normImg   = normTap.applying(invertedT)
        let tapDX     = max(0, min(dW - 1, Int(Float(normImg.x) * Float(dW))))
        let tapDY     = max(0, min(dH - 1, Int(Float(normImg.y) * Float(dH))))

        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        let rb   = CVPixelBufferGetBytesPerRow(depthBuffer)
        let base = CVPixelBufferGetBaseAddress(depthBuffer)!

        func depthAt(_ px: Int, _ py: Int) -> Float {
            base.advanced(by: py * rb).assumingMemoryBound(to: Float32.self)[px]
        }
        func unproject(_ px: Int, _ py: Int, _ d: Float) -> simd_float3 {
            let ix  = Float(px) / Float(dW) * bufW
            let iy  = Float(py) / Float(dH) * bufH
            let cam = simd_float4((ix - cx) / fx * d, (iy - cy) / fy * d, -d, 1)
            let w   = T * cam
            return simd_float3(w.x, w.y, w.z) / w.w
        }

        let tapD = depthAt(tapDX, tapDY)
        guard tapD > 0.10, tapD < 8.0 else {
            CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly)
            return nil
        }

        // ── PHASE 1: tight flood fill ─────────────────────────────────────────
        // ±4 cm keeps the fill on the tapped surface and stops at depth edges
        // (box→floor or box→wall discontinuity is always >5 cm when camera ≥30 cm up).
        let tightTol: Float = 0.04
        var visited  = [Bool](repeating: false, count: dW * dH)
        var stack    = [(Int, Int)]()
        stack.reserveCapacity(256)
        stack.append((tapDX, tapDY))

        var minDX = dW, maxDX = 0, minDY = dH, maxDY = 0
        var faceCount = 0

        while !stack.isEmpty {
            let (px, py) = stack.removeLast()
            guard px >= 0, px < dW, py >= 0, py < dH else { continue }
            let idx = py * dW + px
            guard !visited[idx] else { continue }
            visited[idx] = true

            let d = depthAt(px, py)
            guard d > 0.10, d < 8.0, abs(d - tapD) < tightTol else { continue }

            faceCount += 1
            if px < minDX { minDX = px }; if px > maxDX { maxDX = px }
            if py < minDY { minDY = py }; if py > maxDY { maxDY = py }

            stack.append((px + 1, py)); stack.append((px - 1, py))
            stack.append((px, py + 1)); stack.append((px, py - 1))
        }

        guard faceCount >= 15 else {
            debugOut?.pointee = "tapD=\(String(format:"%.2f",tapD))m face=\(faceCount)<15"
            CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly)
            return nil
        }

        // ── PHASE 2: full collection inside the Phase-1 bounding box ─────────
        let anchorFloorY: Float? = frame.anchors
            .compactMap { $0 as? ARPlaneAnchor }
            .filter { $0.alignment == .horizontal }
            .map { Float($0.transform.columns.3.y) }.min()

        // Expand upward by full face height to include the top face of the box.
        // Phase-1 only found the front face; the top face is directly above it in the
        // depth image and is shallower by sin(camera_angle) × box_height — which fits
        // within the tapD-0.30 depth window used in Phase-2.
        let faceH  = maxDY - minDY          // front-face pixel height in depth map
        let margin = 4
        let pxLo = max(0, minDX - margin),    pxHi = min(dW - 1, maxDX + margin)
        let pyLo = max(0, minDY - faceH - margin)   // look UP by one full face height
        let pyHi = min(dH - 1, maxDY + margin)

        var pts = [simd_float3]()
        pts.reserveCapacity((pxHi - pxLo + 1) * (pyHi - pyLo + 1))

        for py in pyLo...pyHi {
            for px in pxLo...pxHi {
                let d = depthAt(px, py)
                // Depth window: up to 30 cm closer (top face) and 20 cm deeper (box depth)
                guard d > 0.10, d < 8.0,
                      d >= tapD - 0.30,
                      d <= tapD + 0.20 else { continue }

                let p3d = unproject(px, py, d)

                // Height filter: drop pixels at floor level
                if let floorY = anchorFloorY {
                    guard p3d.y > floorY + 0.06 else { continue }
                }
                pts.append(p3d)
            }
        }
        CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly)

        guard pts.count >= 20 else {
            debugOut?.pointee = "tapD=\(String(format:"%.2f",tapD))m face=\(faceCount) pts=\(pts.count)<20"
            return nil
        }

        // Floor removal (fallback when no ARPlane anchor)
        let rawYMin = pts.map { $0.y }.min()!
        let rawYMax = pts.map { $0.y }.max()!
        let floorY: Float
        let floorSrc: String
        if let a = anchorFloorY, a <= rawYMin + 0.10 {
            floorY = a; floorSrc = "plane"
        } else {
            floorY = rawYMin; floorSrc = "raw"
        }
        let above = pts.filter { $0.y > floorY + 0.05 }
        guard above.count >= 20 else {
            debugOut?.pointee = "tapD=\(String(format:"%.2f",tapD))m floorY=\(String(format:"%.2f",floorY))(\(floorSrc)) above=\(above.count)<20"
            return nil
        }

        // ── PCA OBB in XZ ────────────────────────────────────────────────────
        func pct(_ vals: [Float], _ p: Float) -> Float {
            let s = vals.sorted()
            return s[max(0, Int(Float(s.count) * p))]
        }
        let yMax   = pct(above.map { $0.y }, 0.97)
        let height = yMax - floorY
        guard height > 0.03, height < 3.0 else {
            debugOut?.pointee = "floorY=\(String(format:"%.2f",floorY))(\(floorSrc)) yMax=\(String(format:"%.2f",yMax)) h=\(String(format:"%.0f",height*100))cm OOB"
            return nil
        }

        let (width, depth, center2D, axis1) = obbXZ(points: above)
        guard width > 0.03, width < 4.0,
              depth > 0.03, depth < 4.0 else {
            debugOut?.pointee = "h=\(String(format:"%.0f",height*100))cm w=\(String(format:"%.0f",width*100)) d=\(String(format:"%.0f",depth*100)) OOB"
            return nil
        }
        debugOut?.pointee = "D=\(String(format:"%.2f",tapD))m fl=\(String(format:"%.2f",floorY))(\(floorSrc)) rawY[\(String(format:"%.2f",rawYMin)),\(String(format:"%.2f",rawYMax))] pts=\(pts.count)/\(above.count)"

        let centerY = (floorY + yMax) / 2
        let center  = simd_float3(center2D.x, centerY, center2D.y)

        let yaw  = atan2(axis1.y, axis1.x)
        let cosY = cos(yaw), sinY = sin(yaw)
        let worldTransform = simd_float4x4(
            simd_float4( cosY, 0, sinY, 0),
            simd_float4(    0, 1,    0, 0),
            simd_float4(-sinY, 0, cosY, 0),
            simd_float4(center.x, center.y, center.z, 1)
        )

        return Detection3D(center: center,
                           size: simd_float3(width, height, depth),
                           yaw: yaw,
                           confidence: 1.0,
                           worldTransform: worldTransform,
                           label: "caja")
    }

    // MARK: - PCA OBB in XZ

    private static func obbXZ(
        points: [simd_float3]
    ) -> (width: Float, depth: Float, center: simd_float2, axis1: simd_float2) {

        let n  = Float(points.count)
        let cx = points.map { $0.x }.reduce(0, +) / n
        let cz = points.map { $0.z }.reduce(0, +) / n

        var cxx: Float = 0, cxz: Float = 0, czz: Float = 0
        for p in points {
            let dx = p.x - cx, dz = p.z - cz
            cxx += dx * dx; cxz += dx * dz; czz += dz * dz
        }
        cxx /= n; cxz /= n; czz /= n

        let trace   = cxx + czz
        let det     = cxx * czz - cxz * cxz
        let disc    = max(0, trace * trace / 4 - det)
        let lambda1 = trace / 2 + sqrt(disc)

        var axis1: simd_float2
        if abs(cxz) > 1e-6 {
            axis1 = simd_normalize(simd_float2(lambda1 - czz, cxz))
        } else {
            axis1 = cxx >= czz ? simd_float2(1, 0) : simd_float2(0, 1)
        }
        let axis2 = simd_float2(-axis1.y, axis1.x)

        var min1: Float = .infinity, max1: Float = -.infinity
        var min2: Float = .infinity, max2: Float = -.infinity
        for p in points {
            let dx = p.x - cx, dz = p.z - cz
            let p1 = dx * axis1.x + dz * axis1.y
            let p2 = dx * axis2.x + dz * axis2.y
            min1 = min(min1, p1); max1 = max(max1, p1)
            min2 = min(min2, p2); max2 = max(max2, p2)
        }

        return (max1 - min1, max2 - min2, simd_float2(cx, cz), axis1)
    }
}
