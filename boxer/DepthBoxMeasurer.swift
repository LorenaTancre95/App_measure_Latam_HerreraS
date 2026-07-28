import ARKit
import simd

/// Tap-to-select measurement using LiDAR depth flood fill with inline floor rejection.
///
/// The flood fill follows depth discontinuities (box edge vs wall/other box) AND
/// rejects pixels whose 3D world Y is at floor height. This combination handles the
/// common case where camera is nearly level and box/floor have similar LiDAR depth.
struct DepthBoxMeasurer {

    static func measure(
        frame: ARFrame,
        tapPoint: CGPoint,
        viewportSize: CGSize
    ) -> Detection3D? {

        guard let depthBuffer = frame.sceneDepth?.depthMap else { return nil }

        let pixelBuffer = frame.capturedImage
        let bufW = Float(CVPixelBufferGetWidth(pixelBuffer))
        let bufH = Float(CVPixelBufferGetHeight(pixelBuffer))

        let dW = CVPixelBufferGetWidth(depthBuffer)
        let dH = CVPixelBufferGetHeight(depthBuffer)

        let intr = frame.camera.intrinsics
        let fx = intr[0][0], fy = intr[1][1], cx = intr[2][0], cy = intr[2][1]
        let T  = frame.camera.transform

        // 1. Portrait tap → depth map pixel
        let displayT  = frame.displayTransform(for: .portrait, viewportSize: viewportSize)
        let invertedT = displayT.inverted()
        let normTap   = CGPoint(x: tapPoint.x / viewportSize.width,
                                y: tapPoint.y / viewportSize.height)
        let normImg   = normTap.applying(invertedT)
        let tapDX     = max(0, min(dW - 1, Int(Float(normImg.x) * Float(dW))))
        let tapDY     = max(0, min(dH - 1, Int(Float(normImg.y) * Float(dH))))

        // Floor Y from ARPlaneAnchor — used inline during flood fill to stop at floor.
        // IMPORTANT: apuntá al suelo antes de medir para que ARKit lo detecte.
        let anchorFloorY: Float? = frame.anchors
            .compactMap { $0 as? ARPlaneAnchor }
            .filter { $0.alignment == .horizontal }
            .map { Float($0.transform.columns.3.y) }.min()

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

        // 2. Flood fill with dual filter:
        //    a) Depth: tapD−25cm (top/side face) to tapD+5cm (surface variation only)
        //    b) World Y: stop if pixel projects to floor height (even at similar depth)
        let maxPts = dW * dH / 5        // bail if >20% of depth map (runaway)
        var visited = [Bool](repeating: false, count: dW * dH)
        var stack   = [(Int, Int)]()
        stack.reserveCapacity(512)
        stack.append((tapDX, tapDY))

        var pts = [simd_float3]()
        pts.reserveCapacity(500)

        while !stack.isEmpty {
            let (px, py) = stack.removeLast()
            guard px >= 0, px < dW, py >= 0, py < dH else { continue }
            let flatIdx = py * dW + px
            guard !visited[flatIdx] else { continue }
            visited[flatIdx] = true

            let d = depthAt(px, py)
            guard d > 0.10, d < 8.0,
                  d >= tapD - 0.25,
                  d <= tapD + 0.05 else { continue }

            let p3d = unproject(px, py, d)

            // Floor barrier: if this pixel is at floor height, mark visited and stop
            // expanding — acts as a wall that prevents the fill from reaching the floor.
            if let floorY = anchorFloorY, p3d.y <= floorY + 0.07 { continue }

            pts.append(p3d)
            guard pts.count < maxPts else { break }

            stack.append((px + 1, py))
            stack.append((px - 1, py))
            stack.append((px, py + 1))
            stack.append((px, py - 1))
        }
        CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly)

        guard pts.count >= 20, pts.count < maxPts else { return nil }

        // 3. Floor removal (safety net for pixels at boundary)
        let rawYMin = pts.map { $0.y }.min()!
        let floorY: Float
        if let a = anchorFloorY, a <= rawYMin + 0.10 { floorY = a } else { floorY = rawYMin }

        let above = pts.filter { $0.y > floorY + 0.06 }
        guard above.count >= 20 else { return nil }

        // 4. PCA OBB in XZ
        func pct(_ vals: [Float], _ p: Float) -> Float {
            let s = vals.sorted()
            return s[max(0, Int(Float(s.count) * p))]
        }
        let yMax   = pct(above.map { $0.y }, 0.97)
        let height = yMax - floorY
        guard height > 0.03, height < 3.0 else { return nil }

        let (width, depth, center2D, axis1) = obbXZ(points: above)
        guard width > 0.03, width < 4.0,
              depth > 0.03, depth < 4.0 else { return nil }

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
