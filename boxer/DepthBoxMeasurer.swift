import ARKit
import simd

/// Tap-to-select measurement using LiDAR depth flood fill.
///
/// Flow:
///   1. Portrait tap → depth map pixel via ARKit displayTransform
///   2. Flood fill from that pixel in depth space (±10 cm tolerance)
///      — stops naturally at depth discontinuities (box edge vs floor/wall)
///      — works even when box and floor have the same color
///   3. Unproject flood-fill pixels to 3D world space
///   4. Floor removal via ARPlaneAnchor
///   5. PCA OBB in XZ plane
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

        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        let rb   = CVPixelBufferGetBytesPerRow(depthBuffer)
        let base = CVPixelBufferGetBaseAddress(depthBuffer)!

        func depthAt(_ px: Int, _ py: Int) -> Float {
            base.advanced(by: py * rb).assumingMemoryBound(to: Float32.self)[px]
        }

        let tapD = depthAt(tapDX, tapDY)
        guard tapD > 0.10, tapD < 8.0 else {
            CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly)
            return nil
        }

        // 2. Flood fill in depth image space
        // Depth discontinuities at box edges stop the fill — robust to color similarity.
        let maxRegion = dW * dH * 2 / 5        // bail if >40% of depth map (runaway)
        var visited   = [Bool](repeating: false, count: dW * dH)
        var stack     = [(Int, Int)]()
        stack.reserveCapacity(512)
        stack.append((tapDX, tapDY))

        var region = [(px: Int, py: Int, d: Float)]()
        region.reserveCapacity(500)

        while !stack.isEmpty {
            let (px, py) = stack.removeLast()
            guard px >= 0, px < dW, py >= 0, py < dH else { continue }
            let flatIdx = py * dW + px
            guard !visited[flatIdx] else { continue }
            visited[flatIdx] = true

            let d = depthAt(px, py)
            // Asymmetric tolerance: capture surfaces closer to camera (top/side face of box)
            // but stop quickly when depth increases (floor, wall behind box).
            // tapD − 0.25: 25 cm closer captures top face at most viewing angles.
            // tapD + 0.08: 8 cm deeper allows natural surface variation without bleeding into floor.
            guard d > 0.10, d < 8.0,
                  d >= tapD - 0.25,
                  d <= tapD + 0.08 else { continue }

            region.append((px, py, d))
            guard region.count < maxRegion else { break }

            stack.append((px + 1, py))
            stack.append((px - 1, py))
            stack.append((px, py + 1))
            stack.append((px, py - 1))
        }

        // 3. Unproject to world space
        var pts = [simd_float3]()
        pts.reserveCapacity(region.count)
        for (px, py, d) in region {
            let ix  = Float(px) / Float(dW) * bufW
            let iy  = Float(py) / Float(dH) * bufH
            let cam = simd_float4((ix - cx) / fx * d, (iy - cy) / fy * d, -d, 1)
            let w   = T * cam
            pts.append(simd_float3(w.x, w.y, w.z) / w.w)
        }
        CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly)

        // Too few → didn't hit anything; too many → runaway into floor
        guard pts.count >= 20, region.count < maxRegion else { return nil }

        // 4. Floor removal
        let rawYMin = pts.map { $0.y }.min()!
        let anchorY = frame.anchors
            .compactMap { $0 as? ARPlaneAnchor }
            .filter { $0.alignment == .horizontal }
            .map { Float($0.transform.columns.3.y) }.min()
        let floorY: Float
        if let a = anchorY, a <= rawYMin + 0.08 { floorY = a } else { floorY = rawYMin }

        let above = pts.filter { $0.y > floorY + 0.05 }
        guard above.count >= 20 else { return nil }

        // 5. PCA OBB in XZ
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

    // MARK: - PCA OBB in XZ (2×2 covariance, analytic eigenvalues)

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
