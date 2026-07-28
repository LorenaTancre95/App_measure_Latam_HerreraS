import ARKit
import simd

/// Tap-to-select measurement using only LiDAR — no ML models required.
///
/// Flow:
///   1. Portrait tap → image pixel via ARKit displayTransform
///   2. LiDAR depth at that pixel → anchor 3D world point
///   3. Collect all LiDAR points within `spatialRadius` metres of the anchor
///   4. Floor removal via ARPlaneAnchor
///   5. Axis-aligned OBB (percentile-trimmed, same as PalletMeasurer)
struct DepthBoxMeasurer {

    static func measure(
        frame: ARFrame,
        tapPoint: CGPoint,
        viewportSize: CGSize,
        spatialRadius: Float = 0.45
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

        // 1. Map portrait tap → landscape image pixel
        let displayT  = frame.displayTransform(for: .portrait, viewportSize: viewportSize)
        let invertedT = displayT.inverted()
        let normTap   = CGPoint(x: tapPoint.x / viewportSize.width,
                                y: tapPoint.y / viewportSize.height)
        let normImg   = normTap.applying(invertedT)
        let tapImgX   = Float(normImg.x) * bufW
        let tapImgY   = Float(normImg.y) * bufH

        // 2. Map image pixel → depth map pixel and read depth
        let tapDX = max(0, min(dW - 1, Int(tapImgX / bufW * Float(dW))))
        let tapDY = max(0, min(dH - 1, Int(tapImgY / bufH * Float(dH))))

        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        let rb    = CVPixelBufferGetBytesPerRow(depthBuffer)
        let base  = CVPixelBufferGetBaseAddress(depthBuffer)!
        let tapD  = base.advanced(by: tapDY * rb).assumingMemoryBound(to: Float32.self)[tapDX]
        guard tapD > 0.10, tapD < 8.0 else {
            CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly)
            return nil
        }

        // Unproject tap pixel to world space
        let tapCam  = simd_float4((tapImgX - cx) / fx * tapD,
                                  (tapImgY - cy) / fy * tapD,
                                  -tapD, 1)
        let tapW    = T * tapCam
        let tapPos  = simd_float3(tapW.x, tapW.y, tapW.z) / tapW.w

        // 3. Collect all nearby LiDAR points
        var pts: [simd_float3] = []
        pts.reserveCapacity(3000)

        for py in 0..<dH {
            let row = base.advanced(by: py * rb).assumingMemoryBound(to: Float32.self)
            for px in 0..<dW {
                let d = row[px]
                guard d > 0.10, d < 8.0 else { continue }

                let ix = Float(px) / Float(dW) * bufW
                let iy = Float(py) / Float(dH) * bufH
                let cam = simd_float4((ix - cx) / fx * d, (iy - cy) / fy * d, -d, 1)
                let w   = T * cam
                let p   = simd_float3(w.x, w.y, w.z) / w.w

                if simd_distance(p, tapPos) < spatialRadius {
                    pts.append(p)
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly)

        guard pts.count >= 30 else { return nil }

        // 4. Floor removal (same logic as PalletMeasurer)
        let rawYMin = pts.map { $0.y }.min()!
        let anchorY: Float? = frame.anchors
            .compactMap { $0 as? ARPlaneAnchor }
            .filter { $0.alignment == .horizontal }
            .map { Float($0.transform.columns.3.y) }.min()
        let floorY: Float
        if let a = anchorY, a <= rawYMin + 0.08 { floorY = a } else { floorY = rawYMin }

        let above = pts.filter { $0.y > floorY + 0.04 }
        guard above.count >= 20 else { return nil }

        // 5. Percentile-trimmed AABB (same as PalletMeasurer)
        func pct(_ vals: [Float], _ p: Float) -> Float {
            let s = vals.sorted(); return s[max(0, Int(Float(s.count) * p))]
        }
        let xs = above.map { $0.x }
        let ys = above.map { $0.y }
        let zs = above.map { $0.z }

        let xMin = pct(xs, 0.02), xMax = pct(xs, 0.98)
        let yMin = floorY
        let yMax = pct(ys, 0.97)
        let zMin = pct(zs, 0.02), zMax = pct(zs, 0.98)

        let width  = xMax - xMin
        let height = yMax - yMin
        let depth  = zMax - zMin

        guard width  > 0.03, width  < 4.0,
              height > 0.03, height < 3.0,
              depth  > 0.03, depth  < 4.0 else { return nil }

        let center = simd_float3((xMin + xMax) / 2, (yMin + yMax) / 2, (zMin + zMax) / 2)
        let worldTransform = simd_float4x4(
            simd_float4(1, 0, 0, 0),
            simd_float4(0, 1, 0, 0),
            simd_float4(0, 0, 1, 0),
            simd_float4(center.x, center.y, center.z, 1)
        )

        return Detection3D(center: center,
                           size: simd_float3(width, height, depth),
                           yaw: 0,
                           confidence: 1.0,
                           worldTransform: worldTransform,
                           label: "caja")
    }
}
