import Vision
import ARKit
import simd

/// Tap-to-select measurement using VNGenerateForegroundInstanceMaskRequest (iOS 17+) + LiDAR.
///
/// Flow:
///   1. Portrait tap → landscape image pixel via ARKit displayTransform
///   2. VNGenerateForegroundInstanceMaskRequest segments all foreground objects
///   3. Sample each instance mask at the tap pixel → identify the tapped object
///   4. Filter LiDAR depth map through that instance mask → clean point cloud
///   5. Floor removal + PCA OBB → Detection3D
@available(iOS 17.0, *)
struct VisionBoxMeasurer {

    static func measure(
        frame: ARFrame,
        tapPoint: CGPoint,
        viewportSize: CGSize
    ) throws -> Detection3D? {

        let pixelBuffer = frame.capturedImage
        let bufW = Float(CVPixelBufferGetWidth(pixelBuffer))
        let bufH = Float(CVPixelBufferGetHeight(pixelBuffer))

        // 1. Portrait tap → landscape image pixel
        let displayT  = frame.displayTransform(for: .portrait, viewportSize: viewportSize)
        let invertedT = displayT.inverted()
        let normTap   = CGPoint(x: tapPoint.x / viewportSize.width,
                                y: tapPoint.y / viewportSize.height)
        let normImg   = normTap.applying(invertedT)
        let tapImgX   = Int(max(0, min(Float(normImg.x) * bufW, bufW - 1)))
        let tapImgY   = Int(max(0, min(Float(normImg.y) * bufH, bufH - 1)))

        // 2. Foreground instance segmentation on landscape image
        let request = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up)
        try handler.perform([request])

        guard let observation = request.results?.first,
              !observation.allInstances.isEmpty else {
            print("[VISION] No foreground instances found")
            return nil
        }
        print("[VISION] \(observation.allInstances.count) instance(s)")

        // 3. Find which instance the tap falls in; keep the mask we already generated
        var selectedMask: CVPixelBuffer? = nil
        for instanceLabel in observation.allInstances {
            let mask = try observation.generateScaledMaskForImage(
                forInstances: IndexSet(integer: instanceLabel),
                from: handler
            )
            CVPixelBufferLockBaseAddress(mask, .readOnly)
            let mW  = CVPixelBufferGetWidth(mask)
            let mH  = CVPixelBufferGetHeight(mask)
            let rb  = CVPixelBufferGetBytesPerRow(mask)
            let ptr = CVPixelBufferGetBaseAddress(mask)!
                        .assumingMemoryBound(to: UInt8.self)
            let mx  = min(mW - 1, tapImgX)
            let my  = min(mH - 1, tapImgY)
            let hit = ptr[my * rb + mx] > 127
            CVPixelBufferUnlockBaseAddress(mask, .readOnly)

            if hit {
                selectedMask = mask
                print("[VISION] tapped instance \(instanceLabel)")
                break
            }
        }

        guard let instanceMask = selectedMask else {
            print("[VISION] tap did not hit any foreground instance")
            return nil
        }

        // 4. LiDAR + mask → point cloud → OBB
        return measureWithMask(frame: frame, maskBuffer: instanceMask)
    }

    // MARK: - LiDAR + mask → OBB

    private static func measureWithMask(
        frame: ARFrame,
        maskBuffer: CVPixelBuffer
    ) -> Detection3D? {

        guard let depthBuffer = frame.sceneDepth?.depthMap else { return nil }

        let bufW = Float(CVPixelBufferGetWidth(frame.capturedImage))
        let bufH = Float(CVPixelBufferGetHeight(frame.capturedImage))
        let mW   = CVPixelBufferGetWidth(maskBuffer)
        let mH   = CVPixelBufferGetHeight(maskBuffer)
        let dW   = CVPixelBufferGetWidth(depthBuffer)
        let dH   = CVPixelBufferGetHeight(depthBuffer)

        let intr = frame.camera.intrinsics
        let fx = intr[0][0], fy = intr[1][1], cx = intr[2][0], cy = intr[2][1]
        let T  = frame.camera.transform

        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        CVPixelBufferLockBaseAddress(maskBuffer,  .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly)
            CVPixelBufferUnlockBaseAddress(maskBuffer,  .readOnly)
        }

        let dBase = CVPixelBufferGetBaseAddress(depthBuffer)!
        let dRB   = CVPixelBufferGetBytesPerRow(depthBuffer)
        let mBase = CVPixelBufferGetBaseAddress(maskBuffer)!
        let mRB   = CVPixelBufferGetBytesPerRow(maskBuffer)

        var pts: [simd_float3] = []
        pts.reserveCapacity(2000)

        for py in 0..<dH {
            let dRow = dBase.advanced(by: py * dRB)
                            .assumingMemoryBound(to: Float32.self)
            // depth row py → mask row (proportional to mask height)
            let my   = min(mH - 1, py * mH / dH)
            let mRow = mBase.advanced(by: my * mRB)
                            .assumingMemoryBound(to: UInt8.self)

            for px in 0..<dW {
                let d = dRow[px]
                guard d > 0.1, d < 8.0 else { continue }

                let mx = min(mW - 1, px * mW / dW)
                guard mRow[mx] > 127 else { continue }

                let ix  = Float(px) / Float(dW) * bufW
                let iy  = Float(py) / Float(dH) * bufH
                let cam = simd_float4((ix - cx) / fx * d,
                                      (iy - cy) / fy * d, -d, 1)
                let w   = T * cam
                pts.append(simd_float3(w.x, w.y, w.z) / w.w)
            }
        }

        guard pts.count >= 20 else { return nil }

        // Floor removal
        let rawYMin = pts.map { $0.y }.min()!
        let anchorY = frame.anchors
            .compactMap { $0 as? ARPlaneAnchor }
            .filter { $0.alignment == .horizontal }
            .map { Float($0.transform.columns.3.y) }.min()
        let floorY: Float
        if let a = anchorY, a <= rawYMin + 0.08 { floorY = a } else { floorY = rawYMin }

        let above = pts.filter { $0.y > floorY + 0.05 }
        guard above.count >= 20 else { return nil }

        // PCA OBB in XZ
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
