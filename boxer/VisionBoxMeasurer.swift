import Vision
import ARKit
import simd

/// Tap-to-select measurement using VNGenerateForegroundInstanceMaskRequest (iOS 17+) + LiDAR.
///
/// Flow:
///   1. Portrait tap → landscape image pixel via ARKit displayTransform
///   2. VNGenerateForegroundInstanceMaskRequest (.right = portrait content in landscape buffer)
///   3. generateScaledMaskForImage returns mask at original buffer dimensions (landscape coords)
///   4. Sample mask at landscape tap pixel → identify tapped instance
///   5. Filter LiDAR depth through that mask → clean point cloud (no floor bleed)
///   6. Percentile Y range for height (no ARPlane floor removal — Vision mask already excludes floor)
///   7. PCA OBB in XZ
@available(iOS 17.0, *)
struct VisionBoxMeasurer {

    static func measure(
        frame: ARFrame,
        tapPoint: CGPoint,
        viewportSize: CGSize,
        maskOut: ((CVPixelBuffer) -> Void)? = nil
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

        // Get LiDAR depth at tap point — used later to reject background points.
        let tapD: Float
        if let depthBuf = frame.sceneDepth?.depthMap {
            let dW = CVPixelBufferGetWidth(depthBuf)
            let dH = CVPixelBufferGetHeight(depthBuf)
            let dpx = max(0, min(dW - 1, Int(Float(tapImgX) / bufW * Float(dW))))
            let dpy = max(0, min(dH - 1, Int(Float(tapImgY) / bufH * Float(dH))))
            CVPixelBufferLockBaseAddress(depthBuf, .readOnly)
            let rb  = CVPixelBufferGetBytesPerRow(depthBuf)
            let d   = CVPixelBufferGetBaseAddress(depthBuf)!
                        .advanced(by: dpy * rb)
                        .assumingMemoryBound(to: Float32.self)[dpx]
            CVPixelBufferUnlockBaseAddress(depthBuf, .readOnly)
            tapD = (d > 0.10 && d < 8.0) ? d : 0
        } else {
            tapD = 0
        }

        // 2. Foreground segmentation
        // .right = buffer is landscape but represents portrait content (rotate 90° CW to view)
        // generateScaledMaskForImage always returns at the original buffer size (landscape coords)
        let request = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right)
        try handler.perform([request])

        guard let observation = request.results?.first,
              !observation.allInstances.isEmpty else {
            print("[VISION] no foreground instances")
            return nil
        }
        print("[VISION] \(observation.allInstances.count) instance(s)")

        // 3. Find which instance the tap hits; reuse the already-generated mask
        var selectedMask: CVPixelBuffer? = nil
        for instanceLabel in observation.allInstances {
            let mask = try observation.generateScaledMaskForImage(
                forInstances: IndexSet(integer: instanceLabel),
                from: handler
            )
            let hit = sampleMask(mask, x: tapImgX, y: tapImgY)
            if hit {
                selectedMask = mask
                print("[VISION] tapped instance \(instanceLabel)")
                break
            }
        }

        guard let instanceMask = selectedMask else {
            print("[VISION] tap missed all instances (tapImgX=\(tapImgX) tapImgY=\(tapImgY) bufW=\(Int(bufW)) bufH=\(Int(bufH)))")
            return nil
        }

        maskOut?(instanceMask)
        return measureWithMask(frame: frame, maskBuffer: instanceMask, tapD: tapD)
    }

    // MARK: - Mask sampling (handles UInt8 and Float32 buffers)

    private static func sampleMask(_ mask: CVPixelBuffer, x: Int, y: Int) -> Bool {
        CVPixelBufferLockBaseAddress(mask, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }

        let mW  = CVPixelBufferGetWidth(mask)
        let mH  = CVPixelBufferGetHeight(mask)
        let mRB = CVPixelBufferGetBytesPerRow(mask)
        let fmt = CVPixelBufferGetPixelFormatType(mask)
        let basePtr = CVPixelBufferGetBaseAddress(mask)!

        let mx = min(mW - 1, max(0, x))
        let my = min(mH - 1, max(0, y))

        if fmt == kCVPixelFormatType_OneComponent32Float {
            let fPtr = basePtr.advanced(by: my * mRB).assumingMemoryBound(to: Float32.self)
            return fPtr[mx] > 0.5
        } else {
            let bPtr = basePtr.advanced(by: my * mRB).assumingMemoryBound(to: UInt8.self)
            return bPtr[mx] > 127
        }
    }

    // MARK: - LiDAR + mask → OBB

    private static func measureWithMask(
        frame: ARFrame,
        maskBuffer: CVPixelBuffer,
        tapD: Float = 0
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
        let fmt   = CVPixelBufferGetPixelFormatType(maskBuffer)

        var pts: [simd_float3] = []
        pts.reserveCapacity(2000)

        for py in 0..<dH {
            let dRow = dBase.advanced(by: py * dRB).assumingMemoryBound(to: Float32.self)
            let my   = min(mH - 1, py * mH / dH)

            for px in 0..<dW {
                let d = dRow[px]
                guard d > 0.1, d < 8.0 else { continue }

                // Depth window around the tap point: rejects wall/background behind the box.
                // Skip if tapD is unknown (0) — fallback to no depth constraint.
                if tapD > 0 {
                    guard d >= tapD - 0.35, d <= tapD + 0.35 else { continue }
                }

                let mx = min(mW - 1, px * mW / dW)

                let inMask: Bool
                if fmt == kCVPixelFormatType_OneComponent32Float {
                    let fRow = mBase.advanced(by: my * mRB).assumingMemoryBound(to: Float32.self)
                    inMask = fRow[mx] > 0.5
                } else {
                    let bRow = mBase.advanced(by: my * mRB).assumingMemoryBound(to: UInt8.self)
                    inMask = bRow[mx] > 127
                }
                guard inMask else { continue }

                let ix  = Float(px) / Float(dW) * bufW
                let iy  = Float(py) / Float(dH) * bufH
                let cam = simd_float4((ix - cx) / fx * d, (iy - cy) / fy * d, -d, 1)
                let w   = T * cam
                pts.append(simd_float3(w.x, w.y, w.z) / w.w)
            }
        }

        print("[VISION] \(pts.count) LiDAR points inside mask")
        guard pts.count >= 20 else { return nil }

        // Floor Y from ARPlane anchor (same approach as DepthBoxMeasurer).
        // Vision mask often includes floor pixels when the box rests on the floor.
        let anchorFloorY: Float? = frame.anchors
            .compactMap { $0 as? ARPlaneAnchor }
            .filter { $0.alignment == .horizontal }
            .map { Float($0.transform.columns.3.y) }.min()

        // Remove floor points (10 cm margin).
        let above: [simd_float3]
        if let floorY = anchorFloorY {
            above = pts.filter { $0.y > floorY + 0.10 }
        } else {
            // No plane anchor: trim bottom 5% to remove floor leaks.
            let rawYMin = pts.map { $0.y }.min()!
            above = pts.filter { $0.y > rawYMin + 0.05 }
        }
        guard above.count >= 20 else { return nil }

        func pct(_ vals: [Float], _ p: Float) -> Float {
            let s = vals.sorted()
            return s[max(0, Int(Float(s.count) * p))]
        }
        let ys     = above.map { $0.y }
        let yLo    = anchorFloorY ?? pct(ys, 0.03)
        let yHi    = pct(ys, 0.97)
        let height = yHi - yLo
        guard height > 0.03, height < 3.0 else { return nil }

        let (width, depth, center2D, axis1) = obbXZ(points: above)
        guard width > 0.03, width < 4.0,
              depth > 0.03, depth < 4.0 else { return nil }

        let centerY = (yLo + yHi) / 2
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
