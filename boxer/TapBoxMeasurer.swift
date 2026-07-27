import ARKit
import Vision
import simd

/// Tap-to-select box measurement using VNGenerateForegroundInstanceMaskRequest (iOS 17+).
///
/// Flow:
///   1. User taps on screen → screen point passed here
///   2. VNGenerateForegroundInstanceMaskRequest segments all foreground objects
///   3. The instance label at the tapped point is identified
///   4. A binary mask is built for ONLY that instance
///   5. LiDAR depth map is filtered through the mask
///   6. Floor-removed point cloud → OBB via PCA in XZ plane → Detection3D
///
/// Coordinate note: capturedImage is landscape (width > height).
/// PalletMeasurer uses the same depth-map → world pattern reused here.
@available(iOS 17.0, *)
struct TapBoxMeasurer {

    // MARK: - Entry point

    static func measure(
        frame: ARFrame,
        tapPoint: CGPoint,
        viewportSize: CGSize,
        minPoints: Int = 30
    ) -> Detection3D? {

        guard let depthBuffer = frame.sceneDepth?.depthMap else { return nil }

        // 1. Run instance segmentation
        guard let (instanceMask, selectedLabel) = segmentAtPoint(
            pixelBuffer: frame.capturedImage,
            tapPoint: tapPoint,
            viewportSize: viewportSize
        ) else { return nil }

        // 2. Collect filtered 3D point cloud (same pattern as PalletMeasurer)
        let pts = buildPointCloud(
            frame: frame,
            depthBuffer: depthBuffer,
            instanceMask: instanceMask,
            selectedLabel: selectedLabel
        )
        guard pts.count >= minPoints else { return nil }

        // 3. Floor removal
        let rawYMin = pts.map { $0.y }.min()!
        let anchorY: Float? = frame.anchors
            .compactMap { $0 as? ARPlaneAnchor }
            .filter { $0.alignment == .horizontal }
            .map { Float($0.transform.columns.3.y) }.min()

        let floorY: Float
        if let a = anchorY, a <= rawYMin + 0.08 { floorY = a } else { floorY = rawYMin }

        let above = pts.filter { $0.y > floorY + 0.04 }
        guard above.count >= minPoints else { return nil }

        // 4. Height using 97th-percentile top (same as BoxMeasurer2)
        var ySorted = above.map { $0.y }; ySorted.sort()
        let yTop  = ySorted[Int(Float(ySorted.count) * 0.97)]
        let height = yTop - floorY
        guard height > 0.02, height < 3.0 else { return nil }

        // 5. Horizontal OBB via 2×2 PCA on XZ
        let (width, depth, center2D, axis1) = obbXZ(points: above)
        guard width > 0.02, width < 4.0,
              depth > 0.02, depth < 4.0 else { return nil }

        // 6. Build Detection3D
        let centerY = (floorY + yTop) / 2
        let center  = simd_float3(center2D.x, centerY, center2D.y)

        // Yaw from principal axis
        let yaw = atan2(axis1.y, axis1.x)
        let cosY = cos(yaw), sinY = sin(yaw)
        let worldTransform = simd_float4x4(
            simd_float4( cosY, 0, sinY, 0),
            simd_float4(    0, 1,    0, 0),
            simd_float4(-sinY, 0, cosY, 0),
            simd_float4(center.x, center.y, center.z, 1)
        )

        // size.x = width along axis1, size.z = depth along axis2
        return Detection3D(
            center: center,
            size: simd_float3(width, height, depth),
            yaw: yaw,
            confidence: 1.0,
            worldTransform: worldTransform,
            label: "caja"
        )
    }

    // MARK: - Segmentation

    /// Runs VNGenerateForegroundInstanceMaskRequest and returns the instance mask
    /// CVPixelBuffer + the label (UInt8) of the instance under the tap point.
    private static func segmentAtPoint(
        pixelBuffer: CVPixelBuffer,
        tapPoint: CGPoint,
        viewportSize: CGSize
    ) -> (CVPixelBuffer, UInt8)? {

        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: .right,    // capturedImage is landscape-right; rotate to portrait
            options: [:]
        )
        let request = VNGenerateForegroundInstanceMaskRequest()
        guard (try? handler.perform([request])) != nil,
              let obs = request.results?.first as? VNInstanceMaskObservation
        else { return nil }

        let instanceMask = obs.instanceMask

        // Map portrait-screen tap → normalized landscape-image coords
        // (same portrait→landscape mapping used throughout the codebase)
        let normX = Float(tapPoint.y / viewportSize.height)
        let normY = Float(1.0 - tapPoint.x / viewportSize.width)

        let maskW = CVPixelBufferGetWidth(instanceMask)
        let maskH = CVPixelBufferGetHeight(instanceMask)

        let px = max(0, min(maskW - 1, Int(normX * Float(maskW))))
        let py = max(0, min(maskH - 1, Int(normY * Float(maskH))))

        CVPixelBufferLockBaseAddress(instanceMask, .readOnly)
        let ptr = CVPixelBufferGetBaseAddress(instanceMask)!
            .assumingMemoryBound(to: UInt8.self)
        let label = ptr[py * maskW + px]
        CVPixelBufferUnlockBaseAddress(instanceMask, .readOnly)

        guard label != 0 else { return nil }   // 0 = background

        return (instanceMask, label)
    }

    // MARK: - Point cloud

    private static func buildPointCloud(
        frame: ARFrame,
        depthBuffer: CVPixelBuffer,
        instanceMask: CVPixelBuffer,
        selectedLabel: UInt8
    ) -> [simd_float3] {

        let bufW = Float(CVPixelBufferGetWidth(frame.capturedImage))
        let bufH = Float(CVPixelBufferGetHeight(frame.capturedImage))

        let dW = CVPixelBufferGetWidth(depthBuffer)
        let dH = CVPixelBufferGetHeight(depthBuffer)
        let maskW = CVPixelBufferGetWidth(instanceMask)
        let maskH = CVPixelBufferGetHeight(instanceMask)

        let intr = frame.camera.intrinsics
        let fx = intr[0][0], fy = intr[1][1], cx = intr[2][0], cy = intr[2][1]
        let T  = frame.camera.transform

        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        CVPixelBufferLockBaseAddress(instanceMask, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly)
            CVPixelBufferUnlockBaseAddress(instanceMask, .readOnly)
        }

        guard let depthBase = CVPixelBufferGetBaseAddress(depthBuffer),
              let maskBase  = CVPixelBufferGetBaseAddress(instanceMask)
        else { return [] }

        let depthRB = CVPixelBufferGetBytesPerRow(depthBuffer)
        let maskPtr = maskBase.assumingMemoryBound(to: UInt8.self)

        var pts: [simd_float3] = []
        pts.reserveCapacity(2000)

        for py in 0 ..< dH {
            let depthRow = depthBase
                .advanced(by: py * depthRB)
                .assumingMemoryBound(to: Float32.self)

            for px in 0 ..< dW {
                let d = depthRow[px]
                guard d > 0.05, d < 8.0 else { continue }

                // depth pixel → full-res camera image pixel
                let ix = Float(px) / Float(dW) * bufW
                let iy = Float(py) / Float(dH) * bufH

                // full-res camera (landscape) → instanceMask pixel
                // instanceMask is in the same orientation as capturedImage (landscape)
                let mx = max(0, min(maskW - 1, Int(ix / bufW * Float(maskW))))
                let my = max(0, min(maskH - 1, Int(iy / bufH * Float(maskH))))

                guard maskPtr[my * maskW + mx] == selectedLabel else { continue }

                // Unproject (same convention as PalletMeasurer / BoxMeasurer2)
                let cam = simd_float4((ix - cx) / fx * d,
                                     (iy - cy) / fy * d,
                                     -d, 1)
                let w = T * cam
                pts.append(simd_float3(w.x, w.y, w.z) / w.w)
            }
        }

        return pts
    }

    // MARK: - PCA OBB in XZ plane

    /// Returns (width, depth, centerXZ, principalAxis1) in metres.
    private static func obbXZ(
        points: [simd_float3]
    ) -> (width: Float, depth: Float, center: simd_float2, axis1: simd_float2) {

        let n = Float(points.count)
        let cx = points.map { $0.x }.reduce(0, +) / n
        let cz = points.map { $0.z }.reduce(0, +) / n

        // 2×2 covariance matrix
        var cxx: Float = 0, cxz: Float = 0, czz: Float = 0
        for p in points {
            let dx = p.x - cx, dz = p.z - cz
            cxx += dx * dx; cxz += dx * dz; czz += dz * dz
        }
        cxx /= n; cxz /= n; czz /= n

        // Analytic eigenvalue / eigenvector for 2×2 symmetric matrix
        let trace    = cxx + czz
        let det      = cxx * czz - cxz * cxz
        let disc     = max(0, trace * trace / 4 - det)
        let lambda1  = trace / 2 + sqrt(disc)

        var axis1: simd_float2
        if abs(cxz) > 1e-6 {
            axis1 = simd_normalize(simd_float2(lambda1 - czz, cxz))
        } else {
            axis1 = cxx >= czz ? simd_float2(1, 0) : simd_float2(0, 1)
        }
        let axis2 = simd_float2(-axis1.y, axis1.x)

        // Project all points onto principal axes → extents
        var min1: Float = .infinity, max1: Float = -.infinity
        var min2: Float = .infinity, max2: Float = -.infinity
        for p in points {
            let dx = p.x - cx, dz = p.z - cz
            let p1 = dx * axis1.x + dz * axis1.y
            let p2 = dx * axis2.x + dz * axis2.y
            min1 = min(min1, p1); max1 = max(max1, p1)
            min2 = min(min2, p2); max2 = max(max2, p2)
        }

        // Trim 1% outliers each side
        // (PalletMeasurer uses 2%–98%; approximate here with simple min/max)

        return (max1 - min1, max2 - min2, simd_float2(cx, cz), axis1)
    }
}
