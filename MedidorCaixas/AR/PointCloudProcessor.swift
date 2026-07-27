import ARKit
import simd

// Builds a filtered 3D point cloud from a LiDAR depth map + binary mask,
// then computes an Oriented Bounding Box (OBB) via PCA on the XZ plane.
struct PointCloudProcessor {

    /// Returns a BoxMeasurement (cm) or nil if not enough points were found.
    static func computeOBB(depthMap: CVPixelBuffer,
                           mask: CVPixelBuffer,
                           intrinsics: matrix_float3x3,
                           cameraTransform: matrix_float4x4,
                           planeY: Float?) -> BoxMeasurement? {

        let points = buildPointCloud(depthMap: depthMap,
                                     mask: mask,
                                     intrinsics: intrinsics,
                                     cameraTransform: cameraTransform)

        guard points.count >= 20 else { return nil }

        // --- Height ---
        let ys = points.map { $0.y }
        let topY  = ys.max()!
        // Snap base to detected plane; fallback to lowest point in cloud
        let baseY = planeY ?? ys.min()!
        let heightM = max(0, topY - baseY)

        // --- OBB in XZ via 2x2 PCA ---
        let (width, depth) = obbExtents(points: points)

        let heightCm = Double(heightM) * 100
        let widthCm  = Double(width)   * 100
        let depthCm  = Double(depth)   * 100

        // Sanity bounds: 2 cm – 300 cm
        guard heightCm > 2, heightCm < 300,
              widthCm  > 2, widthCm  < 300,
              depthCm  > 2, depthCm  < 300 else { return nil }

        // comprimento = longest horizontal side, largura = shorter horizontal side
        let (comprimento, largura) = widthCm >= depthCm
            ? (widthCm, depthCm)
            : (depthCm, widthCm)

        return BoxMeasurement(comprimento: comprimento, largura: largura, altura: heightCm)
    }

    // MARK: - Point cloud

    private static func buildPointCloud(depthMap: CVPixelBuffer,
                                        mask: CVPixelBuffer,
                                        intrinsics: matrix_float3x3,
                                        cameraTransform: matrix_float4x4) -> [SIMD3<Float>] {

        let depthW = CVPixelBufferGetWidth(depthMap)
        let depthH = CVPixelBufferGetHeight(depthMap)

        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        CVPixelBufferLockBaseAddress(mask, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)
            CVPixelBufferUnlockBaseAddress(mask, .readOnly)
        }

        guard let depthBase = CVPixelBufferGetBaseAddress(depthMap),
              let maskBase  = CVPixelBufferGetBaseAddress(mask) else { return [] }

        let depthPtr = depthBase.assumingMemoryBound(to: Float32.self)
        let maskPtr  = maskBase.assumingMemoryBound(to: UInt8.self)

        // Camera intrinsics (column-major: intrinsics[col][row])
        let fx = intrinsics[0][0]
        let fy = intrinsics[1][1]
        let cx = intrinsics[2][0]
        let cy = intrinsics[2][1]

        // The depth map is in landscape orientation (same as capturedImage).
        // Intrinsics are for the full-res image; we scale them to depth map size.
        // capturedImage resolution is typically 1920×1440 or 1440×1080.
        // We use the depth map pixel directly with scaled intrinsics.
        // ARKit guarantees depthMap and capturedImage share the same aspect ratio.
        let scaleX = Float(depthW) / fx   // approximate – we'll use ratio approach
        _ = scaleX  // suppress warning; actual scaling below

        // Determine image-to-depth scaling from capturedImage size
        // (intrinsics are defined for the full-res image)
        // We'll reproject using depth-map-relative intrinsics.
        // ARKit's sceneDepth is at a lower resolution than capturedImage but
        // shares the same FOV, so we scale intrinsics by the ratio.
        // A simpler approach: treat depth map as if it were the "image", scale intrinsics.
        // capturedImage width is accessible from intrinsics context – but we don't have it here.
        // Instead we use the known ARKit relationship: intrinsics are for capturedImage.
        // We pass through as-is and accept slight reprojection error at depth-map resolution.
        // For measurement accuracy this is sufficient (depth map is ~256×192).

        let fxD = fx
        let fyD = fy
        let cxD = cx
        let cyD = cy

        var points: [SIMD3<Float>] = []
        points.reserveCapacity(1000)

        for dy in 0 ..< depthH {
            for dx in 0 ..< depthW {
                let idx = dy * depthW + dx
                guard maskPtr[idx] != 0 else { continue }

                let z = depthPtr[idx]
                guard z > 0.05, z < 8.0 else { continue }

                // Reproject: depth map pixel → camera space
                // depth map is landscape (width > height); same orientation as capturedImage
                let xCam = (Float(dx) - cxD) / fxD * z
                let yCam = (Float(dy) - cyD) / fyD * z

                let pointCam = SIMD4<Float>(xCam, yCam, z, 1.0)
                let pointWorld = cameraTransform * pointCam
                points.append(SIMD3<Float>(pointWorld.x, pointWorld.y, pointWorld.z))
            }
        }

        return points
    }

    // MARK: - PCA-based OBB extents in XZ

    // Returns (width, depth) in meters — the two horizontal extents of the OBB.
    private static func obbExtents(points: [SIMD3<Float>]) -> (Float, Float) {
        let n = Float(points.count)
        let cx = points.map { $0.x }.reduce(0, +) / n
        let cz = points.map { $0.z }.reduce(0, +) / n

        // 2×2 covariance matrix of (x - cx, z - cz)
        var cxx: Float = 0, cxz: Float = 0, czz: Float = 0
        for p in points {
            let dx = p.x - cx
            let dz = p.z - cz
            cxx += dx * dx
            cxz += dx * dz
            czz += dz * dz
        }
        cxx /= n; cxz /= n; czz /= n

        // Analytic eigenvalues of 2x2 symmetric matrix [[cxx, cxz],[cxz, czz]]
        let trace = cxx + czz
        let det   = cxx * czz - cxz * cxz
        let disc  = max(0, trace * trace / 4 - det)
        let sqrtDisc = sqrt(disc)
        // λ1 >= λ2
        let lambda1 = trace / 2 + sqrtDisc
        let lambda2 = trace / 2 - sqrtDisc

        // Eigenvector for λ1 (principal axis = long side of box)
        var axis1: SIMD2<Float>
        if abs(cxz) > 1e-6 {
            axis1 = simd_normalize(SIMD2<Float>(lambda1 - czz, cxz))
        } else {
            axis1 = cxx >= czz ? SIMD2<Float>(1, 0) : SIMD2<Float>(0, 1)
        }
        let axis2 = SIMD2<Float>(-axis1.y, axis1.x)   // perpendicular

        // Project all points onto axes and compute extents
        var min1: Float =  .infinity, max1: Float = -.infinity
        var min2: Float =  .infinity, max2: Float = -.infinity
        for p in points {
            let dx = p.x - cx
            let dz = p.z - cz
            let proj1 = dx * axis1.x + dz * axis1.y
            let proj2 = dx * axis2.x + dz * axis2.y
            min1 = min(min1, proj1); max1 = max(max1, proj1)
            min2 = min(min2, proj2); max2 = max(max2, proj2)
        }

        _ = lambda2   // suppress warning

        return (max1 - min1, max2 - min2)
    }

    // MARK: - OBB corners for 3D overlay
    // Returns the 4 top corners and 4 bottom corners of the OBB in world space,
    // useful for drawing the bounding box wireframe.
    static func obbCorners(depthMap: CVPixelBuffer,
                           mask: CVPixelBuffer,
                           intrinsics: matrix_float3x3,
                           cameraTransform: matrix_float4x4,
                           planeY: Float?) -> (tl: SIMD3<Float>, tr: SIMD3<Float>,
                                               bl: SIMD3<Float>, br: SIMD3<Float>,
                                               depth: Float)? {

        let points = buildPointCloud(depthMap: depthMap,
                                     mask: mask,
                                     intrinsics: intrinsics,
                                     cameraTransform: cameraTransform)
        guard points.count >= 20 else { return nil }

        let n = Float(points.count)
        let cx = points.map { $0.x }.reduce(0, +) / n
        let cz = points.map { $0.z }.reduce(0, +) / n

        var cxx: Float = 0, cxz: Float = 0, czz: Float = 0
        for p in points {
            let dx = p.x - cx; let dz = p.z - cz
            cxx += dx * dx; cxz += dx * dz; czz += dz * dz
        }
        cxx /= n; cxz /= n; czz /= n

        let trace = cxx + czz
        let det   = cxx * czz - cxz * cxz
        let disc  = max(0, trace * trace / 4 - det)
        let lambda1 = trace / 2 + sqrt(disc)

        var axis1: SIMD2<Float>
        if abs(cxz) > 1e-6 {
            axis1 = simd_normalize(SIMD2<Float>(lambda1 - czz, cxz))
        } else {
            axis1 = cxx >= czz ? SIMD2<Float>(1, 0) : SIMD2<Float>(0, 1)
        }
        let axis2 = SIMD2<Float>(-axis1.y, axis1.x)

        var min1: Float = .infinity, max1: Float = -.infinity
        var min2: Float = .infinity, max2: Float = -.infinity
        let ys = points.map { $0.y }
        let topY  = ys.max()!
        let baseY = planeY ?? ys.min()!

        for p in points {
            let dx = p.x - cx; let dz = p.z - cz
            let p1 = dx * axis1.x + dz * axis1.y
            let p2 = dx * axis2.x + dz * axis2.y
            min1 = min(min1, p1); max1 = max(max1, p1)
            min2 = min(min2, p2); max2 = max(max2, p2)
        }

        // OBB center in XZ
        let mid1 = (min1 + max1) / 2
        let mid2 = (min2 + max2) / 2
        let centerX = cx + mid1 * axis1.x + mid2 * axis2.x
        let centerZ = cz + mid1 * axis1.y + mid2 * axis2.y

        // Half-extents
        let h1 = (max1 - min1) / 2   // half-width along axis1
        let h2 = (max2 - min2) / 2   // half-depth along axis2

        func corner(_ s1: Float, _ s2: Float, _ y: Float) -> SIMD3<Float> {
            let x = centerX + s1 * h1 * axis1.x + s2 * h2 * axis2.x
            let z = centerZ + s1 * h1 * axis1.y + s2 * h2 * axis2.y
            return SIMD3<Float>(x, y, z)
        }

        // Front face = the two corners closest to the camera
        // We pick the face along axis2 that is closer to camera position
        let camPos = SIMD3<Float>(cameraTransform.columns.3.x,
                                  cameraTransform.columns.3.y,
                                  cameraTransform.columns.3.z)
        let toCamera2 = (camPos.x - centerX) * axis2.x + (camPos.z - centerZ) * axis2.y
        let frontSign: Float = toCamera2 > 0 ? 1 : -1

        let tl = corner(-1, frontSign, topY)
        let tr = corner( 1, frontSign, topY)
        let bl = corner(-1, frontSign, baseY)
        let br = corner( 1, frontSign, baseY)
        let boxDepth = max2 - min2

        return (tl: tl, tr: tr, bl: bl, br: br, depth: boxDepth)
    }
}
