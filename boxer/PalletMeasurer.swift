import ARKit
import simd

/// Measures the 3D bounding box of arbitrary cargo using a SAM segmentation mask.
/// Unlike BoxMeasurer2 (tight YOLO bbox → single box geometry), this measurer
/// collects ALL LiDAR points that project into the SAM mask and returns the
/// axis-aligned outer bounding box — ideal for pallets and loose cargo.
struct PalletMeasurer {

    static func measure(
        frame: ARFrame,
        mask: [[Bool]],       // binary mask at decoder output resolution
        maskImageSize: Int = SAMSegmenter.imageSize   // 1024
    ) -> Detection3D? {

        guard let depthBuffer = frame.sceneDepth?.depthMap else { return nil }

        let bufW = Float(CVPixelBufferGetWidth(frame.capturedImage))
        let bufH = Float(CVPixelBufferGetHeight(frame.capturedImage))
        let side = min(bufW, bufH)
        let ox   = (bufW - side) / 2
        let oy   = (bufH - side) / 2

        let dW = CVPixelBufferGetWidth(depthBuffer)
        let dH = CVPixelBufferGetHeight(depthBuffer)

        let intr = frame.camera.intrinsics
        let fx = intr[0][0], fy = intr[1][1], cx = intr[2][0], cy = intr[2][1]
        let T  = frame.camera.transform

        let mH = mask.count
        let mW = mH > 0 ? mask[0].count : 0
        guard mH > 0, mW > 0 else { return nil }

        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }
        let base = CVPixelBufferGetBaseAddress(depthBuffer)!
        let rb   = CVPixelBufferGetBytesPerRow(depthBuffer)

        var pts: [simd_float3] = []

        for py in 0..<dH {
            let row = base.advanced(by: py * rb).assumingMemoryBound(to: Float32.self)
            for px in 0..<dW {
                let d = row[px]
                guard d > 0.1, d < 8.0 else { continue }

                // Depth pixel → camera image pixel
                let ix = Float(px) / Float(dW) * bufW
                let iy = Float(py) / Float(dH) * bufH

                // Camera image → center-crop space (same crop used for SAM encoder)
                let cropX = ix - ox
                let cropY = iy - oy
                guard cropX >= 0, cropX < side, cropY >= 0, cropY < side else { continue }

                // Crop → mask pixel
                let maskX = Int(cropX / side * Float(mW))
                let maskY = Int(cropY / side * Float(mH))
                guard maskX >= 0, maskX < mW, maskY >= 0, maskY < mH else { continue }
                guard mask[maskY][maskX] else { continue }

                // Unproject to ARKit world space
                let cam = simd_float4((ix - cx) / fx * d, (iy - cy) / fy * d, -d, 1)
                let w   = T * cam
                pts.append(simd_float3(w.x, w.y, w.z) / w.w)
            }
        }

        guard pts.count >= 20 else { return nil }

        // Floor removal (same logic as BoxMeasurer2)
        let rawYMin = pts.map { $0.y }.min()!
        let anchorY: Float? = frame.anchors
            .compactMap { $0 as? ARPlaneAnchor }
            .filter { $0.alignment == .horizontal }
            .map { Float($0.transform.columns.3.y) }.min()
        let floorY: Float
        if let a = anchorY, a <= rawYMin + 0.08 { floorY = a } else { floorY = rawYMin }

        let above = pts.filter { $0.y > floorY + 0.04 }
        guard above.count >= 20 else { return nil }

        func pct(_ vals: [Float], _ p: Float) -> Float {
            let s = vals.sorted(); return s[max(0, min(s.count-1, Int(Float(s.count) * p)))]
        }
        let yMin = floorY
        let yMax = pct(above.map { $0.y }, 0.97)
        let height = yMax - yMin
        guard height > 0.05, height < 3.0 else { return nil }

        // PCA en plano XZ para obtener orientación real de la carga
        let n = Float(above.count)
        let xMean = above.map { $0.x }.reduce(0, +) / n
        let zMean = above.map { $0.z }.reduce(0, +) / n
        var cxx: Float = 0, czz: Float = 0, cxz: Float = 0
        for p in above {
            let dx = p.x - xMean, dz = p.z - zMean
            cxx += dx*dx; czz += dz*dz; cxz += dx*dz
        }
        cxx /= n; czz /= n; cxz /= n

        // Power iteration → eigenvector dominante (eje de ancho)
        var ex: Float = 1, ez: Float = 0
        for _ in 0..<30 {
            let nx = cxx*ex + cxz*ez, nz = cxz*ex + czz*ez
            let len = sqrt(nx*nx + nz*nz)
            guard len > 1e-9 else { break }
            ex = nx/len; ez = nz/len
        }
        let px = -ez, pz = ex   // eje perpendicular (profundidad)

        // Proyectar puntos sobre los dos ejes orientados
        var wVals = [Float](), dVals = [Float]()
        for p in above {
            let dx = p.x - xMean, dz = p.z - zMean
            wVals.append(dx*ex + dz*ez)
            dVals.append(dx*px + dz*pz)
        }
        wVals.sort(); dVals.sort()
        let lo = max(0, Int(Float(wVals.count)*0.02))
        let hi = min(wVals.count-1, Int(Float(wVals.count)*0.98))
        let wMin = wVals[lo], wMax = wVals[hi]
        let dMin = dVals[lo], dMax = dVals[hi]

        let width = wMax - wMin
        let depth = dMax - dMin
        guard width > 0.05, width < 4.0,
              depth > 0.05, depth < 4.0 else { return nil }

        // Centro OBB en espacio mundo
        let cX = xMean + (wMin+wMax)/2 * ex + (dMin+dMax)/2 * px
        let cZ = zMean + (wMin+wMax)/2 * ez + (dMin+dMax)/2 * pz
        let cY = (yMin + yMax) / 2
        let center = simd_float3(cX, cY, cZ)

        let yaw = atan2(ez, ex)
        let cosY = cos(yaw), sinY = sin(yaw)
        let worldTransform = simd_float4x4(
            simd_float4( cosY, 0, sinY, 0),
            simd_float4(    0, 1,    0, 0),
            simd_float4(-sinY, 0, cosY, 0),
            simd_float4(cX, cY, cZ, 1)
        )

        return Detection3D(center: center,
                           size: simd_float3(width, height, depth),
                           yaw: yaw,
                           confidence: 1.0,
                           worldTransform: worldTransform,
                           label: "oversize")
    }
}
