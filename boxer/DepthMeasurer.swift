import Foundation
import ARKit
import simd

/// Measures a box directly from the LiDAR depth map — no external model needed.
///
/// Algorithm:
///   1. Collect depth pixels inside the YOLO bbox.
///   2. Find the front surface (10th-percentile depth).
///   3. Keep only points ≤ front + 30 cm (box surface, not background).
///   4. Unproject to 3D world space using camera intrinsics.
///   5. Fit a gravity-aligned bounding box via PCA on the XZ footprint.
struct DepthMeasurer {

    static func measure(
        frame: ARFrame,
        yoloBox: YOLOBox,
        minPoints: Int = 40
    ) -> Detection3D? {

        guard let sceneDepth = frame.sceneDepth else { return nil }
        let depthBuffer = sceneDepth.depthMap

        let bufW = Float(CVPixelBufferGetWidth(frame.capturedImage))
        let bufH = Float(CVPixelBufferGetHeight(frame.capturedImage))
        let depthW = CVPixelBufferGetWidth(depthBuffer)
        let depthH = CVPixelBufferGetHeight(depthBuffer)

        // Map YOLO bbox (640×640 square crop) → full landscape buffer coords.
        let side = min(bufW, bufH)
        let ox = (bufW - side) / 2
        let oy = (bufH - side) / 2

        let bx0 = yoloBox.xmin / 640.0 * side + ox
        let by0 = yoloBox.ymin / 640.0 * side + oy
        let bx1 = yoloBox.xmax / 640.0 * side + ox
        let by1 = yoloBox.ymax / 640.0 * side + oy

        // Depth-map pixel range for the bbox.
        let dx0 = max(0, Int(bx0 / bufW * Float(depthW)))
        let dy0 = max(0, Int(by0 / bufH * Float(depthH)))
        let dx1 = min(depthW - 1, Int(bx1 / bufW * Float(depthW)))
        let dy1 = min(depthH - 1, Int(by1 / bufH * Float(depthH)))

        guard dx1 > dx0, dy1 > dy0 else { return nil }

        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }

        let base      = CVPixelBufferGetBaseAddress(depthBuffer)!
        let rowBytes  = CVPixelBufferGetBytesPerRow(depthBuffer)

        // Pass 1: collect all valid depths to find d_front.
        var allDepths: [Float] = []
        allDepths.reserveCapacity((dx1 - dx0) * (dy1 - dy0))

        for dy in dy0...dy1 {
            let row = base.advanced(by: dy * rowBytes).assumingMemoryBound(to: Float32.self)
            for dx in dx0...dx1 {
                let d = row[dx]
                if d > 0.15 && d < 7.0 { allDepths.append(d) }
            }
        }
        guard allDepths.count >= minPoints else { return nil }

        allDepths.sort()
        let dFront = allDepths[allDepths.count / 10]          // 10th-percentile = near surface
        let dCutoff = dFront + 0.30                           // keep ≤ 30 cm behind front face

        // Pass 2: unproject box-surface pixels to 3D world points.
        let intr = frame.camera.intrinsics
        let fx = intr[0][0], fy = intr[1][1]
        let cx = intr[2][0], cy = intr[2][1]
        let camT = frame.camera.transform

        var points3D: [simd_float3] = []
        points3D.reserveCapacity(allDepths.count / 2)

        for dy in dy0...dy1 {
            let row = base.advanced(by: dy * rowBytes).assumingMemoryBound(to: Float32.self)
            for dx in dx0...dx1 {
                let d = row[dx]
                guard d >= dFront - 0.03, d <= dCutoff else { continue }

                // Map depth pixel → landscape image pixel.
                let imgX = Float(dx) / Float(depthW) * bufW
                let imgY = Float(dy) / Float(depthH) * bufH

                // Unproject (ARKit: camera looks in –Z).
                let cam = simd_float4((imgX - cx) / fx * d,
                                      (imgY - cy) / fy * d,
                                      -d, 1)
                let w = camT * cam
                points3D.append(simd_float3(w.x, w.y, w.z) / w.w)
            }
        }
        guard points3D.count >= minPoints else { return nil }

        return fitGravityAlignedBox(
            points: points3D,
            label: yoloBox.label,
            confidence: yoloBox.score
        )
    }

    // MARK: - PCA gravity-aligned box fit

    private static func fitGravityAlignedBox(
        points: [simd_float3],
        label: String,
        confidence: Float
    ) -> Detection3D? {

        // Height (Y axis, gravity = –Y in ARKit world).
        let yVals = points.map { $0.y }
        let yMin = yVals.min()!, yMax = yVals.max()!
        let height = yMax - yMin
        guard height > 0.02 else { return nil }

        // XZ centroid.
        let n = Float(points.count)
        let xMean = points.map { $0.x }.reduce(0, +) / n
        let zMean = points.map { $0.z }.reduce(0, +) / n

        // 2×2 covariance on XZ.
        var c00: Float = 0, c01: Float = 0, c11: Float = 0
        for p in points {
            let dx = p.x - xMean, dz = p.z - zMean
            c00 += dx * dx; c01 += dx * dz; c11 += dz * dz
        }
        c00 /= n; c01 /= n; c11 /= n

        // Closed-form 2×2 eigenvector for the larger eigenvalue.
        let trace = c00 + c11
        let disc  = sqrt(max(0, trace * trace / 4 - (c00 * c11 - c01 * c01)))
        let lambda1 = trace / 2 + disc

        var axisX: Float, axisZ: Float
        if abs(c01) > 1e-6 {
            let e = lambda1 - c11
            let len = sqrt(e * e + c01 * c01)
            axisX = e / len; axisZ = c01 / len
        } else {
            axisX = c00 >= c11 ? 1 : 0
            axisZ = c00 >= c11 ? 0 : 1
        }

        let perpX = -axisZ, perpZ = axisX

        // Project all points onto both axes.
        var pMin: Float = .greatestFiniteMagnitude, pMax: Float = -.greatestFiniteMagnitude
        var qMin: Float = .greatestFiniteMagnitude, qMax: Float = -.greatestFiniteMagnitude
        for p in points {
            let dx = p.x - xMean, dz = p.z - zMean
            let pp = dx * axisX + dz * axisZ
            let qq = dx * perpX + dz * perpZ
            pMin = min(pMin, pp); pMax = max(pMax, pp)
            qMin = min(qMin, qq); qMax = max(qMax, qq)
        }

        let widthA = pMax - pMin
        let widthB = qMax - qMin
        guard widthA > 0.02, widthB > 0.02 else { return nil }

        // 3D center.
        let pc = (pMin + pMax) / 2, qc = (qMin + qMax) / 2
        let cx3 = xMean + pc * axisX + qc * perpX
        let cz3 = zMean + pc * axisZ + qc * perpZ
        let center = simd_float3(cx3, (yMin + yMax) / 2, cz3)

        let yaw = atan2(axisZ, axisX)
        let cosY = cos(yaw), sinY = sin(yaw)
        let worldTransform = simd_float4x4(
            simd_float4( cosY, 0, sinY, 0),
            simd_float4(    0, 1,    0, 0),
            simd_float4(-sinY, 0, cosY, 0),
            simd_float4(center.x, center.y, center.z, 1)
        )

        return Detection3D(
            center: center,
            size: simd_float3(widthA, height, widthB),
            yaw: yaw,
            confidence: confidence,
            worldTransform: worldTransform,
            label: label
        )
    }
}
