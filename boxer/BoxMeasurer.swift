import Foundation
import ARKit
import SceneKit
import simd

/// Measures a cardboard box using ARMeshAnchor vertices filtered by a 2D YOLO bbox.
/// Replaces BoxerNet — no external model needed, runs entirely on-device geometry.
struct BoxMeasurer {

    // MARK: - Public API

    /// Measure a box from ARKit mesh anchors projected through the YOLO 2D bbox.
    ///
    /// - Parameters:
    ///   - frame: Current ARFrame (has camera, capturedImage, anchors).
    ///   - yoloBox: YOLO detection in 640×640 space.
    ///   - minPoints: Minimum mesh vertices required inside bbox to attempt fitting.
    /// - Returns: Detection3D or nil if not enough geometry is found.
    static func measure(
        frame: ARFrame,
        yoloBox: YOLOBox,
        minPoints: Int = 20
    ) -> Detection3D? {
        let meshAnchors = frame.anchors.compactMap { $0 as? ARMeshAnchor }
        guard !meshAnchors.isEmpty else { return nil }

        // 1. Convert YOLO bbox (640×640 square crop) → normalized image coords [0,1].
        let bufW = Float(CVPixelBufferGetWidth(frame.capturedImage))
        let bufH = Float(CVPixelBufferGetHeight(frame.capturedImage))
        let side = min(bufW, bufH)
        let ox = (bufW - side) / 2
        let oy = (bufH - side) / 2

        // YOLO coords are in the 640×640 square; map back to full image [0,1].
        let normXmin = (yoloBox.xmin / 640.0 * side + ox) / bufW
        let normYmin = (yoloBox.ymin / 640.0 * side + oy) / bufH
        let normXmax = (yoloBox.xmax / 640.0 * side + ox) / bufW
        let normYmax = (yoloBox.ymax / 640.0 * side + oy) / bufH

        // 2. Collect mesh vertices inside the bbox.
        var points3D: [simd_float3] = []

        let imgSize = frame.camera.imageResolution  // real resolution for projectPoint
        let vp = CGSize(width: imgSize.width, height: imgSize.height)

        for anchor in meshAnchors {
            let geometry = anchor.geometry
            let vertexCount = geometry.vertices.count
            let stride = geometry.vertices.stride
            let src = geometry.vertices.buffer.contents()

            for i in 0..<vertexCount {
                // Read local vertex position.
                let ptr = src.advanced(by: i * stride)
                var local = simd_float3(
                    ptr.load(as: Float.self),
                    ptr.advanced(by: 4).load(as: Float.self),
                    ptr.advanced(by: 8).load(as: Float.self)
                )

                // Transform to world space.
                let world4 = anchor.transform * simd_float4(local.x, local.y, local.z, 1)
                local = simd_float3(world4.x, world4.y, world4.z) / world4.w

                // Project to 2D image coords (landscape right = default ARKit orientation).
                let projected = frame.camera.projectPoint(
                    simd_float3(local.x, local.y, local.z),
                    orientation: .landscapeRight,
                    viewportSize: vp
                )

                // Normalize to [0,1].
                let nx = Float(projected.x / imgSize.width)
                let ny = Float(projected.y / imgSize.height)

                // Filter: must be inside YOLO bbox.
                guard nx >= normXmin, nx <= normXmax,
                      ny >= normYmin, ny <= normYmax else { continue }

                // Filter: must be in front of camera.
                let camSpaceZ = (frame.camera.transform.inverse * simd_float4(local.x, local.y, local.z, 1)).z
                guard camSpaceZ < 0 else { continue }   // ARKit camera: -Z is forward

                points3D.append(local)
            }
        }

        guard points3D.count >= minPoints else { return nil }

        // 3. Fit a gravity-aligned bounding box via PCA on the XZ footprint.
        return fitGravityAlignedBox(points: points3D, label: yoloBox.label, confidence: yoloBox.score)
    }

    // MARK: - Geometry Fitting

    private static func fitGravityAlignedBox(
        points: [simd_float3],
        label: String,
        confidence: Float
    ) -> Detection3D? {
        // Height (Y axis, gravity-aligned) is trivial.
        let yMin = points.map { $0.y }.min()!
        let yMax = points.map { $0.y }.max()!
        let height = yMax - yMin
        let yCentroid = (yMin + yMax) / 2

        guard height > 0.01 else { return nil }   // sanity check: >1 cm tall

        // XZ centroid.
        let n = Float(points.count)
        let xMean = points.map { $0.x }.reduce(0, +) / n
        let zMean = points.map { $0.z }.reduce(0, +) / n

        // 2×2 covariance on XZ.
        var c00: Float = 0, c01: Float = 0, c11: Float = 0
        for p in points {
            let dx = p.x - xMean
            let dz = p.z - zMean
            c00 += dx * dx
            c01 += dx * dz
            c11 += dz * dz
        }
        c00 /= n; c01 /= n; c11 /= n

        // Principal axis via closed-form 2×2 eigen.
        let trace = c00 + c11
        let det   = c00 * c11 - c01 * c01
        let disc  = sqrt(max(0, trace * trace / 4 - det))
        let lambda1 = trace / 2 + disc  // larger eigenvalue

        // Eigenvector for lambda1.
        var axisX: Float
        var axisZ: Float
        if abs(c01) > 1e-6 {
            let ex = lambda1 - c11
            let len = sqrt(ex * ex + c01 * c01)
            axisX = ex / len
            axisZ = c01 / len
        } else {
            // Axes already aligned.
            axisX = c00 >= c11 ? 1 : 0
            axisZ = c00 >= c11 ? 0 : 1
        }

        // Project all XZ points onto (axisX, axisZ) and its perpendicular.
        let perpX = -axisZ
        let perpZ =  axisX

        var pMin: Float = .greatestFiniteMagnitude
        var pMax: Float = -.greatestFiniteMagnitude
        var qMin: Float = .greatestFiniteMagnitude
        var qMax: Float = -.greatestFiniteMagnitude

        for p in points {
            let dx = p.x - xMean
            let dz = p.z - zMean
            let pp = dx * axisX + dz * axisZ
            let qq = dx * perpX + dz * perpZ
            pMin = min(pMin, pp); pMax = max(pMax, pp)
            qMin = min(qMin, qq); qMax = max(qMax, qq)
        }

        let widthA = pMax - pMin
        let widthB = qMax - qMin

        guard widthA > 0.01, widthB > 0.01 else { return nil }  // sanity: >1 cm wide

        // 3D center in world space.
        let pCenter = (pMin + pMax) / 2
        let qCenter = (qMin + qMax) / 2
        let cx = xMean + pCenter * axisX + qCenter * perpX
        let cz = zMean + pCenter * axisZ + qCenter * perpZ

        let center = simd_float3(cx, yCentroid, cz)

        // Yaw angle around Y axis (ARKit world up = +Y).
        let yaw = atan2(axisZ, axisX)

        // Build world transform: rotate by yaw around Y, translate to center.
        let cosY = cos(yaw), sinY = sin(yaw)
        let worldTransform = simd_float4x4(
            simd_float4( cosY, 0, sinY, 0),
            simd_float4(    0, 1,    0, 0),
            simd_float4(-sinY, 0, cosY, 0),
            simd_float4(center.x, center.y, center.z, 1)
        )

        // Convention: x = widthA (main axis), y = height, z = widthB.
        let size = simd_float3(widthA, height, widthB)

        return Detection3D(
            center: center,
            size: size,
            yaw: yaw,
            confidence: confidence,
            worldTransform: worldTransform,
            label: label
        )
    }
}
