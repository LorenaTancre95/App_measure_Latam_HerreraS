import ARKit
import SceneKit
import UIKit

// Coordinates tap-to-select segmentation + LiDAR point cloud measurement.
// Requires iPhone with LiDAR (iPhone 12 Pro+) and iOS 17+.
final class BoxDetectionCoordinator: NSObject {

    // MARK: References
    weak var sceneView: ARSCNView?
    var viewModel: ARMeasurementViewModel

    // MARK: Processors
    private let segmentation = SegmentationProcessor()

    // MARK: SceneKit overlay nodes
    private var labelNodes: [SCNNode] = []

    // MARK: Init
    init(viewModel: ARMeasurementViewModel) {
        self.viewModel = viewModel
        super.init()
    }

    // MARK: - Tap handler (called from ARViewContainer gesture)
    func handleObjectTap(at location: CGPoint) {
        guard !viewModel.isConfirmed,
              viewModel.state != .processing,
              let sceneView = sceneView,
              let frame = sceneView.session.currentFrame,
              frame.sceneDepth != nil else { return }

        viewModel.startProcessing()

        let viewSize = sceneView.bounds.size
        let planeY   = viewModel.nearestPlaneY

        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }

            // 1. Segment the object at the tap point
            guard let mask = self.segmentation.segment(frame: frame,
                                                       tapPoint: location,
                                                       viewSize: viewSize) else {
                await MainActor.run {
                    self.viewModel.selectionFailed(message: "Nenhum objeto detectado neste ponto")
                }
                return
            }

            // 2. Build OBB from filtered LiDAR point cloud
            let depthMap        = frame.sceneDepth!.depthMap
            let intrinsics      = frame.camera.intrinsics
            let cameraTransform = frame.camera.transform

            guard let measurement = PointCloudProcessor.computeOBB(
                depthMap: depthMap,
                mask: mask,
                intrinsics: intrinsics,
                cameraTransform: cameraTransform,
                planeY: planeY
            ) else {
                await MainActor.run {
                    self.viewModel.selectionFailed(message: "Não foi possível medir o objeto")
                }
                return
            }

            // 3. Compute OBB corners for the 3D wireframe
            let corners = PointCloudProcessor.obbCorners(
                depthMap: depthMap,
                mask: mask,
                intrinsics: intrinsics,
                cameraTransform: cameraTransform,
                planeY: planeY
            )

            await MainActor.run {
                self.viewModel.updateMeasurement(measurement)
                if let c = corners {
                    self.updateOverlay(tl: c.tl, tr: c.tr, bl: c.bl, br: c.br, depthM: c.depth)
                }
            }
        }
    }

    // MARK: - SceneKit 3D Overlay
    func updateOverlay(tl: SIMD3<Float>, tr: SIMD3<Float>,
                       bl: SIMD3<Float>, br: SIMD3<Float>,
                       depthM: Float) {
        guard let sceneView = sceneView else { return }

        clearOverlay()

        let isConfirmed = viewModel.isConfirmed
        let color: UIColor = isConfirmed
            ? UIColor(red: 0, green: 0.9, blue: 0.3, alpha: 1)
            : UIColor(red: 0, green: 0.85, blue: 0.95, alpha: 1)

        // Direction from front face toward back of box
        let faceRight = simd_normalize(SIMD3<Float>(br.x - bl.x, 0, br.z - bl.z))
        var depthDir  = SIMD3<Float>(-faceRight.z, 0, faceRight.x)

        if let cam = sceneView.session.currentFrame?.camera.transform.columns.3 {
            let camPt     = SIMD3<Float>(cam.x, cam.y, cam.z)
            let faceCenter = (bl + br + tl + tr) / 4
            let toCamera  = simd_normalize(camPt - faceCenter)
            if simd_dot(depthDir, toCamera) > 0 { depthDir = -depthDir }
        }

        let offset = depthDir * depthM

        let bl_b = bl + offset; let br_b = br + offset
        let tl_b = tl + offset; let tr_b = tr + offset

        let edges: [(SIMD3<Float>, SIMD3<Float>)] = [
            (bl, br), (br, tr), (tr, tl), (tl, bl),
            (bl_b, br_b), (br_b, tr_b), (tr_b, tl_b), (tl_b, bl_b),
            (bl, bl_b), (br, br_b), (tl, tl_b), (tr, tr_b)
        ]

        for (start, end) in edges {
            let node = makeLine(from: start, to: end, color: color)
            sceneView.scene.rootNode.addChildNode(node)
            labelNodes.append(node)
        }

        if let m = viewModel.measurement {
            addDimensionLabel(text: "\(Int(m.comprimento.rounded())) cm",
                              position: midpoint(bl, br) + SIMD3<Float>(0, 0.03, 0),
                              sceneView: sceneView)
            addDimensionLabel(text: "\(Int(m.altura.rounded())) cm",
                              position: midpoint(br, tr) + SIMD3<Float>(0.05, 0, 0),
                              sceneView: sceneView)
            addDimensionLabel(text: "\(Int(m.largura.rounded())) cm",
                              position: midpoint(br, br_b) + SIMD3<Float>(0, 0.03, 0),
                              sceneView: sceneView)
        }
    }

    func clearOverlay() {
        labelNodes.forEach { $0.removeFromParentNode() }
        labelNodes.removeAll()
    }

    // MARK: - SceneKit helpers
    private func makeLine(from start: SIMD3<Float>,
                          to end: SIMD3<Float>,
                          color: UIColor) -> SCNNode {
        let vector = end - start
        let length = simd_length(vector)
        guard length > 0.001 else { return SCNNode() }

        let cyl = SCNCylinder(radius: 0.002, height: CGFloat(length))
        cyl.firstMaterial?.diffuse.contents = color
        cyl.firstMaterial?.lightingModel = .constant

        let node = SCNNode(geometry: cyl)
        let mid = (start + end) / 2
        node.position = SCNVector3(mid.x, mid.y, mid.z)

        let dir  = simd_normalize(vector)
        let up   = SIMD3<Float>(0, 1, 0)
        let dot  = simd_dot(up, dir)

        if dot > 0.9999 {
            // Already aligned with up — no rotation needed
        } else if dot < -0.9999 {
            // Pointing straight down — rotate 180° around X
            node.rotation = SCNVector4(1, 0, 0, Float.pi)
        } else {
            let axis  = simd_normalize(simd_cross(up, dir))
            let angle = acos(dot)
            node.rotation = SCNVector4(axis.x, axis.y, axis.z, angle)
        }

        return node
    }

    private func addDimensionLabel(text: String,
                                   position: SIMD3<Float>,
                                   sceneView: ARSCNView) {
        let textGeo = SCNText(string: text, extrusionDepth: 0)
        textGeo.font = UIFont.boldSystemFont(ofSize: 6)
        textGeo.firstMaterial?.diffuse.contents = UIColor.white
        textGeo.firstMaterial?.lightingModel = .constant

        let textNode = SCNNode(geometry: textGeo)
        textNode.position = SCNVector3(position.x, position.y, position.z)
        textNode.scale = SCNVector3(0.005, 0.005, 0.005)

        let constraint = SCNBillboardConstraint()
        constraint.freeAxes = .all
        textNode.constraints = [constraint]

        sceneView.scene.rootNode.addChildNode(textNode)
        labelNodes.append(textNode)
    }
}

// MARK: - ARSCNViewDelegate
extension BoxDetectionCoordinator: ARSCNViewDelegate {
    func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
        guard let plane = anchor as? ARPlaneAnchor,
              plane.alignment == .horizontal else { return }
        let planeY = plane.transform.columns.3.y
        DispatchQueue.main.async { [weak self] in
            self?.viewModel.planeFound(y: planeY)
        }
    }

    func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
        guard let plane = anchor as? ARPlaneAnchor,
              plane.alignment == .horizontal else { return }
        let planeY = plane.transform.columns.3.y
        DispatchQueue.main.async { [weak self] in
            self?.viewModel.planeFound(y: planeY)
        }
    }
}

// MARK: - ARSessionDelegate
extension BoxDetectionCoordinator: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // No continuous processing — measurement is triggered by user tap only.
    }
}

// MARK: - Helpers
private func midpoint(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> SIMD3<Float> {
    (a + b) / 2
}
