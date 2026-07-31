import SwiftUI
import ARKit
import SceneKit

struct ARViewContainer: UIViewRepresentable {
    @ObservedObject var viewModel: ARViewModel

    func makeCoordinator() -> Coordinator { Coordinator(viewModel) }

    func makeUIView(context: Context) -> ARSCNView {
        let sceneView = ARSCNView()
        sceneView.autoenablesDefaultLighting = true
        sceneView.automaticallyUpdatesLighting = true
        sceneView.delegate = context.coordinator

        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = [.sceneDepth]
        config.planeDetection = [.horizontal, .vertical]
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            config.sceneReconstruction = .mesh
        }

        sceneView.session.run(config)
        viewModel.setup(sceneView: sceneView)
        context.coordinator.sceneView = sceneView

        // Tap gesture for TAP mode: fallback finger-tap (crosshair button is primary)
        let tap = UITapGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handleTap(_:))
        )
        sceneView.addGestureRecognizer(tap)

        return sceneView
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {
        viewModel.viewportSize = uiView.bounds.size
    }

    // MARK: - Delegate
    class Coordinator: NSObject, ARSCNViewDelegate {
        weak var viewModel: ARViewModel?
        weak var sceneView: ARSCNView?
        private var lastHitCheck: TimeInterval = 0
        init(_ viewModel: ARViewModel) { self.viewModel = viewModel }

        func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
            guard let plane = anchor as? ARPlaneAnchor,
                  plane.alignment == .horizontal else { return }
            Task { @MainActor [weak self] in self?.viewModel?.floorDetected() }
        }

        // Check every 0.15s if there's a surface at the crosshair center → drives crosshairHit color.
        func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
            guard time - lastHitCheck > 0.15 else { return }
            lastHitCheck = time
            guard let sv = sceneView else { return }
            let center = CGPoint(x: sv.bounds.midX, y: sv.bounds.midY)
            let hit: Bool
            if let q = sv.raycastQuery(from: center, allowing: .existingPlaneGeometry, alignment: .any) {
                hit = !sv.session.raycast(q).isEmpty
            } else if let q = sv.raycastQuery(from: center, allowing: .estimatedPlane, alignment: .any) {
                hit = !sv.session.raycast(q).isEmpty
            } else {
                hit = false
            }
            Task { @MainActor [weak self] in self?.viewModel?.crosshairHit = hit }
        }

        @objc func handleTap(_ recognizer: UITapGestureRecognizer) {
            guard let vm = viewModel, vm.measureMode == .tap,
                  let sv = recognizer.view as? ARSCNView else { return }
            let point = recognizer.location(in: sv)
            Task { @MainActor in vm.measureAtTap(at: point) }
        }
    }
}
