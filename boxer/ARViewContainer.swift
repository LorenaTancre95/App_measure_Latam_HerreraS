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

        // Tap gesture for TAP mode: user touches the object they want to measure
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
        init(_ viewModel: ARViewModel) { self.viewModel = viewModel }

        func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
            guard let plane = anchor as? ARPlaneAnchor,
                  plane.alignment == .horizontal else { return }
            Task { @MainActor [weak self] in self?.viewModel?.floorDetected() }
        }

        @objc func handleTap(_ recognizer: UITapGestureRecognizer) {
            guard let vm = viewModel, vm.measureMode == .tap,
                  let sv = recognizer.view as? ARSCNView else { return }
            let point = recognizer.location(in: sv)
            Task { @MainActor in vm.measureAtTap(at: point) }
        }
    }
}
