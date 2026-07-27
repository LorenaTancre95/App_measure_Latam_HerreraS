import SwiftUI
import ARKit
import SceneKit

struct ARViewContainer: UIViewRepresentable {
    @ObservedObject var viewModel: ARMeasurementViewModel

    func makeCoordinator() -> BoxDetectionCoordinator {
        BoxDetectionCoordinator(viewModel: viewModel)
    }

    func makeUIView(context: Context) -> ARSCNView {
        let sceneView = ARSCNView()
        sceneView.delegate         = context.coordinator
        sceneView.session.delegate = context.coordinator
        sceneView.autoenablesDefaultLighting    = true
        sceneView.automaticallyUpdatesLighting  = true

        context.coordinator.sceneView = sceneView

        // Tap → select the object under the finger
        let tap = UITapGestureRecognizer(
            target: context.coordinator,
            action: #selector(BoxDetectionCoordinator.handleTapGesture(_:))
        )
        sceneView.addGestureRecognizer(tap)

        startARSession(sceneView)
        return sceneView
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {
        // Clear overlay when returning to readyToSelect after remedir
        if viewModel.state == .readyToSelect {
            context.coordinator.clearOverlay()
        }
    }

    // MARK: - ARKit session with LiDAR
    private func startARSession(_ sceneView: ARSCNView) {
        guard ARWorldTrackingConfiguration.isSupported else {
            print("ARKit não suportado neste dispositivo")
            return
        }

        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal]

        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics = [.sceneDepth, .smoothedSceneDepth]
        }

        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            config.sceneReconstruction = .mesh
        }

        sceneView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }
}

// MARK: - Gesture forwarding
extension BoxDetectionCoordinator {
    @objc func handleTapGesture(_ recognizer: UITapGestureRecognizer) {
        guard let sceneView = sceneView else { return }
        let location = recognizer.location(in: sceneView)
        handleObjectTap(at: location)
    }
}
