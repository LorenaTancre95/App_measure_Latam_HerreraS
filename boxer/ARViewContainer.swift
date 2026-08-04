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
        config.frameSemantics = [.sceneDepth, .smoothedSceneDepth]
        config.planeDetection = [.horizontal, .vertical]
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            config.sceneReconstruction = .mesh  // mesh reconstruido = raycast más estable
        }

        sceneView.session.run(config)
        viewModel.setup(sceneView: sceneView)
        context.coordinator.sceneView = sceneView
        return sceneView
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {
        viewModel.viewportSize = uiView.bounds.size
    }

    // MARK: - Coordinator

    class Coordinator: NSObject, ARSCNViewDelegate {
        weak var viewModel: ARViewModel?
        weak var sceneView: ARSCNView?
        private var lastHitCheck: TimeInterval = 0
        /// Último tap 3D cacheado para proyectar a pantalla en el render thread
        var cachedLastTap: simd_float3? = nil

        init(_ viewModel: ARViewModel) { self.viewModel = viewModel }

        /// Cada 0.1s: actualiza liveAimPoint (raycast+LiDAR combinados) y lastTapScreen.
        func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
            guard time - lastHitCheck > 0.1 else { return }
            lastHitCheck = time
            guard let sv = sceneView else { return }

            let center = CGPoint(x: sv.bounds.midX, y: sv.bounds.midY)
            let frame  = sv.session.currentFrame

            // Intenta raycast contra mesh (más estable en bordes de caja)
            var hitPoint: simd_float3? = nil
            for target: ARRaycastQuery.Target in [.existingPlaneGeometry, .estimatedPlane] {
                if let q = sv.raycastQuery(from: center, allowing: target, alignment: .any),
                   let r = sv.session.raycast(q).first {
                    let c = r.worldTransform.columns.3
                    hitPoint = simd_float3(c.x, c.y, c.z)
                    break
                }
            }

            // Si no hay raycast, usa LiDAR
            if hitPoint == nil, let f = frame {
                hitPoint = ARViewModel.lidarPoint(frame: f, screenPoint: center, viewportSize: sv.bounds.size)
            }

            // Proyectar el último tap colocado a coordenadas de pantalla
            var lastTapScreenPt: CGPoint? = nil
            if let lt = cachedLastTap {
                let proj = renderer.projectPoint(SCNVector3(lt.x, lt.y, lt.z))
                if proj.z > 0, proj.z < 1 {
                    lastTapScreenPt = CGPoint(x: CGFloat(proj.x), y: CGFloat(proj.y))
                }
            }

            let hit = hitPoint != nil
            Task { @MainActor [weak self] in
                guard let self, let vm = self.viewModel else { return }
                vm.crosshairHit  = hit
                vm.liveAimPoint  = hitPoint
                vm.lastTapScreen = lastTapScreenPt
                // Cachear el último tap para el siguiente frame
                self.cachedLastTap = vm.tapPoints.last
            }
        }
    }
}
