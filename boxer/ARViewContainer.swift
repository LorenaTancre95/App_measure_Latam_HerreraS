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
        // Cached on main thread, read on render thread (write-once per UI cycle, safe for reads)
        var cachedLastCorner: simd_float3? = nil
        init(_ viewModel: ARViewModel) { self.viewModel = viewModel }

        func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
            guard let plane = anchor as? ARPlaneAnchor,
                  plane.alignment == .horizontal else { return }
            Task { @MainActor [weak self] in self?.viewModel?.floorDetected() }
        }

        // Every 0.1s: raycast from crosshair center → live 3D point + project last corner to 2D.
        func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
            guard time - lastHitCheck > 0.1 else { return }
            lastHitCheck = time
            guard let sv = sceneView else { return }

            let center = CGPoint(x: sv.bounds.midX, y: sv.bounds.midY)

            // Live aim point: LiDAR gives the actual visible surface depth.
            // Plane raycast can overshoot through objects onto the wall/floor behind.
            // We take whichever is closer to the camera.
            var hitPoint: simd_float3? = nil
            let frame = sv.session.currentFrame
            if let frame {
                hitPoint = ARViewModel.lidarPoint(frame: frame, screenPoint: center, viewportSize: sv.bounds.size)
            }
            // Also try plane raycast — use it only if it's closer than LiDAR (or LiDAR failed)
            for target: ARRaycastQuery.Target in [.existingPlaneGeometry, .estimatedPlane] {
                if let q = sv.raycastQuery(from: center, allowing: target, alignment: .any),
                   let r = sv.session.raycast(q).first {
                    let c = r.worldTransform.columns.3
                    let rayPt = simd_float3(c.x, c.y, c.z)
                    if let existing = hitPoint, let f = frame {
                        let camPos = simd_float3(f.camera.transform.columns.3.x,
                                                  f.camera.transform.columns.3.y,
                                                  f.camera.transform.columns.3.z)
                        if simd_distance(rayPt, camPos) < simd_distance(existing, camPos) {
                            hitPoint = rayPt
                        }
                    } else {
                        hitPoint = rayPt
                    }
                    break
                }
            }

            // Project last placed corner to 2D screen coordinates
            var projectedCorner: CGPoint? = nil
            if let lc = cachedLastCorner {
                let proj = renderer.projectPoint(SCNVector3(lc.x, lc.y, lc.z))
                if proj.z > 0 && proj.z < 1 {
                    projectedCorner = CGPoint(x: CGFloat(proj.x), y: CGFloat(proj.y))
                }
            }

            let hit = hitPoint != nil
            Task { @MainActor [weak self] in
                guard let self, let vm = self.viewModel else { return }
                vm.crosshairHit = hit
                vm.liveAimPoint = hitPoint
                vm.lastCornerScreen = projectedCorner
                self.cachedLastCorner = vm.cornerPts.last
            }
        }

        @objc func handleTap(_ recognizer: UITapGestureRecognizer) {
            guard let vm = viewModel, vm.measureMode == .tap,
                  let sv = recognizer.view as? ARSCNView else { return }
            let point = recognizer.location(in: sv)
            Task { @MainActor in vm.measureAtTap(at: point) }
        }
    }
}
