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

        let tap = UITapGestureRecognizer(target: context.coordinator,
                                         action: #selector(Coordinator.handleScreenTap(_:)))
        tap.cancelsTouchesInView = false
        sceneView.addGestureRecognizer(tap)

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

        // Punto de snap suavizado con lerp (evita saltos entre frames)
        private var smoothedSnapPt: CGPoint? = nil

        // Caché del frame anterior para filtrado 3D del snap
        var cachedLastTap:  simd_float3? = nil
        var cachedAimPoint: simd_float3? = nil

        // Buffer de los últimos puntos LiDAR para calcular la mediana → punto estable
        private var aimBuffer: [simd_float3] = []
        private let bufferSize = 8
        private let stableThreshold: Float = 0.015

        init(_ viewModel: ARViewModel) { self.viewModel = viewModel }

        // Materializa los ARAnchor "marker" como esferas amarillas.
        func renderer(_ renderer: SCNSceneRenderer, nodeFor anchor: ARAnchor) -> SCNNode? {
            guard anchor.name == "marker" else { return nil }
            let sphere = SCNSphere(radius: 0.006)
            let mat = SCNMaterial()
            mat.diffuse.contents = UIColor.systemYellow
            sphere.materials = [mat]
            return SCNNode(geometry: sphere)
        }

        @objc func handleScreenTap(_ gesture: UITapGestureRecognizer) {
            guard let vm = viewModel, let sv = sceneView,
                  vm.dimPhase == .alto, vm.tapPhase != .preview else { return }
            let pt = gesture.location(in: sv)
            Task { @MainActor in vm.captureTap(at: pt) }
        }

        func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
            guard let plane = anchor as? ARPlaneAnchor, plane.alignment == .horizontal else { return }
            let y = plane.transform.columns.3.y
            Task { @MainActor in self.viewModel?.updateFloorY(y) }
        }

        func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
            guard let plane = anchor as? ARPlaneAnchor, plane.alignment == .horizontal else { return }
            let y = plane.transform.columns.3.y
            Task { @MainActor in self.viewModel?.updateFloorY(y) }
        }

        /// Cada 0.1s: calcula punto estable + snap a feature points o bordes de plano.
        func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
            guard time - lastHitCheck > 0.1 else { return }
            lastHitCheck = time
            guard let sv = sceneView, let frame = sv.session.currentFrame else { return }

            let center = CGPoint(x: sv.bounds.midX, y: sv.bounds.midY)

            // ── SNAP: feature points trackeados por ARKit + bordes de planos detectados ──
            // Las dos fuentes más estables disponibles sin APIs privadas.
            var bestSnapWP: simd_float3? = nil
            if let aim = cachedAimPoint {

                // 1) Feature points de ARKit (trackeados frame a frame, muy estables)
                //    Se concentran en zonas de alto contraste → esquinas y bordes de cajas
                var bestFP: Float = 0.035   // 3.5 cm
                if let pts = frame.rawFeaturePoints?.points {
                    for p in pts {
                        let d = simd_distance(p, aim)
                        if d < bestFP { bestFP = d; bestSnapWP = p }
                    }
                }

                // 2) Vértices de borde de planos ARKit (si no hay feature point cercano)
                //    ARPlaneAnchor.geometry.boundaryVertices = perímetro exacto del plano
                if bestSnapWP == nil {
                    var bestBV: Float = 0.05   // 5 cm
                    for anchor in frame.anchors.compactMap({ $0 as? ARPlaneAnchor }) {
                        let T = anchor.transform
                        for v in anchor.geometry.boundaryVertices {
                            let w  = T * simd_float4(v.x, v.y, v.z, 1)
                            let wp = simd_float3(w.x, w.y, w.z) / w.w
                            let d  = simd_distance(aim, wp)
                            if d < bestBV { bestBV = d; bestSnapWP = wp }
                        }
                    }
                }
            }

            // Proyectar punto snap 3D → pantalla y suavizar con lerp (evita saltos)
            let rawSnapPt: CGPoint? = bestSnapWP.flatMap {
                projectToScreen($0, camera: frame.camera, size: sv.bounds.size)
            }
            if let rsp = rawSnapPt {
                let prev = smoothedSnapPt ?? rsp
                smoothedSnapPt = CGPoint(x: prev.x + (rsp.x - prev.x) * 0.35,
                                          y: prev.y + (rsp.y - prev.y) * 0.35)
            } else {
                smoothedSnapPt = nil
            }
            let snapPt = smoothedSnapPt

            // ── PROFUNDIDAD: LiDAR en centro de pantalla + buffer mediana ──
            var rawHit: simd_float3? = ARViewModel.lidarPoint(frame: frame,
                                                               screenPoint: center,
                                                               viewportSize: sv.bounds.size)
            if rawHit == nil {
                for target: ARRaycastQuery.Target in [.existingPlaneGeometry, .estimatedPlane] {
                    if let q = sv.raycastQuery(from: center, allowing: target, alignment: .any),
                       let r = sv.session.raycast(q).first {
                        let c = r.worldTransform.columns.3
                        rawHit = simd_float3(c.x, c.y, c.z)
                        break
                    }
                }
            }

            if let pt = rawHit {
                aimBuffer.append(pt)
                if aimBuffer.count > bufferSize { aimBuffer.removeFirst() }
            } else {
                aimBuffer.removeAll()
            }

            var stablePoint: simd_float3? = nil
            var isStable = false
            if aimBuffer.count >= 3 {
                let xs = aimBuffer.map(\.x).sorted()
                let ys = aimBuffer.map(\.y).sorted()
                let zs = aimBuffer.map(\.z).sorted()
                let mid = aimBuffer.count / 2
                let med = simd_float3(xs[mid], ys[mid], zs[mid])
                // Si hay snap cercano, usarlo como punto de medición (más preciso)
                stablePoint = bestSnapWP ?? med
                let maxDist = aimBuffer.map { simd_distance($0, med) }.max() ?? 0
                isStable = aimBuffer.count >= bufferSize && maxDist < stableThreshold
            }

            // Proyectar el tap anterior a pantalla (para la línea de preview)
            var lastTapScreenPt: CGPoint? = nil
            if let lt = cachedLastTap {
                let proj = renderer.projectPoint(SCNVector3(lt.x, lt.y, lt.z))
                if proj.z > 0, proj.z < 1 {
                    lastTapScreenPt = CGPoint(x: CGFloat(proj.x), y: CGFloat(proj.y))
                }
            }

            Task { @MainActor [weak self] in
                guard let self, let vm = self.viewModel else { return }
                vm.crosshairHit    = stablePoint != nil
                vm.liveAimPoint    = stablePoint
                vm.isAimStable     = isStable
                vm.lastTapScreen   = lastTapScreenPt
                vm.crosshairSnapPt = snapPt
                self.cachedLastTap  = vm.tapPhase == .waitingSecond ? vm.firstPoint : nil
                self.cachedAimPoint = stablePoint
            }
        }

        // MARK: - Helpers

        private func projectToScreen(_ wp: simd_float3, camera: ARCamera, size: CGSize) -> CGPoint? {
            let view = simd_inverse(camera.transform)
            let cam  = view * simd_float4(wp.x, wp.y, wp.z, 1)
            guard cam.z < -0.05 else { return nil }
            let proj = camera.projectionMatrix(for: .portrait, viewportSize: size,
                                               zNear: 0.001, zFar: 100)
            let clip = proj * cam
            let ndc  = simd_float2(clip.x, clip.y) / clip.w
            guard abs(ndc.x) <= 1.3, abs(ndc.y) <= 1.3 else { return nil }
            return CGPoint(
                x: CGFloat((ndc.x + 1) * 0.5) * size.width,
                y: CGFloat((1 - ndc.y) * 0.5) * size.height
            )
        }
    }
}
