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

        // Tap directo para medición ALTO (sin mover el teléfono → Y preciso)
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
        var cachedLastTap: simd_float3? = nil

        // Buffer de los últimos puntos LiDAR para calcular la mediana → punto estable
        private var aimBuffer: [simd_float3] = []
        private let bufferSize = 6           // máximo historial (descarta el más viejo)
        private let stableFrames = 3         // solo miro los últimos 3 frames → 0.3s
        private let stableThreshold: Float = 0.020  // 2 cm — suficiente para carga

        init(_ viewModel: ARViewModel) { self.viewModel = viewModel }

        @objc func handleScreenTap(_ gesture: UITapGestureRecognizer) {
            guard let vm = viewModel, let sv = sceneView,
                  vm.dimPhase == .alto, vm.tapPhase != .preview else { return }
            let pt = gesture.location(in: sv)
            Task { @MainActor in vm.captureTap(at: pt) }
        }

        // Materializa los ARAnchor de tipo "marker" como esferas amarillas.
        // ARKit reposiciona el nodo automáticamente cuando refina el mapa del mundo.
        func renderer(_ renderer: SCNSceneRenderer, nodeFor anchor: ARAnchor) -> SCNNode? {
            guard anchor.name == "marker" else { return nil }
            let sphere = SCNSphere(radius: 0.006)
            let mat = SCNMaterial()
            mat.diffuse.contents = UIColor.systemYellow
            sphere.materials = [mat]
            return SCNNode(geometry: sphere)
        }

        // Captura el plano del piso cada vez que ARKit detecta o actualiza un plano horizontal
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

        /// Cada 0.1s: acumula el punto LiDAR en el buffer y publica la mediana.
        /// La mediana es estable incluso si el teléfono vibra levemente.
        func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
            guard time - lastHitCheck > 0.1 else { return }
            lastHitCheck = time
            guard let sv = sceneView else { return }

            let center = CGPoint(x: sv.bounds.midX, y: sv.bounds.midY)
            let frame  = sv.session.currentFrame

            // LiDAR primero (no atraviesa la caja hacia el fondo), raycast como fallback
            var rawHit: simd_float3? = nil
            if let f = frame {
                rawHit = ARViewModel.lidarPoint(frame: f, screenPoint: center, viewportSize: sv.bounds.size)
            }
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

            // Actualizar el buffer con el nuevo punto (o vaciar si no hay hit)
            if let pt = rawHit {
                aimBuffer.append(pt)
                if aimBuffer.count > bufferSize { aimBuffer.removeFirst() }
            } else {
                aimBuffer.removeAll()
            }

            // Mediana 3D: más robusta que el promedio frente a lecturas outlier
            var stablePoint: simd_float3? = nil
            var isStable = false
            if aimBuffer.count >= 3 {
                let xs = aimBuffer.map(\.x).sorted()
                let ys = aimBuffer.map(\.y).sorted()
                let zs = aimBuffer.map(\.z).sorted()
                let mid = aimBuffer.count / 2
                let med = simd_float3(xs[mid], ys[mid], zs[mid])
                stablePoint = med
                // Estable si los últimos `stableFrames` puntos están dentro del threshold.
                // Solo miro el sufijo reciente → responde en ~0.3s sin esperar historial completo.
                let recent = aimBuffer.suffix(stableFrames)
                let maxDist = recent.map { simd_distance($0, med) }.max() ?? 0
                isStable = recent.count >= stableFrames && maxDist < stableThreshold
            }

            // Proyectar último tap a pantalla (para la línea de preview)
            var lastTapScreenPt: CGPoint? = nil
            if let lt = cachedLastTap {
                let proj = renderer.projectPoint(SCNVector3(lt.x, lt.y, lt.z))
                if proj.z > 0, proj.z < 1 {
                    lastTapScreenPt = CGPoint(x: CGFloat(proj.x), y: CGFloat(proj.y))
                }
            }

            Task { @MainActor [weak self] in
                guard let self, let vm = self.viewModel else { return }
                vm.crosshairHit  = stablePoint != nil
                vm.liveAimPoint  = stablePoint          // mediana, no el raw
                vm.isAimStable   = isStable
                vm.lastTapScreen = lastTapScreenPt
                self.cachedLastTap = vm.tapPhase == .waitingSecond ? vm.firstPoint : nil
            }
        }
    }
}
