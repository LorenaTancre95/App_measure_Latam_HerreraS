import SwiftUI
import ARKit
import SceneKit
import Vision

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

        // Tap para LARGO (modo tap, estado largoTap)
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
        private var lastRectCheck: TimeInterval = 0
        private var isRunningRect = false

        /// VNDetectRectanglesRequest configurado para caras de cajas.
        private lazy var rectRequest: VNDetectRectanglesRequest = {
            let req = VNDetectRectanglesRequest()
            req.minimumAspectRatio = 0.15
            req.maximumAspectRatio = 1.0
            req.minimumSize       = 0.10   // mín 10% del ancho/alto de la imagen
            req.maximumObservations = 1
            req.minimumConfidence  = 0.6
            return req
        }()

        init(_ viewModel: ARViewModel) { self.viewModel = viewModel }

        // Crosshair hit cada 0.1s + detección de rectángulo cada 0.3s
        func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
            guard let sv = sceneView else { return }

            // — Crosshair hit (liveAimPoint) —
            if time - lastHitCheck > 0.1 {
                lastHitCheck = time
                let center = CGPoint(x: sv.bounds.midX, y: sv.bounds.midY)
                var hitPoint: simd_float3? = nil
                let frame = sv.session.currentFrame
                if let frame {
                    hitPoint = ARViewModel.lidarPoint(frame: frame,
                                                      screenPoint: center,
                                                      viewportSize: sv.bounds.size)
                }
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
                        } else { hitPoint = rayPt }
                        break
                    }
                }
                let hit = hitPoint != nil
                Task { @MainActor [weak self] in
                    self?.viewModel?.crosshairHit = hit
                    self?.viewModel?.liveAimPoint = hitPoint
                }
            }

            // — Detección de rectángulo cada 0.3s (solo en estado .detecting) —
            guard time - lastRectCheck > 0.3, !isRunningRect else { return }
            lastRectCheck = time

            guard let vm = viewModel,
                  vm.measureMode == .tap else { return }
            // Verificar estado en main antes de disparar
            let shouldRun = vm.rectState == .detecting   // @MainActor published, safe to read from render thread as snapshot
            guard shouldRun else { return }

            guard let frame = sv.session.currentFrame else { return }
            let vp = sv.bounds.size

            isRunningRect = true
            let capturedFrame = frame
            DispatchQueue.global(qos: .userInteractive).async { [weak self] in
                guard let self else { return }
                defer { self.isRunningRect = false }

                let handler = VNImageRequestHandler(
                    cvPixelBuffer: capturedFrame.capturedImage,
                    orientation: .right,   // imagen ARKit está en landscape, .right la pone portrait
                    options: [:]
                )
                try? handler.perform([self.rectRequest])

                guard let obs = self.rectRequest.results?.first else {
                    Task { @MainActor [weak self] in self?.viewModel?.clearLiveRect() }
                    return
                }

                // Convertir esquinas Vision → pantalla → 3D
                // Vision: origen bottom-left, y hacia arriba
                // Necesitamos origen top-left (flip Y) para pasar por displayTransform
                let visionCorners: [CGPoint] = [
                    obs.topLeft, obs.topRight, obs.bottomLeft, obs.bottomRight
                ]

                var screenPts: [CGPoint] = []
                var world3D:   [simd_float3] = []

                let displayT = capturedFrame.displayTransform(for: .portrait, viewportSize: vp)

                for vc in visionCorners {
                    let imgNorm = CGPoint(x: vc.x, y: 1.0 - vc.y)  // flip Y
                    let sNorm   = imgNorm.applying(displayT)
                    let screen  = CGPoint(x: sNorm.x * vp.width, y: sNorm.y * vp.height)
                    screenPts.append(screen)
                    if let pt3D = ARViewModel.lidarPoint(frame: capturedFrame,
                                                         screenPoint: screen,
                                                         viewportSize: vp) {
                        world3D.append(pt3D)
                    }
                }

                guard world3D.count == 4 else {
                    Task { @MainActor [weak self] in self?.viewModel?.clearLiveRect() }
                    return
                }

                Task { @MainActor [weak self] in
                    self?.viewModel?.updateLiveRect(corners3D: world3D, cornersScreen: screenPts)
                }
            }
        }

        @objc func handleTap(_ recognizer: UITapGestureRecognizer) {
            guard let vm = viewModel, vm.measureMode == .tap,
                  vm.rectState == .largoTap,
                  let sv = recognizer.view as? ARSCNView else { return }
            let point = recognizer.location(in: sv)
            Task { @MainActor in vm.measureLargoAt(point: point) }
        }
    }
}
