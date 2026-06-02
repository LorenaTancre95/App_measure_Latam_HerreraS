import ExpoModulesCore
import ARKit
import AVFoundation
import Vision
import SceneKit

enum ARMode { case auto, manual }

class ARBoxView: ExpoView {

    // MARK: - Events
    let onMeasurementUpdate    = EventDispatcher()
    let onMeasurementConfirmed = EventDispatcher()
    let onPlaneFound           = EventDispatcher()

    // MARK: - Properties
    var arMode: ARMode = .auto {
        didSet { coordinator.mode = arMode }
    }

    // MARK: - Internals
    private let sceneView  = ARSCNView()
    private lazy var coordinator = BoxDetectionCoordinator(
        sceneView: sceneView,
        onUpdate: { [weak self] m in
            self?.onMeasurementUpdate([
                "comprimento": m.comprimento,
                "largura":     m.largura,
                "altura":      m.altura,
            ])
        },
        onPlaneFound: { [weak self] in
            self?.onPlaneFound([:])
        }
    )

    // MARK: - Init
    required init(appContext: AppContext? = nil) {
        super.init(appContext: appContext)
        setupSceneView()
        setupGesture()
        startSession()
    }

    private func setupSceneView() {
        sceneView.translatesAutoresizingMaskIntoConstraints = false
        addSubview(sceneView)
        NSLayoutConstraint.activate([
            sceneView.topAnchor.constraint(equalTo: topAnchor),
            sceneView.bottomAnchor.constraint(equalTo: bottomAnchor),
            sceneView.leadingAnchor.constraint(equalTo: leadingAnchor),
            sceneView.trailingAnchor.constraint(equalTo: trailingAnchor),
        ])
        sceneView.delegate = coordinator
        sceneView.session.delegate = coordinator
        sceneView.autoenablesDefaultLighting = true
    }

    private func setupGesture() {
        let tap = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        sceneView.addGestureRecognizer(tap)
    }

    private func startSession() {
        guard ARWorldTrackingConfiguration.isSupported else { return }

        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            runARSession()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                guard granted else { return }
                DispatchQueue.main.async { self?.runARSession() }
            }
        default:
            break
        }
    }

    private func runARSession() {
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal]

        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics = [.sceneDepth, .smoothedSceneDepth]
        }

        sceneView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    // MARK: - Manual tap
    @objc private func handleTap(_ r: UITapGestureRecognizer) {
        guard arMode == .manual else { return }

        let loc = r.location(in: sceneView)
        guard
            let query = sceneView.raycastQuery(from: loc,
                                               allowing: .existingPlaneGeometry,
                                               alignment: .any),
            let result = sceneView.session.raycast(query).first
        else { return }

        let p = result.worldTransform.columns.3
        coordinator.addManualPoint(SIMD3<Float>(p.x, p.y, p.z))
    }

    // MARK: - Public: confirm (called from RN ref)
    func confirm() {
        guard let m = coordinator.lastMeasurement else { return }
        onMeasurementConfirmed([
            "comprimento": m.comprimento,
            "largura":     m.largura,
            "altura":      m.altura,
        ])
        coordinator.clearOverlay()
    }

    func reset() {
        coordinator.reset()
        startSession()
    }
}
