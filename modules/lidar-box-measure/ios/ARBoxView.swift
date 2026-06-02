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
        didSet { coordinator?.mode = arMode }
    }

    // MARK: - Internals
    private var sceneView: ARSCNView?
    private var coordinator: BoxDetectionCoordinator?
    private var sessionStarted = false

    // MARK: - Init — no UIKit here, safe for any thread
    required init(appContext: AppContext? = nil) {
        super.init(appContext: appContext)
    }

    // MARK: - Lifecycle — guaranteed main thread
    override func didMoveToSuperview() {
        super.didMoveToSuperview()
        guard superview != nil, !sessionStarted else { return }
        sessionStarted = true
        backgroundColor = .black
        requestCameraAndStart()
    }

    // MARK: - Permission → setup → session
    private func requestCameraAndStart() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            setupAndStart()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    if granted { self?.setupAndStart() }
                }
            }
        default:
            break
        }
    }

    private func setupAndStart() {
        guard ARWorldTrackingConfiguration.isSupported else { return }

        let sv = ARSCNView()
        sv.translatesAutoresizingMaskIntoConstraints = false
        sv.autoenablesDefaultLighting = true
        addSubview(sv)
        NSLayoutConstraint.activate([
            sv.topAnchor.constraint(equalTo: topAnchor),
            sv.bottomAnchor.constraint(equalTo: bottomAnchor),
            sv.leadingAnchor.constraint(equalTo: leadingAnchor),
            sv.trailingAnchor.constraint(equalTo: trailingAnchor),
        ])

        let coord = BoxDetectionCoordinator(
            sceneView: sv,
            onUpdate: { [weak self] m in
                self?.onMeasurementUpdate([
                    "comprimento": m.comprimento,
                    "largura":     m.largura,
                    "altura":      m.altura,
                ])
            },
            onPlaneFound: { [weak self] in self?.onPlaneFound([:]) }
        )
        sv.delegate = coord
        sv.session.delegate = coord
        sv.addGestureRecognizer(
            UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        )
        sceneView  = sv
        coordinator = coord
        coord.mode  = arMode

        runARSession(sv)
    }

    private func runARSession(_ sv: ARSCNView) {
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal]
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics = [.sceneDepth, .smoothedSceneDepth]
        }
        sv.session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    // MARK: - Manual tap
    @objc private func handleTap(_ r: UITapGestureRecognizer) {
        guard arMode == .manual, let sv = sceneView else { return }
        let loc = r.location(in: sv)
        guard
            let query = sv.raycastQuery(from: loc, allowing: .existingPlaneGeometry, alignment: .any),
            let result = sv.session.raycast(query).first
        else { return }
        let p = result.worldTransform.columns.3
        coordinator?.addManualPoint(SIMD3<Float>(p.x, p.y, p.z))
    }

    // MARK: - Public
    func confirm() {
        guard let m = coordinator?.lastMeasurement else { return }
        onMeasurementConfirmed([
            "comprimento": m.comprimento,
            "largura":     m.largura,
            "altura":      m.altura,
        ])
        coordinator?.clearOverlay()
    }

    func reset() {
        coordinator?.reset()
        if let sv = sceneView { runARSession(sv) }
    }
}
