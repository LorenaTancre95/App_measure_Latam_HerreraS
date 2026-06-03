import UIKit
import ARKit
import AVFoundation

final class ARCameraViewController: UIViewController {

    // MARK: - Callbacks
    var onMeasurement: ((NativeMeasurement) -> Void)?
    var onCancel:      (() -> Void)?

    // MARK: - AR
    private var sceneView:      ARSCNView!
    private var coordinator:    BoxDetectionCoordinator!
    private var didStartSession = false

    // MARK: - UI
    private let statusLabel = UILabel()
    private let confirmBtn  = UIButton(type: .custom)
    private let modeControl = UISegmentedControl(items: ["AUTO", "MANUAL"])
    private let frameView   = CornerFrameView()
    private let bottomPanel = UIView()

    // MARK: - Lifecycle
    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        buildARView()
        buildUI()
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        guard !didStartSession else { return }
        didStartSession = true
        requestCameraAndStart()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        sceneView?.session.pause()
    }

    // MARK: - AR
    private func buildARView() {
        sceneView = ARSCNView(frame: view.bounds)
        sceneView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        sceneView.autoenablesDefaultLighting = true
        view.insertSubview(sceneView, at: 0)

        coordinator = BoxDetectionCoordinator(
            sceneView: sceneView,
            onUpdate:  { [weak self] m in self?.handleMeasurement(m) },
            onPlaneFound: {}
        )
        sceneView.delegate         = coordinator
        sceneView.session.delegate = coordinator

        sceneView.addGestureRecognizer(
            UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        )
    }

    // MARK: - UI
    private func buildUI() {
        // Scan-frame overlay (UIView, no CALayer override needed)
        frameView.translatesAutoresizingMaskIntoConstraints = false
        frameView.isUserInteractionEnabled = false
        view.addSubview(frameView)

        // X close button
        let closeBtn = UIButton(type: .custom)
        closeBtn.setImage(
            UIImage(systemName: "xmark.circle.fill",
                    withConfiguration: UIImage.SymbolConfiguration(pointSize: 30)),
            for: .normal)
        closeBtn.tintColor = .white
        closeBtn.addTarget(self, action: #selector(didTapCancel), for: .touchUpInside)
        closeBtn.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(closeBtn)

        // Mode selector
        modeControl.selectedSegmentIndex     = 0
        modeControl.backgroundColor          = UIColor.black.withAlphaComponent(0.6)
        modeControl.selectedSegmentTintColor = .white
        modeControl.setTitleTextAttributes(
            [.foregroundColor: UIColor.white],
            for: .normal)
        modeControl.setTitleTextAttributes(
            [.foregroundColor: UIColor.black,
             .font: UIFont.boldSystemFont(ofSize: 13)],
            for: .selected)
        modeControl.addTarget(self, action: #selector(modeChanged), for: .valueChanged)
        modeControl.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(modeControl)

        // Bottom panel
        bottomPanel.backgroundColor = UIColor.black.withAlphaComponent(0.78)
        bottomPanel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(bottomPanel)

        // Status label
        statusLabel.text          = "Aponte para a caixa"
        statusLabel.textColor     = .white
        statusLabel.font          = UIFont.systemFont(ofSize: 17, weight: .semibold)
        statusLabel.textAlignment = .center
        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        bottomPanel.addSubview(statusLabel)

        // Confirm button
        confirmBtn.setTitle("Confirmar medição", for: .normal)
        confirmBtn.setTitleColor(.black, for: .normal)
        confirmBtn.titleLabel?.font = UIFont.boldSystemFont(ofSize: 15)
        confirmBtn.backgroundColor  = UIColor(red: 1, green: 0.8, blue: 0, alpha: 1)
        confirmBtn.layer.cornerRadius = 22
        confirmBtn.clipsToBounds = true
        confirmBtn.isHidden = true
        confirmBtn.addTarget(self, action: #selector(didTapConfirm), for: .touchUpInside)
        confirmBtn.translatesAutoresizingMaskIntoConstraints = false
        bottomPanel.addSubview(confirmBtn)

        let safe = view.safeAreaLayoutGuide
        NSLayoutConstraint.activate([
            closeBtn.topAnchor.constraint(equalTo: safe.topAnchor, constant: 6),
            closeBtn.leadingAnchor.constraint(equalTo: safe.leadingAnchor, constant: 10),
            closeBtn.widthAnchor.constraint(equalToConstant: 44),
            closeBtn.heightAnchor.constraint(equalToConstant: 44),

            modeControl.centerYAnchor.constraint(equalTo: closeBtn.centerYAnchor),
            modeControl.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            modeControl.widthAnchor.constraint(equalToConstant: 190),
            modeControl.heightAnchor.constraint(equalToConstant: 36),

            bottomPanel.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            bottomPanel.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            bottomPanel.bottomAnchor.constraint(equalTo: view.bottomAnchor),

            frameView.topAnchor.constraint(equalTo: closeBtn.bottomAnchor, constant: 16),
            frameView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 40),
            frameView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -40),
            frameView.bottomAnchor.constraint(equalTo: bottomPanel.topAnchor, constant: -16),

            statusLabel.topAnchor.constraint(equalTo: bottomPanel.topAnchor, constant: 14),
            statusLabel.leadingAnchor.constraint(equalTo: bottomPanel.leadingAnchor, constant: 16),
            statusLabel.trailingAnchor.constraint(equalTo: bottomPanel.trailingAnchor, constant: -16),

            confirmBtn.topAnchor.constraint(equalTo: statusLabel.bottomAnchor, constant: 10),
            confirmBtn.centerXAnchor.constraint(equalTo: bottomPanel.centerXAnchor),
            confirmBtn.widthAnchor.constraint(greaterThanOrEqualToConstant: 200),
            confirmBtn.heightAnchor.constraint(equalToConstant: 44),
            confirmBtn.bottomAnchor.constraint(equalTo: safe.bottomAnchor, constant: -14),
        ])
    }

    // MARK: - Camera → ARKit
    private func requestCameraAndStart() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            startARSession()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async { if granted { self?.startARSession() } }
            }
        default:
            statusLabel.text = "Permissão de câmera negada"
        }
    }

    private func startARSession() {
        guard ARWorldTrackingConfiguration.isSupported else {
            statusLabel.text = "ARKit não suportado neste dispositivo"
            return
        }
        let cfg = ARWorldTrackingConfiguration()
        cfg.planeDetection = [.horizontal]
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            cfg.frameSemantics = [.sceneDepth, .smoothedSceneDepth]
        }
        sceneView.session.run(cfg, options: [.resetTracking, .removeExistingAnchors])
    }

    // MARK: - Measurement callback
    private func handleMeasurement(_ m: NativeMeasurement) {
        let c = Int(m.comprimento.rounded())
        let l = Int(m.largura.rounded())
        let a = Int(m.altura.rounded())
        statusLabel.text = "C: \(c)  ×  L: \(l)  ×  A: \(a) cm"
        confirmBtn.isHidden = false
        frameView.setGreen()
    }

    // MARK: - Actions
    @objc private func didTapCancel() {
        sceneView.session.pause()
        dismiss(animated: true) { [weak self] in self?.onCancel?() }
    }

    @objc private func didTapConfirm() {
        guard let m = coordinator.lastMeasurement else { return }
        sceneView.session.pause()
        dismiss(animated: true) { [weak self] in self?.onMeasurement?(m) }
    }

    @objc private func modeChanged() {
        coordinator.mode = modeControl.selectedSegmentIndex == 0 ? .auto : .manual
    }

    @objc private func handleTap(_ r: UITapGestureRecognizer) {
        guard coordinator.mode == .manual else { return }
        let loc = r.location(in: sceneView)
        guard
            let q   = sceneView.raycastQuery(from: loc,
                                             allowing: .existingPlaneGeometry,
                                             alignment: .any),
            let hit = sceneView.session.raycast(q).first
        else { return }
        let p = hit.worldTransform.columns.3
        coordinator.addManualPoint(SIMD3<Float>(p.x, p.y, p.z))
    }
}

// MARK: - Corner-frame overlay (UIView with Core Graphics drawing — no CALayer subclassing)
private final class CornerFrameView: UIView {
    private var strokeColor: UIColor = UIColor(red: 1, green: 0.8, blue: 0, alpha: 1)
    private var isGreen              = false

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .clear
        isOpaque        = false
    }
    required init?(coder: NSCoder) { fatalError() }

    func setGreen() {
        guard !isGreen else { return }
        isGreen     = true
        strokeColor = UIColor(red: 0.2, green: 0.85, blue: 0.2, alpha: 1)
        setNeedsDisplay()
    }

    override func draw(_ rect: CGRect) {
        guard let ctx = UIGraphicsGetCurrentContext() else { return }
        ctx.setStrokeColor(strokeColor.cgColor)
        ctx.setLineWidth(3)
        ctx.setLineCap(.square)

        let b   = bounds
        let len: CGFloat = 28

        // Top-left
        ctx.move(to: CGPoint(x: 0, y: len))
        ctx.addLine(to: CGPoint(x: 0, y: 0))
        ctx.addLine(to: CGPoint(x: len, y: 0))

        // Top-right
        ctx.move(to: CGPoint(x: b.maxX - len, y: 0))
        ctx.addLine(to: CGPoint(x: b.maxX, y: 0))
        ctx.addLine(to: CGPoint(x: b.maxX, y: len))

        // Bottom-left
        ctx.move(to: CGPoint(x: 0, y: b.maxY - len))
        ctx.addLine(to: CGPoint(x: 0, y: b.maxY))
        ctx.addLine(to: CGPoint(x: len, y: b.maxY))

        // Bottom-right
        ctx.move(to: CGPoint(x: b.maxX - len, y: b.maxY))
        ctx.addLine(to: CGPoint(x: b.maxX, y: b.maxY))
        ctx.addLine(to: CGPoint(x: b.maxX, y: b.maxY - len))

        ctx.strokePath()
    }
}
