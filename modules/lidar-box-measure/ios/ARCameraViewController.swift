import UIKit
import ARKit
import AVFoundation

final class ARCameraViewController: UIViewController {

    // MARK: - Callbacks
    var onMeasurement: ((NativeMeasurement) -> Void)?
    var onCancel:      (() -> Void)?

    // MARK: - AR
    private var sceneView:   ARSCNView!
    private var coordinator: BoxDetectionCoordinator!
    private var didStartSession = false

    // MARK: - UI
    private let statusLabel = UILabel()
    private let confirmBtn  = UIButton(type: .system)
    private let modeControl = UISegmentedControl(items: ["AUTO", "MANUAL"])
    private let frameLayer  = CornerFrameLayer()

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

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        let inset: CGFloat = 40
        frameLayer.frame = view.bounds.insetBy(dx: inset, dy: inset + 40)
    }

    // MARK: - AR setup
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

    // MARK: - UI setup
    private func buildUI() {
        view.layer.addSublayer(frameLayer)

        // X button
        let closeBtn = UIButton(type: .system)
        closeBtn.setImage(UIImage(systemName: "xmark.circle.fill"), for: .normal)
        closeBtn.tintColor = .white
        closeBtn.contentVerticalAlignment   = .fill
        closeBtn.contentHorizontalAlignment = .fill
        closeBtn.addTarget(self, action: #selector(didTapCancel), for: .touchUpInside)
        closeBtn.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(closeBtn)

        // Mode selector
        modeControl.selectedSegmentIndex    = 0
        modeControl.backgroundColor         = UIColor.black.withAlphaComponent(0.6)
        modeControl.selectedSegmentTintColor = .white
        modeControl.setTitleTextAttributes(
            [.foregroundColor: UIColor.white], for: .normal)
        modeControl.setTitleTextAttributes(
            [.foregroundColor: UIColor.black, .font: UIFont.boldSystemFont(ofSize: 13)],
            for: .selected)
        modeControl.addTarget(self, action: #selector(modeChanged), for: .valueChanged)
        modeControl.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(modeControl)

        // Bottom panel
        let panel = UIView()
        panel.backgroundColor = UIColor.black.withAlphaComponent(0.78)
        panel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(panel)

        // Status label
        statusLabel.text          = "Aponte para a caixa"
        statusLabel.textColor     = .white
        statusLabel.font          = .systemFont(ofSize: 17, weight: .semibold)
        statusLabel.textAlignment = .center
        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        panel.addSubview(statusLabel)

        // Confirm button
        confirmBtn.setTitle("Confirmar medição", for: .normal)
        confirmBtn.titleLabel?.font = .boldSystemFont(ofSize: 15)
        confirmBtn.backgroundColor  = UIColor(red: 1, green: 0.8, blue: 0, alpha: 1)
        confirmBtn.setTitleColor(.black, for: .normal)
        confirmBtn.layer.cornerRadius = 22
        confirmBtn.isHidden = true
        confirmBtn.addTarget(self, action: #selector(didTapConfirm), for: .touchUpInside)
        confirmBtn.translatesAutoresizingMaskIntoConstraints = false
        panel.addSubview(confirmBtn)

        let safe = view.safeAreaLayoutGuide

        NSLayoutConstraint.activate([
            // X button — top-left
            closeBtn.topAnchor.constraint(equalTo: safe.topAnchor, constant: 6),
            closeBtn.leadingAnchor.constraint(equalTo: safe.leadingAnchor, constant: 10),
            closeBtn.widthAnchor.constraint(equalToConstant: 40),
            closeBtn.heightAnchor.constraint(equalToConstant: 40),

            // Mode control — top-center
            modeControl.centerYAnchor.constraint(equalTo: closeBtn.centerYAnchor),
            modeControl.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            modeControl.widthAnchor.constraint(equalToConstant: 190),
            modeControl.heightAnchor.constraint(equalToConstant: 36),

            // Bottom panel — sticks to bottom
            panel.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            panel.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            panel.bottomAnchor.constraint(equalTo: view.bottomAnchor),

            // Status label inside panel
            statusLabel.topAnchor.constraint(equalTo: panel.topAnchor, constant: 14),
            statusLabel.leadingAnchor.constraint(equalTo: panel.leadingAnchor, constant: 16),
            statusLabel.trailingAnchor.constraint(equalTo: panel.trailingAnchor, constant: -16),

            // Confirm button
            confirmBtn.topAnchor.constraint(equalTo: statusLabel.bottomAnchor, constant: 10),
            confirmBtn.centerXAnchor.constraint(equalTo: panel.centerXAnchor),
            confirmBtn.widthAnchor.constraint(greaterThanOrEqualToConstant: 200),
            confirmBtn.heightAnchor.constraint(equalToConstant: 44),
            confirmBtn.bottomAnchor.constraint(equalTo: safe.bottomAnchor, constant: -14),
        ])
    }

    // MARK: - Camera permission → ARKit
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

    // MARK: - Measurement callback (already on main thread via coordinator)
    private func handleMeasurement(_ m: NativeMeasurement) {
        let c = Int(m.comprimento.rounded())
        let l = Int(m.largura.rounded())
        let a = Int(m.altura.rounded())
        statusLabel.text = "C: \(c)  ×  L: \(l)  ×  A: \(a) cm"
        confirmBtn.isHidden = false
        frameLayer.setGreen()
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
            let q = sceneView.raycastQuery(from: loc, allowing: .existingPlaneGeometry, alignment: .any),
            let hit = sceneView.session.raycast(q).first
        else { return }
        let p = hit.worldTransform.columns.3
        coordinator.addManualPoint(SIMD3<Float>(p.x, p.y, p.z))
    }
}

// MARK: - Corner-frame overlay (CAShapeLayer, no UIView embedding required)
private final class CornerFrameLayer: CALayer {
    private let shape = CAShapeLayer()
    private var isGreen = false

    override init() {
        super.init()
        shape.fillColor   = UIColor.clear.cgColor
        shape.strokeColor = UIColor(red: 1, green: 0.8, blue: 0, alpha: 1).cgColor
        shape.lineWidth   = 3
        addSublayer(shape)
        isOpaque = false
    }
    required init?(coder: NSCoder) { fatalError() }

    func setGreen() {
        guard !isGreen else { return }
        isGreen = true
        CATransaction.begin()
        CATransaction.setAnimationDuration(0.3)
        shape.strokeColor = UIColor(red: 0.2, green: 0.85, blue: 0.2, alpha: 1).cgColor
        CATransaction.commit()
    }

    override var frame: CGRect {
        didSet { shape.frame = bounds; redraw() }
    }

    private func redraw() {
        let b   = bounds
        let len: CGFloat = 28
        let p   = CGMutablePath()

        // Top-left
        p.move(to: CGPoint(x: 0, y: len));     p.addLine(to: .zero)
        p.addLine(to: CGPoint(x: len, y: 0))
        // Top-right
        p.move(to: CGPoint(x: b.maxX - len, y: 0)); p.addLine(to: CGPoint(x: b.maxX, y: 0))
        p.addLine(to: CGPoint(x: b.maxX, y: len))
        // Bottom-left
        p.move(to: CGPoint(x: 0, y: b.maxY - len)); p.addLine(to: CGPoint(x: 0, y: b.maxY))
        p.addLine(to: CGPoint(x: len, y: b.maxY))
        // Bottom-right
        p.move(to: CGPoint(x: b.maxX - len, y: b.maxY)); p.addLine(to: CGPoint(x: b.maxX, y: b.maxY))
        p.addLine(to: CGPoint(x: b.maxX, y: b.maxY - len))

        shape.path = p
    }
}
