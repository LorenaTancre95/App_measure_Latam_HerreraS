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
    private let statusLabel    = UILabel()
    private let confirmedLabel = UILabel()
    private let volumeLabel    = UILabel()
    private let confirmBtn     = UIButton(type: .custom)
    private let remeasureBtn   = UIButton(type: .custom)
    private let modeControl    = UISegmentedControl(items: ["AUTO", "MANUAL"])
    private let frameView      = CornerFrameView()
    private let bottomPanel    = UIView()
    private let reticleView    = ReticleView()

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
        // Scan-frame overlay
        frameView.translatesAutoresizingMaskIntoConstraints = false
        frameView.isUserInteractionEnabled = false
        view.addSubview(frameView)

        // Reticle (green circle, always centered in camera area)
        reticleView.translatesAutoresizingMaskIntoConstraints = false
        reticleView.isUserInteractionEnabled = false
        view.addSubview(reticleView)

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
        modeControl.setTitleTextAttributes([.foregroundColor: UIColor.white], for: .normal)
        modeControl.setTitleTextAttributes(
            [.foregroundColor: UIColor.black, .font: UIFont.boldSystemFont(ofSize: 13)],
            for: .selected)
        modeControl.addTarget(self, action: #selector(modeChanged), for: .valueChanged)
        modeControl.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(modeControl)

        // Bottom panel
        bottomPanel.backgroundColor = UIColor(red: 0.07, green: 0.10, blue: 0.18, alpha: 0.95)
        bottomPanel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(bottomPanel)

        // Status label — scanning state
        statusLabel.text          = "Aponte para a caixa"
        statusLabel.textColor     = .white
        statusLabel.font          = UIFont.systemFont(ofSize: 17, weight: .semibold)
        statusLabel.textAlignment = .center
        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        bottomPanel.addSubview(statusLabel)

        // Confirmed label — shown after measurement locks
        confirmedLabel.text          = "✅  Medição confirmada"
        confirmedLabel.textColor     = UIColor(red: 0.2, green: 0.85, blue: 0.4, alpha: 1)
        confirmedLabel.font          = UIFont.boldSystemFont(ofSize: 17)
        confirmedLabel.textAlignment = .center
        confirmedLabel.isHidden      = true
        confirmedLabel.translatesAutoresizingMaskIntoConstraints = false
        bottomPanel.addSubview(confirmedLabel)

        // Volume / cubic weight label
        volumeLabel.textColor     = UIColor.white.withAlphaComponent(0.85)
        volumeLabel.font          = UIFont.systemFont(ofSize: 14)
        volumeLabel.textAlignment = .center
        volumeLabel.isHidden      = true
        volumeLabel.translatesAutoresizingMaskIntoConstraints = false
        bottomPanel.addSubview(volumeLabel)

        // Remedir button
        remeasureBtn.setTitle("↺  Remedir", for: .normal)
        remeasureBtn.setTitleColor(.white, for: .normal)
        remeasureBtn.titleLabel?.font   = UIFont.boldSystemFont(ofSize: 15)
        remeasureBtn.backgroundColor    = UIColor.white.withAlphaComponent(0.15)
        remeasureBtn.layer.cornerRadius = 22
        remeasureBtn.layer.borderWidth  = 1
        remeasureBtn.layer.borderColor  = UIColor.white.withAlphaComponent(0.3).cgColor
        remeasureBtn.clipsToBounds      = true
        remeasureBtn.isHidden           = true
        remeasureBtn.addTarget(self, action: #selector(didTapRemeasure), for: .touchUpInside)
        remeasureBtn.translatesAutoresizingMaskIntoConstraints = false
        bottomPanel.addSubview(remeasureBtn)

        // Usar button (green)
        confirmBtn.setTitle("✓  Usar", for: .normal)
        confirmBtn.setTitleColor(.black, for: .normal)
        confirmBtn.titleLabel?.font   = UIFont.boldSystemFont(ofSize: 15)
        confirmBtn.backgroundColor    = UIColor(red: 0.2, green: 0.85, blue: 0.4, alpha: 1)
        confirmBtn.layer.cornerRadius = 22
        confirmBtn.clipsToBounds      = true
        confirmBtn.isHidden           = true
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
            bottomPanel.heightAnchor.constraint(equalToConstant: 160),

            frameView.topAnchor.constraint(equalTo: closeBtn.bottomAnchor, constant: 16),
            frameView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 40),
            frameView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -40),
            frameView.bottomAnchor.constraint(equalTo: bottomPanel.topAnchor, constant: -16),

            reticleView.centerXAnchor.constraint(equalTo: frameView.centerXAnchor),
            reticleView.centerYAnchor.constraint(equalTo: frameView.centerYAnchor),
            reticleView.widthAnchor.constraint(equalToConstant: 44),
            reticleView.heightAnchor.constraint(equalToConstant: 44),

            // Scanning state: statusLabel near top, vertically centered in usable panel space
            statusLabel.centerYAnchor.constraint(equalTo: bottomPanel.topAnchor, constant: 50),
            statusLabel.leadingAnchor.constraint(equalTo: bottomPanel.leadingAnchor, constant: 16),
            statusLabel.trailingAnchor.constraint(equalTo: bottomPanel.trailingAnchor, constant: -16),

            // Confirmed state
            confirmedLabel.topAnchor.constraint(equalTo: bottomPanel.topAnchor, constant: 14),
            confirmedLabel.leadingAnchor.constraint(equalTo: bottomPanel.leadingAnchor, constant: 16),
            confirmedLabel.trailingAnchor.constraint(equalTo: bottomPanel.trailingAnchor, constant: -16),

            volumeLabel.topAnchor.constraint(equalTo: confirmedLabel.bottomAnchor, constant: 5),
            volumeLabel.leadingAnchor.constraint(equalTo: bottomPanel.leadingAnchor, constant: 16),
            volumeLabel.trailingAnchor.constraint(equalTo: bottomPanel.trailingAnchor, constant: -16),

            remeasureBtn.leadingAnchor.constraint(equalTo: bottomPanel.leadingAnchor, constant: 16),
            remeasureBtn.bottomAnchor.constraint(equalTo: safe.bottomAnchor, constant: -14),
            remeasureBtn.heightAnchor.constraint(equalToConstant: 44),

            confirmBtn.leadingAnchor.constraint(equalTo: remeasureBtn.trailingAnchor, constant: 12),
            confirmBtn.trailingAnchor.constraint(equalTo: bottomPanel.trailingAnchor, constant: -16),
            confirmBtn.bottomAnchor.constraint(equalTo: safe.bottomAnchor, constant: -14),
            confirmBtn.heightAnchor.constraint(equalToConstant: 44),
            confirmBtn.widthAnchor.constraint(equalTo: remeasureBtn.widthAnchor, multiplier: 1.6),
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
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            cfg.sceneReconstruction = .mesh
        }
        sceneView.session.run(cfg, options: [.resetTracking, .removeExistingAnchors])
    }

    // MARK: - Measurement callback
    private func handleMeasurement(_ m: NativeMeasurement) {
        let volM3      = (m.comprimento / 100) * (m.largura / 100) * (m.altura / 100)
        let pesoCubado = volM3 * 167   // standard road-freight factor kg/m³

        let volStr  = String(format: "%.4f", volM3).replacingOccurrences(of: ".", with: ",")
        let pesoStr = String(format: "%.1f", pesoCubado).replacingOccurrences(of: ".", with: ",")
        volumeLabel.text = "Vol: \(volStr) m³  ·  Peso Cubado: \(pesoStr) kg"

        statusLabel.isHidden    = true
        confirmedLabel.isHidden = false
        volumeLabel.isHidden    = false
        remeasureBtn.isHidden   = false
        confirmBtn.isHidden     = false
        frameView.setGreen()
    }

    // MARK: - Actions
    @objc private func didTapRemeasure() {
        coordinator.reset()
        statusLabel.isHidden    = false
        confirmedLabel.isHidden = true
        volumeLabel.isHidden    = true
        remeasureBtn.isHidden   = true
        confirmBtn.isHidden     = true
        frameView.resetColor()
    }

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

    func resetColor() {
        guard isGreen else { return }
        isGreen     = false
        strokeColor = UIColor(red: 1, green: 0.8, blue: 0, alpha: 1)
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

// MARK: - Reticle overlay (green circle + center dot)
private final class ReticleView: UIView {
    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .clear
        isOpaque        = false
    }
    required init?(coder: NSCoder) { fatalError() }

    override func draw(_ rect: CGRect) {
        guard let ctx = UIGraphicsGetCurrentContext() else { return }
        let green = UIColor(red: 0.2, green: 0.85, blue: 0.2, alpha: 1)

        ctx.setStrokeColor(green.cgColor)
        ctx.setLineWidth(2.5)
        ctx.strokeEllipse(in: rect.insetBy(dx: 2, dy: 2))

        ctx.setFillColor(green.cgColor)
        let d: CGFloat = 6
        ctx.fillEllipse(in: CGRect(x: rect.midX - d / 2,
                                   y: rect.midY - d / 2,
                                   width: d, height: d))
    }
}
