import ExpoModulesCore
import UIKit

// Minimal test: no ARKit — just a plain UIView.
// If this mounts without crashing we know the issue is in ARKit/ARSCNView init.
enum ARMode { case auto, manual }

class ARBoxView: ExpoView {

    // MARK: - Events
    let onMeasurementUpdate    = EventDispatcher()
    let onMeasurementConfirmed = EventDispatcher()
    let onPlaneFound           = EventDispatcher()

    // MARK: - Props
    var arMode: ARMode = .auto

    // MARK: - Init
    required init(appContext: AppContext? = nil) {
        super.init(appContext: appContext)
    }

    // MARK: - Lifecycle
    override func didMoveToSuperview() {
        super.didMoveToSuperview()
        guard superview != nil else { return }

        // Dark blue background — visible proof the view mounted correctly.
        backgroundColor = UIColor(red: 0.05, green: 0.05, blue: 0.25, alpha: 1)

        addSimulatedLabel()

        // After 2 s emit a fake measurement to verify the JS event bridge works.
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in
            self?.onMeasurementUpdate([
                "comprimento": 60.0,
                "largura":     40.0,
                "altura":      30.0,
            ])
        }
    }

    private func addSimulatedLabel() {
        let label = UILabel()
        label.text = "ARBoxView montado ✓\n(sem ARKit — teste de isolamento)"
        label.textColor = .white
        label.numberOfLines = 0
        label.textAlignment = .center
        label.font = UIFont.systemFont(ofSize: 16, weight: .medium)
        label.translatesAutoresizingMaskIntoConstraints = false
        addSubview(label)
        NSLayoutConstraint.activate([
            label.centerXAnchor.constraint(equalTo: centerXAnchor),
            label.centerYAnchor.constraint(equalTo: centerYAnchor),
            label.leadingAnchor.constraint(greaterThanOrEqualTo: leadingAnchor, constant: 24),
            label.trailingAnchor.constraint(lessThanOrEqualTo: trailingAnchor, constant: -24),
        ])
    }

    // MARK: - Public API (stubs — no ARKit)
    func confirm() {
        onMeasurementConfirmed([
            "comprimento": 60.0,
            "largura":     40.0,
            "altura":      30.0,
        ])
    }
    func reset() {}
}
