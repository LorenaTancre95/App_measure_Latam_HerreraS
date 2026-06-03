import ExpoModulesCore
import UIKit

public class LidarBoxMeasureModule: Module {
    public func definition() -> ModuleDefinition {
        Name("LidarBoxMeasure")

        AsyncFunction("openARCamera") { () async throws -> [String: Double] in
            try await withCheckedThrowingContinuation { continuation in
                DispatchQueue.main.async {
                    let vc = ARCameraViewController()
                    vc.modalPresentationStyle = .fullScreen

                    var done = false

                    vc.onMeasurement = { m in
                        guard !done else { return }
                        done = true
                        continuation.resume(returning: [
                            "comprimento": m.comprimento,
                            "largura":     m.largura,
                            "altura":      m.altura,
                        ])
                    }

                    vc.onCancel = {
                        guard !done else { return }
                        done = true
                        continuation.resume(throwing: ARCancelledError())
                    }

                    guard let root = lidarTopViewController() else {
                        done = true
                        continuation.resume(throwing: ARCancelledError())
                        return
                    }
                    root.present(vc, animated: true)
                }
            }
        }
    }
}

// MARK: - Helpers (module-level, avoids Self in closure ambiguity)

private struct ARCancelledError: Error {}

private func lidarTopViewController() -> UIViewController? {
    let scene = UIApplication.shared.connectedScenes
        .compactMap { $0 as? UIWindowScene }.first
    var vc = scene?.windows.first(where: { $0.isKeyWindow })?.rootViewController
          ?? scene?.windows.first?.rootViewController
    while let p = vc?.presentedViewController { vc = p }
    return vc
}
