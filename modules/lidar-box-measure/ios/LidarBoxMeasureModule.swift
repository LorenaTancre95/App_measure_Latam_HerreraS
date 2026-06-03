import ExpoModulesCore
import UIKit

// Return type from the async function — Record is the correct ExpoModulesCore way
// to return a structured object. @Field requires AnyArgument; Double conforms.
struct ARMeasurementResult: Record {
    @Field var comprimento: Double = 0
    @Field var largura:     Double = 0
    @Field var altura:      Double = 0
}

public class LidarBoxMeasureModule: Module {
    public func definition() -> ModuleDefinition {
        Name("LidarBoxMeasure")

        // AsyncFunction with Swift async/await — this is the correct API in
        // ExpoModulesCore 1.12 (Expo SDK 51). The Promise-based closure style
        // does not exist in this version.
        AsyncFunction("openARCamera") { () async throws -> ARMeasurementResult in
            try await withCheckedThrowingContinuation { continuation in
                DispatchQueue.main.async {
                    let vc = ARCameraViewController()
                    vc.modalPresentationStyle = .fullScreen

                    var done = false   // guard against double-resume

                    vc.onMeasurement = { m in
                        guard !done else { return }
                        done = true
                        var result = ARMeasurementResult()
                        result.comprimento = m.comprimento
                        result.largura     = m.largura
                        result.altura      = m.altura
                        continuation.resume(returning: result)
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
