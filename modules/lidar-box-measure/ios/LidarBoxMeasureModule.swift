import ExpoModulesCore
import UIKit

public class LidarBoxMeasureModule: Module {
    public func definition() -> ModuleDefinition {
        Name("LidarBoxMeasure")

        // Present a fully-native UIViewController with ARKit.
        // This avoids embedding any native view in the RN hierarchy (which crashes on iOS 26).
        AsyncFunction("openARCamera") { (promise: Promise) in
            DispatchQueue.main.async {
                let vc = ARCameraViewController()
                vc.modalPresentationStyle = .fullScreen
                vc.onMeasurement = { m in
                    promise.resolve([
                        "comprimento": m.comprimento,
                        "largura":     m.largura,
                        "altura":      m.altura,
                    ])
                }
                vc.onCancel = {
                    promise.reject("CANCELLED", "cancelled", nil)
                }
                guard let root = Self.topVC() else {
                    promise.reject("NO_VC", "no root view controller found", nil)
                    return
                }
                root.present(vc, animated: true)
            }
        }
    }

    private static func topVC() -> UIViewController? {
        let scene = UIApplication.shared.connectedScenes
            .compactMap { $0 as? UIWindowScene }.first
        var vc: UIViewController? = scene?.keyWindow?.rootViewController
            ?? scene?.windows.first?.rootViewController
        while let presented = vc?.presentedViewController { vc = presented }
        return vc
    }
}
