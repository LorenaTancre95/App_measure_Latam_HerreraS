import ExpoModulesCore
import ARKit

public class LidarBoxMeasureModule: Module {
    public func definition() -> ModuleDefinition {
        Name("LidarBoxMeasure")

        View(ARBoxView.self) {
            Prop("mode") { (view: ARBoxView, mode: String) in
                view.arMode = mode == "manual" ? .manual : .auto
            }

            Events("onMeasurementUpdate", "onMeasurementConfirmed", "onPlaneFound")
        }
    }
}
