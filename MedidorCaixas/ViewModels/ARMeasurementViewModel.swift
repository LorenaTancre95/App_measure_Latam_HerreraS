import Foundation
import Combine

// Single selection-based mode: user taps the object they want to measure.
enum ARMode { case selecting }

enum ARState {
    case searchingPlane     // scanning floor/table to detect horizontal plane
    case readyToSelect      // plane found, waiting for user tap
    case processing         // running segmentation + OBB computation
    case measuring          // result ready, waiting for confirm
    case confirmed          // measurement confirmed by user
}

class ARMeasurementViewModel: ObservableObject {
    @Published var state: ARState = .searchingPlane
    @Published var measurement: BoxMeasurement?
    @Published var errorMessage: String?

    // Set by ARSessionDelegate when a horizontal plane is detected
    var nearestPlaneY: Float?

    var mode: ARMode { .selecting }

    var statusText: String {
        switch state {
        case .searchingPlane: return "Aponte para o chão para calibrar"
        case .readyToSelect:  return "Toque na caixa que deseja medir"
        case .processing:     return "Analisando..."
        case .measuring:      return measurement?.displayText ?? "Medindo..."
        case .confirmed:      return "Medição confirmada"
        }
    }

    var isConfirmed: Bool { state == .confirmed }

    // Called when the user taps on screen
    func startProcessing() {
        errorMessage = nil
        state = .processing
    }

    // Called when segmentation + OBB succeeded
    func updateMeasurement(_ m: BoxMeasurement) {
        measurement = m
        if state != .confirmed { state = .measuring }
    }

    // Called when segmentation found nothing at the tap point
    func selectionFailed(message: String) {
        errorMessage = message
        state = .readyToSelect
    }

    // Called by ARSessionDelegate when a horizontal plane is found
    func planeFound(y: Float) {
        if nearestPlaneY == nil || abs(y - (nearestPlaneY ?? 0)) < 0.5 {
            nearestPlaneY = y
        }
        if state == .searchingPlane {
            state = .readyToSelect
        }
    }

    func confirm() {
        guard measurement != nil else { return }
        state = .confirmed
    }

    func remedir() {
        state = .readyToSelect
        measurement = nil
        errorMessage = nil
    }
}
