import Foundation

struct BoxMeasurement {
    var comprimento: Double  // C - cm
    var largura: Double      // L - cm
    var altura: Double       // A - cm

    var volumeM3: Double { comprimento * largura * altura / 1_000_000 }
    var pesoCubado: Double { comprimento * largura * altura / 6_000 }

    var displayText: String {
        "C: \(Int(comprimento.rounded())) × L: \(Int(largura.rounded())) × A: \(Int(altura.rounded())) cm"
    }
}
