import Foundation

struct Volume: Identifiable {
    let id = UUID()
    var comprimento: Double
    var largura: Double
    var altura: Double
    var quantidade: Int
    var pesoUnit: Double

    var pesoTotal: Double { Double(quantidade) * pesoUnit }
    var volumeM3: Double { comprimento * largura * altura / 1_000_000 }
    var pesoCubadoUnit: Double { comprimento * largura * altura / 6_000 }
    var pesoCubadoTotal: Double { pesoCubadoUnit * Double(quantidade) }
    var dimensoesText: String { "\(Int(comprimento))×\(Int(largura))×\(Int(altura))" }
}
