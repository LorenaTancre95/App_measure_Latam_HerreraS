import Foundation
import Combine

class MeditionFormViewModel: ObservableObject {
    @Published var minutaNumber: String = ""
    @Published var comprimento: String = ""
    @Published var largura: String = ""
    @Published var altura: String = ""
    @Published var quantidadeStr: String = ""
    @Published var pesoUnitStr: String = ""
    @Published var volumes: [Volume] = []
    @Published var isLoading: Bool = false
    @Published var photoAttached: Bool = false

    var pesoTotal: String {
        guard let q = Double(quantidadeStr), let p = Double(pesoUnitStr) else { return "" }
        return String(format: "%.3f", q * p)
    }

    var totalVolumes: Int { volumes.reduce(0) { $0 + $1.quantidade } }
    var totalPesoReal: Double { volumes.reduce(0) { $0 + $1.pesoTotal } }
    var totalPesoCubado: Double { volumes.reduce(0) { $0 + $1.pesoCubadoTotal } }

    func aplicarMedicao(_ m: BoxMeasurement) {
        comprimento = String(Int(m.comprimento.rounded()))
        largura     = String(Int(m.largura.rounded()))
        altura      = String(Int(m.altura.rounded()))
        photoAttached = true
    }

    func adicionar() {
        guard
            let c = Double(comprimento), c > 0,
            let l = Double(largura),     l > 0,
            let a = Double(altura),      a > 0,
            let q = Int(quantidadeStr),  q > 0,
            let p = Double(pesoUnitStr), p > 0
        else { return }

        volumes.append(Volume(
            comprimento: c, largura: l, altura: a,
            quantidade: q, pesoUnit: p
        ))
        clearCurrentEntry()
    }

    func finalizar() {
        // Aqui vai a chamada para a API real
        isLoading = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            self.isLoading = false
        }
    }

    func novaMinnuta() {
        minutaNumber = ""
        volumes = []
        clearCurrentEntry()
    }

    private func clearCurrentEntry() {
        comprimento   = ""
        largura       = ""
        altura        = ""
        quantidadeStr = ""
        pesoUnitStr   = ""
        photoAttached = false
    }
}
