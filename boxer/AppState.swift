import SwiftUI
import Combine

// MARK: - Colores LATAM
extension Color {
    static let latamBlue = Color(red: 0.06, green: 0.06, blue: 0.40)
    static let latamCard = Color(white: 1.0, opacity: 0.10)
    static let latamCardBorder = Color(white: 1.0, opacity: 0.18)
}

// MARK: - Rutas de navegación
enum AppRoute: Hashable {
    case minuta
    case medicion(String)
    case consultar
}

// MARK: - Modelo de ítem
struct MedicionItem: Identifiable {
    let id = UUID()
    var c: Double   // largo (cm)
    var l: Double   // ancho (cm)
    var a: Double   // alto  (cm)
    var vols: Int
    var pesoUnit: Double
    var pesoTotal: Double { Double(vols) * pesoUnit }
    // CBM = cm³ / 1_000_000 → peso cubado aéreo = CBM × 166.67 kg/m³
    var pesoCubado: Double { (c * l * a / 1_000_000.0) * 166.67 * Double(vols) }
    var cla: String { "\(Int(c))×\(Int(l))×\(Int(a))" }
}

// MARK: - Registro de una minuta finalizada
struct MinutaRecord: Identifiable {
    let id = UUID()
    let numero: String
    let fecha: Date
    let items: [MedicionItem]
    var totalVols: Int    { items.reduce(0) { $0 + $1.vols } }
    var totalPesoReal: Double { items.reduce(0) { $0 + $1.pesoTotal } }
    var totalPesoCubado: Double { items.reduce(0) { $0 + $1.pesoCubado } }
}

// MARK: - Estado global de la app
class AppState: ObservableObject {
    @Published var path: [AppRoute] = []
    @Published var userName: String = "Silvina Herrera"
    @Published var userEmail: String = "silvinaherrera.acidlabs@latam.com"
    @Published var minutas: [MinutaRecord] = []
}
