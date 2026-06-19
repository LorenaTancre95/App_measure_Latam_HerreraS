import SwiftUI

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
}

// MARK: - Modelo de ítem
struct MedicionItem: Identifiable {
    let id = UUID()
    var c: Double   // largo (m)
    var l: Double   // ancho (m)
    var a: Double   // alto (m)
    var vols: Int
    var pesoUnit: Double
    var pesoTotal: Double { Double(vols) * pesoUnit }
    var pesoCubado: Double { c * l * a * 166.67 * Double(vols) }
    var cla: String { "\(Int(c*100))+\(Int(l*100))+\(Int(a*100))" }
}

// MARK: - Estado global de la app
class AppState: ObservableObject {
    @Published var path: [AppRoute] = []
    @Published var userName: String = "Usuario"
    @Published var userEmail: String = "usuario@latam.com"
}
