import Foundation

/// Guarda mediciones en Google Sheets via Apps Script Web App.
/// No requiere OAuth — el script corre bajo la cuenta del dueño del sheet.
///
/// Columnas: AWB | Medida# | Bultos | Peso | Largo | Ancho | Alto | PesoVol | Fecha | User | FotoURL
actor SheetsUploader {
    static let shared = SheetsUploader()

    private let webAppURL = "https://script.google.com/a/macros/latam.com/s/AKfycbxg4jfqH1uybsHysAXuCONdFbpt-sEBR6oZn-6Pvhu7BRZIWKpMumT461Y4Il-J5sbwrA/exec"

    private static let dateFmt: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "dd/MM/yyyy HH:mm"
        return f
    }()

    // MARK: - Public

    func appendRows(minuta: MinutaRecord, userEmail: String) async throws {
        var rows: [[Any]] = []
        for (idx, item) in minuta.items.enumerated() {
            // PesoVol = peso cubado aéreo: (cm³ / 1_000_000 m³) × 166.67 kg/m³ × bultos
            let pesoVol = (item.c * item.l * item.a / 1_000_000.0) * 166.67 * Double(item.vols)
            rows.append([
                minuta.numero,                            // AWB
                idx + 1,                                  // Medida#
                item.vols,                                // Bultos
                item.pesoUnit,                            // Peso (por unidad)
                Int(item.c),                              // Largo (cm)
                Int(item.l),                              // Ancho (cm)
                Int(item.a),                              // Alto  (cm)
                String(format: "%.6f", pesoVol),          // PesoVol
                Self.dateFmt.string(from: minuta.fecha),  // Fecha
                userEmail,                                 // User
                ""                                        // FotoURL
            ])
        }

        guard let url = URL(string: webAppURL) else {
            throw SheetsError.invalidURL
        }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: ["rows": rows])

        let (data, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            let msg = String(data: data, encoding: .utf8) ?? "unknown"
            throw SheetsError.appendFailed(msg)
        }
        // El script devuelve {ok: true} o {error: "..."}
        if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let errMsg = json["error"] as? String {
            throw SheetsError.appendFailed(errMsg)
        }
    }

    // MARK: - Errors

    enum SheetsError: LocalizedError {
        case invalidURL, appendFailed(String)
        var errorDescription: String? {
            switch self {
            case .invalidURL:           return "URL del script inválida"
            case .appendFailed(let m):  return "Error al guardar: \(m)"
            }
        }
    }
}
