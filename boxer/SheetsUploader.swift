import Foundation

/// Guarda mediciones en Google Sheets via Apps Script Web App.
/// No requiere OAuth — el script corre bajo la cuenta del dueño del sheet.
///
/// Columnas: AWB | Medida# | Bultos | Peso | Largo | Ancho | Alto | PesoVol | Fecha | User | FotoURL
actor SheetsUploader {
    static let shared = SheetsUploader()

    private let webAppURL = "https://script.google.com/macros/s/AKfycbz5jgT3D41EAwpeyIq3TdP-G1duQic2gGCvE7Qx2OnHQ3WpCtaT0dmfsXhmTc5y__99lA/exec"

    private static let dateFmt: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "dd/MM/yyyy HH:mm"
        return f
    }()

    private static let fileNameFmt: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyyMMdd_HHmmss"
        return f
    }()

    // MARK: - Public

    func appendRows(minuta: MinutaRecord, userEmail: String, photo: Data? = nil) async throws {
        var rows: [[Any]] = []
        for (idx, item) in minuta.items.enumerated() {
            // PesoVol = peso cubado aéreo: (cm³ / 1_000_000 m³) × 166.67 kg/m³ × bultos
            let pesoVol = (item.c * item.l * item.a / 1_000_000.0) * 166.67 * Double(item.vols)
            rows.append([
                minuta.numero,                            // AWB
                idx + 1,                                  // Medida#
                item.vols,                                // Bultos
                item.pesoUnit > 0 ? item.pesoUnit : "",  // Peso (por unidad) — vacío si no se ingresó
                Int(item.c),                              // Largo (cm)
                Int(item.l),                              // Ancho (cm)
                Int(item.a),                              // Alto  (cm)
                String(format: "%.6f", pesoVol),          // PesoVol
                Self.dateFmt.string(from: minuta.fecha),  // Fecha
                userEmail,                                 // User
                "",                                       // FotoURL (el script lo reemplaza si hay foto)
                item.isOversize ? "SI" : ""               // SOBREDIM
            ])
        }

        guard let url = URL(string: webAppURL) else {
            throw SheetsError.invalidURL
        }
        var req = URLRequest(url: url, timeoutInterval: 120)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")

        var body: [String: Any] = ["rows": rows]
        if let photoData = photo {
            body["photo"] = photoData.base64EncodedString()
            let stamp = Self.fileNameFmt.string(from: minuta.fecha)
            body["photoName"] = "CUBAJE_\(minuta.numero)_\(stamp).jpg"
        }
        req.httpBody = try JSONSerialization.data(withJSONObject: body)

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
