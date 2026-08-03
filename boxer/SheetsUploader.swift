import Foundation
import GoogleSignIn

/// Appends measurement rows to the LATAM cargo Google Sheet.
/// Each call to appendRows() adds one row per MedicionItem in the minuta.
///
/// Sheet columns: AWB | Medida# | Bultos | Peso | Largo | Ancho | Alto | PesoVol | Fecha | User | FotoURL
actor SheetsUploader {
    static let shared = SheetsUploader()

    private let spreadsheetID = "1rtOz2WUu0uvKN0ZmnPVHdDidHwxlGhb3Ztw6do757z8"
    private let tabGID: Int    = 2102533928

    // Cached sheet name resolved from tabGID on first use
    private var cachedSheetName: String? = nil

    private static let dateFmt: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "dd/MM/yyyy HH:mm"
        return f
    }()

    // MARK: - Public

    /// Appends one row per item in the minuta to the Google Sheet.
    /// Fails silently if the user is not signed in or lacks scope — data is already saved locally.
    func appendRows(minuta: MinutaRecord, userEmail: String) async throws {
        let token     = try await getAccessToken()
        let sheetName = try await resolveSheetName(token: token)
        let range     = "\(sheetName)!A:K"

        var values: [[Any]] = []
        for (idx, item) in minuta.items.enumerated() {
            let pesoVol = (item.c * item.l * item.a / 1_000_000.0) * 166.67 * Double(item.vols)
            let row: [Any] = [
                minuta.numero,                              // A: AWB
                idx + 1,                                   // B: Medida#
                item.vols,                                 // C: Bultos
                item.pesoUnit,                             // D: Peso (por unidad)
                Int(item.c),                               // E: Largo (cm)
                Int(item.l),                               // F: Ancho (cm)
                Int(item.a),                               // G: Alto  (cm)
                String(format: "%.6f", pesoVol),           // H: PesoVol
                Self.dateFmt.string(from: minuta.fecha),   // I: Fecha
                userEmail,                                  // J: User
                ""                                         // K: FotoURL (se agrega por separado)
            ]
            values.append(row)
        }

        let endpoint = "https://sheets.googleapis.com/v4/spreadsheets/\(spreadsheetID)/values/\(encode(range)):append?valueInputOption=USER_ENTERED&insertDataOption=INSERT_ROWS"
        guard let url = URL(string: endpoint) else { return }

        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: ["values": values])

        let (data, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            let msg = String(data: data, encoding: .utf8) ?? "unknown"
            throw SheetsError.appendFailed(msg)
        }
    }

    // MARK: - Sheet name resolution

    // Resolves the sheet name for the configured GID (cached after first call).
    private func resolveSheetName(token: String) async throws -> String {
        if let cached = cachedSheetName { return cached }

        let urlStr = "https://sheets.googleapis.com/v4/spreadsheets/\(spreadsheetID)?fields=sheets.properties"
        guard let url = URL(string: urlStr) else { return "Sheet1" }

        var req = URLRequest(url: url)
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")

        let (data, _) = try await URLSession.shared.data(for: req)
        let json    = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let sheets  = json?["sheets"] as? [[String: Any]] ?? []

        for sheet in sheets {
            if let props   = sheet["properties"] as? [String: Any],
               let sheetId = props["sheetId"] as? Int,
               sheetId == tabGID,
               let title   = props["title"] as? String {
                cachedSheetName = title
                return title
            }
        }
        // Fallback: use the first sheet
        let fallback = (sheets.first?["properties"] as? [String: Any])?["title"] as? String ?? "Sheet1"
        cachedSheetName = fallback
        return fallback
    }

    // MARK: - Auth

    private func getAccessToken() async throws -> String {
        guard let user = await MainActor.run(body: { GIDSignIn.sharedInstance.currentUser }) else {
            throw SheetsError.notSignedIn
        }
        let refreshed = try await user.refreshTokensIfNeeded()
        return refreshed.accessToken.tokenString
    }

    // URL-encodes the range string (e.g. "Mediciones!A:K" → "Mediciones%21A%3AK")
    private func encode(_ range: String) -> String {
        range.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? range
    }

    // MARK: - Errors

    enum SheetsError: LocalizedError {
        case notSignedIn, appendFailed(String)
        var errorDescription: String? {
            switch self {
            case .notSignedIn:          return "Iniciá sesión con Google para guardar en el sheet"
            case .appendFailed(let m):  return "Error al guardar en sheet: \(m)"
            }
        }
    }
}
