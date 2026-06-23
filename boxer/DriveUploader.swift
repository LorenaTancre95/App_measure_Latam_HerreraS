import Foundation
import GoogleSignIn

/// Uploads JPEG images to Google Drive under BOXER3D_DATASET/CAJA/ or BOXER3D_DATASET/OVERSIZE/
actor DriveUploader {
    static let shared = DriveUploader()
    private let rootName = "BOXER3D_DATASET"
    private var folderCache: [String: String] = [:]  // "CAJA" / "OVERSIZE" → Drive folder ID

    // MARK: - Public

    func upload(imageData: Data, mode: String) async throws {
        let token   = try await getAccessToken()
        let rootID  = try await getOrCreate(name: rootName, parent: nil, token: token)
        let modeID  = try await getOrCreate(name: mode, parent: rootID, token: token)

        let ts = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        let filename = "\(mode)_\(ts).jpg"
        try await uploadFile(data: imageData, name: filename, parentID: modeID, token: token)
    }

    // MARK: - Auth

    private func getAccessToken() async throws -> String {
        guard let user = await MainActor.run(body: { GIDSignIn.sharedInstance.currentUser }) else {
            throw DriveError.notSignedIn
        }
        let refreshed = try await user.refreshTokensIfNeeded()
        return refreshed.accessToken.tokenString
    }

    // MARK: - Folder management

    private func getOrCreate(name: String, parent: String?, token: String) async throws -> String {
        let cacheKey = (parent ?? "root") + "/" + name
        if let cached = folderCache[cacheKey] { return cached }

        // Search for existing folder
        var query = "name='\(name)' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        if let p = parent { query += " and '\(p)' in parents" }

        var comps = URLComponents(string: "https://www.googleapis.com/drive/v3/files")!
        comps.queryItems = [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "fields", value: "files(id)")
        ]
        var req = URLRequest(url: comps.url!)
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")

        let (data, _) = try await URLSession.shared.data(for: req)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        if let files = json?["files"] as? [[String: Any]],
           let id = files.first?["id"] as? String {
            folderCache[cacheKey] = id
            return id
        }

        // Create new folder
        var body: [String: Any] = [
            "name": name,
            "mimeType": "application/vnd.google-apps.folder"
        ]
        if let p = parent { body["parents"] = [p] }

        var createReq = URLRequest(url: URL(string: "https://www.googleapis.com/drive/v3/files")!)
        createReq.httpMethod = "POST"
        createReq.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        createReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
        createReq.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (createData, _) = try await URLSession.shared.data(for: createReq)
        let created = try JSONSerialization.jsonObject(with: createData) as? [String: Any]
        guard let newID = created?["id"] as? String else {
            let msg = String(data: createData, encoding: .utf8) ?? "unknown"
            throw DriveError.uploadFailed("folder create: \(msg)")
        }
        folderCache[cacheKey] = newID
        return newID
    }

    // MARK: - File upload

    private func uploadFile(data: Data, name: String, parentID: String, token: String) async throws {
        let boundary = UUID().uuidString
        let metadata: [String: Any] = ["name": name, "parents": [parentID]]
        let metaJSON = try JSONSerialization.data(withJSONObject: metadata)

        var body = Data()
        body.append("--\(boundary)\r\nContent-Type: application/json\r\n\r\n".data(using: .utf8)!)
        body.append(metaJSON)
        body.append("\r\n--\(boundary)\r\nContent-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(data)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)

        var req = URLRequest(url: URL(string: "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart")!)
        req.httpMethod = "POST"
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        req.setValue("multipart/related; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        req.httpBody = body

        let (respData, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse, http.statusCode == 200 else {
            let msg = String(data: respData, encoding: .utf8) ?? "unknown"
            throw DriveError.uploadFailed(msg)
        }
    }

    // MARK: - Errors

    enum DriveError: LocalizedError {
        case notSignedIn, folderCreateFailed, uploadFailed(String)
        var errorDescription: String? {
            switch self {
            case .notSignedIn:          return "Iniciá sesión con Google primero"
            case .folderCreateFailed:   return "No se pudo crear la carpeta en Drive"
            case .uploadFailed(let m):  return "Error al subir: \(m)"
            }
        }
    }
}
