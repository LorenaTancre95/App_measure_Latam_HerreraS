import Foundation
import CoreVideo
import CoreImage
import UIKit

// MARK: - Types

struct GeminiBox {
    // Cara frontal (más cercana a la cámara)
    let fbl: CGPoint   // front-bottom-left
    let fbr: CGPoint   // front-bottom-right
    let ftl: CGPoint   // front-top-left
    let ftr: CGPoint   // front-top-right
    // Cara trasera / vértices visibles por cara superior o lateral
    let bbl: CGPoint   // back-bottom-left
    let bbr: CGPoint   // back-bottom-right
    let btl: CGPoint   // back-top-left
    let btr: CGPoint   // back-top-right
}

// MARK: - GeminiDetector

final class GeminiDetector {

    private let apiKey  = Bundle.main.infoDictionary?["GEMINI_API_KEY"] as? String ?? ""
    private let model   = "gemini-2.5-pro"
    private let baseURL = "https://generativelanguage.googleapis.com/v1beta/models"

    private(set) var status = "Gemini: init"

    // MARK: - API pública

    func detectBox(pixelBuffer: CVPixelBuffer,
                   completion: @escaping (GeminiBox?) -> Void) {
        guard let jpeg = pixelBufferToJPEG(pixelBuffer) else {
            status = "Gemini: img err"; completion(nil); return
        }
        guard let bodyData = buildRequestBody(jpeg: jpeg) else {
            status = "Gemini: body err"; completion(nil); return
        }
        let urlStr = "\(baseURL)/\(model):generateContent?key=\(apiKey)"
        guard let url = URL(string: urlStr) else {
            status = "Gemini: url err"; completion(nil); return
        }

        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = bodyData
        req.timeoutInterval = 15

        let t0 = Date()
        URLSession.shared.dataTask(with: req) { [weak self] data, _, error in
            guard let self = self else { return }
            let ms = Int(Date().timeIntervalSince(t0) * 1000)
            if let error = error {
                self.status = "Gemini: \(error.localizedDescription.prefix(40))"
                completion(nil); return
            }
            guard let data = data else {
                self.status = "Gemini: no data \(ms)ms"; completion(nil); return
            }
            self.parseResponse(data: data, ms: ms, completion: completion)
        }.resume()
    }

    // MARK: - Request body

    private func buildRequestBody(jpeg: Data) -> Data? {
        let b64 = jpeg.base64EncodedString()
        let inlineData: [String: Any] = ["mime_type": "image/jpeg", "data": b64]
        let imagePart:  [String: Any] = ["inline_data": inlineData]
        let textPart:   [String: Any] = ["text": detectionPrompt]
        let content:    [String: Any] = ["parts": [imagePart, textPart]]
        let body:       [String: Any] = ["contents": [content]]
        return try? JSONSerialization.data(withJSONObject: body)
    }

    // MARK: - Response parsing

    private func parseResponse(data: Data, ms: Int,
                                completion: (GeminiBox?) -> Void) {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            status = "Gemini: bad JSON \(ms)ms"; completion(nil); return
        }
        if let err = json["error"] as? [String: Any],
           let msg = err["message"] as? String {
            status = "API err: \(msg.prefix(120))"
            completion(nil); return
        }
        guard let cands = json["candidates"] as? [[String: Any]] else {
            let keys = json.keys.joined(separator: ",")
            status = "Gemini: no candidates [\(keys)] \(ms)ms"
            completion(nil); return
        }
        guard let content = cands.first?["content"] as? [String: Any],
              let parts   = content["parts"] as? [[String: Any]],
              let text    = parts.first?["text"] as? String else {
            status = "Gemini: no text \(ms)ms"; completion(nil); return
        }

        var clean = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if let r = clean.range(of: "```json") { clean = String(clean[r.upperBound...]) }
        if let r = clean.range(of: "```")     { clean = String(clean[clean.startIndex..<r.lowerBound]) }
        clean = clean.trimmingCharacters(in: .whitespacesAndNewlines)

        guard let rData  = clean.data(using: .utf8),
              let result = try? JSONSerialization.jsonObject(with: rData) as? [String: Any] else {
            let preview = String(clean.prefix(80)).replacingOccurrences(of: "\n", with: " ")
            status = "Gemini: JSON err: \(preview)"
            completion(nil); return
        }

        let detected = (result["boxDetected"] as? Bool)
                    ?? ((result["boxDetected"] as? NSNumber)?.boolValue ?? false)
        guard detected, let verts = result["vertices"] as? [String: Any] else {
            status = "Gemini: no box \(ms)ms"; completion(nil); return
        }

        func pt(_ k: String) -> CGPoint? {
            guard let d = verts[k] as? [String: Any],
                  let x = (d["x"] as? NSNumber)?.doubleValue,
                  let y = (d["y"] as? NSNumber)?.doubleValue
            else { return nil }
            return CGPoint(x: x / 100.0, y: y / 100.0)
        }

        guard let fbl = pt("fbl"), let fbr = pt("fbr"),
              let ftl = pt("ftl"), let ftr = pt("ftr"),
              let bbl = pt("bbl"), let bbr = pt("bbr"),
              let btl = pt("btl"), let btr = pt("btr") else {
            status = "Gemini: vertex err \(ms)ms"; completion(nil); return
        }

        status = "Gemini OK \(ms)ms"
        completion(GeminiBox(fbl: fbl, fbr: fbr, ftl: ftl, ftr: ftr,
                             bbl: bbl, bbr: bbr, btl: btl, btr: btr))
    }

    // MARK: - Prompt

    private var detectionPrompt: String {
        return "Find the cardboard box in this image and locate all 8 corners of the 3D box. " +
               "Reply with ONLY a raw JSON object, no markdown, no explanation. " +
               "The 8 vertices are: " +
               "fbl=front-bottom-left, fbr=front-bottom-right, ftl=front-top-left, ftr=front-top-right " +
               "(the face closest to the camera), " +
               "bbl=back-bottom-left, bbr=back-bottom-right, btl=back-top-left, btr=back-top-right " +
               "(the face farthest from camera, visible via top or side faces). " +
               "Estimate hidden corners by extending the visible edges. " +
               "x and y are percentages 0-100 of image width and height. " +
               "Format: {\"boxDetected\":true,\"vertices\":{" +
               "\"fbl\":{\"x\":10,\"y\":80},\"fbr\":{\"x\":60,\"y\":80}," +
               "\"ftl\":{\"x\":10,\"y\":30},\"ftr\":{\"x\":60,\"y\":30}," +
               "\"bbl\":{\"x\":60,\"y\":85},\"bbr\":{\"x\":90,\"y\":85}," +
               "\"btl\":{\"x\":60,\"y\":20},\"btr\":{\"x\":90,\"y\":20}}} " +
               "If no box: {\"boxDetected\":false,\"vertices\":{}}"
    }

    // MARK: - Imagen

    private func pixelBufferToJPEG(_ pb: CVPixelBuffer) -> Data? {
        let ci  = CIImage(cvPixelBuffer: pb).oriented(.right)
        let ctx = CIContext(options: [.useSoftwareRenderer: false])
        let maxDim = max(ci.extent.width, ci.extent.height)
        let scale  = maxDim > 640 ? 640.0 / maxDim : 1.0
        let img    = scale < 1.0
            ? ci.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            : ci
        guard let cg = ctx.createCGImage(img, from: img.extent) else { return nil }
        return UIImage(cgImage: cg).jpegData(compressionQuality: 0.75)
    }
}
