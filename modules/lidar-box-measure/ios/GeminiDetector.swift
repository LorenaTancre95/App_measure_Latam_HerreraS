import Foundation
import CoreVideo
import CoreImage
import UIKit

// MARK: - Types

struct GeminiCorners {
    let bottomLeft:  CGPoint   // normalized [0,1] x,y en pantalla portrait
    let bottomRight: CGPoint
    let topLeft:     CGPoint
    let topRight:    CGPoint
}

// MARK: - GeminiDetector

final class GeminiDetector {

    // swiftlint:disable:next line_length
    private let apiKey  = "AIzaSyBiiL7jIZzl436vNrp6ZUCo0_-EpSAkrig"
    private let model   = "gemini-2.0-flash"
    private let baseURL = "https://generativelanguage.googleapis.com/v1beta/models"

    private(set) var status = "Gemini: init"

    // MARK: - API pública

    func detectCorners(pixelBuffer: CVPixelBuffer,
                       completion: @escaping (GeminiCorners?) -> Void) {
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
        req.httpMethod  = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody     = bodyData
        req.timeoutInterval = 12

        let t0 = Date()
        URLSession.shared.dataTask(with: req) { [weak self] data, _, error in
            guard let self = self else { return }
            let ms = Int(Date().timeIntervalSince(t0) * 1000)

            if let error = error {
                self.status = "Gemini: \(error.localizedDescription.prefix(28))"
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

        let genConfig: [String: Any] = [
            "responseMimeType": "application/json",
            "responseSchema": buildSchema()
        ]

        let body: [String: Any] = [
            "contents": [content],
            "generationConfig": genConfig
        ]

        return try? JSONSerialization.data(withJSONObject: body)
    }

    private func buildSchema() -> [String: Any] {
        let numType: [String: Any]  = ["type": "NUMBER"]
        let boolType: [String: Any] = ["type": "BOOLEAN"]

        let xyProps: [String: Any]  = ["x": numType, "y": numType]
        let cornerSchema: [String: Any] = [
            "type": "OBJECT",
            "properties": xyProps,
            "required": ["x", "y"]
        ]

        let cornersProps: [String: Any] = [
            "bottomLeft":  cornerSchema,
            "bottomRight": cornerSchema,
            "topLeft":     cornerSchema,
            "topRight":    cornerSchema
        ]
        let cornersSchema: [String: Any] = [
            "type": "OBJECT",
            "properties": cornersProps,
            "required": ["bottomLeft", "bottomRight", "topLeft", "topRight"]
        ]

        let rootProps: [String: Any] = [
            "boxDetected": boolType,
            "corners": cornersSchema
        ]
        return [
            "type": "OBJECT",
            "properties": rootProps,
            "required": ["boxDetected", "corners"]
        ]
    }

    // MARK: - Response parsing

    private func parseResponse(data: Data, ms: Int,
                                completion: (GeminiCorners?) -> Void) {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            status = "Gemini: bad JSON \(ms)ms"; completion(nil); return
        }
        // API-level error (invalid key, quota, bad model, bad schema)
        if let err = json["error"] as? [String: Any],
           let msg = err["message"] as? String {
            status = "API err: \(msg.prefix(50))"
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
        guard let rData  = text.data(using: .utf8),
              let result = try? JSONSerialization.jsonObject(with: rData) as? [String: Any] else {
            let preview = String(text.prefix(60)).replacingOccurrences(of: "\n", with: " ")
            status = "Gemini: inner JSON err: \(preview)"
            completion(nil); return
        }
        let detected = (result["boxDetected"] as? Bool)
                    ?? ((result["boxDetected"] as? NSNumber)?.boolValue ?? false)
        guard detected, let cd = result["corners"] as? [String: Any] else {
            status = "Gemini: no box \(ms)ms"; completion(nil); return
        }

        func pt(_ k: String) -> CGPoint? {
            guard let d = cd[k] as? [String: Any],
                  let x = (d["x"] as? NSNumber)?.doubleValue,
                  let y = (d["y"] as? NSNumber)?.doubleValue
            else { return nil }
            return CGPoint(x: x / 100.0, y: y / 100.0)
        }

        guard let bl = pt("bottomLeft"),  let br = pt("bottomRight"),
              let tl = pt("topLeft"),     let tr = pt("topRight") else {
            status = "Gemini: corner err \(ms)ms"; completion(nil); return
        }

        status = "Gemini OK \(ms)ms"
        completion(GeminiCorners(bottomLeft: bl, bottomRight: br,
                                 topLeft: tl,    topRight: tr))
    }

    // MARK: - Prompt

    private var detectionPrompt: String {
        return "Analyze this image and find the cardboard box or rectangular package. " +
               "Return the 4 corners of the FRONT FACE (the face closest to the camera) " +
               "as percentage coordinates 0 to 100 of image width and height. " +
               "bottomLeft is the bottom-left corner, bottomRight is the bottom-right corner, " +
               "topLeft is the top-left corner, topRight is the top-right corner. " +
               "Place corners exactly at the physical edges of the box. " +
               "If no box is visible set boxDetected to false."
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
