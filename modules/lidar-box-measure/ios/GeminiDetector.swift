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

    // ⚠️ No compartir esta key públicamente
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

        let body: [String: Any] = [
            "contents": [[
                "parts": [
                    ["inline_data": ["mime_type": "image/jpeg",
                                     "data": jpeg.base64EncodedString()]],
                    ["text": detectionPrompt]
                ]
            ]],
            "generationConfig": [
                "responseMimeType": "application/json",
                "responseSchema": responseSchema
            ]
        ]

        guard let url = URL(string: "\(baseURL)/\(model):generateContent?key=\(apiKey)"),
              let bodyData = try? JSONSerialization.data(withJSONObject: body) else {
            status = "Gemini: req err"; completion(nil); return
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
            guard let data = data,
                  let json      = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let cands     = json["candidates"] as? [[String: Any]],
                  let content   = cands.first?["content"] as? [String: Any],
                  let parts     = content["parts"] as? [[String: Any]],
                  let text      = parts.first?["text"] as? String,
                  let rData     = text.data(using: .utf8),
                  let result    = try? JSONSerialization.jsonObject(with: rData) as? [String: Any]
            else {
                self.status = "Gemini: parse err \(ms)ms"; completion(nil); return
            }

            guard let detected = result["boxDetected"] as? Bool, detected,
                  let cd = result["corners"] as? [String: Any]
            else {
                self.status = "Gemini: no box \(ms)ms"; completion(nil); return
            }

            func pt(_ k: String) -> CGPoint? {
                guard let d = cd[k] as? [String: Any],
                      let x = d["x"] as? Double,
                      let y = d["y"] as? Double
                else { return nil }
                return CGPoint(x: x / 100.0, y: y / 100.0)
            }

            guard let bl = pt("bottomLeft"),  let br = pt("bottomRight"),
                  let tl = pt("topLeft"),     let tr = pt("topRight")
            else {
                self.status = "Gemini: corner err \(ms)ms"; completion(nil); return
            }

            self.status = "Gemini OK \(ms)ms"
            completion(GeminiCorners(bottomLeft: bl, bottomRight: br,
                                     topLeft: tl,    topRight: tr))
        }.resume()
    }

    // MARK: - Prompt y schema

    private var detectionPrompt: String {
        """
        Analyze this image and find the cardboard box or rectangular package \
        with the most clearly visible front face.
        Return the 4 corners of the FRONT FACE (the face closest to the camera) \
        as percentage coordinates (0–100) of image width and height:
        • bottomLeft:  bottom-left corner of the front face
        • bottomRight: bottom-right corner of the front face
        • topLeft:     top-left corner of the front face
        • topRight:    top-right corner of the front face
        Place corners exactly at the physical edges of the box, not inside it.
        If no box or rectangular package is visible, set boxDetected to false.
        """
    }

    private var responseSchema: [String: Any] {
        func corner() -> [String: Any] {
            ["type": "OBJECT",
             "properties": ["x": ["type": "NUMBER"], "y": ["type": "NUMBER"]],
             "required": ["x", "y"]]
        }
        return [
            "type": "OBJECT",
            "properties": [
                "boxDetected": ["type": "BOOLEAN"],
                "corners": [
                    "type": "OBJECT",
                    "properties": [
                        "bottomLeft":  corner(),
                        "bottomRight": corner(),
                        "topLeft":     corner(),
                        "topRight":    corner()
                    ],
                    "required": ["bottomLeft", "bottomRight", "topLeft", "topRight"]
                ]
            ],
            "required": ["boxDetected", "corners"]
        ]
    }

    // MARK: - Imagen

    private func pixelBufferToJPEG(_ pb: CVPixelBuffer) -> Data? {
        let ci  = CIImage(cvPixelBuffer: pb).oriented(.right)
        let ctx = CIContext(options: [.useSoftwareRenderer: false])
        // Escalar a max 640px para reducir tamaño del payload
        let scale = min(1.0, 640.0 / max(ci.extent.width, ci.extent.height))
        let img   = scale < 1.0
            ? ci.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            : ci
        guard let cg = ctx.createCGImage(img, from: img.extent) else { return nil }
        return UIImage(cgImage: cg).jpegData(compressionQuality: 0.75)
    }
}
