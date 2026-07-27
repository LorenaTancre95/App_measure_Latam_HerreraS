import CoreML
import ARKit

/// Runs MobileSAM encoder + decoder on device via CoreML.
/// Encoder: image [1,3,1024,1024] → image_embeddings
/// Decoder: image_embeddings + point_coords [1,2,2] + point_labels [1,2] → masks [1,4,H,W]
final class SAMSegmenter {

    private let encoder: MLModel
    private let decoder: MLModel

    static let imageSize = 1024

    init() throws {
        let config = MLModelConfiguration()
        // cpuOnly: spikes 400–500 MB RAM on device → jetsam kill.
        // NeuralEngine: also spikes RAM (original comment). GPU uses Metal VRAM, no RAM pressure.
        config.computeUnits = .cpuAndGPU

        guard let encURL = Bundle.main.url(forResource: "sam_encoder", withExtension: "mlpackage")
                        ?? Bundle.main.url(forResource: "sam_encoder", withExtension: "mlmodelc")
        else { throw SAMError.modelNotFound("sam_encoder") }

        guard let decURL = Bundle.main.url(forResource: "sam_decoder", withExtension: "mlpackage")
                        ?? Bundle.main.url(forResource: "sam_decoder", withExtension: "mlmodelc")
        else { throw SAMError.modelNotFound("sam_decoder") }

        encoder = try MLModel(contentsOf: encURL, configuration: config)
        decoder = try MLModel(contentsOf: decURL, configuration: config)
    }

    // MARK: - Public

    /// Segment the object at `promptPoint` (in 1024×1024 SAM pixel coordinates).
    /// Returns a binary mask at the decoder output resolution.
    func segment(pixelBuffer: CVPixelBuffer, promptPoint: CGPoint) throws -> [[Bool]] {
        let embeddings = try encodeImage(pixelBuffer)
        return try decodeMask(embeddings: embeddings, promptPoint: promptPoint)
    }

    // MARK: - Encoder

    private func encodeImage(_ buffer: CVPixelBuffer) throws -> MLMultiArray {
        let S = SAMSegmenter.imageSize

        // Center-crop to square, scale to 1024×1024, extract RGB float [0,255]
        let ci    = CIImage(cvPixelBuffer: buffer)
        let ctx   = CIContext()
        let w = ci.extent.width, h = ci.extent.height, side = min(w, h)
        let cropped = ci.cropped(to: CGRect(x: (w - side) / 2,
                                            y: (h - side) / 2,
                                            width: side, height: side))
        let scale   = CGFloat(S) / side
        let scaled  = cropped.transformed(by: CGAffineTransform(scaleX: scale, y: scale))

        var rgba = [UInt8](repeating: 0, count: S * S * 4)
        ctx.render(scaled, toBitmap: &rgba, rowBytes: S * 4,
                   bounds: CGRect(x: scaled.extent.origin.x,
                                  y: scaled.extent.origin.y,
                                  width: CGFloat(S), height: CGFloat(S)),
                   format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())

        // CHW float32, values in [0, 255] — SAM normalizes internally
        let arr  = try MLMultiArray(shape: [1, 3, S, S] as [NSNumber], dataType: .float32)
        let ptr  = arr.dataPointer.assumingMemoryBound(to: Float32.self)
        let n    = S * S
        for i in 0..<n {
            ptr[i]       = Float(rgba[i * 4])
            ptr[n + i]   = Float(rgba[i * 4 + 1])
            ptr[2 * n + i] = Float(rgba[i * 4 + 2])
        }

        let input  = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(multiArray: arr)])
        let output = try autoreleasepool { try encoder.prediction(from: input) }

        guard let emb = output.featureValue(for: "image_embeddings")?.multiArrayValue
        else { throw SAMError.outputMissing("image_embeddings") }
        return emb
    }

    // MARK: - Decoder

    private func decodeMask(embeddings: MLMultiArray, promptPoint: CGPoint) throws -> [[Bool]] {
        // point_coords [1, 2, 2]: foreground point + required padding point
        let coords = try MLMultiArray(shape: [1, 2, 2] as [NSNumber], dataType: .float32)
        let cPtr   = coords.dataPointer.assumingMemoryBound(to: Float32.self)
        cPtr[0] = Float(promptPoint.x)   // fg x (pixel in 1024 space)
        cPtr[1] = Float(promptPoint.y)   // fg y
        cPtr[2] = 0                       // padding
        cPtr[3] = 0

        // point_labels [1, 2]: 1 = foreground, -1 = padding
        let labels = try MLMultiArray(shape: [1, 2] as [NSNumber], dataType: .float32)
        let lPtr   = labels.dataPointer.assumingMemoryBound(to: Float32.self)
        lPtr[0] =  1
        lPtr[1] = -1

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image_embeddings": MLFeatureValue(multiArray: embeddings),
            "point_coords":     MLFeatureValue(multiArray: coords),
            "point_labels":     MLFeatureValue(multiArray: labels)
        ])
        let output = try autoreleasepool { try decoder.prediction(from: input) }

        guard let masksArr = output.featureValue(for: "masks")?.multiArrayValue
        else { throw SAMError.outputMissing("masks") }

        // masks shape [1, 4, H, W] — take mask 0 (best), threshold at 0
        let mH  = masksArr.shape[2].intValue
        let mW  = masksArr.shape[3].intValue
        let ptr = masksArr.dataPointer.assumingMemoryBound(to: Float32.self)

        var result = [[Bool]](repeating: [Bool](repeating: false, count: mW), count: mH)
        for y in 0..<mH {
            for x in 0..<mW {
                result[y][x] = ptr[y * mW + x] > 0
            }
        }
        return result
    }

    // MARK: - Errors

    enum SAMError: LocalizedError {
        case modelNotFound(String)
        case outputMissing(String)
        var errorDescription: String? {
            switch self {
            case .modelNotFound(let n): return "\(n).mlpackage no encontrado en el bundle"
            case .outputMissing(let n): return "Output '\(n)' no encontrado en el modelo SAM"
            }
        }
    }
}
