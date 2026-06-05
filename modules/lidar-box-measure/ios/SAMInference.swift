import CoreML
import CoreImage

// MARK: - SAMInference
final class SAMInference {

    private var encoderModel: MLModel?
    private var decoderModel: MLModel?

    private var cachedEmbedding: MLMultiArray?
    private var encoderFrameCounter = 0
    private let encoderRefreshInterval = 4

    static let maskW = 256
    static let maskH = 256
    static let samInputSize = 1024

    // Estado visible para el debug layer
    private(set) var status = "SAM: init"

    // MARK: - Carga
    func load() {
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        let cfg = MLModelConfiguration()
        if #available(iOS 16.0, *) { cfg.computeUnits = .cpuAndNeuralEngine }
        else { cfg.computeUnits = .all }

        var loaded = [String]()
        for name in ["sam_encoder", "sam_decoder"] {
            guard let url = bundleURL(for: name) else {
                status = "SAM: \(name) not found"; continue
            }
            do {
                let compiled = cacheDir.appendingPathComponent("\(name).mlmodelc")
                let loadURL: URL
                if FileManager.default.fileExists(atPath: compiled.path) {
                    loadURL = compiled
                } else {
                    let tmp = try MLModel.compileModel(at: url)
                    try? FileManager.default.moveItem(at: tmp, to: compiled)
                    loadURL = compiled
                }
                let m = try MLModel(contentsOf: loadURL, configuration: cfg)
                if name == "sam_encoder" { encoderModel = m }
                else                     { decoderModel = m }
                loaded.append(name)
            } catch {
                status = "SAM ERR \(name): \(error.localizedDescription.prefix(40))"
            }
        }
        if loaded.count == 2 { status = "SAM: OK (enc+dec)" }
        else if loaded.isEmpty { status = "SAM: no models" }
    }

    private func bundleURL(for name: String) -> URL? {
        var bundles: [Bundle] = [Bundle.main, Bundle(for: SAMInference.self)]
        for bName in ["LidarBoxMeasure", "LidarBoxMeasureResources"] {
            for parent in bundles.prefix(2) {
                if let u = parent.url(forResource: bName, withExtension: "bundle"),
                   let b = Bundle(url: u) { bundles.append(b) }
            }
        }
        for b in bundles {
            let url = URL(fileURLWithPath: b.bundlePath)
                .appendingPathComponent("\(name).mlpackage")
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        return nil
    }

    // MARK: - API pública (llamar desde background thread)
    func getMask(pixelBuffer: CVPixelBuffer,
                 screenBox: CGRect,
                 viewportSize: CGSize) -> [Bool]? {
        guard encoderModel != nil, decoderModel != nil else { return nil }

        encoderFrameCounter += 1
        if encoderFrameCounter % encoderRefreshInterval == 1 || cachedEmbedding == nil {
            let t0 = Date()
            cachedEmbedding = runEncoder(pixelBuffer: pixelBuffer)
            let ms = Int(Date().timeIntervalSince(t0) * 1000)
            if cachedEmbedding == nil {
                status = "SAM: enc nil"
            } else {
                status = "SAM enc \(ms)ms"
            }
        }
        guard let emb = cachedEmbedding else { return nil }

        let S = Float(SAMInference.samInputSize)
        let x1 = Float(screenBox.minX / viewportSize.width)  * S
        let y1 = Float(screenBox.minY / viewportSize.height) * S
        let x2 = Float(screenBox.maxX / viewportSize.width)  * S
        let y2 = Float(screenBox.maxY / viewportSize.height) * S

        guard let mask = runDecoder(embedding: emb, x1: x1, y1: y1, x2: x2, y2: y2) else {
            status = "SAM: dec nil"
            return nil
        }
        let coverage = mask.filter { $0 }.count * 100 / max(mask.count, 1)
        status = "SAM mask \(coverage)%"
        return mask
    }

    // MARK: - Encoder
    private func runEncoder(pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
        guard let model = encoderModel,
              let mlArr = preprocessImage(pixelBuffer),
              let input = try? MLDictionaryFeatureProvider(dictionary: [
                  "image": MLFeatureValue(multiArray: mlArr)
              ]),
              let output = try? model.prediction(from: input)
        else { return nil }

        // Intentar por nombre conocido; si falla, tomar el primer MLMultiArray disponible
        if let v = output.featureValue(for: "image_embeddings")?.multiArrayValue { return v }
        for key in output.featureNames {
            if let v = output.featureValue(for: key)?.multiArrayValue { return v }
        }
        return nil
    }

    // MARK: - Decoder
    private func runDecoder(embedding: MLMultiArray,
                             x1: Float, y1: Float,
                             x2: Float, y2: Float) -> [Bool]? {
        guard let model = decoderModel,
              let coords = try? MLMultiArray(shape: [1, 2, 2], dataType: .float32),
              let labels = try? MLMultiArray(shape: [1, 2],    dataType: .float32)
        else { return nil }

        let cPtr = coords.dataPointer.assumingMemoryBound(to: Float32.self)
        cPtr[0] = x1; cPtr[1] = y1
        cPtr[2] = x2; cPtr[3] = y2

        let lPtr = labels.dataPointer.assumingMemoryBound(to: Float32.self)
        lPtr[0] = 2.0   // top-left
        lPtr[1] = 3.0   // bottom-right

        guard let input = try? MLDictionaryFeatureProvider(dictionary: [
            "image_embeddings": MLFeatureValue(multiArray: embedding),
            "point_coords":     MLFeatureValue(multiArray: coords),
            "point_labels":     MLFeatureValue(multiArray: labels),
        ]),
              let output = try? model.prediction(from: input)
        else { return nil }

        // Intentar por nombre conocido; si falla, tomar el primer MLMultiArray con 4 dims
        var masksArr: MLMultiArray?
        if let v = output.featureValue(for: "masks")?.multiArrayValue {
            masksArr = v
        } else {
            for key in output.featureNames {
                if let v = output.featureValue(for: key)?.multiArrayValue,
                   v.shape.count == 4 { masksArr = v; break }
            }
        }
        guard let arr = masksArr else { return nil }

        let H = arr.shape[2].intValue
        let W = arr.shape[3].intValue
        let mPtr = arr.dataPointer.assumingMemoryBound(to: Float32.self)
        let total = H * W
        var mask = [Bool](repeating: false, count: total)
        for i in 0..<total { mask[i] = mPtr[i] > 0 }
        return mask
    }

    // MARK: - Preprocesamiento
    private func preprocessImage(_ pb: CVPixelBuffer) -> MLMultiArray? {
        let S = SAMInference.samInputSize
        let ciImage = CIImage(cvPixelBuffer: pb).oriented(.right)
        let ctx = CIContext(options: [.useSoftwareRenderer: false])
        let sx = CGFloat(S) / ciImage.extent.width
        let sy = CGFloat(S) / ciImage.extent.height
        let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: sx, y: sy))

        var rawRGBA = [UInt8](repeating: 0, count: S * S * 4)
        ctx.render(scaled, toBitmap: &rawRGBA, rowBytes: S * 4,
                   bounds: CGRect(x: 0, y: 0, width: S, height: S),
                   format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())

        guard let mlArr = try? MLMultiArray(shape: [1, 3, S, S] as [NSNumber],
                                             dataType: .float32) else { return nil }
        let ptr = mlArr.dataPointer.assumingMemoryBound(to: Float32.self)
        let mean: (Float, Float, Float) = (123.675, 116.28,  103.53)
        let std:  (Float, Float, Float) = (58.395,  57.12,   57.375)
        let plane = S * S
        for i in 0..<plane {
            let b = i * 4
            ptr[            i] = (Float(rawRGBA[b])     - mean.0) / std.0
            ptr[plane     + i] = (Float(rawRGBA[b + 1]) - mean.1) / std.1
            ptr[plane * 2 + i] = (Float(rawRGBA[b + 2]) - mean.2) / std.2
        }
        return mlArr
    }
}
