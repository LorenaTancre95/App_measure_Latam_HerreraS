import CoreML
import CoreImage
import Accelerate

// MARK: - SAMInference
// Encapsula MobileSAM encoder + decoder.
// Encoder: imagen 1024×1024 → embedding [1,256,64,64]  (se cachea cada N frames)
// Decoder: embedding + bbox YOLO → máscara binaria 256×256
final class SAMInference {

    private var encoderModel: MLModel?
    private var decoderModel: MLModel?

    private var cachedEmbedding: MLMultiArray?
    private var encoderFrameCounter = 0
    private let encoderRefreshInterval = 4   // re-corre el encoder cada 4 frames (~0.8 s)

    static let maskW = 256
    static let maskH = 256
    static let samInputSize = 1024

    // MARK: - Carga de modelos
    func load() {
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        let cfg = MLModelConfiguration()
        if #available(iOS 16.0, *) { cfg.computeUnits = .cpuAndNeuralEngine }
        else { cfg.computeUnits = .all }

        for (name, keyPath) in [("sam_encoder", \SAMInference.encoderModel),
                                 ("sam_decoder", \SAMInference.decoderModel)] {
            guard let url = bundleURL(for: name) else {
                print("SAM: \(name).mlpackage no encontrado en bundle"); continue
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
                self[keyPath: keyPath] = try MLModel(contentsOf: loadURL, configuration: cfg)
                print("SAM: \(name) ✅")
            } catch {
                print("SAM: \(name) error — \(error)")
            }
        }
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

    // MARK: - API principal
    // Llama esto desde el background thread antes de measure3D.
    // screenBox: bbox YOLO en coordenadas de pantalla (portrait)
    // viewportSize: tamaño de la vista AR en pantalla
    // Retorna: máscara binaria [256×256] (row-major) o nil si SAM no está listo
    func getMask(pixelBuffer: CVPixelBuffer,
                 screenBox: CGRect,
                 viewportSize: CGSize) -> [Bool]? {
        guard encoderModel != nil, decoderModel != nil else { return nil }

        encoderFrameCounter += 1
        if encoderFrameCounter % encoderRefreshInterval == 1 || cachedEmbedding == nil {
            cachedEmbedding = runEncoder(pixelBuffer: pixelBuffer)
        }
        guard let emb = cachedEmbedding else { return nil }

        // Convertir bbox de screen-space a espacio SAM 1024×1024
        let S = Float(SAMInference.samInputSize)
        let sx1 = Float(screenBox.minX / viewportSize.width)  * S
        let sy1 = Float(screenBox.minY / viewportSize.height) * S
        let sx2 = Float(screenBox.maxX / viewportSize.width)  * S
        let sy2 = Float(screenBox.maxY / viewportSize.height) * S

        return runDecoder(embedding: emb, x1: sx1, y1: sy1, x2: sx2, y2: sy2)
    }

    // MARK: - Encoder
    private func runEncoder(pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
        guard let model = encoderModel else { return nil }
        guard let mlArr = preprocessImage(pixelBuffer) else { return nil }
        guard let input = try? MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(multiArray: mlArr)
        ]) else { return nil }
        guard let output = try? model.prediction(from: input) else { return nil }
        return output.featureValue(for: "image_embeddings")?.multiArrayValue
    }

    // MARK: - Decoder
    private func runDecoder(embedding: MLMultiArray,
                             x1: Float, y1: Float,
                             x2: Float, y2: Float) -> [Bool]? {
        guard let model = decoderModel else { return nil }

        guard let coords = try? MLMultiArray(shape: [1, 2, 2], dataType: .float32),
              let labels = try? MLMultiArray(shape: [1, 2],    dataType: .float32)
        else { return nil }

        // SAM bbox prompt: punto 0 = top-left (label 2), punto 1 = bottom-right (label 3)
        let cPtr = coords.dataPointer.assumingMemoryBound(to: Float32.self)
        cPtr[0] = x1; cPtr[1] = y1   // top-left x, y
        cPtr[2] = x2; cPtr[3] = y2   // bottom-right x, y

        let lPtr = labels.dataPointer.assumingMemoryBound(to: Float32.self)
        lPtr[0] = 2.0   // top-left corner label
        lPtr[1] = 3.0   // bottom-right corner label

        guard let input = try? MLDictionaryFeatureProvider(dictionary: [
            "image_embeddings": MLFeatureValue(multiArray: embedding),
            "point_coords":     MLFeatureValue(multiArray: coords),
            "point_labels":     MLFeatureValue(multiArray: labels),
        ]) else { return nil }

        guard let output = try? model.prediction(from: input),
              let masksArr = output.featureValue(for: "masks")?.multiArrayValue
        else { return nil }

        // masksArr: [1, 1, 256, 256] logits (sin sigmoid)
        // sigmoid(x) > 0.5  ↔  x > 0
        let H = masksArr.shape[2].intValue
        let W = masksArr.shape[3].intValue
        let mPtr = masksArr.dataPointer.assumingMemoryBound(to: Float32.self)
        let total = H * W
        var mask = [Bool](repeating: false, count: total)
        for i in 0..<total { mask[i] = mPtr[i] > 0 }
        return mask
    }

    // MARK: - Preprocesamiento imagen para SAM
    // CVPixelBuffer (landscape) → MLMultiArray [1, 3, 1024, 1024] normalizado con mean/std SAM
    private func preprocessImage(_ pb: CVPixelBuffer) -> MLMultiArray? {
        let S = SAMInference.samInputSize

        // Rotar a portrait (.right) y resize a 1024×1024 via CIImage
        let ciImage = CIImage(cvPixelBuffer: pb).oriented(.right)
        let ctx = CIContext(options: [.useSoftwareRenderer: false])
        let scaleX = CGFloat(S) / ciImage.extent.width
        let scaleY = CGFloat(S) / ciImage.extent.height
        let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var rawRGBA = [UInt8](repeating: 0, count: S * S * 4)
        ctx.render(scaled,
                   toBitmap: &rawRGBA,
                   rowBytes: S * 4,
                   bounds: CGRect(x: 0, y: 0, width: S, height: S),
                   format: .RGBA8,
                   colorSpace: colorSpace)

        guard let mlArr = try? MLMultiArray(shape: [1, 3, S, S] as [NSNumber],
                                             dataType: .float32) else { return nil }
        let ptr = mlArr.dataPointer.assumingMemoryBound(to: Float32.self)

        // SAM mean / std por canal (en escala 0-255)
        let mean: (Float, Float, Float) = (123.675, 116.28, 103.53)
        let std:  (Float, Float, Float) = (58.395,  57.12,  57.375)
        let plane = S * S

        for i in 0..<plane {
            let base = i * 4
            ptr[            i] = (Float(rawRGBA[base])     - mean.0) / std.0
            ptr[plane     + i] = (Float(rawRGBA[base + 1]) - mean.1) / std.1
            ptr[plane * 2 + i] = (Float(rawRGBA[base + 2]) - mean.2) / std.2
        }
        return mlArr
    }
}
