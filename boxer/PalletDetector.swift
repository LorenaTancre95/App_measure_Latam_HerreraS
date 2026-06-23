import CoreML
import ARKit

/// Runs pallet_seg.mlpackage (YOLO11n-seg, class "cargo") to detect and
/// segment palletized cargo. Returns a binary mask compatible with PalletMeasurer.
final class PalletDetector {
    private let model: MLModel
    static let imageSize = 640

    private let kDetOut = "var_1323"
    private let kMaskOut = "var_1361"

    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine

        guard let url = Bundle.main.url(forResource: "pallet_seg", withExtension: "mlpackage")
                     ?? Bundle.main.url(forResource: "pallet_seg", withExtension: "mlmodelc")
        else { throw DetectorError.modelNotFound }

        model = try MLModel(contentsOf: url, configuration: config)
    }

    /// Detects cargo and returns a binary mask at the model's mask resolution.
    /// Returns nil if no cargo found above the confidence threshold.
    func detect(pixelBuffer: CVPixelBuffer, confThreshold: Float = 0.3) throws -> [[Bool]]? {
        let S = PalletDetector.imageSize

        // Center-crop to square → scale to 640×640 → RGBA bytes
        let ci  = CIImage(cvPixelBuffer: pixelBuffer)
        let ctx = CIContext()
        let w = ci.extent.width, h = ci.extent.height, side = min(w, h)
        let cropped = ci.cropped(to: CGRect(x: (w-side)/2, y: (h-side)/2, width: side, height: side))
        let scaled  = cropped.transformed(by: CGAffineTransform(scaleX: CGFloat(S)/side,
                                                                y: CGFloat(S)/side))
        var rgba = [UInt8](repeating: 0, count: S * S * 4)
        ctx.render(scaled, toBitmap: &rgba, rowBytes: S * 4,
                   bounds: CGRect(x: scaled.extent.origin.x, y: scaled.extent.origin.y,
                                  width: CGFloat(S), height: CGFloat(S)),
                   format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())

        // CHW float32 in [0, 255] — model scales internally
        let arr = try MLMultiArray(shape: [1, 3, S, S] as [NSNumber], dataType: .float32)
        let ptr = arr.dataPointer.assumingMemoryBound(to: Float32.self)
        let n   = S * S
        for i in 0..<n {
            ptr[i]       = Float(rgba[i*4])
            ptr[n + i]   = Float(rgba[i*4 + 1])
            ptr[2*n + i] = Float(rgba[i*4 + 2])
        }

        let input  = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(multiArray: arr)])
        let output = try autoreleasepool { try model.prediction(from: input) }

        // Log shapes once for debugging
        for name in output.featureNames {
            if let v = output.featureValue(for: name)?.multiArrayValue {
                print("PalletDetector '\(name)': \(v.shape)")
            }
        }

        guard let detArr  = output.featureValue(for: kDetOut)?.multiArrayValue,
              let maskArr = output.featureValue(for: kMaskOut)?.multiArrayValue
        else { return nil }

        return parseMask(detArr: detArr, maskArr: maskArr, conf: confThreshold)
    }

    // MARK: - Parsing

    private func parseMask(detArr: MLMultiArray, maskArr: MLMultiArray, conf: Float) -> [[Bool]]? {
        let dShape = detArr.shape.map { $0.intValue }
        let mShape = maskArr.shape.map { $0.intValue }
        let dPtr   = detArr.dataPointer.assumingMemoryBound(to: Float32.self)
        let mPtr   = maskArr.dataPointer.assumingMemoryBound(to: Float32.self)

        var bestIdx  = -1
        var bestConf = conf

        // Format A: [1, N, K] where K >= 5
        if dShape.count == 3 && dShape[2] >= 5 {
            let N = dShape[1], K = dShape[2]
            for i in 0..<N {
                let c = dPtr[i * K + 4]
                if c > bestConf { bestConf = c; bestIdx = i }
            }
        }
        // Format B: [1, K, N] transposed
        else if dShape.count == 3 && dShape[1] >= 5 {
            let K = dShape[1], N = dShape[2]
            for i in 0..<N {
                let c = dPtr[4 * N + i]
                if c > bestConf { bestConf = c; bestIdx = i }
            }
        }

        guard bestIdx >= 0 else { return nil }

        // Mask format A: [1, N, mH, mW] — one mask per detection
        if mShape.count == 4 && mShape[1] > 1 {
            let N = mShape[1], mH = mShape[2], mW = mShape[3]
            guard bestIdx < N else { return nil }
            let offset = bestIdx * mH * mW
            return threshold(mPtr, offset: offset, h: mH, w: mW)
        }

        // Mask format B: [1, 1, mH, mW] — single combined mask
        if mShape.count == 4 && mShape[1] == 1 {
            let mH = mShape[2], mW = mShape[3]
            return threshold(mPtr, offset: 0, h: mH, w: mW)
        }

        // Mask format C: [1, mH, mW]
        if mShape.count == 3 {
            let mH = mShape[1], mW = mShape[2]
            return threshold(mPtr, offset: 0, h: mH, w: mW)
        }

        return nil
    }

    private func threshold(_ ptr: UnsafePointer<Float32>, offset: Int, h: Int, w: Int) -> [[Bool]] {
        var result = [[Bool]](repeating: [Bool](repeating: false, count: w), count: h)
        for y in 0..<h {
            for x in 0..<w {
                result[y][x] = ptr[offset + y * w + x] > 0
            }
        }
        return result
    }

    // MARK: - Errors

    enum DetectorError: LocalizedError {
        case modelNotFound
        var errorDescription: String? { "pallet_seg.mlpackage no encontrado en el bundle" }
    }
}
