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

        // Center-crop to square → scale to 640×640 → CVPixelBuffer (model expects Image type)
        let ci  = CIImage(cvPixelBuffer: pixelBuffer)
        let ctx = CIContext()
        let w = ci.extent.width, h = ci.extent.height, side = min(w, h)
        let cropped = ci.cropped(to: CGRect(x: (w-side)/2, y: (h-side)/2, width: side, height: side))
        let scaled  = cropped.transformed(by: CGAffineTransform(scaleX: CGFloat(S)/side,
                                                                y: CGFloat(S)/side))

        var outBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, S, S, kCVPixelFormatType_32BGRA,
                            [kCVPixelBufferCGImageCompatibilityKey: true,
                             kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary,
                            &outBuffer)
        guard let buf = outBuffer else { return nil }
        ctx.render(scaled, to: buf)

        let imageValue = MLFeatureValue(pixelBuffer: buf)

        let input  = try MLDictionaryFeatureProvider(dictionary: ["image": imageValue])
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

        // Centro del visor en espacio de imagen YOLO (640×640)
        let cx = Float(PalletDetector.imageSize) / 2
        let cy = Float(PalletDetector.imageSize) / 2

        var bestIdx  = -1
        var bestDist = Float.greatestFiniteMagnitude

        // Format A: [1, N, K] where K >= 5  →  [x1,y1,x2,y2,conf,...]
        if dShape.count == 3 && dShape[2] >= 5 {
            let N = dShape[1], K = dShape[2]
            for i in 0..<N {
                let c = dPtr[i * K + 4]
                guard c >= conf else { continue }
                let midX = (dPtr[i * K + 0] + dPtr[i * K + 2]) / 2
                let midY = (dPtr[i * K + 1] + dPtr[i * K + 3]) / 2
                let dist = (midX - cx) * (midX - cx) + (midY - cy) * (midY - cy)
                if dist < bestDist { bestDist = dist; bestIdx = i }
            }
        }
        // Format B: [1, K, N] transposed
        else if dShape.count == 3 && dShape[1] >= 5 {
            let N = dShape[2]
            for i in 0..<N {
                let c = dPtr[4 * N + i]
                guard c >= conf else { continue }
                let midX = (dPtr[0 * N + i] + dPtr[2 * N + i]) / 2
                let midY = (dPtr[1 * N + i] + dPtr[3 * N + i]) / 2
                let dist = (midX - cx) * (midX - cx) + (midY - cy) * (midY - cy)
                if dist < bestDist { bestDist = dist; bestIdx = i }
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
