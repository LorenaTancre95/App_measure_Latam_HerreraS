import Foundation
import OnnxRuntimeBindings

/// YOLO detection result.
struct YOLOBox {
    let xmin: Float
    let ymin: Float
    let xmax: Float
    let ymax: Float
    let label: String
    let score: Float
}

/// YOLO ONNX inference wrapper.
final class YOLODetector {
    private let session: ORTSession
    private let env: ORTEnv

    // 1-class model trained on "caja" (models_yolo/best.onnx, 100 epochs).
    static let classNames: [String] = ["caja"]

    init(modelPath: String) throws {
        env = try ORTEnv(loggingLevel: .warning)
        let opts = try ORTSessionOptions()
        try opts.setIntraOpNumThreads(2)
        session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: opts)
    }

    /// Run YOLO detection.
    /// - Parameters:
    ///   - image: CHW float32 array in [0, 1], length = 3 * 640 * 640.
    ///   - imageWidth: 640
    ///   - imageHeight: 640
    ///   - confThreshold: Minimum confidence.
    ///   - iouThreshold: NMS IoU threshold.
    /// - Returns: Array of 2D bounding boxes.
    func detect(
        image: [Float],
        imageWidth: Int = 640,
        imageHeight: Int = 640,
        confThreshold: Float = 0.25,
        iouThreshold: Float = 0.45
    ) throws -> [YOLOBox] {
        let imageData = Data(bytes: image, count: image.count * MemoryLayout<Float>.stride)
        let inputTensor = try ORTValue(
            tensorData: NSMutableData(data: imageData),
            elementType: .float,
            shape: [1, 3, NSNumber(value: imageHeight), NSNumber(value: imageWidth)]
        )

        var boxes: [YOLOBox] = []

        // YOLOv9 native: 3 outputs por escala (p3=80×80, p4=40×40, p5=20×20)
        if let v9out = try? session.run(
            withInputs: ["images": inputTensor],
            outputNames: ["p3_box", "p4_box", "p5_box"],
            runOptions: nil
        ) {
            for (key, anchors) in [("p3_box", 6400), ("p4_box", 1600), ("p5_box", 400)] {
                if let tensor = v9out[key],
                   let data = try? tensor.tensorData() as Data {
                    let vals = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                    boxes += parseAnchors(vals, numAnchors: anchors, confThreshold: confThreshold)
                }
            }
        } else {
            // Fallback: YOLOv8 / YOLOv11 — output único (1, 4+nc, 8400)
            let out = try session.run(
                withInputs: ["images": inputTensor],
                outputNames: ["output0"],
                runOptions: nil
            )
            let data = try out["output0"]!.tensorData() as Data
            let vals = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            boxes = parseAnchors(vals, numAnchors: 8400, confThreshold: confThreshold)
        }

        return nms(boxes: boxes, iouThreshold: iouThreshold)
    }

    // Decodifica un tensor YOLO de shape (1, 4+nc, numAnchors) → YOLOBox array.
    private func parseAnchors(_ values: [Float], numAnchors: Int,
                               confThreshold: Float) -> [YOLOBox] {
        let numClasses = Self.classNames.count
        var boxes: [YOLOBox] = []
        for a in 0..<numAnchors {
            let cx = values[0 * numAnchors + a]
            let cy = values[1 * numAnchors + a]
            let w  = values[2 * numAnchors + a]
            let h  = values[3 * numAnchors + a]
            var bestClass = 0, bestScore: Float = 0
            for c in 0..<numClasses {
                let s = values[(4 + c) * numAnchors + a]
                if s > bestScore { bestScore = s; bestClass = c }
            }
            guard bestScore >= confThreshold else { continue }
            boxes.append(YOLOBox(
                xmin: cx - w/2, ymin: cy - h/2,
                xmax: cx + w/2, ymax: cy + h/2,
                label: bestClass < Self.classNames.count ? Self.classNames[bestClass] : "class_\(bestClass)",
                score: bestScore
            ))
        }
        return boxes
    }

    /// Non-Maximum Suppression.
    private func nms(boxes: [YOLOBox], iouThreshold: Float) -> [YOLOBox] {
        let sorted = boxes.sorted { $0.score > $1.score }
        var keep: [YOLOBox] = []

        for box in sorted {
            var shouldKeep = true
            for kept in keep {
                if iou(box, kept) > iouThreshold && box.label == kept.label {
                    shouldKeep = false
                    break
                }
            }
            if shouldKeep {
                keep.append(box)
            }
        }
        return keep
    }

    private func iou(_ a: YOLOBox, _ b: YOLOBox) -> Float {
        let x1 = max(a.xmin, b.xmin)
        let y1 = max(a.ymin, b.ymin)
        let x2 = min(a.xmax, b.xmax)
        let y2 = min(a.ymax, b.ymax)
        let intersection = max(0, x2 - x1) * max(0, y2 - y1)
        let areaA = (a.xmax - a.xmin) * (a.ymax - a.ymin)
        let areaB = (b.xmax - b.xmin) * (b.ymax - b.ymin)
        let union = areaA + areaB - intersection
        return union > 0 ? intersection / union : 0
    }
}
