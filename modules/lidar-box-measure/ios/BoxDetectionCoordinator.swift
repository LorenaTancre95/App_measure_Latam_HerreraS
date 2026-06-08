import ARKit
import SceneKit
import CoreML
import Vision
import Accelerate

// MARK: - Data types

struct NativeMeasurement {
    var comprimento: Double
    var largura: Double
    var altura: Double
}

private struct YOLOPrediction {
    let classIndex: Int
    let score: Float
    let x1: Float, y1: Float, x2: Float, y2: Float
    let maskCoefficients: [Float]
}

// MARK: - ARMeshGeometry helpers

private extension ARMeshGeometry {
    /// ARMeshClassification por índice de cara (UInt8 per face)
    func classificationAt(faceIndex i: Int) -> ARMeshClassification {
        guard let cls = classification else { return .none }
        let byteOffset = cls.offset + i * cls.stride
        let raw = cls.buffer.contents()
            .advanced(by: byteOffset)
            .assumingMemoryBound(to: UInt8.self)
            .pointee
        return ARMeshClassification(rawValue: Int(raw)) ?? .none
    }
}

private extension ARGeometrySource {
    /// Vértice 3D en espacio local del anchor
    func vertex(at index: UInt32) -> SIMD3<Float> {
        buffer.contents()
            .advanced(by: offset + Int(index) * stride)
            .assumingMemoryBound(to: SIMD3<Float>.self)
            .pointee
    }
}

private extension ARGeometryElement {
    /// Índice de vértice para la cara i, vértice v (0, 1, 2)
    func vertexIndex(at faceIndex: Int, vertex v: Int) -> UInt32 {
        let ptr = buffer.contents()
            .advanced(by: (faceIndex * 3 + v) * bytesPerIndex)
        return bytesPerIndex == 2
            ? UInt32(ptr.assumingMemoryBound(to: UInt16.self).pointee)
            : ptr.assumingMemoryBound(to: UInt32.self).pointee
    }
}

// MARK: - BoxDetectionCoordinator

final class BoxDetectionCoordinator: NSObject {

    // MARK: References
    weak var sceneView: ARSCNView?
    var mode: ARMode = .auto
    private(set) var lastMeasurement: NativeMeasurement?

    // MARK: Callbacks
    var onUpdate: (NativeMeasurement) -> Void
    var onPlaneFound: () -> Void

    // MARK: Timing
    private var lastScanTime: TimeInterval = 0
    private let scanInterval: TimeInterval = 0.20
    private var scanInFlight = false

    // MARK: YOLO detection persistence
    private var lastDetection: CGRect?
    private var missedFrames   = 0
    private let maxMissedFrames = 8

    // MARK: YOLO model
    private var yoloModel: VNCoreMLModel?
    private let modelInputSize: CGFloat = 640
    private var modelStatusText = "YOLO: loading..."

    // MARK: SAM segmentation
    private let sam = SAMInference()

    // MARK: Gemini detection
    private let gemini          = GeminiDetector()
    private var cachedBox:        GeminiBox? = nil
    private var lastGeminiCall:   TimeInterval   = 0
    private let geminiInterval:   TimeInterval   = 2.5
    private var lastMeasureTime:  TimeInterval   = 0
    private let measureInterval:  TimeInterval   = 0.10

    // MARK: 2D overlay (CALayer)
    private let detectionLayer = CAShapeLayer()
    private let labelLayer     = CATextLayer()
    private let debugLayer     = CATextLayer()
    private let wireframeLayer = CAShapeLayer()

    // MARK: 3D wireframe (SceneKit)
    private var overlayNodes: [SCNNode] = []

    // MARK: Stability / EMA / Lock
    private var buffer: [NativeMeasurement] = []
    private let stabilityWindow = 8
    private let thresholdCm     = 2.0
    private var smoothBL: SIMD3<Float>?
    private var smoothBR: SIMD3<Float>?
    private var smoothTL: SIMD3<Float>?
    private var smoothTR: SIMD3<Float>?
    private var smoothBBL: SIMD3<Float>?
    private var smoothBBR: SIMD3<Float>?
    private var smoothBTL: SIMD3<Float>?
    private var smoothBTR: SIMD3<Float>?
    private let smoothAlpha: Float = 0.30

    // MARK: Point cloud accumulation (multi-frame fusion)
    private var accPoints: [SIMD3<Float>] = []
    private let accMaxPoints = 3000

    // MARK: Floor plane
    private var floorWorldY: Float? = nil

    private var isLocked = false
    private var lockedTransform: simd_float4x4?
    private let lockMovementThreshold: Float = 0.08

    // MARK: Manual mode
    private var manualPoints: [SIMD3<Float>] = []

    // MARK: Init
    init(sceneView: ARSCNView,
         onUpdate: @escaping (NativeMeasurement) -> Void,
         onPlaneFound: @escaping () -> Void) {
        self.sceneView = sceneView
        self.onUpdate  = onUpdate
        self.onPlaneFound = onPlaneFound
        super.init()
        loadModel()
        sam.load()
        setupLayers()
    }

    // MARK: - Model loading
    private func loadModel() {
        let names = ["box_detector",
                     "coco128-yolo11n-seg", "coco128_yolo11n_seg",
                     "coco128-yolov8n-seg", "coco128_yolov8n_seg",
                     "model_core"]
        let exts  = ["mlmodelc", "mlpackage", "mlmodel"]

        var searchBundles: [Bundle] = [Bundle.main, Bundle(for: BoxDetectionCoordinator.self)]
        for bName in ["LidarBoxMeasure", "LidarBoxMeasureResources", "lidar-box-measure"] {
            for parent in [Bundle.main, Bundle(for: BoxDetectionCoordinator.self)] {
                if let u = parent.url(forResource: bName, withExtension: "bundle"),
                   let b = Bundle(url: u) { searchBundles.append(b) }
            }
        }

        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        var lastError = ""

        for b in searchBundles {
            let root = URL(fileURLWithPath: b.bundlePath)
            for name in names {
                for ext in exts {
                    let url = root.appendingPathComponent("\(name).\(ext)")
                    guard FileManager.default.fileExists(atPath: url.path) else { continue }
                    do {
                        let loadURL: URL
                        if ext == "mlpackage" {
                            let cached = cacheDir.appendingPathComponent("\(name).mlmodelc")
                            if FileManager.default.fileExists(atPath: cached.path) {
                                loadURL = cached
                            } else {
                                modelStatusText = "Compilando modelo..."
                                let tmp = try MLModel.compileModel(at: url)
                                try? FileManager.default.moveItem(at: tmp, to: cached)
                                loadURL = cached
                            }
                        } else {
                            loadURL = url
                        }
                        let cfg = MLModelConfiguration()
                        if #available(iOS 16.0, *) { cfg.computeUnits = .cpuAndNeuralEngine }
                        else                        { cfg.computeUnits = .all }
                        let ml = try MLModel(contentsOf: loadURL, configuration: cfg)
                        yoloModel = try VNCoreMLModel(for: ml)
                        modelStatusText = "YOLO OK: \(name).\(ext)"
                        return
                    } catch {
                        lastError = "\(name).\(ext): \(error.localizedDescription)"
                    }
                }
            }
        }
        modelStatusText = lastError.isEmpty ? "NOT LOADED" : "LOAD ERR: \(lastError)"
    }

    // MARK: - Layer setup
    private func setupLayers() {
        detectionLayer.fillColor   = UIColor.clear.cgColor
        detectionLayer.strokeColor = UIColor(red: 0, green: 0.9, blue: 0.3, alpha: 1).cgColor
        detectionLayer.lineWidth   = 3
        detectionLayer.cornerRadius = 4

        labelLayer.fontSize        = 14
        labelLayer.foregroundColor = UIColor.white.cgColor
        labelLayer.backgroundColor = UIColor(red: 0, green: 0.7, blue: 0.2, alpha: 0.8).cgColor
        labelLayer.alignmentMode   = .center
        labelLayer.contentsScale   = UIScreen.main.scale

        debugLayer.fontSize        = 12
        debugLayer.foregroundColor = UIColor.yellow.cgColor
        debugLayer.backgroundColor = UIColor.black.withAlphaComponent(0.6).cgColor
        debugLayer.alignmentMode   = .left
        debugLayer.contentsScale   = UIScreen.main.scale
        debugLayer.isWrapped       = true

        wireframeLayer.fillColor   = UIColor.clear.cgColor
        wireframeLayer.strokeColor = UIColor(red: 0.0, green: 0.90, blue: 0.3, alpha: 1).cgColor
        wireframeLayer.lineWidth   = 2.5
        wireframeLayer.lineCap     = .round
    }

    private func attachLayersIfNeeded() {
        guard let sv = sceneView, detectionLayer.superlayer == nil else { return }
        detectionLayer.frame  = sv.bounds
        wireframeLayer.frame  = sv.bounds
        labelLayer.frame      = CGRect(x: 0, y: 0, width: 120, height: 22)
        debugLayer.frame      = CGRect(x: 8, y: 60, width: sv.bounds.width - 16, height: 120)
        sv.layer.addSublayer(detectionLayer)
        sv.layer.addSublayer(wireframeLayer)
        sv.layer.addSublayer(labelLayer)
        sv.layer.addSublayer(debugLayer)
        debugLayer.string = gemini.status
    }

    private func updateDebug(_ text: String) {
        DispatchQueue.main.async {
            CATransaction.begin(); CATransaction.setDisableActions(true)
            self.debugLayer.string = text
            CATransaction.commit()
        }
    }

    // MARK: - Main pipeline (Gemini corners + LiDAR depth)
    private func measureFromCenter(frame: ARFrame, depth: ARDepthData) {
        guard let sv = sceneView else { return }
        attachLayersIfNeeded()
        let vp = sv.bounds.size

        // ── Por frame: medir con la caja cacheada de Gemini ─────────────────
        if let box = cachedBox,
           frame.timestamp - lastMeasureTime > measureInterval {
            lastMeasureTime = frame.timestamp
            measureWithBox(box, frame: frame, depth: depth, vp: vp)
        }

        // ── Periódico: llamar a Gemini cada geminiInterval segundos ──────────
        guard !scanInFlight,
              frame.timestamp - lastGeminiCall > geminiInterval else { return }
        lastGeminiCall = frame.timestamp
        scanInFlight   = true
        let pb = frame.capturedImage

        updateDebug("Gemini: detectando...")

        gemini.detectBox(pixelBuffer: pb) { [weak self] box in
            guard let self = self else { return }
            DispatchQueue.main.async {
                self.scanInFlight = false
                if let b = box {
                    self.cachedBox    = b
                    self.missedFrames = 0
                } else {
                    self.missedFrames += 1
                    if self.missedFrames > self.maxMissedFrames {
                        self.cachedBox = nil
                        self.clearDetectionLayer()
                    }
                }
                self.updateDebug(self.gemini.status)
            }
        }
    }

    // MARK: - Medición: Gemini 8 vértices → wireframe 2D + PCA LiDAR para medidas
    private func measureWithBox(_ box: GeminiBox,
                                 frame: ARFrame, depth: ARDepthData, vp: CGSize) {
        // Dibujar wireframe 2D inmediatamente desde los vértices de Gemini
        DispatchQueue.main.async { self.draw2DBox(box, in: vp) }

        // Lock check
        if isLocked {
            if let lt = lockedTransform {
                let cur = frame.camera.transform.columns.3
                let prv = lt.columns.3
                if simd_length(SIMD3<Float>(cur.x-prv.x, cur.y-prv.y, cur.z-prv.z)) < lockMovementThreshold { return }
            }
            isLocked = false; lockedTransform = nil; buffer.removeAll()
            smoothBL = nil; smoothBR = nil; smoothTL = nil; smoothTR = nil
            smoothBBL = nil; smoothBBR = nil; smoothBTL = nil; smoothBTR = nil
        }
        lockedTransform = frame.camera.transform

        // Cara frontal en coordenadas pantalla
        func sp(_ p: CGPoint) -> CGPoint { CGPoint(x: p.x * vp.width, y: p.y * vp.height) }
        let sFBL = sp(box.fbl), sFBR = sp(box.fbr)
        let sFTL = sp(box.ftl), sFTR = sp(box.ftr)
        let frontXs = [sFBL.x, sFBR.x, sFTL.x, sFTR.x]
        let frontYs = [sFBL.y, sFBR.y, sFTL.y, sFTR.y]
        let screenBox = CGRect(x: frontXs.min()!, y: frontYs.min()!,
                               width: frontXs.max()! - frontXs.min()!,
                               height: frontYs.max()! - frontYs.min()!)
        guard screenBox.width > 20, screenBox.height > 20 else { return }

        // Profundidad del centro de la cara frontal
        guard let centerD = sampleDepth(at: CGPoint(x: screenBox.midX, y: screenBox.midY),
                                         frame: frame, depth: depth, vp: vp),
              centerD > 0.15, centerD < 4.0 else { return }

        // Recolectar puntos 3D de la cara frontal usando el polígono Gemini como máscara
        let camPos3 = SIMD3<Float>(frame.camera.transform.columns.3.x,
                                    frame.camera.transform.columns.3.y,
                                    frame.camera.transform.columns.3.z)
        let geminiQuad: [CGPoint] = [sFTL, sFTR, sFBR, sFBL]
        let frontPts = collectFrontFacePoints(centerD: centerD, yoloBox: screenBox,
                                               quadMask: geminiQuad, samMask: nil,
                                               frame: frame, depth: depth, vp: vp)
        guard frontPts.count >= 15 else { return }

        // Acumular puntos multi-frame y ejecutar PCA
        accPoints.append(contentsOf: frontPts)
        if accPoints.count > accMaxPoints { accPoints.removeFirst(accPoints.count - accMaxPoints) }
        let ptsForPCA = accPoints.count >= 200 ? accPoints : frontPts

        let centroid = ptsForPCA.reduce(.zero, +) / Float(ptsForPCA.count)
        let (faceR, faceU, faceN) = computeFacePCA(ptsForPCA, centroid: centroid, camPos: camPos3)

        var minR: Float = .infinity, maxR: Float = -.infinity
        var minU: Float = .infinity, maxU: Float = -.infinity
        for p in ptsForPCA {
            let d = p - centroid
            let pr = simd_dot(d, faceR), pu = simd_dot(d, faceU)
            if pr < minR { minR = pr }; if pr > maxR { maxR = pr }
            if pu < minU { minU = pu }; if pu > maxU { maxU = pu }
        }

        var bl = centroid + faceR * minR + faceU * minU
        var br = centroid + faceR * maxR + faceU * minU
        var tl = centroid + faceR * minR + faceU * maxU
        var tr = centroid + faceR * maxR + faceU * maxU

        let c = Double(maxR - minR) * 100
        var a = Double(maxU - minU) * 100
        guard c > 3, c < 300, a > 3, a < 300 else { return }

        // Snap to floor
        if let fy = floorWorldY {
            let bottomY = min(bl.y, br.y)
            let snap = fy - bottomY
            if abs(snap) < 0.25 {
                bl.y += snap; br.y += snap
                tl.y += snap; tr.y += snap
                a = Double(max(tl.y, tr.y) - fy) * 100
            }
        }
        guard a > 3 else { return }

        // Profundidad: samples LiDAR más profundos dentro del bbox
        let dm = depth.depthMap
        let dW = CVPixelBufferGetWidth(dm), dH = CVPixelBufferGetHeight(dm)
        let invTx = frame.displayTransform(for: .portrait, viewportSize: vp).inverted()
        let bboxPts = [CGPoint(x: screenBox.minX/vp.width, y: screenBox.minY/vp.height),
                       CGPoint(x: screenBox.maxX/vp.width, y: screenBox.maxY/vp.height)].map { $0.applying(invTx) }
        let dxS = max(0,    Int(bboxPts.map { $0.x }.min()! * CGFloat(dW)))
        let dxE = min(dW-1, Int(bboxPts.map { $0.x }.max()! * CGFloat(dW)))
        let dyS = max(0,    Int(bboxPts.map { $0.y }.min()! * CGFloat(dH)))
        let dyE = min(dH-1, Int(bboxPts.map { $0.y }.max()! * CGFloat(dH)))
        var deeperSamples: [Float] = []
        CVPixelBufferLockBaseAddress(dm, .readOnly)
        if let base = CVPixelBufferGetBaseAddress(dm) {
            let ptr = base.assumingMemoryBound(to: Float32.self)
            for dy in stride(from: dyS, through: dyE, by: 3) {
                for dx in stride(from: dxS, through: dxE, by: 3) {
                    let v = ptr[dy * dW + dx]
                    if v > centerD + 0.06 && v < centerD + 0.50 { deeperSamples.append(v) }
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(dm, .readOnly)

        let toBox = simd_normalize(centroid - camPos3)
        let cosTheta = min(abs(simd_dot(faceN, -toBox)), 1.0)
        let sinTheta = sqrt(max(0, 1 - cosTheta * cosTheta))
        let cCorrected = cosTheta > 0.25 ? c / Double(cosTheta) : c
        let maxDepthM = Float(cCorrected / 100.0)
        let depthEst: Float
        if deeperSamples.count >= 6 {
            let ds = deeperSamples.sorted()
            let rawApparent = ds[min(ds.count-1, ds.count*3/4)] - centerD
            let rawCorrected = sinTheta > 0.25 ? rawApparent / sinTheta : rawApparent
            depthEst = max(min(rawCorrected, maxDepthM), 0.03)
        } else {
            depthEst = Float(cCorrected / 100.0) * 0.80
        }
        let l = max(Double(depthEst) * 100, 3.0)

        let ext  = -faceN * Float(l / 100)
        let bbl  = bl + ext, bbr = br + ext
        let btl  = tl + ext, btr = tr + ext

        let α = smoothAlpha
        smoothBL  = smoothBL.map  { α*bl  + (1-α)*$0 } ?? bl
        smoothBR  = smoothBR.map  { α*br  + (1-α)*$0 } ?? br
        smoothTL  = smoothTL.map  { α*tl  + (1-α)*$0 } ?? tl
        smoothTR  = smoothTR.map  { α*tr  + (1-α)*$0 } ?? tr
        smoothBBL = smoothBBL.map { α*bbl + (1-α)*$0 } ?? bbl
        smoothBBR = smoothBBR.map { α*bbr + (1-α)*$0 } ?? bbr
        smoothBTL = smoothBTL.map { α*btl + (1-α)*$0 } ?? btl
        smoothBTR = smoothBTR.map { α*btr + (1-α)*$0 } ?? btr

        let m = NativeMeasurement(comprimento: cCorrected, largura: l, altura: a)
        addToBuffer(m)
        DispatchQueue.main.async {
            self.updateOverlay(bl: self.smoothBL!, br: self.smoothBR!,
                               tl: self.smoothTL!, tr: self.smoothTR!,
                               bbl: self.smoothBBL!, bbr: self.smoothBBR!,
                               btl: self.smoothBTL!, btr: self.smoothBTR!,
                               measurement: m)
        }
    }

    // MARK: - Muestra profundidad LiDAR en vecindario de un punto pantalla
    // Percentil 25 para preferir la cara frontal (más cercana) sobre bordes ruidosos.
    private func sampleDepthNeighborhood(at pt: CGPoint, frame: ARFrame,
                                          depth: ARDepthData, vp: CGSize) -> Float? {
        let offsets: [CGFloat] = [-8, -4, 0, 4, 8]
        var depths: [Float] = []
        for dy in offsets {
            for dx in offsets {
                if let d = sampleDepth(at: CGPoint(x: pt.x + dx, y: pt.y + dy),
                                       frame: frame, depth: depth, vp: vp) {
                    depths.append(d)
                }
            }
        }
        guard depths.count >= 3 else { return nil }
        let sorted = depths.sorted()
        return sorted[sorted.count / 4]
    }

    // MARK: - YOLO inference
    private func detectClosestBox(in pixelBuffer: CVPixelBuffer,
                                   viewportSize vp: CGSize,
                                   cx: CGFloat, cy: CGFloat) -> YOLOPrediction? {
        guard let model = yoloModel else {
            updateDebug("\(modelStatusText)\nNO MODEL\n\(sam.status)")
            return nil
        }

        var boxesOutput: MLMultiArray?
        var allObsTypes = [String]()
        let confidenceThreshold: Float = 0.08
        let iouThreshold:        Float = 0.60

        let request = VNCoreMLRequest(model: model) { req, err in
            if let err = err { allObsTypes.append("ERR:\(err.localizedDescription)") }
            let obs = req.results ?? []
            allObsTypes.append(contentsOf: obs.map { String(describing: type(of: $0)) })
            let observations = obs.compactMap { $0 as? VNCoreMLFeatureValueObservation }
            boxesOutput = observations
                .first { $0.featureValue.multiArrayValue?.shape.count == 3 }?
                .featureValue.multiArrayValue
        }
        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: .right, options: [:])
        do { try handler.perform([request]) }
        catch { updateDebug("\(modelStatusText)\nperform ERR: \(error)"); return nil }

        guard let boxes = boxesOutput else {
            let types = allObsTypes.prefix(3).joined(separator: ", ")
            updateDebug("\(modelStatusText)\nnil output | [\(types)]\n\(sam.status)")
            return nil
        }

        let shape = boxes.shape.map { $0.intValue }
        guard shape.count == 3 else { return nil }
        let C = shape[1]; let N = shape[2]
        let numSegMasks = C > 36 ? 32 : 0
        let numClasses  = C - 4 - numSegMasks
        guard numClasses > 0 else { return nil }

        let strides = boxes.strides.map { $0.intValue }
        let ptr     = boxes.dataPointer.assumingMemoryBound(to: Float.self)

        @inline(__always)
        func idx(_ ch: Int, _ anchor: Int) -> Int { ch * strides[1] + anchor * strides[2] }

        var predictions = [YOLOPrediction]()
        for i in 0..<N {
            let cx640 = ptr[idx(0, i)], cy640 = ptr[idx(1, i)]
            let w640  = ptr[idx(2, i)], h640  = ptr[idx(3, i)]
            var classScores = [Float](repeating: 0, count: numClasses)
            for j in 0..<numClasses { classScores[j] = ptr[idx(4 + j, i)] }
            var best: Float = 0; var bestCls: vDSP_Length = 0
            vDSP_maxvi(classScores, 1, &best, &bestCls, vDSP_Length(numClasses))
            guard best >= confidenceThreshold else { continue }
            var coefs = [Float](repeating: 0, count: numSegMasks)
            for k in 0..<numSegMasks { coefs[k] = ptr[idx(4 + numClasses + k, i)] }
            predictions.append(YOLOPrediction(
                classIndex: Int(bestCls), score: best,
                x1: cx640 - w640*0.5, y1: cy640 - h640*0.5,
                x2: cx640 + w640*0.5, y2: cy640 + h640*0.5,
                maskCoefficients: coefs))
        }
        guard !predictions.isEmpty else {
            updateDebug("\(modelStatusText)\nC=\(C) N=\(N) cls=\(numClasses) no det\n\(sam.status)")
            return nil
        }

        let grouped = Dictionary(grouping: predictions) { $0.classIndex }
        var nms = [YOLOPrediction]()
        for (_, g) in grouped { nms.append(contentsOf: nonMaxSuppression(g, iou: iouThreshold)) }
        guard !nms.isEmpty else { return nil }

        let best = nms.min(by: {
            let ra = $0.screenRect(viewportSize: vp, modelSize: modelInputSize)
            let rb = $1.screenRect(viewportSize: vp, modelSize: modelInputSize)
            return hypot(ra.midX - cx, ra.midY - cy) < hypot(rb.midX - cx, rb.midY - cy)
        })!
        updateDebug("\(modelStatusText)\ncls=\(Int(best.score*100))% \(sam.status)")
        return best
    }

    private func nonMaxSuppression(_ preds: [YOLOPrediction], iou: Float) -> [YOLOPrediction] {
        let sorted = preds.sorted { $0.score > $1.score }
        var kept = [YOLOPrediction](); var active = [Bool](repeating: true, count: sorted.count)
        for i in 0..<sorted.count {
            guard active[i] else { continue }
            kept.append(sorted[i])
            for j in (i+1)..<sorted.count {
                guard active[j] else { continue }
                if iouBetween(sorted[i], sorted[j]) > iou { active[j] = false }
            }
        }
        return kept
    }

    private func iouBetween(_ a: YOLOPrediction, _ b: YOLOPrediction) -> Float {
        let ix1 = max(a.x1, b.x1), iy1 = max(a.y1, b.y1)
        let ix2 = min(a.x2, b.x2), iy2 = min(a.y2, b.y2)
        let inter = max(ix2-ix1, 0) * max(iy2-iy1, 0)
        return inter / ((a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - inter + 1e-6)
    }

    // MARK: - 2D detection overlay
    private func drawDetectionRect(_ rect: CGRect, label: String, in vp: CGSize) {
        detectionLayer.frame = CGRect(origin: .zero, size: vp)
        CATransaction.begin(); CATransaction.setDisableActions(true)
        detectionLayer.path = UIBezierPath(roundedRect: rect, cornerRadius: 6).cgPath
        labelLayer.string   = label
        labelLayer.position = CGPoint(x: rect.midX, y: rect.minY - 14)
        CATransaction.commit()
    }

    private func clearDetectionLayer() {
        CATransaction.begin(); CATransaction.setDisableActions(true)
        detectionLayer.path = nil; labelLayer.string = nil; wireframeLayer.path = nil
        CATransaction.commit()
    }

    // MARK: - 2D wireframe: solo cara frontal (Gemini detecta bien los 4 corners frontales;
    // los traseros son estimados y poco fiables). El cubo completo lo dibuja updateOverlay.
    private func draw2DBox(_ box: GeminiBox, in vp: CGSize) {
        wireframeLayer.frame = CGRect(origin: .zero, size: vp)
        func sp(_ p: CGPoint) -> CGPoint { CGPoint(x: p.x * vp.width, y: p.y * vp.height) }
        let fbl = sp(box.fbl), fbr = sp(box.fbr)
        let ftl = sp(box.ftl), ftr = sp(box.ftr)

        let path = UIBezierPath()
        for (a, b) in [(fbl,fbr),(fbr,ftr),(ftr,ftl),(ftl,fbl)] {
            path.move(to: a); path.addLine(to: b)
        }

        CATransaction.begin(); CATransaction.setDisableActions(true)
        wireframeLayer.path = path.cgPath
        CATransaction.commit()
    }

    // MARK: - ARKit raycast
    // Intenta primero contra geometría de malla existente, luego plano estimado.
    // Debe llamarse desde el main thread.
    private func raycast(from screenPt: CGPoint) -> SIMD3<Float>? {
        guard let sv = sceneView else { return nil }
        for target: ARRaycastQuery.Target in [.existingPlaneGeometry, .estimatedPlane] {
            guard let q = sv.raycastQuery(from: screenPt, allowing: target, alignment: .any),
                  let r = sv.session.raycast(q).first else { continue }
            let c = r.worldTransform.columns.3
            return SIMD3<Float>(c.x, c.y, c.z)
        }
        return nil
    }

    // MARK: - Medición: YOLO bbox + SAM mask + LiDAR depth map
    private func measure3D(box: CGRect, samMask: [Bool]?,
                            frame: ARFrame, depth: ARDepthData, vp: CGSize) {
        // Lock mode
        if isLocked {
            if let lt = lockedTransform {
                let cur = frame.camera.transform.columns.3
                let prv = lt.columns.3
                let moved = simd_length(SIMD3<Float>(cur.x-prv.x, cur.y-prv.y, cur.z-prv.z))
                if moved < lockMovementThreshold { return }
            }
            isLocked = false; lockedTransform = nil; buffer.removeAll()
            smoothBL = nil; smoothBR = nil; smoothTL = nil; smoothTR = nil
            smoothBBL = nil; smoothBBR = nil; smoothBTL = nil; smoothBTR = nil
        }
        lockedTransform = frame.camera.transform

        guard box.width > 15, box.height > 15 else { return }

        // 1. Grilla 14×14 para estimar centerD (rápido)
        let G = 14
        let maskW = SAMInference.maskW, maskH = SAMInference.maskH
        var rawDepths:    [Float] = []
        var maskedDepths: [Float] = []
        rawDepths.reserveCapacity(G * G)
        maskedDepths.reserveCapacity(G * G)

        for r in 0..<G {
            for c in 0..<G {
                let sx = box.minX + box.width  * CGFloat(c) / CGFloat(G - 1)
                let sy = box.minY + box.height * CGFloat(r) / CGFloat(G - 1)
                guard let d = sampleDepth(at: CGPoint(x: sx, y: sy),
                                          frame: frame, depth: depth, vp: vp) else { continue }
                rawDepths.append(d)
                if let mask = samMask {
                    let mx = max(0, min(maskW - 1, Int(sx / vp.width  * CGFloat(maskW))))
                    let my = max(0, min(maskH - 1, Int(sy / vp.height * CGFloat(maskH))))
                    if mask[my * maskW + mx] { maskedDepths.append(d) }
                }
            }
        }

        let allDepths = (samMask != nil && maskedDepths.count >= 10) ? maskedDepths : rawDepths
        guard allDepths.count >= 10 else { return }

        let minD = allDepths.min()!
        let frontCluster = allDepths.filter { abs($0 - minD) < 0.15 }
        let sortedFront = frontCluster.sorted()
        let centerD = sortedFront[sortedFront.count / 2]

        // 2. Proyectar pixels del depth map (cara frontal) directamente a 3D
        // Mucho más preciso que proyectar 4 corners 2D de YOLO: usa cientos de puntos reales.
        let camPos3 = SIMD3<Float>(frame.camera.transform.columns.3.x,
                                    frame.camera.transform.columns.3.y,
                                    frame.camera.transform.columns.3.z)
        let frontPts3D = collectFrontFacePoints(centerD: centerD, yoloBox: box,
                                                 samMask: samMask, frame: frame,
                                                 depth: depth, vp: vp)
        guard frontPts3D.count >= 15 else { return }

        // 3. Medir cara frontal en 3D: PCA da los verdaderos ejes del plano de la cara
        let centroid3D = frontPts3D.reduce(.zero, +) / Float(frontPts3D.count)
        let (faceR, faceU, faceN) = computeFacePCA(frontPts3D, centroid: centroid3D, camPos: camPos3)

        var minR: Float = .infinity, maxR: Float = -.infinity
        var minU: Float = .infinity, maxU: Float = -.infinity
        for p in frontPts3D {
            let d = p - centroid3D
            let pr = simd_dot(d, faceR), pu = simd_dot(d, faceU)
            if pr < minR { minR = pr }; if pr > maxR { maxR = pr }
            if pu < minU { minU = pu }; if pu > maxU { maxU = pu }
        }

        let bl = centroid3D + faceR * minR + faceU * minU
        let br = centroid3D + faceR * maxR + faceU * minU
        let tl = centroid3D + faceR * minR + faceU * maxU
        let tr = centroid3D + faceR * maxR + faceU * maxU

        let c = Double(maxR - minR) * 100
        var a = Double(maxU - minU) * 100
        // If floor plane is known, use it to cross-check height
        if let fy = floorWorldY {
            let topY = Double(max(tl.y, tr.y))
            let hFloor = (topY - Double(fy)) * 100
            if hFloor > 5 && hFloor < 300 { a = hFloor }
        }
        guard c > 3, c < 300, a > 3, a < 300 else { return }

        // 4. Profundidad de la caja: percentil 75 de deeper samples para evitar outliers de fondo
        let deeperSamples = allDepths.filter { $0 > centerD + 0.06 && $0 < centerD + 0.65 }
        let depthEst: Float
        if deeperSamples.count >= 6 {
            let ds = deeperSamples.sorted()
            let p75 = ds[min(ds.count - 1, ds.count * 3 / 4)]
            depthEst = max(p75 - centerD, 0.03)
        } else {
            depthEst = Float(min(c, a) / 100.0) * 0.65
        }
        let l = max(Double(depthEst) * 100, 3.0)

        // 4. Extruir cara trasera a lo largo de la normal
        let ext = -faceN * Float(l / 100)
        let bbl = bl + ext, bbr = br + ext
        let btl = tl + ext, btr = tr + ext

        // 5. EMA smoothing
        let α = smoothAlpha
        smoothBL  = smoothBL.map  { α*bl  + (1-α)*$0 } ?? bl
        smoothBR  = smoothBR.map  { α*br  + (1-α)*$0 } ?? br
        smoothTL  = smoothTL.map  { α*tl  + (1-α)*$0 } ?? tl
        smoothTR  = smoothTR.map  { α*tr  + (1-α)*$0 } ?? tr
        smoothBBL = smoothBBL.map { α*bbl + (1-α)*$0 } ?? bbl
        smoothBBR = smoothBBR.map { α*bbr + (1-α)*$0 } ?? bbr
        smoothBTL = smoothBTL.map { α*btl + (1-α)*$0 } ?? btl
        smoothBTR = smoothBTR.map { α*btr + (1-α)*$0 } ?? btr

        let m = NativeMeasurement(comprimento: c, largura: l, altura: a)
        addToBuffer(m)
        updateOverlay(bl: smoothBL!, br: smoothBR!, tl: smoothTL!, tr: smoothTR!,
                      bbl: smoothBBL!, bbr: smoothBBR!, btl: smoothBTL!, btr: smoothBTR!,
                      measurement: m)
    }

    // MARK: - Colectar puntos 3D de la cara frontal desde el depth map
    // Itera el depth map con step=2 dentro del yoloBox, filtrando depth ≈ centerD (±10cm)
    // y máscara SAM. Proyecta directamente a mundo 3D sin pasar por pantalla.
    private func pointInConvexQuad(_ p: CGPoint, _ q: [CGPoint]) -> Bool {
        func cross(_ a: CGPoint, _ b: CGPoint, _ c: CGPoint) -> CGFloat {
            (b.x-a.x)*(c.y-a.y) - (b.y-a.y)*(c.x-a.x)
        }
        let n = q.count
        var pos = 0, neg = 0
        for i in 0..<n {
            let c = cross(q[i], q[(i+1)%n], p)
            if c > 0 { pos += 1 } else if c < 0 { neg += 1 }
        }
        return pos == n || neg == n
    }

    private func collectFrontFacePoints(centerD: Float, yoloBox: CGRect,
                                         quadMask: [CGPoint]? = nil,
                                         samMask: [Bool]?,
                                         frame: ARFrame, depth: ARDepthData,
                                         vp: CGSize) -> [SIMD3<Float>] {
        let dm = depth.depthMap
        let dW = CVPixelBufferGetWidth(dm), dH = CVPixelBufferGetHeight(dm)

        let invTx = frame.displayTransform(for: .portrait, viewportSize: vp).inverted()
        let corners: [CGPoint] = [
            CGPoint(x: yoloBox.minX / vp.width, y: yoloBox.minY / vp.height),
            CGPoint(x: yoloBox.maxX / vp.width, y: yoloBox.minY / vp.height),
            CGPoint(x: yoloBox.minX / vp.width, y: yoloBox.maxY / vp.height),
            CGPoint(x: yoloBox.maxX / vp.width, y: yoloBox.maxY / vp.height),
        ].map { $0.applying(invTx) }
        let dxS = max(0, Int(corners.map { $0.x }.min()! * CGFloat(dW)))
        let dxE = min(dW - 1, Int(corners.map { $0.x }.max()! * CGFloat(dW)))
        let dyS = max(0, Int(corners.map { $0.y }.min()! * CGFloat(dH)))
        let dyE = min(dH - 1, Int(corners.map { $0.y }.max()! * CGFloat(dH)))

        CVPixelBufferLockBaseAddress(dm, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(dm, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(dm) else { return [] }
        let dptr = base.assumingMemoryBound(to: Float32.self)

        let displayTx = frame.displayTransform(for: .portrait, viewportSize: vp)
        let maskW = SAMInference.maskW, maskH = SAMInference.maskH
        let intr = frame.camera.intrinsics
        let iW   = Float(frame.camera.imageResolution.width)
        let iH   = Float(frame.camera.imageResolution.height)

        var points: [SIMD3<Float>] = []

        let step = 2
        for dy in stride(from: dyS, through: dyE, by: step) {
            for dx in stride(from: dxS, through: dxE, by: step) {
                let d = dptr[dy * dW + dx]
                let depthTol: Float = samMask != nil ? 0.25 : 0.12
                guard d > 0.02, d < 8.0, abs(d - centerD) < depthTol else { continue }

                // Filtrar con polígono de Gemini o máscara SAM
                if quadMask != nil || samMask != nil {
                    let nc = CGPoint(x: CGFloat(dx) / CGFloat(dW), y: CGFloat(dy) / CGFloat(dH))
                    let ns = nc.applying(displayTx)
                    let sp = CGPoint(x: ns.x * vp.width, y: ns.y * vp.height)
                    if let quad = quadMask, !pointInConvexQuad(sp, quad) { continue }
                    if let mask = samMask {
                        let mx = max(0, min(maskW - 1, Int(sp.x / vp.width  * CGFloat(maskW))))
                        let my = max(0, min(maskH - 1, Int(sp.y / vp.height * CGFloat(maskH))))
                        guard mask[my * maskW + mx] else { continue }
                    }
                }

                // Depth pixel → cámara imagen → cámara 3D → mundo (sin pasar por pantalla)
                let imgX = Float(dx) / Float(dW) * iW
                let imgY = Float(dy) / Float(dH) * iH
                let xCam = (imgX - intr[2][0]) / intr[0][0] * d
                let yCam = (imgY - intr[2][1]) / intr[1][1] * d
                let pt   = frame.camera.transform * SIMD4<Float>(xCam, yCam, -d, 1)
                points.append(SIMD3<Float>(pt.x, pt.y, pt.z))
            }
        }
        return points
    }

    // MARK: - PCA: verdaderos ejes del plano frontal de la caja
    // El eigenvector de mayor varianza → faceR (ancho), segundo → faceU (alto),
    // tercero → faceN (normal, menor varianza porque los puntos están en un plano).
    private func computeFacePCA(_ pts: [SIMD3<Float>], centroid: SIMD3<Float>,
                                  camPos: SIMD3<Float>)
        -> (right: SIMD3<Float>, up: SIMD3<Float>, normal: SIMD3<Float>)
    {
        var Cxx: Float = 0, Cyy: Float = 0, Czz: Float = 0
        var Cxy: Float = 0, Cxz: Float = 0, Cyz: Float = 0
        for p in pts {
            let d = p - centroid
            Cxx += d.x*d.x; Cyy += d.y*d.y; Czz += d.z*d.z
            Cxy += d.x*d.y; Cxz += d.x*d.z; Cyz += d.y*d.z
        }
        let n = Float(pts.count)
        Cxx /= n; Cyy /= n; Czz /= n; Cxy /= n; Cxz /= n; Cyz /= n

        func cov(_ v: SIMD3<Float>) -> SIMD3<Float> {
            SIMD3(Cxx*v.x + Cxy*v.y + Cxz*v.z,
                  Cxy*v.x + Cyy*v.y + Cyz*v.z,
                  Cxz*v.x + Cyz*v.y + Czz*v.z)
        }

        // Power iteration: eigenvector de mayor varianza
        var e1 = simd_normalize(SIMD3<Float>(1, 0.01, 0.01))
        for _ in 0..<30 { let t = cov(e1); guard simd_length(t) > 1e-8 else { break }; e1 = simd_normalize(t) }

        // Segundo eigenvector via deflación
        var e2 = simd_normalize(SIMD3<Float>(0.01, 1, 0.01))
        for _ in 0..<30 {
            let t = cov(e2); guard simd_length(t) > 1e-8 else { break }
            let td = t - simd_dot(t, e1) * e1; guard simd_length(td) > 1e-8 else { break }
            e2 = simd_normalize(td)
        }

        // Tercer eigenvector = normal del plano (menor varianza)
        var e3 = simd_normalize(simd_cross(e1, e2))
        if simd_dot(e3, camPos - centroid) < 0 { e3 = -e3 }

        // faceU = eje más vertical, faceR = eje más horizontal
        var faceU = abs(e1.y) >= abs(e2.y) ? e1 : e2
        var faceR = abs(e1.y) >= abs(e2.y) ? e2 : e1
        if faceU.y < 0 { faceU = -faceU }
        if simd_dot(simd_cross(faceR, faceU), e3) < 0 { faceR = -faceR }
        return (faceR, faceU, e3)
    }

    // MARK: - LiDAR helpers (fallback)
    private func sampleDepth(at screenPt: CGPoint, frame: ARFrame,
                              depth: ARDepthData, vp: CGSize) -> Float? {
        let inv = frame.displayTransform(for: .portrait, viewportSize: vp).inverted()
        let nc  = CGPoint(x: screenPt.x/vp.width, y: screenPt.y/vp.height).applying(inv)
        let dm  = depth.depthMap
        let dW  = CVPixelBufferGetWidth(dm), dH = CVPixelBufferGetHeight(dm)
        let sx  = max(0, min(dW-1, Int(nc.x * CGFloat(dW))))
        let sy  = max(0, min(dH-1, Int(nc.y * CGFloat(dH))))
        CVPixelBufferLockBaseAddress(dm, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(dm, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(dm) else { return nil }
        let v = base.assumingMemoryBound(to: Float32.self)[sy * dW + sx]
        return v > 0.02 && v < 8 ? v : nil
    }

    private func worldPointAtDepth(_ screenPt: CGPoint, depth depthVal: Float,
                                    frame: ARFrame, vp: CGSize) -> SIMD3<Float>? {
        let inv  = frame.displayTransform(for: .portrait, viewportSize: vp).inverted()
        let nc   = CGPoint(x: screenPt.x/vp.width, y: screenPt.y/vp.height).applying(inv)
        let intr = frame.camera.intrinsics
        let iW   = Float(frame.camera.imageResolution.width)
        let iH   = Float(frame.camera.imageResolution.height)
        let imgX = Float(nc.x) * iW, imgY = Float(nc.y) * iH
        let xCam = (imgX - intr[2][0]) / intr[0][0] * depthVal
        let yCam = (imgY - intr[2][1]) / intr[1][1] * depthVal
        let pt   = frame.camera.transform * SIMD4<Float>(xCam, yCam, -depthVal, 1)
        return SIMD3<Float>(pt.x, pt.y, pt.z)
    }

    // MARK: - Stability buffer
    private func addToBuffer(_ m: NativeMeasurement) {
        buffer.append(m)
        if buffer.count > stabilityWindow { buffer.removeFirst() }
        guard buffer.count == stabilityWindow else { return }
        func range(_ kp: KeyPath<NativeMeasurement, Double>) -> Double {
            let v = buffer.map { $0[keyPath: kp] }
            return (v.max() ?? 0) - (v.min() ?? 0)
        }
        guard range(\.comprimento) < thresholdCm,
              range(\.altura)      < thresholdCm,
              range(\.largura)     < thresholdCm * 2 else { return }
        let stable = NativeMeasurement(
            comprimento: median(buffer.map(\.comprimento)),
            largura:     median(buffer.map(\.largura)),
            altura:      median(buffer.map(\.altura)))
        lastMeasurement = stable
        isLocked = true
        DispatchQueue.main.async { [weak self] in self?.onUpdate(stable) }
    }

    private func median(_ arr: [Double]) -> Double {
        let s = arr.sorted(); let m = s.count / 2
        return s.count % 2 == 0 ? (s[m-1]+s[m])/2 : s[m]
    }

    // MARK: - 3D Wireframe
    // Recibe los 8 corners ya calculados del AABB → dibuja directamente sin extrusión.
    func updateOverlay(bl: SIMD3<Float>, br: SIMD3<Float>,
                       tl: SIMD3<Float>, tr: SIMD3<Float>,
                       bbl: SIMD3<Float>, bbr: SIMD3<Float>,
                       btl: SIMD3<Float>, btr: SIMD3<Float>,
                       measurement: NativeMeasurement) {
        guard let sv = sceneView else { return }
        clearOverlay()
        let green = UIColor(red: 0.0, green: 0.90, blue: 0.3, alpha: 1)
        let right = simd_normalize(br - bl)
        let up    = simd_normalize(tl - bl)

        let edges: [(SIMD3<Float>, SIMD3<Float>)] = [
            (bl,br),(br,tr),(tr,tl),(tl,bl),
            (bbl,bbr),(bbr,btr),(btr,btl),(btl,bbl),
            (bl,bbl),(br,bbr),(tl,btl),(tr,btr)
        ]
        for (s, e) in edges {
            let n = makeLine(from: s, to: e, color: green)
            sv.scene.rootNode.addChildNode(n); overlayNodes.append(n)
        }
        let cStr = "\(Int(measurement.comprimento.rounded())) cm"
        let aStr = "\(Int(measurement.altura.rounded())) cm"
        let lStr = "\(Int(measurement.largura.rounded())) cm"
        let labels: [(String, SIMD3<Float>)] = [
            (cStr, (bl+br)/2 + up * (-0.06)),
            (aStr, (bl+tl)/2 + right * (-0.07)),
            (lStr, (br+bbr)/2 + right * 0.07)
        ]
        for (txt, pos) in labels {
            let n = makeTextNode(txt, color: green)
            n.position = SCNVector3(pos.x, pos.y, pos.z)
            sv.scene.rootNode.addChildNode(n); overlayNodes.append(n)
        }
    }

    func clearOverlay() {
        overlayNodes.forEach { $0.removeFromParentNode() }
        overlayNodes.removeAll()
        clearDetectionLayer()
        CATransaction.begin(); CATransaction.setDisableActions(true)
        wireframeLayer.path = nil
        CATransaction.commit()
    }

    func reset() {
        clearOverlay(); buffer.removeAll(); manualPoints.removeAll()
        lastMeasurement = nil; isLocked = false; lockedTransform = nil
        smoothBL = nil; smoothBR = nil; smoothTL = nil; smoothTR = nil
        smoothBBL = nil; smoothBBR = nil; smoothBTL = nil; smoothBTR = nil
        cachedBox = nil; lastGeminiCall = 0; accPoints.removeAll()
    }

    private func makeLine(from s: SIMD3<Float>, to e: SIMD3<Float>, color: UIColor) -> SCNNode {
        let v = e - s; let len = simd_length(v)
        let cyl = SCNCylinder(radius: 0.002, height: CGFloat(len))
        cyl.firstMaterial?.diffuse.contents = color
        cyl.firstMaterial?.lightingModel    = .constant
        let node = SCNNode(geometry: cyl)
        let mid = (s+e)/2; node.position = SCNVector3(mid.x, mid.y, mid.z)
        let dir = simd_normalize(v); let up = SIMD3<Float>(0,1,0)
        let dot = simd_dot(up, dir)
        if abs(dot) > 0.9999 { if dot < 0 { node.rotation = SCNVector4(1,0,0,Float.pi) } }
        else {
            let ax = simd_normalize(simd_cross(up, dir))
            node.rotation = SCNVector4(ax.x, ax.y, ax.z, acos(max(-1, min(1, dot))))
        }
        return node
    }

    private func makeTextNode(_ text: String, color: UIColor) -> SCNNode {
        let geo = SCNText(string: text, extrusionDepth: 0)
        geo.font = UIFont.boldSystemFont(ofSize: 48)
        geo.flatness = 0.1
        geo.firstMaterial?.diffuse.contents = color
        geo.firstMaterial?.lightingModel    = .constant
        geo.firstMaterial?.isDoubleSided    = true
        let node = SCNNode(geometry: geo)
        let s: Float = 0.001; node.scale = SCNVector3(s,s,s)
        let (mn, mx) = node.boundingBox
        node.pivot = SCNMatrix4MakeTranslation((mx.x-mn.x)/2+mn.x, (mx.y-mn.y)/2+mn.y, 0)
        let c = SCNBillboardConstraint(); c.freeAxes = .all; node.constraints = [c]
        return node
    }

    // MARK: - Manual mode
    func addManualPoint(_ p: SIMD3<Float>) {
        manualPoints.append(p)
        guard manualPoints.count == 3 else { return }
        let p0 = manualPoints[0], p1 = manualPoints[1], p2 = manualPoints[2]
        let c = Double(simd_distance(p0, p1)) * 100
        let a = Double(simd_distance(p1, p2)) * 100
        let l = min(c, a) * 0.5
        manualPoints.removeAll()
        overlayNodes.forEach { $0.removeFromParentNode() }
        overlayNodes.removeAll()
        let m = NativeMeasurement(comprimento: c, largura: l, altura: a)
        lastMeasurement = m
        DispatchQueue.main.async { [weak self] in self?.onUpdate(m) }
    }

    func clearManualPoints() { manualPoints.removeAll() }
}

// MARK: - YOLOPrediction screen coordinates
private extension YOLOPrediction {
    func screenRect(viewportSize vp: CGSize, modelSize: CGFloat) -> CGRect {
        let scaleX = vp.width / modelSize, scaleY = vp.height / modelSize
        return CGRect(x: CGFloat(x1)*scaleX, y: CGFloat(y1)*scaleY,
                      width: CGFloat(x2-x1)*scaleX, height: CGFloat(y2-y1)*scaleY)
    }
}

// MARK: - ARKit delegates
extension BoxDetectionCoordinator: ARSCNViewDelegate {
    func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
        if let plane = anchor as? ARPlaneAnchor {
            DispatchQueue.main.async { [weak self] in self?.onPlaneFound() }
            if plane.alignment == .horizontal {
                let worldY = anchor.transform.columns.3.y
                DispatchQueue.main.async { [weak self] in
                    guard let s = self else { return }
                    if s.floorWorldY == nil || worldY < s.floorWorldY! { s.floorWorldY = worldY }
                }
            }
        }
    }
    func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
        if let plane = anchor as? ARPlaneAnchor, plane.alignment == .horizontal {
            let worldY = anchor.transform.columns.3.y
            DispatchQueue.main.async { [weak self] in
                guard let s = self else { return }
                if s.floorWorldY == nil || worldY < s.floorWorldY! { s.floorWorldY = worldY }
            }
        }
    }
}

extension BoxDetectionCoordinator: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard mode == .auto,
              let depthData = frame.sceneDepth
        else { return }
        measureFromCenter(frame: frame, depth: depthData)
    }
    func session(_ session: ARSession, didFailWithError error: Error) {
        print("[ARKit] \(error.localizedDescription)")
    }
    func sessionWasInterrupted(_ session: ARSession)    {}
    func sessionInterruptionEnded(_ session: ARSession) {}
}
