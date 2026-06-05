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

    // MARK: 2D overlay (CALayer)
    private let detectionLayer = CAShapeLayer()
    private let labelLayer     = CATextLayer()
    private let debugLayer     = CATextLayer()

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
    private let smoothAlpha: Float = 0.12

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
    }

    private func attachLayersIfNeeded() {
        guard let sv = sceneView, detectionLayer.superlayer == nil else { return }
        detectionLayer.frame = sv.bounds
        labelLayer.frame     = CGRect(x: 0, y: 0, width: 120, height: 22)
        debugLayer.frame     = CGRect(x: 8, y: 60, width: sv.bounds.width - 16, height: 56)
        sv.layer.addSublayer(detectionLayer)
        sv.layer.addSublayer(labelLayer)
        sv.layer.addSublayer(debugLayer)
        debugLayer.string = "\(modelStatusText)\n\(sam.status)"
    }

    private func updateDebug(_ text: String) {
        DispatchQueue.main.async {
            CATransaction.begin(); CATransaction.setDisableActions(true)
            self.debugLayer.string = text
            CATransaction.commit()
        }
    }

    // MARK: - Main pipeline
    private func measureFromCenter(frame: ARFrame, depth: ARDepthData) {
        guard let sv = sceneView, !scanInFlight else { return }
        attachLayersIfNeeded()
        updateDebug("\(modelStatusText)\n\(sam.status)")
        let vp = sv.bounds.size
        let cx = vp.width / 2, cy = vp.height / 2
        scanInFlight = true
        let pb = frame.capturedImage

        DispatchQueue.global(qos: .userInteractive).async { [weak self] in
            guard let self = self else { return }
            defer { DispatchQueue.main.async { self.scanInFlight = false } }

            // ── Fase 1: YOLO → bbox 2D ──────────────────────────────────────
            let pred = self.detectClosestBox(in: pb, viewportSize: vp, cx: cx, cy: cy)

            let screenBox: CGRect
            if let p = pred {
                let box = p.screenRect(viewportSize: vp, modelSize: self.modelInputSize)
                DispatchQueue.main.async {
                    self.lastDetection = box
                    self.missedFrames  = 0
                    self.drawDetectionRect(box, label: "caja \(Int(p.score * 100))%", in: vp)
                }
                screenBox = box
            } else {
                DispatchQueue.main.async { self.missedFrames += 1 }
                guard let last = self.lastDetection,
                      self.missedFrames <= self.maxMissedFrames else {
                    DispatchQueue.main.async { self.clearDetectionLayer() }
                    return
                }
                screenBox = last
            }

            // ── Fase 2: SAM segmentación (background) ───────────────────────
            let samMask = self.sam.getMask(pixelBuffer: pb,
                                           screenBox: screenBox,
                                           viewportSize: vp)
            self.updateDebug("\(self.modelStatusText)\n\(self.sam.status)")

            // ── Fase 3: medición 3D (main thread por acceso a ARFrame) ───────
            DispatchQueue.main.async {
                self.measure3D(box: screenBox, samMask: samMask,
                               frame: frame, depth: depth, vp: vp)
            }
        }
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
        detectionLayer.path = nil; labelLayer.string = nil
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
        guard frontPts3D.count >= 20 else { return }

        // 3. Medir cara frontal en 3D: proyectar puntos sobre el plano de la cara
        let centroid3D = frontPts3D.reduce(.zero, +) / Float(frontPts3D.count)
        let faceN = simd_normalize(camPos3 - centroid3D)
        let faceRRaw = simd_cross(SIMD3<Float>(0, 1, 0), faceN)
        let faceR = simd_length(faceRRaw) > 0.001 ? simd_normalize(faceRRaw)
                                                   : SIMD3<Float>(1, 0, 0)
        let faceU = simd_cross(faceN, faceR)  // ≈ world-up proyectado al plano de cara

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
        let a = Double(maxU - minU) * 100
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
    private func collectFrontFacePoints(centerD: Float, yoloBox: CGRect, samMask: [Bool]?,
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
                guard d > 0.02, d < 8.0, abs(d - centerD) < 0.10 else { continue }

                // Filtrar con máscara SAM (necesita posición en pantalla)
                if let mask = samMask {
                    let nc = CGPoint(x: CGFloat(dx) / CGFloat(dW), y: CGFloat(dy) / CGFloat(dH))
                    let ns = nc.applying(displayTx)
                    let sx = ns.x * vp.width, sy = ns.y * vp.height
                    let mx = max(0, min(maskW - 1, Int(sx / vp.width  * CGFloat(maskW))))
                    let my = max(0, min(maskH - 1, Int(sy / vp.height * CGFloat(maskH))))
                    guard mask[my * maskW + mx] else { continue }
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
    }

    func reset() {
        clearOverlay(); buffer.removeAll(); manualPoints.removeAll()
        lastMeasurement = nil; isLocked = false; lockedTransform = nil
        smoothBL = nil; smoothBR = nil; smoothTL = nil; smoothTR = nil
        smoothBBL = nil; smoothBBR = nil; smoothBTL = nil; smoothBTR = nil
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
        if anchor is ARPlaneAnchor {
            DispatchQueue.main.async { [weak self] in self?.onPlaneFound() }
        }
    }
}

extension BoxDetectionCoordinator: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard mode == .auto,
              frame.timestamp - lastScanTime > scanInterval,
              let depthData = frame.sceneDepth
        else { return }
        lastScanTime = frame.timestamp
        measureFromCenter(frame: frame, depth: depthData)
    }
    func session(_ session: ARSession, didFailWithError error: Error) {
        print("[ARKit] \(error.localizedDescription)")
    }
    func sessionWasInterrupted(_ session: ARSession)    {}
    func sessionInterruptionEnded(_ session: ARSession) {}
}
