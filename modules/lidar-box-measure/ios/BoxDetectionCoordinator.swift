import ARKit
import SceneKit
import CoreML
import Vision
import Accelerate

struct NativeMeasurement {
    var comprimento: Double
    var largura: Double
    var altura: Double
}

// Bounding box en coordenadas de la imagen de entrada del modelo (píxeles, 0-640)
private struct YOLOPrediction {
    let classIndex: Int
    let score: Float
    let x1: Float, y1: Float, x2: Float, y2: Float   // en píxeles del modelo (0-640)
    let maskCoefficients: [Float]
}

final class BoxDetectionCoordinator: NSObject {

    // MARK: - References
    weak var sceneView: ARSCNView?
    var mode: ARMode = .auto
    private(set) var lastMeasurement: NativeMeasurement?

    // MARK: - Callbacks
    var onUpdate: (NativeMeasurement) -> Void
    var onPlaneFound: () -> Void

    // MARK: - Timing
    private var lastScanTime: TimeInterval = 0
    private let scanInterval: TimeInterval = 0.20
    private var scanInFlight = false

    // MARK: - Persistencia de detección
    private var lastDetection: CGRect?
    private var missedFrames   = 0
    private let maxMissedFrames = 8

    // MARK: - YOLO model
    private var yoloModel: VNCoreMLModel?
    private let modelInputSize: CGFloat = 640
    private var modelStatusText = "YOLO: loading..."

    // MARK: - 2D detection overlay
    private let detectionLayer = CAShapeLayer()
    private let labelLayer     = CATextLayer()
    private let debugLayer     = CATextLayer()

    // MARK: - 3D overlay (SceneKit)
    private var overlayNodes: [SCNNode] = []

    // MARK: - Stability / EMA
    private var buffer: [NativeMeasurement] = []
    private let stabilityWindow = 8
    private let thresholdCm     = 2.0
    private var smoothBL: SIMD3<Float>?
    private var smoothBR: SIMD3<Float>?
    private var smoothTL: SIMD3<Float>?
    private var smoothTR: SIMD3<Float>?
    private let smoothAlpha: Float = 0.10   // suave: 10% por frame

    // MARK: - Lock mode (congela wireframe cuando es estable)
    private var isLocked = false
    private var lockedTransform: simd_float4x4?
    private let lockMovementThreshold: Float = 0.08  // 8cm mueve la cámara → desbloquea

    // MARK: - Manual mode
    private var manualPoints: [SIMD3<Float>] = []

    // MARK: - Init
    init(sceneView: ARSCNView,
         onUpdate: @escaping (NativeMeasurement) -> Void,
         onPlaneFound: @escaping () -> Void) {
        self.sceneView = sceneView
        self.onUpdate = onUpdate
        self.onPlaneFound = onPlaneFound
        super.init()
        loadModel()
        setupLayers()
    }

    // MARK: - Cargar coco128-yolo11n-seg
    private func loadModel() {
        let names = ["box_detector",
                     "coco128-yolo11n-seg", "coco128_yolo11n_seg",
                     "coco128-yolov8n-seg", "coco128_yolov8n_seg",
                     "model_core"]
        let exts  = ["mlmodelc", "mlpackage", "mlmodel"]

        // Buscar en todos los bundles candidatos
        var searchBundles: [Bundle] = [Bundle.main, Bundle(for: BoxDetectionCoordinator.self)]
        let subBundleNames = ["LidarBoxMeasure", "LidarBoxMeasureResources", "lidar-box-measure"]
        for bName in subBundleNames {
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
                        // .mlpackage necesita compilarse; usar cache para no recompilar cada vez
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
                        if #available(iOS 16.0, *) {
                            cfg.computeUnits = .cpuAndNeuralEngine
                        } else {
                            cfg.computeUnits = .all
                        }
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

        modelStatusText = lastError.isEmpty ? "NOT LOADED - no file found" : "LOAD ERR: \(lastError)"
    }

    // MARK: - Setup layers
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

        debugLayer.fontSize        = 13
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
        debugLayer.frame     = CGRect(x: 8, y: 60, width: sv.bounds.width - 16, height: 100)
        sv.layer.addSublayer(detectionLayer)
        sv.layer.addSublayer(labelLayer)
        sv.layer.addSublayer(debugLayer)
        debugLayer.string = modelStatusText
    }

    private func updateDebug(_ text: String) {
        DispatchQueue.main.async {
            CATransaction.begin()
            CATransaction.setDisableActions(true)
            self.debugLayer.string = text
            CATransaction.commit()
        }
    }

    // MARK: - Pipeline principal
    private func measureFromCenter(frame: ARFrame, depth: ARDepthData) {
        guard let sv = sceneView, !scanInFlight else { return }
        attachLayersIfNeeded()

        let vp = sv.bounds.size
        let cx = vp.width / 2, cy = vp.height / 2
        scanInFlight = true

        let pb     = frame.capturedImage
        let cfr    = frame
        let cdepth = depth

        DispatchQueue.global(qos: .userInteractive).async { [weak self] in
            guard let self = self else { return }
            defer { DispatchQueue.main.async { self.scanInFlight = false } }

            // ── FASE 1: YOLO → bbox 2D ─────────────────────────────────────
            let pred = self.detectClosestBox(in: pb, viewportSize: vp, cx: cx, cy: cy)

            let screenBox: CGRect
            if let p = pred {
                screenBox = p.screenRect(viewportSize: vp, modelSize: self.modelInputSize)
                DispatchQueue.main.async {
                    self.lastDetection = screenBox
                    self.missedFrames  = 0
                    self.drawDetectionRect(screenBox, label: "box \(Int(p.score * 100))%", in: vp)
                }
            } else {
                // Sin detección: usar última bbox conocida hasta maxMissedFrames
                DispatchQueue.main.async { self.missedFrames += 1 }
                guard let last = self.lastDetection, self.missedFrames <= self.maxMissedFrames
                else {
                    DispatchQueue.main.async { self.clearDetectionLayer() }
                    return
                }
                screenBox = last
            }

            // ── FASE 2: LiDAR → medición 3D ────────────────────────────────
            // Samplear profundidad en el centro del bbox YOLO, no en el centro de pantalla
            let bboxCenter = CGPoint(x: screenBox.midX, y: screenBox.midY)
            guard let centerD = self.sampleDepth(
                at: bboxCenter, frame: cfr, depth: cdepth
            ), centerD > 0.15, centerD < 4.0 else { return }

            DispatchQueue.main.async {
                self.measure3D(box: screenBox, cx: cx, cy: cy, centerD: centerD,
                               frame: cfr, depth: cdepth, vp: vp)
            }
        }
    }

    // MARK: - YOLO inference (lógica de parsing tomada de MaciDE/YOLOv8-seg-iOS)
    private func detectClosestBox(in pixelBuffer: CVPixelBuffer,
                                  viewportSize vp: CGSize,
                                  cx: CGFloat, cy: CGFloat) -> YOLOPrediction? {
        guard let model = yoloModel else {
            updateDebug("\(modelStatusText)\nNO MODEL")
            return nil
        }

        var boxesOutput: MLMultiArray?
        var allObservationTypes = [String]()
        let confidenceThreshold: Float = 0.08
        let iouThreshold:        Float = 0.60

        let request = VNCoreMLRequest(model: model) { req, err in
            if let err = err { allObservationTypes.append("ERR:\(err.localizedDescription)") }
            let obs = req.results ?? []
            allObservationTypes.append(contentsOf: obs.map { String(describing: type(of: $0)) })
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
            let types = allObservationTypes.prefix(3).joined(separator: ", ")
            updateDebug("\(modelStatusText)\noutput=nil obs=[\(types)]")
            return nil
        }

        // Parse [1, C, N] con strides (exactamente como MaciDE)
        let shape   = boxes.shape.map { $0.intValue }
        guard shape.count == 3 else { return nil }
        let C = shape[1]   // canales: 4 + numClasses (yolov8n) o 4 + numClasses + 32 (seg)
        let N = shape[2]   // anchors: 8400
        // Detectar automáticamente si es modelo seg (C >= 4+1+32=37) o detection puro (C=4+numCls)
        // yolov8n-seg COCO: C=116 (4+80+32); yolov8n cardboard: C=5 (4+1)
        let numSegMasks = C > 36 ? 32 : 0
        let numClasses  = C - 4 - numSegMasks
        guard numClasses > 0 else { return nil }

        let strides = boxes.strides.map { $0.intValue }
        let ptr     = boxes.dataPointer.assumingMemoryBound(to: Float.self)

        @inline(__always)
        func idx(_ channel: Int, _ anchor: Int) -> Int {
            channel * strides[1] + anchor * strides[2]
        }

        // Recolectar predicciones con score > threshold
        var predictions = [YOLOPrediction]()
        for i in 0..<N {
            let cx640  = ptr[idx(0, i)]
            let cy640  = ptr[idx(1, i)]
            let w640   = ptr[idx(2, i)]
            let h640   = ptr[idx(3, i)]

            var classScores = [Float](repeating: 0, count: numClasses)
            for j in 0..<numClasses { classScores[j] = ptr[idx(4 + j, i)] }

            var best: Float = 0
            var bestCls: vDSP_Length = 0
            vDSP_maxvi(classScores, 1, &best, &bestCls, vDSP_Length(numClasses))
            guard best >= confidenceThreshold else { continue }

            var coefs = [Float](repeating: 0, count: numSegMasks)
            for k in 0..<numSegMasks { coefs[k] = ptr[idx(4 + numClasses + k, i)] }

            // xywh → xyxy
            let x1 = cx640 - w640 * 0.5
            let y1 = cy640 - h640 * 0.5
            let x2 = cx640 + w640 * 0.5
            let y2 = cy640 + h640 * 0.5

            predictions.append(YOLOPrediction(classIndex: Int(bestCls), score: best,
                                               x1: x1, y1: y1, x2: x2, y2: y2,
                                               maskCoefficients: coefs))
        }
        guard !predictions.isEmpty else {
            updateDebug("\(modelStatusText)\nC=\(C) N=\(N) cls=\(numClasses)\nno preds >thr=\(confidenceThreshold)")
            return nil
        }

        // NMS por clase
        let grouped = Dictionary(grouping: predictions) { $0.classIndex }
        var nms = [YOLOPrediction]()
        for (_, group) in grouped {
            nms.append(contentsOf: nonMaxSuppression(group, iou: iouThreshold))
        }
        guard !nms.isEmpty else {
            updateDebug("\(modelStatusText)\n\(predictions.count) preds, all NMS-filtered")
            return nil
        }

        let best = nms.min(by: { a, b in
            let ra = a.screenRect(viewportSize: vp, modelSize: modelInputSize)
            let rb = b.screenRect(viewportSize: vp, modelSize: modelInputSize)
            return hypot(ra.midX - cx, ra.midY - cy) < hypot(rb.midX - cx, rb.midY - cy)
        })!
        let topScore = String(format: "%.0f%%", best.score * 100)
        updateDebug("\(modelStatusText)\n\(nms.count) det | cls=\(best.classIndex) \(topScore)")
        return best
    }

    // MARK: - NMS
    private func nonMaxSuppression(_ preds: [YOLOPrediction], iou: Float) -> [YOLOPrediction] {
        let sorted = preds.sorted { $0.score > $1.score }
        var kept = [YOLOPrediction]()
        var active = [Bool](repeating: true, count: sorted.count)
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
        let inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
        let areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
        let areaB = (b.x2 - b.x1) * (b.y2 - b.y1)
        return inter / (areaA + areaB - inter + 1e-6)
    }

    // MARK: - Dibujar 2D overlay
    private func drawDetectionRect(_ rect: CGRect, label: String, in vp: CGSize) {
        detectionLayer.frame = CGRect(origin: .zero, size: vp)
        let path = UIBezierPath(roundedRect: rect, cornerRadius: 6)
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        detectionLayer.path = path.cgPath
        labelLayer.string   = label
        labelLayer.position = CGPoint(x: rect.midX, y: rect.minY - 14)
        CATransaction.commit()
    }

    private func clearDetectionLayer() {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        detectionLayer.path = nil
        labelLayer.string   = nil
        CATransaction.commit()
    }

    // MARK: - FASE 2: medición 3D con LiDAR
    private func measure3D(box: CGRect, cx: CGFloat, cy: CGFloat,
                           centerD: Float, frame: ARFrame,
                           depth: ARDepthData, vp: CGSize) {
        // Si está bloqueado, solo desbloquear si la cámara se movió >8cm
        if isLocked {
            if let lt = lockedTransform {
                let cur = frame.camera.transform.columns.3
                let prv = lt.columns.3
                let moved = simd_length(SIMD3<Float>(cur.x - prv.x, cur.y - prv.y, cur.z - prv.z))
                if moved < lockMovementThreshold { return }
            }
            // Cámara se movió → desbloquear y reiniciar
            isLocked = false
            lockedTransform = nil
            buffer.removeAll()
            smoothBL = nil; smoothBR = nil; smoothTL = nil; smoothTR = nil
        }
        lockedTransform = frame.camera.transform

        guard box.width > 30, box.height > 30 else { return }

        let tol:  Float   = 0.06   // saltar borde cuando profundidad cambia >6cm
        let step: CGFloat = 3      // paso de scan en píxeles de pantalla

        // Escanear en 5 líneas horizontales y 5 verticales, tomar la mediana
        var leftXs = [CGFloat](), rightXs = [CGFloat]()
        var topYs  = [CGFloat](), botYs   = [CGFloat]()

        for frac in [0.3, 0.4, 0.5, 0.6, 0.7] as [CGFloat] {
            let sy = box.minY + box.height * frac
            let sx = box.minX + box.width  * frac

            // ← izquierda desde centro
            var lx = box.midX
            while lx - step >= box.minX {
                lx -= step
                guard let d = sampleDepth(at: CGPoint(x: lx, y: sy),
                                          frame: frame, depth: depth)
                else { break }
                if abs(d - centerD) > tol { lx += step; break }
            }
            // → derecha desde centro
            var rx = box.midX
            while rx + step <= box.maxX {
                rx += step
                guard let d = sampleDepth(at: CGPoint(x: rx, y: sy),
                                          frame: frame, depth: depth)
                else { break }
                if abs(d - centerD) > tol { rx -= step; break }
            }
            // ↑ arriba desde centro
            var ty = box.midY
            while ty - step >= box.minY {
                ty -= step
                guard let d = sampleDepth(at: CGPoint(x: sx, y: ty),
                                          frame: frame, depth: depth)
                else { break }
                if abs(d - centerD) > tol { ty += step; break }
            }
            // ↓ abajo desde centro
            var by = box.midY
            while by + step <= box.maxY {
                by += step
                guard let d = sampleDepth(at: CGPoint(x: sx, y: by),
                                          frame: frame, depth: depth)
                else { break }
                if abs(d - centerD) > tol { by -= step; break }
            }
            leftXs.append(lx); rightXs.append(rx)
            topYs.append(ty);  botYs.append(by)
        }

        guard !leftXs.isEmpty else { return }
        let leftX  = medianCG(leftXs)
        let rightX = medianCG(rightXs)
        let topY   = medianCG(topYs)
        let botY   = medianCG(botYs)

        guard rightX - leftX > 20, botY - topY > 20 else { return }

        let tl = CGPoint(x: leftX,  y: topY)
        let tr = CGPoint(x: rightX, y: topY)
        let bl = CGPoint(x: leftX,  y: botY)
        let br = CGPoint(x: rightX, y: botY)

        // Todos los vértices usan centerD: la cara frontal es perpendicular a la cámara
        guard
            let p3BL = worldPointAtDepth(bl, depth: centerD, frame: frame),
            let p3BR = worldPointAtDepth(br, depth: centerD, frame: frame),
            let p3TL = worldPointAtDepth(tl, depth: centerD, frame: frame),
            let p3TR = worldPointAtDepth(tr, depth: centerD, frame: frame)
        else { return }

        let fRight     = simd_normalize(p3BR - p3BL)
        let fUp        = simd_normalize(p3TL - p3BL)
        let faceNormal = simd_normalize(simd_cross(fRight, fUp))
        guard abs(faceNormal.y) < 0.70 else { return }

        let c = Double(simd_distance(p3BL, p3BR)) * 100
        let a = Double(simd_distance(p3BL, p3TL)) * 100
        let l = estimateDepth(frame: frame, depth: depth,
                              topLeft: tl, topRight: tr,
                              centerD: centerD) ?? min(c, a) * 0.6
        guard c > 5, c < 300, a > 5, a < 300, l > 2 else { return }

        let α = smoothAlpha
        smoothBL = smoothBL.map { α * p3BL + (1-α) * $0 } ?? p3BL
        smoothBR = smoothBR.map { α * p3BR + (1-α) * $0 } ?? p3BR
        smoothTL = smoothTL.map { α * p3TL + (1-α) * $0 } ?? p3TL
        smoothTR = smoothTR.map { α * p3TR + (1-α) * $0 } ?? p3TR

        let m = NativeMeasurement(comprimento: c, largura: l, altura: a)
        addToBuffer(m)
        updateOverlay(bl: smoothBL!, br: smoothBR!, tl: smoothTL!, tr: smoothTR!, measurement: m)
    }

    // MARK: - Estimación profundidad (largura)
    private func estimateDepth(frame: ARFrame, depth: ARDepthData,
                               topLeft: CGPoint, topRight: CGPoint,
                               centerD: Float) -> Double? {
        var estimates = [Double]()
        for frac in [0.25, 0.50, 0.75] as [CGFloat] {
            let sx = topLeft.x + frac * (topRight.x - topLeft.x)
            var sy = topLeft.y
            guard let startD = sampleDepth(at: CGPoint(x: sx, y: sy), frame: frame, depth: depth),
                  let topPt  = worldPointAtDepth(CGPoint(x: sx, y: sy), depth: startD, frame: frame)
            else { continue }
            let topFaceY = topPt.y
            var prevPt   = topPt
            while sy > 8 {
                sy -= 6
                guard let d   = sampleDepth(at: CGPoint(x: sx, y: sy), frame: frame, depth: depth),
                      let wPt = worldPointAtDepth(CGPoint(x: sx, y: sy), depth: d, frame: frame)
                else { break }
                if wPt.y < topFaceY - 0.07 { break }
                if d - startD > 0.22        { break }
                prevPt = wPt
            }
            let d = Double(simd_distance(topPt, prevPt)) * 100
            if d > 3, d < 200 { estimates.append(d) }
        }
        return estimates.isEmpty ? nil : median(estimates)
    }

    // MARK: - LiDAR helpers

    private func sampleDepth(at screenPt: CGPoint,
                             frame: ARFrame,
                             depth: ARDepthData) -> Float? {
        guard let sv = sceneView else { return nil }
        let vp = sv.bounds.size
        let inv = frame.displayTransform(for: .portrait, viewportSize: vp).inverted()
        let nc  = CGPoint(x: screenPt.x / vp.width,
                          y: screenPt.y / vp.height).applying(inv)
        let dm  = depth.depthMap
        let dW  = CVPixelBufferGetWidth(dm)
        let dH  = CVPixelBufferGetHeight(dm)
        let sx  = max(0, min(dW-1, Int(nc.x * CGFloat(dW))))
        let sy  = max(0, min(dH-1, Int(nc.y * CGFloat(dH))))
        CVPixelBufferLockBaseAddress(dm, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(dm, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(dm) else { return nil }
        let v = base.assumingMemoryBound(to: Float32.self)[sy * dW + sx]
        return v > 0.02 && v < 8 ? v : nil
    }

    private func worldPointAtDepth(_ screenPt: CGPoint,
                                   depth depthVal: Float,
                                   frame: ARFrame) -> SIMD3<Float>? {
        guard let sv = sceneView else { return nil }
        let vp  = sv.bounds.size
        let inv = frame.displayTransform(for: .portrait, viewportSize: vp).inverted()
        let nc  = CGPoint(x: screenPt.x / vp.width,
                          y: screenPt.y / vp.height).applying(inv)
        let intr = frame.camera.intrinsics
        let iW   = Float(frame.camera.imageResolution.width)
        let iH   = Float(frame.camera.imageResolution.height)
        let imgX = Float(nc.x) * iW
        let imgY = Float(nc.y) * iH
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
        isLocked = true   // congelar wireframe — medición estable
        DispatchQueue.main.async { [weak self] in self?.onUpdate(stable) }
    }

    private func median(_ arr: [Double]) -> Double {
        let s = arr.sorted(); let m = s.count / 2
        return s.count % 2 == 0 ? (s[m-1] + s[m]) / 2 : s[m]
    }

    private func medianCG(_ arr: [CGFloat]) -> CGFloat {
        let s = arr.sorted(); let m = s.count / 2
        return s.count % 2 == 0 ? (s[m-1] + s[m]) / 2 : s[m]
    }

    // MARK: - Manual points
    func addManualPoint(_ p: SIMD3<Float>) {
        manualPoints.append(p)
        let sphere = SCNSphere(radius: 0.01)
        sphere.firstMaterial?.diffuse.contents = UIColor.yellow
        sphere.firstMaterial?.lightingModel    = .constant
        let node = SCNNode(geometry: sphere)
        node.position = SCNVector3(p.x, p.y, p.z)
        sceneView?.scene.rootNode.addChildNode(node)
        overlayNodes.append(node)
        if manualPoints.count == 3 {
            let p0 = manualPoints[0], p1 = manualPoints[1], p2 = manualPoints[2]
            let m = NativeMeasurement(comprimento: Double(simd_distance(p0, p1)) * 100,
                                      largura:     Double(simd_distance(p0, p2)) * 100,
                                      altura:      Double(simd_distance(p0, p2)) * 100)
            lastMeasurement = m
            DispatchQueue.main.async { [weak self] in self?.onUpdate(m) }
        } else if manualPoints.count == 4 {
            let p0 = manualPoints[0], p1 = manualPoints[1]
            let p2 = manualPoints[2], p3 = manualPoints[3]
            let m = NativeMeasurement(comprimento: Double(simd_distance(p0, p1)) * 100,
                                      largura:     Double(simd_distance(p0, p2)) * 100,
                                      altura:      Double(simd_distance(p0, p3)) * 100)
            lastMeasurement = m
            manualPoints.removeAll()
            overlayNodes.forEach { $0.removeFromParentNode() }
            overlayNodes.removeAll()
            DispatchQueue.main.async { [weak self] in self?.onUpdate(m) }
        }
    }

    // MARK: - 3D Wireframe
    func updateOverlay(bl: SIMD3<Float>, br: SIMD3<Float>,
                       tl: SIMD3<Float>, tr: SIMD3<Float>,
                       measurement: NativeMeasurement) {
        guard let sv = sceneView else { return }
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.clearOverlay()
            let green = UIColor(red: 0.0, green: 0.90, blue: 0.3, alpha: 1)
            let right = simd_normalize(br - bl)
            let up    = simd_normalize(tl - bl)
            let fN    = simd_normalize(simd_cross(right, up))
            let dep   = Float(max(measurement.largura, 2) / 100.0)
            let ext   = -fN
            let bbl = bl + ext * dep, bbr = br + ext * dep
            let btl = tl + ext * dep, btr = tr + ext * dep
            let edges: [(SIMD3<Float>, SIMD3<Float>)] = [
                (bl,br),(br,tr),(tr,tl),(tl,bl),
                (bbl,bbr),(bbr,btr),(btr,btl),(btl,bbl),
                (bl,bbl),(br,bbr),(tl,btl),(tr,btr)
            ]
            for (s, e) in edges {
                let n = self.makeLine(from: s, to: e, color: green)
                sv.scene.rootNode.addChildNode(n)
                self.overlayNodes.append(n)
            }
            let cStr = "\(Int(measurement.comprimento.rounded())) cm"
            let aStr = "\(Int(measurement.altura.rounded())) cm"
            let lStr = "\(Int(measurement.largura.rounded())) cm"
            let labels: [(String, SIMD3<Float>)] = [
                (cStr, (bl+br)/2 + up * (-0.055)),
                (aStr, (bl+tl)/2 + right * (-0.065)),
                (lStr, (br+bbr)/2 + right * 0.065)
            ]
            for (txt, pos) in labels {
                let n = self.makeTextNode(txt, color: green)
                n.position = SCNVector3(pos.x, pos.y, pos.z)
                sv.scene.rootNode.addChildNode(n)
                self.overlayNodes.append(n)
            }
        }
    }

    private func makeTextNode(_ text: String, color: UIColor) -> SCNNode {
        let geo = SCNText(string: text, extrusionDepth: 0)
        geo.font = UIFont.boldSystemFont(ofSize: 48)
        geo.flatness = 0.1
        geo.firstMaterial?.diffuse.contents = color
        geo.firstMaterial?.lightingModel    = .constant
        geo.firstMaterial?.isDoubleSided    = true
        let node = SCNNode(geometry: geo)
        let s: Float = 0.001; node.scale = SCNVector3(s, s, s)
        let (mn, mx) = node.boundingBox
        node.pivot = SCNMatrix4MakeTranslation((mx.x-mn.x)/2+mn.x, (mx.y-mn.y)/2+mn.y, 0)
        let c = SCNBillboardConstraint(); c.freeAxes = .all; node.constraints = [c]
        return node
    }

    func clearOverlay() {
        overlayNodes.forEach { $0.removeFromParentNode() }
        overlayNodes.removeAll()
        clearDetectionLayer()
    }

    func reset() {
        clearOverlay()
        buffer.removeAll(); manualPoints.removeAll()
        lastMeasurement = nil
        smoothBL = nil; smoothBR = nil; smoothTL = nil; smoothTR = nil
    }

    private func makeLine(from s: SIMD3<Float>, to e: SIMD3<Float>, color: UIColor) -> SCNNode {
        let v = e - s; let len = simd_length(v)
        let cyl = SCNCylinder(radius: 0.002, height: CGFloat(len))
        cyl.firstMaterial?.diffuse.contents = color
        cyl.firstMaterial?.lightingModel    = .constant
        let node = SCNNode(geometry: cyl)
        let mid = (s + e) / 2; node.position = SCNVector3(mid.x, mid.y, mid.z)
        let dir = simd_normalize(v); let up = SIMD3<Float>(0, 1, 0)
        let dot = simd_dot(up, dir)
        if abs(dot) > 0.9999 { if dot < 0 { node.rotation = SCNVector4(1,0,0,Float.pi) } }
        else {
            let ax = simd_normalize(simd_cross(up, dir))
            node.rotation = SCNVector4(ax.x, ax.y, ax.z, acos(max(-1, min(1, dot))))
        }
        return node
    }
}

// MARK: - YOLOPrediction screen coordinates
private extension YOLOPrediction {
    // Convierte bbox del modelo (0-640, portrait con orientation:.right)
    // a coordenadas UIKit de pantalla con imageCropAndScaleOption:.scaleFill
    func screenRect(viewportSize vp: CGSize, modelSize: CGFloat) -> CGRect {
        let scaleX = vp.width  / modelSize
        let scaleY = vp.height / modelSize
        return CGRect(
            x:      CGFloat(x1) * scaleX,
            y:      CGFloat(y1) * scaleY,
            width:  CGFloat(x2 - x1) * scaleX,
            height: CGFloat(y2 - y1) * scaleY
        )
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
    func sessionWasInterrupted(_ session: ARSession)    { print("[ARKit] interrupted") }
    func sessionInterruptionEnded(_ session: ARSession) { print("[ARKit] resumido") }
}
