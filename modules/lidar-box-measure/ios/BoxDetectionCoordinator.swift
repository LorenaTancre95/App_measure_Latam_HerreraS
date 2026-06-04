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
        let ptr = cls.buffer.contents()
            .advanced(by: cls.offset + i * cls.stride)
        return ARMeshClassification(rawValue: Int(ptr.load(as: UInt8.self))) ?? .none
    }
}

private extension ARGeometrySource {
    /// Vértice 3D en espacio local del anchor
    func vertex(at index: UInt32) -> SIMD3<Float> {
        buffer.contents()
            .advanced(by: offset + Int(index) * stride)
            .load(as: SIMD3<Float>.self)
    }
}

private extension ARGeometryElement {
    /// Índice de vértice para la cara i, vértice v (0, 1, 2)
    func vertexIndex(at faceIndex: Int, vertex v: Int) -> UInt32 {
        let ptr = buffer.contents()
            .advanced(by: (faceIndex * 3 + v) * bytesPerIndex)
        return bytesPerIndex == 2
            ? UInt32(ptr.load(as: UInt16.self))
            : ptr.load(as: UInt32.self)
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
        debugLayer.frame     = CGRect(x: 8, y: 60, width: sv.bounds.width - 16, height: 44)
        sv.layer.addSublayer(detectionLayer)
        sv.layer.addSublayer(labelLayer)
        sv.layer.addSublayer(debugLayer)
        debugLayer.string = modelStatusText
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

            // ── Fase 2: ARKit raycast + ARMesh bounding box ─────────────────
            // Debe ejecutarse en main thread (ARKit requirement)
            DispatchQueue.main.async {
                self.measure3D(box: screenBox, frame: frame, depth: depth, vp: vp)
            }
        }
    }

    // MARK: - YOLO inference
    private func detectClosestBox(in pixelBuffer: CVPixelBuffer,
                                   viewportSize vp: CGSize,
                                   cx: CGFloat, cy: CGFloat) -> YOLOPrediction? {
        guard let model = yoloModel else {
            updateDebug("\(modelStatusText)\nNO MODEL")
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
            updateDebug("\(modelStatusText)\nnil output | [\(types)]")
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
            updateDebug("\(modelStatusText)\nC=\(C) N=\(N) cls=\(numClasses)\nno preds>thr")
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
        updateDebug("\(modelStatusText)\ncls=\(Int(best.score*100))%")
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

    // MARK: - ARMesh bounding box
    // Extrae los vértices de malla clasificados como objetos genéricos (.none)
    // dentro de `searchRadius` del hit point. Retorna el AABB en world space.
    private func findBoxBounds(near hitPoint: SIMD3<Float>,
                                frame: ARFrame) -> (min: SIMD3<Float>, max: SIMD3<Float>)? {
        let searchRadius: Float = 1.0
        var pts = [SIMD3<Float>]()
        pts.reserveCapacity(512)

        for anchor in frame.anchors.compactMap({ $0 as? ARMeshAnchor }) {
            // Descartar anchors muy lejos del hit point
            let anchorPos = SIMD3<Float>(anchor.transform.columns.3.x,
                                         anchor.transform.columns.3.y,
                                         anchor.transform.columns.3.z)
            guard simd_distance(anchorPos, hitPoint) < searchRadius * 2 else { continue }

            let geo = anchor.geometry
            let t   = anchor.transform

            for fi in 0..<geo.faces.count {
                // Solo objetos genéricos (cajas = .none); excluir piso, paredes, techo
                guard geo.classificationAt(faceIndex: fi) == .none else { continue }

                for vi in 0..<3 {
                    let vIdx   = geo.faces.vertexIndex(at: fi, vertex: vi)
                    let local  = geo.vertices.vertex(at: vIdx)
                    let world4 = t * SIMD4<Float>(local.x, local.y, local.z, 1)
                    let world  = SIMD3<Float>(world4.x, world4.y, world4.z)
                    if simd_distance(world, hitPoint) < searchRadius { pts.append(world) }
                }
            }
        }

        guard pts.count >= 30 else { return nil }

        let minX = pts.map(\.x).min()!, maxX = pts.map(\.x).max()!
        let minY = pts.map(\.y).min()!, maxY = pts.map(\.y).max()!
        let minZ = pts.map(\.z).min()!, maxZ = pts.map(\.z).max()!

        let w = maxX-minX, h = maxY-minY, d = maxZ-minZ
        guard w > 0.03, h > 0.03, d > 0.03, w < 3, h < 3, d < 3 else { return nil }

        return (SIMD3(minX, minY, minZ), SIMD3(maxX, maxY, maxZ))
    }

    // MARK: - Fase 2: ARMesh measure
    // Se llama desde main thread.
    private func measure3D(box: CGRect, frame: ARFrame,
                            depth: ARDepthData, vp: CGSize) {
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

        guard box.width > 20, box.height > 20 else { return }

        // 1. Raycast desde el centro del bbox YOLO
        let bboxCenter = CGPoint(x: box.midX, y: box.midY)
        guard let hitPoint = raycast(from: bboxCenter) else { return }

        // 2. Buscar bounding box de la malla ARKit en ese punto
        guard let bounds = findBoxBounds(near: hitPoint, frame: frame) else {
            // Fallback: depth map si la malla no tiene suficientes vértices
            fallbackDepthMeasure(box: box, hitPoint: hitPoint, frame: frame, depth: depth)
            return
        }

        // 3. Dimensiones desde AABB
        let dimX = bounds.max.x - bounds.min.x
        let dimY = bounds.max.y - bounds.min.y  // altura (gravity-aligned)
        let dimZ = bounds.max.z - bounds.min.z

        // 4. Determinar cara frontal (la más cercana a la cámara)
        let boxCenter = (bounds.min + bounds.max) / 2
        let camPos    = SIMD3<Float>(frame.camera.transform.columns.3.x,
                                     frame.camera.transform.columns.3.y,
                                     frame.camera.transform.columns.3.z)
        let toCam = simd_normalize(camPos - boxCenter)

        // La cara frontal es la que tiene normal más alineada con toCam
        // Para un AABB world-aligned: ±X o ±Z (Y es vertical/arriba)
        let frontInX = abs(toCam.x) >= abs(toCam.z)
        let tl, tr, bl, br: SIMD3<Float>
        let btl, btr, bbl, bbr: SIMD3<Float>

        if frontInX {
            let fx = toCam.x > 0 ? bounds.max.x : bounds.min.x
            let bx = toCam.x > 0 ? bounds.min.x : bounds.max.x
            tl  = SIMD3(fx, bounds.max.y, bounds.min.z)
            tr  = SIMD3(fx, bounds.max.y, bounds.max.z)
            bl  = SIMD3(fx, bounds.min.y, bounds.min.z)
            br  = SIMD3(fx, bounds.min.y, bounds.max.z)
            btl = SIMD3(bx, bounds.max.y, bounds.min.z)
            btr = SIMD3(bx, bounds.max.y, bounds.max.z)
            bbl = SIMD3(bx, bounds.min.y, bounds.min.z)
            bbr = SIMD3(bx, bounds.min.y, bounds.max.z)
        } else {
            let fz = toCam.z > 0 ? bounds.max.z : bounds.min.z
            let bz = toCam.z > 0 ? bounds.min.z : bounds.max.z
            tl  = SIMD3(bounds.min.x, bounds.max.y, fz)
            tr  = SIMD3(bounds.max.x, bounds.max.y, fz)
            bl  = SIMD3(bounds.min.x, bounds.min.y, fz)
            br  = SIMD3(bounds.max.x, bounds.min.y, fz)
            btl = SIMD3(bounds.min.x, bounds.max.y, bz)
            btr = SIMD3(bounds.max.x, bounds.max.y, bz)
            bbl = SIMD3(bounds.min.x, bounds.min.y, bz)
            bbr = SIMD3(bounds.max.x, bounds.min.y, bz)
        }

        // comprimento = ancho de la cara frontal visible
        // altura      = alto (Y)
        // largura     = profundidad de la caja
        let frontWidth = frontInX ? dimZ : dimX
        let c = Double(frontWidth) * 100
        let a = Double(dimY) * 100
        let l = frontInX ? Double(dimX)*100 : Double(dimZ)*100

        guard c > 3, c < 300, a > 3, a < 300, l > 3 else { return }

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

    // MARK: - Fallback: depth map (cuando la malla no tiene vértices suficientes)
    private func fallbackDepthMeasure(box: CGRect, hitPoint: SIMD3<Float>,
                                       frame: ARFrame, depth: ARDepthData) {
        guard let sv = sceneView else { return }
        let vp = sv.bounds.size

        let tl = CGPoint(x: box.minX, y: box.minY)
        let tr = CGPoint(x: box.maxX, y: box.minY)
        let bl = CGPoint(x: box.minX, y: box.maxY)
        let br = CGPoint(x: box.maxX, y: box.maxY)

        let camPos = frame.camera.transform.columns.3
        let centerD = simd_distance(hitPoint, SIMD3<Float>(camPos.x, camPos.y, camPos.z))

        func depthedPoint(_ pt: CGPoint) -> SIMD3<Float>? {
            let d = sampleDepth(at: pt, frame: frame, depth: depth, vp: vp) ?? centerD
            let clamped = abs(d - centerD) < 0.3 ? d : centerD
            return worldPointAtDepth(pt, depth: clamped, frame: frame, vp: vp)
        }

        guard
            let p3BL = depthedPoint(bl), let p3BR = depthedPoint(br),
            let p3TL = depthedPoint(tl), let p3TR = depthedPoint(tr)
        else { return }

        let fRight = simd_normalize(p3BR - p3BL)
        let fUp    = simd_normalize(p3TL - p3BL)
        let fN     = simd_normalize(simd_cross(fRight, fUp))
        guard abs(fN.y) < 0.70 else { return }

        let c = Double(simd_distance(p3BL, p3BR)) * 100
        let a = Double(simd_distance(p3BL, p3TL)) * 100
        let l = min(c, a) * 0.5
        guard c > 3, c < 300, a > 3, a < 300 else { return }

        let dep   = Float(max(l, 3) / 100.0)
        let ext   = -fN
        let bbl   = p3BL + ext*dep, bbr = p3BR + ext*dep
        let fbtl  = p3TL + ext*dep, fbtr = p3TR + ext*dep

        let α = smoothAlpha
        smoothBL  = smoothBL.map  { α*p3BL + (1-α)*$0 } ?? p3BL
        smoothBR  = smoothBR.map  { α*p3BR + (1-α)*$0 } ?? p3BR
        smoothTL  = smoothTL.map  { α*p3TL + (1-α)*$0 } ?? p3TL
        smoothTR  = smoothTR.map  { α*p3TR + (1-α)*$0 } ?? p3TR
        smoothBBL = smoothBBL.map { α*bbl  + (1-α)*$0 } ?? bbl
        smoothBBR = smoothBBR.map { α*bbr  + (1-α)*$0 } ?? bbr
        smoothBTL = smoothBTL.map { α*fbtl + (1-α)*$0 } ?? fbtl
        smoothBTR = smoothBTR.map { α*fbtr + (1-α)*$0 } ?? fbtr

        let m = NativeMeasurement(comprimento: c, largura: l, altura: a)
        addToBuffer(m)
        updateOverlay(bl: smoothBL!, br: smoothBR!, tl: smoothTL!, tr: smoothTR!,
                      bbl: smoothBBL!, bbr: smoothBBR!, btl: smoothBTL!, btr: smoothBTR!,
                      measurement: m)
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
