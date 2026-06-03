import ARKit
import SceneKit
import Vision
import CoreML
import ImageIO

struct NativeMeasurement {
    var comprimento: Double
    var largura: Double
    var altura: Double
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

    // MARK: - Stability
    private var buffer: [NativeMeasurement] = []
    private let stabilityWindow = 6
    private let thresholdCm = 3.0

    // MARK: - EMA smoothing on front-face corners
    private var smoothBL: SIMD3<Float>?
    private var smoothBR: SIMD3<Float>?
    private var smoothTL: SIMD3<Float>?
    private var smoothTR: SIMD3<Float>?
    private let smoothAlpha: Float = 0.25

    // MARK: - Overlay
    private var overlayNodes: [SCNNode] = []

    // MARK: - CoreML / YOLO
    private var yoloModel: VNCoreMLModel?
    private var mlInFlight = false

    // MARK: - Manual
    private var manualPoints: [SIMD3<Float>] = []

    // MARK: - Init
    init(sceneView: ARSCNView,
         onUpdate: @escaping (NativeMeasurement) -> Void,
         onPlaneFound: @escaping () -> Void) {
        self.sceneView = sceneView
        self.onUpdate = onUpdate
        self.onPlaneFound = onPlaneFound
        super.init()
        loadYOLOModel()
    }

    // MARK: - Model loading
    private func loadYOLOModel() {
        // Xcode compila .mlmodel → .mlmodelc durante el build; va al main bundle (static framework)
        let url = Bundle.main.url(forResource: "model_core", withExtension: "mlmodelc")
               ?? Bundle(for: BoxDetectionCoordinator.self).url(forResource: "model_core", withExtension: "mlmodelc")
        guard let modelURL = url else {
            print("[YOLO] model_core.mlmodelc no encontrado en bundle")
            return
        }
        do {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .cpuAndNeuralEngine
            let ml = try MLModel(contentsOf: modelURL, configuration: cfg)
            yoloModel = try VNCoreMLModel(for: ml)
            print("[YOLO] modelo cargado OK")
        } catch {
            print("[YOLO] error al cargar: \(error)")
        }
    }

    // MARK: - Pipeline principal (ejecutado en main queue desde ARSessionDelegate)
    private func measureFromCenter(frame: ARFrame, depth: ARDepthData) {
        guard let sv = sceneView else { return }
        let vp = sv.bounds.size
        let cx = vp.width / 2, cy = vp.height / 2

        guard let centerD = sampleDepth(at: CGPoint(x: cx, y: cy), frame: frame, depth: depth),
              centerD > 0.15, centerD < 4.0
        else { return }

        if let model = yoloModel, !mlInFlight {
            runYOLO(model: model, frame: frame, depth: depth,
                    centerD: centerD, vp: vp, cx: cx, cy: cy)
        } else {
            // Sin modelo: región central fija como fallback
            measureInRegion(box: nil, frame: frame, depth: depth,
                             centerD: centerD, vp: vp, cx: cx, cy: cy)
        }
    }

    // MARK: - YOLO inference
    private func runYOLO(model: VNCoreMLModel, frame: ARFrame, depth: ARDepthData,
                          centerD: Float, vp: CGSize, cx: CGFloat, cy: CGFloat) {
        mlInFlight = true
        let pb = frame.capturedImage
        let capturedFrame = frame
        let capturedDepth = depth

        let request = VNCoreMLRequest(model: model) { [weak self] req, err in
            guard let self = self else { return }
            if let err = err { print("[YOLO] inferencia error: \(err)") }
            let box = self.parseBestDetection(req.results)
            DispatchQueue.main.async {
                self.mlInFlight = false
                self.measureInRegion(box: box, frame: capturedFrame, depth: capturedDepth,
                                      centerD: centerD, vp: vp, cx: cx, cy: cy)
            }
        }
        request.imageCropAndScaleOption = .scaleFill

        DispatchQueue.global(qos: .userInteractive).async {
            // .right = el pixelBuffer de ARKit está en landscape, lo rotamos a portrait
            let handler = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .right)
            try? handler.perform([request])
        }
    }

    // MARK: - Parseo de salida YOLO
    // Soporta VNRecognizedObjectObservation (si CoreML wrappea) o MLMultiArray cruda
    private func parseBestDetection(_ results: [VNObservation]?) -> CGRect? {
        guard let results = results else { return nil }

        // Caso fácil: Vision produce observaciones de objetos directamente
        let objObs = results.compactMap { $0 as? VNRecognizedObjectObservation }
        if !objObs.isEmpty {
            guard let best = objObs.max(by: { $0.confidence < $1.confidence }),
                  best.confidence > 0.25 else { return nil }
            // VNRecognizedObjectObservation.boundingBox: origin bottom-left → convertir a top-left
            let b = best.boundingBox
            return CGRect(x: b.minX, y: 1 - b.maxY, width: b.width, height: b.height)
        }

        // Caso raw: MLMultiArray
        let feats = results.compactMap { $0 as? VNCoreMLFeatureValueObservation }
        for f in feats {
            guard let arr = f.featureValue.multiArrayValue else { continue }
            let shape = arr.shape.map { $0.intValue }
            print("[YOLO] output '\(f.featureName)' shape: \(shape)")
            if let box = extractBestBox(from: arr, shape: shape) { return box }
        }
        return nil
    }

    // Extrae el box de mayor confianza de un MLMultiArray con shape [1,N,C] o [N,C], C>=5
    // Layout esperado: x1, y1, x2, y2, conf, [class, mask_coeffs...]
    // Usa subscript Int (más simple y compatible con WMO): índice lineal i*C+j
    private func extractBestBox(from arr: MLMultiArray, shape: [Int]) -> CGRect? {
        guard shape.count >= 2, shape[shape.count - 1] >= 5 else { return nil }
        let C = shape[shape.count - 1]
        let N = shape.count == 3 ? shape[1] : shape[0]
        guard N > 0 else { return nil }

        var bestConf: Float = 0.25
        var bestBox: CGRect? = nil

        for i in 0..<N {
            // Para [1,N,C] y [N,C] el índice lineal de [_,i,j] es i*C+j (batch=0)
            let base = i * C
            let x1   = arr[base + 0].floatValue
            let y1   = arr[base + 1].floatValue
            let x2   = arr[base + 2].floatValue
            let y2   = arr[base + 3].floatValue
            let conf = arr[base + 4].floatValue

            guard conf > bestConf, x2 > x1, y2 > y1 else { continue }
            bestConf = conf

            let scale: CGFloat = x2 > 2.0 ? 1.0 / 640.0 : 1.0
            bestBox = CGRect(
                x: CGFloat(x1) * scale, y: CGFloat(y1) * scale,
                width: CGFloat(x2 - x1) * scale, height: CGFloat(y2 - y1) * scale
            )
        }
        return bestBox
    }

    // MARK: - Medición dentro de la región detectada (o fallback central)
    private func measureInRegion(box: CGRect?, frame: ARFrame, depth: ARDepthData,
                                  centerD: Float, vp: CGSize, cx: CGFloat, cy: CGFloat) {
        // box en coordenadas [0,1] top-left origin (portrait) → puntos de pantalla UIKit
        let screenBox: CGRect
        if let b = box, b.width > 0.05, b.height > 0.05 {
            screenBox = CGRect(
                x: b.minX * vp.width,
                y: b.minY * vp.height,
                width:  b.width  * vp.width,
                height: b.height * vp.height
            )
        } else {
            let hw = vp.width * 0.38, hh = vp.height * 0.32
            screenBox = CGRect(x: cx - hw, y: cy - hh, width: hw * 2, height: hh * 2)
        }

        guard screenBox.width > 30, screenBox.height > 30 else { return }

        // Esquinas de la cara frontal en pantalla
        let bl = CGPoint(x: screenBox.minX, y: screenBox.maxY)
        let br = CGPoint(x: screenBox.maxX, y: screenBox.maxY)
        let tl = CGPoint(x: screenBox.minX, y: screenBox.minY)
        let tr = CGPoint(x: screenBox.maxX, y: screenBox.minY)

        // Gradiente de profundidad en el centro (para cajas en ángulo)
        let gs: CGFloat = 30
        let dxL = sampleDepth(at: CGPoint(x: cx - gs, y: cy), frame: frame, depth: depth)
        let dxR = sampleDepth(at: CGPoint(x: cx + gs, y: cy), frame: frame, depth: depth)
        let dyT = sampleDepth(at: CGPoint(x: cx, y: cy - gs), frame: frame, depth: depth)
        let dyB = sampleDepth(at: CGPoint(x: cx, y: cy + gs), frame: frame, depth: depth)

        let gx: Float = (dxL != nil && dxR != nil && abs(dxR! - dxL!) < 0.20)
                        ? (dxR! - dxL!) / Float(2 * gs) : 0
        let gy: Float = (dyT != nil && dyB != nil && abs(dyB! - dyT!) < 0.20)
                        ? (dyB! - dyT!) / Float(2 * gs) : 0

        func expDepth(_ p: CGPoint) -> Float {
            centerD + gx * Float(p.x - cx) + gy * Float(p.y - cy)
        }

        guard
            let p3BL = worldPointAtDepth(bl, depth: expDepth(bl), frame: frame),
            let p3BR = worldPointAtDepth(br, depth: expDepth(br), frame: frame),
            let p3TL = worldPointAtDepth(tl, depth: expDepth(tl), frame: frame),
            let p3TR = worldPointAtDepth(tr, depth: expDepth(tr), frame: frame)
        else { return }

        // Rechazar si la cara es mayormente horizontal (piso / tapa de mesa)
        let fRight     = simd_normalize(p3BR - p3BL)
        let fUp        = simd_normalize(p3TL - p3BL)
        let faceNormal = simd_normalize(simd_cross(fRight, fUp))
        guard abs(faceNormal.y) < 0.65 else { return }

        let c = Double(simd_distance(p3BL, p3BR)) * 100   // comprimento (ancho)
        let a = Double(simd_distance(p3BL, p3TL)) * 100   // altura

        // Profundidad: escanear hacia arriba sobre la cara superior de la caja
        let l = estimateDepthFromTopFace(frame: frame, depth: depth,
                                          topLeft: tl, topRight: tr,
                                          centerD: centerD) ?? (min(c, a) * 0.65)

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

    // MARK: - Estimación de profundidad (largura) escaneando la cara superior
    // Escanea hacia arriba desde el borde superior del bounding box detectado.
    // Cuando world.y cae más de 7 cm → cruzamos el borde trasero de la caja.
    private func estimateDepthFromTopFace(frame: ARFrame, depth: ARDepthData,
                                           topLeft: CGPoint, topRight: CGPoint,
                                           centerD: Float) -> Double? {
        var estimates = [Double]()

        for frac in [0.25, 0.50, 0.75] as [CGFloat] {
            let sx = topLeft.x + frac * (topRight.x - topLeft.x)
            var sy = topLeft.y   // empieza en el borde superior de la caja

            guard let startD = sampleDepth(at: CGPoint(x: sx, y: sy), frame: frame, depth: depth),
                  let topPt  = worldPointAtDepth(CGPoint(x: sx, y: sy), depth: startD, frame: frame)
            else { continue }

            let topFaceY = topPt.y
            var prevPt   = topPt

            while sy > 8 {
                sy -= 5
                guard let d   = sampleDepth(at: CGPoint(x: sx, y: sy), frame: frame, depth: depth),
                      let wPt = worldPointAtDepth(CGPoint(x: sx, y: sy), depth: d, frame: frame)
                else { break }
                if wPt.y < topFaceY - 0.07 { break }  // cruzó el borde trasero → suelo/fondo
                if d - startD > 0.22         { break }  // salto grande de profundidad → fondo
                prevPt = wPt
            }

            let boxDepth = Double(simd_distance(topPt, prevPt)) * 100
            if boxDepth > 3, boxDepth < 200 { estimates.append(boxDepth) }
        }

        return estimates.isEmpty ? nil : median(estimates)
    }

    // MARK: - LiDAR helpers

    private func sampleDepth(at screenPt: CGPoint,
                             frame: ARFrame,
                             depth: ARDepthData) -> Float? {
        guard let sv = sceneView else { return nil }
        let vp = sv.bounds.size
        let invDisplay = frame.displayTransform(for: .portrait, viewportSize: vp).inverted()
        let normCam = CGPoint(x: screenPt.x / vp.width,
                              y: screenPt.y / vp.height).applying(invDisplay)
        let dm = depth.depthMap
        let dW = CVPixelBufferGetWidth(dm)
        let dH = CVPixelBufferGetHeight(dm)
        let sx = max(0, min(dW - 1, Int(normCam.x * CGFloat(dW))))
        let sy = max(0, min(dH - 1, Int(normCam.y * CGFloat(dH))))
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
        let vp = sv.bounds.size
        let invDisplay = frame.displayTransform(for: .portrait, viewportSize: vp).inverted()
        let normCam = CGPoint(x: screenPt.x / vp.width,
                              y: screenPt.y / vp.height).applying(invDisplay)
        let intr = frame.camera.intrinsics
        let iW   = Float(frame.camera.imageResolution.width)
        let iH   = Float(frame.camera.imageResolution.height)
        let imgX = Float(normCam.x) * iW
        let imgY = Float(normCam.y) * iH
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
            let vals = buffer.map { $0[keyPath: kp] }
            return (vals.max() ?? 0) - (vals.min() ?? 0)
        }

        guard range(\.comprimento) < thresholdCm,
              range(\.altura)      < thresholdCm,
              range(\.largura)     < thresholdCm * 2 else { return }

        let stable = NativeMeasurement(
            comprimento: median(buffer.map(\.comprimento)),
            largura:     median(buffer.map(\.largura)),
            altura:      median(buffer.map(\.altura))
        )
        lastMeasurement = stable
        DispatchQueue.main.async { [weak self] in
            self?.onUpdate(stable)
        }
    }

    private func median(_ arr: [Double]) -> Double {
        let s = arr.sorted(); let m = s.count / 2
        return s.count % 2 == 0 ? (s[m-1] + s[m]) / 2 : s[m]
    }

    // MARK: - Manual points (3-4 toques)
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
            let c = Double(simd_distance(p0, p1)) * 100
            let l = Double(simd_distance(p0, p2)) * 100
            let m = NativeMeasurement(comprimento: c, largura: l, altura: l)
            lastMeasurement = m
            DispatchQueue.main.async { [weak self] in self?.onUpdate(m) }
        } else if manualPoints.count == 4 {
            let p0 = manualPoints[0], p1 = manualPoints[1]
            let p2 = manualPoints[2], p3 = manualPoints[3]
            let c = Double(simd_distance(p0, p1)) * 100
            let l = Double(simd_distance(p0, p2)) * 100
            let a = Double(simd_distance(p0, p3)) * 100
            let m = NativeMeasurement(comprimento: c, largura: l, altura: a)
            lastMeasurement = m
            manualPoints.removeAll()
            overlayNodes.forEach { $0.removeFromParentNode() }
            overlayNodes.removeAll()
            DispatchQueue.main.async { [weak self] in self?.onUpdate(m) }
        }
    }

    // MARK: - Overlay (12 aristas + 3 etiquetas)
    func updateOverlay(bl: SIMD3<Float>, br: SIMD3<Float>,
                       tl: SIMD3<Float>, tr: SIMD3<Float>,
                       measurement: NativeMeasurement) {
        guard let sv = sceneView else { return }
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.clearOverlay()

            let yellow = UIColor(red: 0.0, green: 0.90, blue: 0.3, alpha: 1)

            let right      = simd_normalize(br - bl)
            let up         = simd_normalize(tl - bl)
            let faceNormal = simd_normalize(simd_cross(right, up))
            let depthM     = Float(max(measurement.largura, 2) / 100.0)
            let extrudeDir = -faceNormal

            let bbl = bl + extrudeDir * depthM
            let bbr = br + extrudeDir * depthM
            let btl = tl + extrudeDir * depthM
            let btr = tr + extrudeDir * depthM

            let edges: [(SIMD3<Float>, SIMD3<Float>)] = [
                (bl, br), (br, tr), (tr, tl), (tl, bl),
                (bbl, bbr), (bbr, btr), (btr, btl), (btl, bbl),
                (bl, bbl), (br, bbr), (tl, btl), (tr, btr),
            ]
            for (s, e) in edges {
                let node = self.makeLine(from: s, to: e, color: yellow)
                sv.scene.rootNode.addChildNode(node)
                self.overlayNodes.append(node)
            }

            let labelData: [(String, SIMD3<Float>)] = [
                ("\(Int(measurement.comprimento.rounded())) cm", (bl + br) / 2 + up * (-0.055)),
                ("\(Int(measurement.altura.rounded())) cm",      (bl + tl) / 2 + right * (-0.065)),
                ("\(Int(measurement.largura.rounded())) cm",     (br + bbr) / 2 + right * 0.065),
            ]
            for (text, pos) in labelData {
                let node = self.makeTextNode(text, color: yellow)
                node.position = SCNVector3(pos.x, pos.y, pos.z)
                sv.scene.rootNode.addChildNode(node)
                self.overlayNodes.append(node)
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

        let node  = SCNNode(geometry: geo)
        let scale: Float = 0.001
        node.scale = SCNVector3(scale, scale, scale)

        let (minB, maxB) = node.boundingBox
        node.pivot = SCNMatrix4MakeTranslation(
            (maxB.x - minB.x) / 2 + minB.x,
            (maxB.y - minB.y) / 2 + minB.y,
            0
        )
        let constraint = SCNBillboardConstraint()
        constraint.freeAxes = .all
        node.constraints = [constraint]
        return node
    }

    func clearOverlay() {
        overlayNodes.forEach { $0.removeFromParentNode() }
        overlayNodes.removeAll()
    }

    func reset() {
        clearOverlay()
        buffer.removeAll()
        manualPoints.removeAll()
        lastMeasurement = nil
        smoothBL = nil; smoothBR = nil; smoothTL = nil; smoothTR = nil
    }

    private func makeLine(from s: SIMD3<Float>,
                          to e: SIMD3<Float>,
                          color: UIColor) -> SCNNode {
        let v   = e - s
        let len = simd_length(v)
        let cyl = SCNCylinder(radius: 0.002, height: CGFloat(len))
        cyl.firstMaterial?.diffuse.contents = color
        cyl.firstMaterial?.lightingModel    = .constant

        let node = SCNNode(geometry: cyl)
        let mid  = (s + e) / 2
        node.position = SCNVector3(mid.x, mid.y, mid.z)

        let dir = simd_normalize(v)
        let up  = SIMD3<Float>(0, 1, 0)
        let dot = simd_dot(up, dir)

        if abs(dot) > 0.9999 {
            if dot < 0 { node.rotation = SCNVector4(1, 0, 0, Float.pi) }
        } else {
            let axis  = simd_normalize(simd_cross(up, dir))
            let angle = acos(max(-1, min(1, dot)))
            node.rotation = SCNVector4(axis.x, axis.y, axis.z, angle)
        }
        return node
    }
}

// MARK: - Delegates
extension BoxDetectionCoordinator: ARSCNViewDelegate {
    func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
        if anchor is ARPlaneAnchor {
            DispatchQueue.main.async { [weak self] in self?.onPlaneFound() }
        }
    }
}

extension BoxDetectionCoordinator: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard
            mode == .auto,
            frame.timestamp - lastScanTime > scanInterval,
            let depthData = frame.sceneDepth
        else { return }
        lastScanTime = frame.timestamp

        let capturedFrame = frame
        let capturedDepth = depthData
        DispatchQueue.main.async { [weak self] in
            self?.measureFromCenter(frame: capturedFrame, depth: capturedDepth)
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        print("[ARKit] session failed: \(error.localizedDescription)")
    }
    func sessionWasInterrupted(_ session: ARSession)    { print("[ARKit] interrupted") }
    func sessionInterruptionEnded(_ session: ARSession) { print("[ARKit] resumido") }
}
