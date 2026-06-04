import ARKit
import SceneKit
import CoreML
import Accelerate

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

    // MARK: - MobileSAM
    private var samEncoder: MLModel?
    private var samDecoder: MLModel?
    private var cachedEmbedding: MLMultiArray?
    private var encoderTickCount = 0
    private let encoderInterval = 4          // re-encode every 4 frames
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
        loadSAMModels()
    }

    // MARK: - SAM model loading
    // Xcode compila .mlmodel → .mlmodelc; buscamos en main bundle y en el bundle del pod
    private func loadSAMModels() {
        let bundles: [Bundle] = [Bundle.main, Bundle(for: BoxDetectionCoordinator.self)]
        var encURL: URL?, decURL: URL?
        for b in bundles {
            encURL = encURL ?? b.url(forResource: "sam_encoder", withExtension: "mlmodelc")
            decURL = decURL ?? b.url(forResource: "sam_decoder", withExtension: "mlmodelc")
        }
        guard let eu = encURL, let du = decURL else {
            print("[SAM] sam_encoder/sam_decoder.mlmodelc no encontrados en bundle")
            return
        }
        do {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .all
            samEncoder = try MLModel(contentsOf: eu, configuration: cfg)
            samDecoder = try MLModel(contentsOf: du, configuration: cfg)
            print("[SAM] encoder + decoder cargados OK")
        } catch {
            print("[SAM] error al cargar: \(error)")
        }
    }

    // MARK: - Pipeline principal
    private func measureFromCenter(frame: ARFrame, depth: ARDepthData) {
        guard let sv = sceneView else { return }
        let vp = sv.bounds.size
        let cx = vp.width / 2, cy = vp.height / 2

        guard let centerD = sampleDepth(at: CGPoint(x: cx, y: cy), frame: frame, depth: depth),
              centerD > 0.15, centerD < 4.0
        else { return }

        guard let encoder = samEncoder, let decoder = samDecoder, !mlInFlight else { return }

        mlInFlight = true
        encoderTickCount += 1
        let needsEncode = encoderTickCount % encoderInterval == 1 || cachedEmbedding == nil

        let pb = frame.capturedImage
        let capturedFrame = frame
        let capturedDepth = depth

        DispatchQueue.global(qos: .userInteractive).async { [weak self] in
            guard let self = self else { return }

            // ── Paso 1: encoder (costoso, cada encoderInterval frames) ──────────
            if needsEncode {
                guard let emb = self.runEncoder(pixelBuffer: pb, model: encoder) else {
                    DispatchQueue.main.async { self.mlInFlight = false }
                    return
                }
                self.cachedEmbedding = emb
            }
            guard let embedding = self.cachedEmbedding else {
                DispatchQueue.main.async { self.mlInFlight = false }
                return
            }

            // ── Paso 2: decoder con punto = centro (barato, cada frame) ─────────
            // Punto en espacio 1024×1024 (SAM canonical)
            // El pixelBuffer ARKit está en landscape; mapeamos centro portrait → SAM
            let samPx = Float(cy / vp.height) * 1024   // portrait Y → SAM X (landscape)
            let samPy = Float(cx / vp.width)  * 1024   // portrait X → SAM Y

            guard let maskArr = self.runDecoder(
                embedding: embedding, px: samPx, py: samPy, model: decoder
            ) else {
                DispatchQueue.main.async { self.mlInFlight = false; self.clearOverlay() }
                return
            }

            // ── Paso 3: máscara → bounding box normalizado ──────────────────────
            guard let box = self.maskToBoundingBox(
                mask: maskArr,
                centerNX: Float(cx / vp.width),
                centerNY: Float(cy / vp.height)
            ) else {
                DispatchQueue.main.async { self.mlInFlight = false; self.clearOverlay() }
                return
            }

            DispatchQueue.main.async {
                self.mlInFlight = false
                self.measureInRegion(box: box, frame: capturedFrame, depth: capturedDepth,
                                     centerD: centerD, vp: vp, cx: cx, cy: cy)
            }
        }
    }

    // MARK: - SAM Encoder
    // Redimensiona el pixelBuffer a 1024×1024, normaliza y corre el encoder TinyViT.
    // Retorna el tensor de embeddings [1, 256, 64, 64].
    private func runEncoder(pixelBuffer: CVPixelBuffer, model: MLModel) -> MLMultiArray? {
        // Pixel mean / std de SAM (ImageNet)
        let mean: [Float] = [123.675, 116.28, 103.53]
        let std:  [Float] = [58.395,  57.12,  57.375]

        // Escalar imagen a 1024×1024
        guard let scaled = resizePixelBuffer(pixelBuffer, to: CGSize(width: 1024, height: 1024)),
              let input  = pixelBufferToSAMTensor(scaled, mean: mean, std: std)
        else { return nil }

        do {
            let feat   = try MLDictionaryFeatureProvider(dictionary: ["image": input])
            let result = try model.prediction(from: feat)
            return result.featureValue(for: "embedding")?.multiArrayValue
        } catch {
            print("[SAM encoder] error: \(error)")
            return nil
        }
    }

    // MARK: - SAM Decoder
    private func runDecoder(embedding: MLMultiArray,
                            px: Float, py: Float,
                            model: MLModel) -> MLMultiArray? {
        do {
            let pxArr = try MLMultiArray(shape: [1], dataType: .float32)
            let pyArr = try MLMultiArray(shape: [1], dataType: .float32)
            pxArr[0] = NSNumber(value: px)
            pyArr[0] = NSNumber(value: py)

            let feat   = try MLDictionaryFeatureProvider(dictionary: [
                "embedding": embedding,
                "point_x":   pxArr,
                "point_y":   pyArr,
            ])
            let result = try model.prediction(from: feat)
            return result.featureValue(for: "mask_logits")?.multiArrayValue
        } catch {
            print("[SAM decoder] error: \(error)")
            return nil
        }
    }

    // MARK: - Mask → bounding box
    // mask_logits tiene shape [1, 1, 256, 256]. Umbral en 0 (sigmoid > 0.5).
    // Retorna el bounding box de la región foreground más cercana al centro,
    // en coordenadas normalizadas [0,1] top-left origin.
    private func maskToBoundingBox(mask: MLMultiArray,
                                   centerNX: Float,
                                   centerNY: Float) -> CGRect? {
        let shape = mask.shape.map { $0.intValue }
        guard shape.count == 4 else { return nil }
        let H = shape[2], W = shape[3]           // 256 × 256

        // Leer valores float del array lineal [1, 1, H, W]
        let count = H * W
        var logits = [Float](repeating: 0, count: count)
        let ptr = mask.dataPointer.assumingMemoryBound(to: Float32.self)
        for i in 0..<count { logits[i] = ptr[i] }

        // Aplicar sigmoid y umbralizar
        var binary = [Bool](repeating: false, count: count)
        for i in 0..<count {
            binary[i] = 1.0 / (1.0 + exp(-logits[i])) > 0.5
        }

        // BFS desde el pixel más cercano al centro que sea foreground
        let cx = Int(centerNX * Float(W))
        let cy = Int(centerNY * Float(H))
        guard cx >= 0, cx < W, cy >= 0, cy < H else { return nil }

        // Encontrar punto de inicio foreground más cercano al centro
        var startX = cx, startY = cy
        if !binary[cy * W + cx] {
            var found = false
            outer: for r in 1..<min(W, H) / 2 {
                for dy in -r...r {
                    for dx in -r...r {
                        let nx = cx + dx, ny = cy + dy
                        guard nx >= 0, nx < W, ny >= 0, ny < H else { continue }
                        if binary[ny * W + nx] { startX = nx; startY = ny; found = true; break outer }
                    }
                }
            }
            guard found else { return nil }
        }

        // BFS limitado para extraer la región conectada
        var visited = [Bool](repeating: false, count: count)
        var queue = [(Int, Int)](); queue.reserveCapacity(2048)
        var minX = startX, maxX = startX, minY = startY, maxY = startY
        queue.append((startX, startY))
        visited[startY * W + startX] = true

        var qi = 0
        while qi < queue.count, qi < W * H / 2 {
            let (x, y) = queue[qi]; qi += 1
            for (dx, dy) in [(-1,0),(1,0),(0,-1),(0,1)] as [(Int,Int)] {
                let nx = x+dx, ny = y+dy
                guard nx >= 0, nx < W, ny >= 0, ny < H else { continue }
                let idx = ny * W + nx
                guard !visited[idx], binary[idx] else { continue }
                visited[idx] = true
                queue.append((nx, ny))
                if nx < minX { minX = nx }; if nx > maxX { maxX = nx }
                if ny < minY { minY = ny }; if ny > maxY { maxY = ny }
            }
        }

        let boxW = CGFloat(maxX - minX + 1) / CGFloat(W)
        let boxH = CGFloat(maxY - minY + 1) / CGFloat(H)
        guard boxW > 0.05, boxH > 0.05 else { return nil }

        return CGRect(
            x:      CGFloat(minX) / CGFloat(W),
            y:      CGFloat(minY) / CGFloat(H),
            width:  boxW,
            height: boxH
        )
    }

    // MARK: - Pixel buffer helpers

    private func resizePixelBuffer(_ input: CVPixelBuffer, to size: CGSize) -> CVPixelBuffer? {
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
        ]
        var output: CVPixelBuffer?
        CVPixelBufferCreate(nil,
                            Int(size.width), Int(size.height),
                            kCVPixelFormatType_32BGRA,
                            attrs as CFDictionary, &output)
        guard let dst = output else { return nil }

        CVPixelBufferLockBaseAddress(input, .readOnly)
        CVPixelBufferLockBaseAddress(dst, [])
        defer {
            CVPixelBufferUnlockBaseAddress(input, .readOnly)
            CVPixelBufferUnlockBaseAddress(dst, [])
        }

        let srcW = CVPixelBufferGetWidth(input)
        let srcH = CVPixelBufferGetHeight(input)
        guard let srcBase = CVPixelBufferGetBaseAddress(input),
              let dstBase = CVPixelBufferGetBaseAddress(dst)
        else { return nil }

        var srcBuf = vImage_Buffer(
            data: srcBase,
            height: vImagePixelCount(srcH),
            width:  vImagePixelCount(srcW),
            rowBytes: CVPixelBufferGetBytesPerRow(input))
        var dstBuf = vImage_Buffer(
            data: dstBase,
            height: vImagePixelCount(Int(size.height)),
            width:  vImagePixelCount(Int(size.width)),
            rowBytes: CVPixelBufferGetBytesPerRow(dst))
        vImageScale_ARGB8888(&srcBuf, &dstBuf, nil, vImage_Flags(0))
        return dst
    }

    // Convierte CVPixelBuffer BGRA 1024×1024 → MLMultiArray [1,3,1024,1024] Float32
    // con la normalización pixel_mean/pixel_std de SAM.
    private func pixelBufferToSAMTensor(_ pb: CVPixelBuffer,
                                        mean: [Float], std: [Float]) -> MLMultiArray? {
        let W = CVPixelBufferGetWidth(pb)
        let H = CVPixelBufferGetHeight(pb)
        guard let arr = try? MLMultiArray(shape: [1, 3, H as NSNumber, W as NSNumber],
                                          dataType: .float32)
        else { return nil }

        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let src = base.assumingMemoryBound(to: UInt8.self)
        let dst = arr.dataPointer.assumingMemoryBound(to: Float32.self)
        let stride = CVPixelBufferGetBytesPerRow(pb)
        let planeSize = H * W

        for y in 0..<H {
            for x in 0..<W {
                let bgraOff = y * stride + x * 4
                let b = Float(src[bgraOff + 0])
                let g = Float(src[bgraOff + 1])
                let r = Float(src[bgraOff + 2])
                let pix = y * W + x
                dst[0 * planeSize + pix] = (r - mean[0]) / std[0]
                dst[1 * planeSize + pix] = (g - mean[1]) / std[1]
                dst[2 * planeSize + pix] = (b - mean[2]) / std[2]
            }
        }
        return arr
    }

    // MARK: - Medición dentro del bounding box detectado por YOLO
    private func measureInRegion(box: CGRect, frame: ARFrame, depth: ARDepthData,
                                  centerD: Float, vp: CGSize, cx: CGFloat, cy: CGFloat) {
        // box en coordenadas [0,1] top-left origin (portrait) → puntos de pantalla UIKit
        guard box.width > 0.04, box.height > 0.04 else { return }
        let screenBox = CGRect(
            x: box.minX * vp.width,
            y: box.minY * vp.height,
            width:  box.width  * vp.width,
            height: box.height * vp.height
        )
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
