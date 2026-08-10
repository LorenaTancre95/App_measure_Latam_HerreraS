import Foundation
import ARKit
import Combine
import SceneKit
import simd

struct DetectionInfo: Identifiable {
    let id = UUID()
    let label: String
    let size: simd_float3
    let confidence: Float
}

@MainActor
final class ARViewModel: ObservableObject {
    @Published var status: String = "Initializing..."
    @Published var isProcessing: Bool = false
    @Published var detections: [DetectionInfo] = []
    @Published var confidenceThreshold: Float = 0.3
    @Published var debugBBoxes: [(rect: CGRect, score: Float)] = []
    @Published var isCalibrated: Bool = false
    @Published var measureMode: MeasureMode = .tap
    @Published var segmentationOverlay: UIImage? = nil
    @Published var crosshairHit:    Bool = false
    @Published var liveAimPoint:   simd_float3? = nil
    @Published var isAimStable:    Bool = false
    @Published var lastTapScreen:  CGPoint? = nil
    @Published var crosshairSnapPt: CGPoint? = nil
    /// Bordes detectados en tiempo real (se usan para el overlay y para CAPTURAR)
    @Published var liveEdges: BoxEdgeDetector.Result? = nil

    // MARK: - TAP mode: ANCHO · LARGO · ALTO de a uno
    // Cada dimensión: capturá 2 puntos → preview distancia → GUARDAR/BORRAR → siguiente

    enum DimPhase: Int { case ancho = 0, largo = 1, alto = 2, done = 3 }
    enum TapPhase { case waitingFirst, waitingSecond, preview }

    @Published var dimPhase: DimPhase = .ancho
    @Published var tapPhase: TapPhase = .waitingFirst
    @Published private(set) var firstPoint:  simd_float3? = nil
    private              var secondPoint: simd_float3? = nil
    @Published private(set) var savedAncho: Float? = nil
    @Published private(set) var savedLargo: Float? = nil
    @Published private(set) var savedAlto:  Float? = nil
    /// Y del plano del piso detectado por ARKit (el plano horizontal más bajo)
    @Published private(set) var floorY: Float? = nil

    func updateFloorY(_ y: Float) {
        if floorY == nil || y < floorY! { floorY = y }
    }

    // Entero compatible con MeasureCrosshairView (0-5)
    var tapStep: Int {
        switch (dimPhase, tapPhase) {
        case (.ancho, .waitingFirst):  return 0
        case (.ancho, _):              return 1
        case (.largo, .waitingFirst):  return 2
        case (.largo, _):              return 3
        case (.alto, .waitingFirst):   return 4
        default:                       return 5
        }
    }

    var activeDim: String {
        switch dimPhase {
        case .ancho: return "ANCHO"
        case .largo: return "LARGO"
        case .alto:  return "ALTO"
        case .done:  return ""
        }
    }

    var currentInstruction: String {
        switch (dimPhase, tapPhase) {
        case (.ancho, .waitingFirst):  return "ANCHO  —  apuntá al borde IZQUIERDO"
        case (.ancho, .waitingSecond): return "ANCHO  —  apuntá al borde DERECHO"
        case (.largo, .waitingFirst):  return "LARGO  —  apuntá al borde MÁS CERCANO"
        case (.largo, .waitingSecond): return "LARGO  —  apuntá al borde MÁS LEJANO"
        case (.alto,  .waitingFirst):  return "ALTO  —  apuntá a la cara SUPERIOR de la caja"
        case (.alto,  .waitingSecond): return "ALTO  —  apuntá ahora a la BASE de la caja"
        default: return ""
        }
    }

    // Captura el punto 3D donde el usuario toca la pantalla directamente (solo para ALTO).
    // Si el piso ya fue detectado por ARKit → 1 solo tap en la esquina superior y listo.
    // Si no → 2 taps manuales (superior + inferior) como fallback.
    func captureTap(at screenPoint: CGPoint) {
        guard dimPhase == .alto, tapPhase != .preview, !isProcessing else { return }
        guard let p = tapTo3D(point: screenPoint) else {
            status = "Sin superficie en ese punto — intentá de nuevo"
            return
        }
        guard let sv = sceneView else { return }
        placeMarker(at: p, in: sv)

        switch tapPhase {
        case .waitingFirst:
            firstPoint = p
            if let fy = floorY {
                // Piso detectado: calcula altura automáticamente con 1 tap
                let floorPt = simd_float3(p.x, fy, p.z)
                secondPoint = floorPt
                placeMarker(at: floorPt, in: sv)
                drawLine(from: p, to: floorPt, in: sv)
                tapPhase = .preview
                let dist = measuredDistance ?? 0
                status = "ALTO: \(measureUnit.format(dist)) \(measureUnit.rawValue)"
            } else {
                // Sin piso detectado: pide el 2° tap manual
                tapPhase = .waitingSecond
                status   = currentInstruction
            }
        case .waitingSecond:
            secondPoint = p
            if let fp = firstPoint { drawLine(from: fp, to: p, in: sv) }
            tapPhase = .preview
            let dist = measuredDistance ?? 0
            status = "ALTO: \(measureUnit.format(dist)) \(measureUnit.rawValue)"
        case .preview: break
        }
    }

    // Distancia en vivo (solo durante el 2° tap de cada dimensión)
    var liveDistance: Float? {
        guard tapPhase == .waitingSecond, let fp = firstPoint, let aim = liveAimPoint else { return nil }
        if dimPhase == .alto { return abs(fp.y - aim.y) }
        return simd_distance(fp, aim)
    }

    // Distancia final capturada (disponible en estado .preview)
    var measuredDistance: Float? {
        guard tapPhase == .preview, let fp = firstPoint, let sp = secondPoint else { return nil }
        if dimPhase == .alto { return abs(fp.y - sp.y) }
        return simd_distance(fp, sp)
    }

    enum MeasureMode { case box, oversize, tap }

    enum MeasureUnit: String, CaseIterable {
        case cm = "cm", m = "m", inches = "in"
        func convert(_ meters: Float) -> Float {
            switch self {
            case .cm:     return meters * 100
            case .m:      return meters
            case .inches: return meters * 39.3701
            }
        }
        func format(_ meters: Float) -> String {
            switch self {
            case .cm:     return String(format: "%.0f",  convert(meters))
            case .m:      return String(format: "%.2f",  convert(meters))
            case .inches: return String(format: "%.1f",  convert(meters))
            }
        }
        func formatBox(_ x: Float, _ y: Float, _ z: Float) -> String {
            "\(format(x))×\(format(y))×\(format(z)) \(rawValue)"
        }
    }

    @Published var measureUnit: MeasureUnit = .cm {
        didSet {
            if let sv = sceneView, !lastDetections3D.isEmpty {
                placeBoxes(lastDetections3D, in: sv)
            }
        }
    }

    var sceneView: ARSCNView?
    var viewportSize: CGSize = UIScreen.main.bounds.size
    let viewfinderNorm = CGRect(x: 0.1, y: 0.18, width: 0.8, height: 0.64)
    private var yoloDetector: YOLODetector?
    private var palletDetector: PalletDetector?
    private var boxNodes: [SCNNode] = []
    private var markerAnchors: [ARAnchor] = []
    private var lineNodes: [SCNNode] = []
    private var lastDetections3D: [Detection3D] = []

    func setup(sceneView: ARSCNView) {
        self.sceneView = sceneView
        Task.detached { await self.loadTAPModels() }
    }

    nonisolated private func loadTAPModels() async {
        let pallet = try? PalletDetector()
        await MainActor.run {
            self.palletDetector = pallet
            self.status = self.currentInstruction
        }
    }

    nonisolated private func loadYOLO() async {
        await MainActor.run { self.status = "Cargando detector CAJA..." }
        guard let yoloPath = Bundle.main.path(forResource: "best", ofType: "onnx") else {
            await MainActor.run { self.status = "best.onnx no encontrado" }; return
        }
        do {
            let yolo = try YOLODetector(modelPath: yoloPath)
            await MainActor.run { self.yoloDetector = yolo; self.status = "Apuntá la caja al visor" }
        } catch {
            await MainActor.run { self.status = "YOLO falló: \(error.localizedDescription)" }
        }
    }

    func detectNow() {
        guard let sceneView, let frame = sceneView.session.currentFrame else { status = "Not ready"; return }
        guard frame.sceneDepth != nil else { status = "No LiDAR depth"; return }
        if measureMode == .oversize { detectPallet(frame: frame, sceneView: sceneView); return }
        guard let yoloDetector else { Task.detached { await self.loadYOLO() }; return }
        isProcessing = true; status = "Detectando..."
        Task.detached {
            do {
                let (img, _, _) = pixelBufferToFloatArray(frame.capturedImage, targetSize: 640)
                let conf = await MainActor.run { self.confidenceThreshold }
                let yoloBoxes = try await MainActor.run {
                    try yoloDetector.detect(image: img, imageWidth: 640, imageHeight: 640, confThreshold: conf)
                }
                guard !yoloBoxes.isEmpty else {
                    await MainActor.run { self.status = "No cajas detectadas"; self.isProcessing = false }; return
                }
                let topBoxes = Array(yoloBoxes.sorted { $0.score > $1.score }.prefix(5))
                let vp = await MainActor.run { self.viewportSize }
                let displayT = frame.displayTransform(for: .portrait, viewportSize: vp)
                let res = frame.camera.imageResolution
                let imgW = Float(res.width), imgH = Float(res.height)
                let side = min(imgW, imgH), ox = (imgW-side)/2, oy = (imgH-side)/2
                let screenBoxes: [(rect: CGRect, score: Float)] = topBoxes.map { box in
                    func pt(_ x: Float, _ y: Float) -> CGPoint {
                        CGPoint(x: CGFloat((x/640*side+ox)/imgW), y: CGFloat((y/640*side+oy)/imgH)).applying(displayT)
                    }
                    let tl = pt(box.xmin, box.ymin), br = pt(box.xmax, box.ymax)
                    return (CGRect(x: min(tl.x,br.x), y: min(tl.y,br.y),
                                  width: abs(br.x-tl.x), height: abs(br.y-br.y)), box.score)
                }
                await MainActor.run { self.debugBBoxes = screenBoxes }
                let vfN = await MainActor.run { self.viewfinderNorm }
                let vfR = CGRect(x: vfN.minX*vp.width, y: vfN.minY*vp.height,
                                 width: vfN.width*vp.width, height: vfN.height*vp.height)
                let vfPassed: [YOLOBox] = zip(topBoxes, screenBoxes).compactMap { box, scr in
                    vfR.contains(CGPoint(x: scr.rect.midX, y: scr.rect.midY)) ? box : nil
                }
                guard let best = vfPassed.first ?? topBoxes.first else {
                    await MainActor.run { self.isProcessing = false }; return
                }
                let nShots = 5; var shots: [Detection3D] = []
                for i in 1...nShots {
                    await MainActor.run { self.status = "Midiendo \(i)/\(nShots)..." }
                    let f = await MainActor.run { self.sceneView?.session.currentFrame }
                    if let f, f.sceneDepth != nil {
                        let det = await MainActor.run { BoxMeasurer2.measure(frame: f, yoloBox: best) }
                        if let det { shots.append(det) }
                    }
                    if i < nShots { try? await Task.sleep(nanoseconds: 250_000_000) }
                }
                let finalShots = shots
                await MainActor.run {
                    if finalShots.isEmpty { self.status = "Sin geometría — acercate más" }
                    else if let sv = self.sceneView { self.placeBoxes([self.medianDetection(finalShots)], in: sv) }
                    self.isProcessing = false
                }
            } catch {
                await MainActor.run { self.status = "Error: \(error.localizedDescription)"; self.isProcessing = false }
            }
        }
    }

    // MARK: - Measurement (TAP mode)

    func captureCenter() {
        guard !isProcessing, tapPhase != .preview, dimPhase != .done else { return }
        guard let sv = sceneView else { return }

        // ALTO: 1 tap en la cara superior + piso detectado = altura automática
        if dimPhase == .alto {
            guard let aim = liveAimPoint else {
                status = "Sin superficie — apuntá directo a la caja"
                return
            }
            placeMarker(at: aim, in: sv)
            switch tapPhase {
            case .waitingFirst:
                firstPoint = aim
                if let fy = floorY {
                    secondPoint = simd_float3(aim.x, fy, aim.z)
                    tapPhase = .preview
                    let dist = measuredDistance ?? 0
                    status = "ALTO: \(measureUnit.format(dist)) \(measureUnit.rawValue)"
                } else {
                    tapPhase = .waitingSecond
                    status = currentInstruction
                }
            case .waitingSecond:
                secondPoint = aim
                if let fp = firstPoint { drawLine(from: fp, to: aim, in: sv) }
                tapPhase = .preview
                let dist = measuredDistance ?? 0
                status = "ALTO: \(measureUnit.format(dist)) \(measureUnit.rawValue)"
            case .preview: break
            }
            return
        }

        // ANCHO / LARGO: 2 taps manuales con crosshair
        let pt3D: simd_float3
        if let aim = liveAimPoint {
            pt3D = aim
        } else {
            let center = CGPoint(x: viewportSize.width / 2, y: viewportSize.height / 2)
            guard let p = tapTo3D(point: center) else {
                status = "Sin superficie — apuntá directo a la caja"
                return
            }
            pt3D = p
        }
        placeMarker(at: pt3D, in: sv)
        switch tapPhase {
        case .waitingFirst:
            firstPoint = pt3D
            tapPhase   = .waitingSecond
            status     = currentInstruction
        case .waitingSecond:
            secondPoint = pt3D
            if let fp = firstPoint { drawLine(from: fp, to: pt3D, in: sv) }
            tapPhase = .preview
            let dist = measuredDistance ?? 0
            status = "\(activeDim): \(measureUnit.format(dist)) \(measureUnit.rawValue)"
        case .preview: break
        }
    }

    // Guarda la dimensión actual y avanza a la primera aún no guardada
    func guardarMedicion() {
        guard let dist = measuredDistance else { return }
        switch dimPhase {
        case .ancho: savedAncho = dist
        case .largo: savedLargo = dist
        case .alto:  savedAlto  = dist
        case .done:  return
        }
        clearCurrentMarkers()
        // Avanza a la primera dimensión que todavía no tiene valor
        if savedAncho == nil      { dimPhase = .ancho }
        else if savedLargo == nil { dimPhase = .largo }
        else if savedAlto  == nil { dimPhase = .alto  }
        else {
            dimPhase = .done
            let sa = savedAncho ?? 0, sl = savedLargo ?? 0, sh = savedAlto ?? 0
            detections.removeAll()
            let det = DetectionInfo(label: "caja",
                                    size: simd_float3(sl, sh, sa),
                                    confidence: 1.0)
            detections.append(det)
            status = "✓ \(measureUnit.formatBox(det.size.x, det.size.y, det.size.z))"
            lastTapScreen = nil
            return
        }
        tapPhase = .waitingFirst
        status   = currentInstruction
    }

    // Permite re-medir una dimensión ya guardada tocando su chip
    func remeasure(_ dim: DimPhase) {
        guard dim != .done else { return }
        clearCurrentMarkers()
        switch dim {
        case .ancho: savedAncho = nil
        case .largo: savedLargo = nil
        case .alto:  savedAlto  = nil
        case .done:  return
        }
        detections.removeAll(); lastDetections3D.removeAll()
        dimPhase = dim
        tapPhase = .waitingFirst
        lastTapScreen = nil
        status = currentInstruction
    }

    // Descarta los 2 puntos del segmento actual y empieza de nuevo esa dimensión
    func borrarMedicion() {
        clearCurrentMarkers()
        tapPhase = .waitingFirst
        status   = currentInstruction
    }

    private func clearCurrentMarkers() {
        markerAnchors.forEach { sceneView?.session.remove(anchor: $0) }; markerAnchors.removeAll()
        lineNodes.forEach { $0.removeFromParentNode() }; lineNodes.removeAll()
        firstPoint = nil; secondPoint = nil
    }

    // MARK: - Utilidades LiDAR / AR

    /// LiDAR primero: lee el depth real del pixel (no atraviesa la caja hacia la cama/suelo).
    /// Raycast solo si LiDAR no tiene datos en esa zona.
    private func tapTo3D(point: CGPoint) -> simd_float3? {
        guard let sceneView, let frame = sceneView.session.currentFrame else { return nil }
        if let lidar = ARViewModel.lidarPoint(frame: frame, screenPoint: point, viewportSize: viewportSize) {
            return lidar
        }
        for target: ARRaycastQuery.Target in [.existingPlaneGeometry, .estimatedPlane] {
            if let q = sceneView.raycastQuery(from: point, allowing: target, alignment: .any),
               let r = sceneView.session.raycast(q).first {
                let col = r.worldTransform.columns.3
                return simd_float3(col.x, col.y, col.z)
            }
        }
        return nil
    }

    nonisolated static func lidarPoint(
        frame: ARFrame,
        screenPoint: CGPoint,
        viewportSize: CGSize
    ) -> simd_float3? {
        guard let depthBuffer = (frame.smoothedSceneDepth ?? frame.sceneDepth)?.depthMap else { return nil }
        let bufW = Float(CVPixelBufferGetWidth(frame.capturedImage))
        let bufH = Float(CVPixelBufferGetHeight(frame.capturedImage))
        let dW   = CVPixelBufferGetWidth(depthBuffer)
        let dH   = CVPixelBufferGetHeight(depthBuffer)
        let normImg = CGPoint(x: screenPoint.x / viewportSize.width,
                              y: screenPoint.y / viewportSize.height)
                      .applying(frame.displayTransform(for: .portrait, viewportSize: viewportSize).inverted())
        let tapDX = max(0, min(dW - 1, Int(Float(normImg.x) * Float(dW))))
        let tapDY = max(0, min(dH - 1, Int(Float(normImg.y) * Float(dH))))
        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }
        let rb   = CVPixelBufferGetBytesPerRow(depthBuffer)
        guard let base = CVPixelBufferGetBaseAddress(depthBuffer) else { return nil }
        // Toma el centro y expande a vecindad 5×5 buscando la profundidad mediana válida
        var valid: [Float] = []
        for dy in -2...2 {
            let py  = max(0, min(dH - 1, tapDY + dy))
            let row = base.advanced(by: py * rb).assumingMemoryBound(to: Float32.self)
            for dx in -2...2 {
                let val = row[max(0, min(dW - 1, tapDX + dx))]
                if val > 0.05, val < 8.0 { valid.append(val) }
            }
        }
        guard !valid.isEmpty else { return nil }
        valid.sort()
        let chosenD = valid[valid.count / 2]  // mediana → más robusto que el centro raw
        let intr = frame.camera.intrinsics
        let fx = intr[0][0], fy = intr[1][1], cx = intr[2][0], cy = intr[2][1]
        let ix  = Float(tapDX) / Float(dW) * bufW
        let iy  = Float(tapDY) / Float(dH) * bufH
        let cam = simd_float4((ix - cx) / fx * chosenD, (iy - cy) / fy * chosenD, -chosenD, 1)
        let w   = frame.camera.transform * cam
        return simd_float3(w.x, w.y, w.z) / w.w
    }

    private func placeMarker(at position: simd_float3, in sceneView: ARSCNView) {
        var t = matrix_identity_float4x4
        t.columns.3 = simd_float4(position.x, position.y, position.z, 1)
        let anchor = ARAnchor(name: "marker", transform: t)
        sceneView.session.add(anchor: anchor)
        markerAnchors.append(anchor)
    }

    private func drawLine(from a: simd_float3, to b: simd_float3, in sceneView: ARSCNView) {
        let dist = simd_distance(a, b)
        guard dist > 0.001 else { return }
        let mat = SCNMaterial(); mat.diffuse.contents = UIColor.systemYellow
        let cyl = SCNCylinder(radius: 0.003, height: CGFloat(dist)); cyl.materials = [mat]
        let node = SCNNode(geometry: cyl); node.simdPosition = (a + b) / 2
        let dir = simd_normalize(b - a), up = simd_float3(0, 1, 0)
        let dot = simd_dot(up, dir)
        if abs(dot) < 0.999 {
            let axis = simd_normalize(simd_cross(up, dir))
            node.simdRotation = simd_float4(axis.x, axis.y, axis.z, acos(dot))
        }
        sceneView.scene.rootNode.addChildNode(node); lineNodes.append(node)
    }

    // MARK: - Rendering (YOLO/Pallet modes)

    private func placeBoxes(_ detections: [Detection3D], in sceneView: ARSCNView) {
        lastDetections3D = detections; clearBoxes()
        let colors: [UIColor] = [.systemGreen, .systemRed, .systemBlue]
        for (i, det) in detections.enumerated() {
            let color = colors[i % colors.count]
            let box = SCNBox(width: CGFloat(det.size.x), height: CGFloat(det.size.y), length: CGFloat(det.size.z), chamferRadius: 0)
            let mat = SCNMaterial(); mat.diffuse.contents = color.withAlphaComponent(0.15); mat.isDoubleSided = true
            box.materials = [mat]
            let node = SCNNode(geometry: box); node.simdWorldTransform = det.worldTransform
            addWireframe(to: node, size: det.size, color: color, radius: 0.004)
            let sz = measureUnit.formatBox(det.size.x, det.size.y, det.size.z)
            addLabel("\(det.label ?? "caja")\n\(sz)", to: node, offset: det.size.y/2+0.04)
            sceneView.scene.rootNode.addChildNode(node); boxNodes.append(node)
        }
        let summary = detections.map { measureUnit.formatBox($0.size.x, $0.size.y, $0.size.z) }.joined(separator:" | ")
        status = detections.isEmpty ? "Sin detecciones" : summary
        detections.forEach { self.detections.append(DetectionInfo(label: $0.label ?? "caja", size: $0.size, confidence: $0.confidence)) }
    }

    private func addWireframe(to parent: SCNNode, size: simd_float3, color: UIColor, radius: Float) {
        let hw=size.x/2, hh=size.y/2, hd=size.z/2
        let mat = SCNMaterial(); mat.diffuse.contents = color
        let edges: [(simd_float3,simd_float3)] = [
            ([-hw,-hh,-hd],[ hw,-hh,-hd]),([ hw,-hh,-hd],[ hw,-hh, hd]),
            ([ hw,-hh, hd],[-hw,-hh, hd]),([-hw,-hh, hd],[-hw,-hh,-hd]),
            ([-hw, hh,-hd],[ hw, hh,-hd]),([ hw, hh,-hd],[ hw, hh, hd]),
            ([ hw, hh, hd],[-hw, hh, hd]),([-hw, hh, hd],[-hw, hh,-hd]),
            ([-hw,-hh,-hd],[-hw, hh,-hd]),([ hw,-hh,-hd],[ hw, hh,-hd]),
            ([ hw,-hh, hd],[ hw, hh, hd]),([-hw,-hh, hd],[-hw, hh, hd]),
        ].map { (simd_float3($0.0), simd_float3($0.1)) }
        for (a,b) in edges {
            let cyl = SCNCylinder(radius: CGFloat(radius), height: CGFloat(simd_distance(a,b))); cyl.materials = [mat]
            let n = SCNNode(geometry: cyl); n.simdPosition = (a+b)/2
            let dir = simd_normalize(b-a), dot = simd_dot(simd_float3(0,1,0), dir)
            if abs(dot) < 0.999 { n.simdRotation = simd_float4(simd_normalize(simd_cross(simd_float3(0,1,0),dir)), acos(dot)) }
            parent.addChildNode(n)
        }
    }

    private func addLabel(_ text: String, to parent: SCNNode, offset: Float) {
        let t = SCNText(string: text, extrusionDepth: 0.005)
        t.font = UIFont.systemFont(ofSize: 0.04, weight: .bold)
        t.firstMaterial?.diffuse.contents = UIColor.white; t.flatness = 0.1
        let n = SCNNode(geometry: t); n.position = SCNVector3(-0.06, offset, 0)
        n.constraints = [SCNBillboardConstraint()]; parent.addChildNode(n)
    }

    private func detectPallet(frame: ARFrame, sceneView: ARSCNView) {
        guard let detector = palletDetector else { status = "PalletDetector no disponible"; return }
        isProcessing = true; status = "Detectando carga..."
        Task.detached {
            do {
                let conf = await MainActor.run { self.confidenceThreshold }
                let optMask = try await MainActor.run { try detector.detect(pixelBuffer: frame.capturedImage, confThreshold: conf) }
                guard let mask = optMask else {
                    await MainActor.run { self.status = "No se detectó carga"; self.isProcessing = false }; return
                }
                let nShots = 3; var shots: [Detection3D] = []
                for i in 1...nShots {
                    await MainActor.run { self.status = "Midiendo \(i)/\(nShots)..." }
                    let f = await MainActor.run { self.sceneView?.session.currentFrame }
                    if let f, f.sceneDepth != nil { let det = await MainActor.run { PalletMeasurer.measure(frame: f, mask: mask) }; if let det { shots.append(det) } }
                    if i < nShots { try? await Task.sleep(nanoseconds: 300_000_000) }
                }
                let finalPalletShots = shots
                await MainActor.run {
                    if finalPalletShots.isEmpty { self.status = "Sin geometría — acercate" }
                    else { self.placeBoxes([self.medianDetection(finalPalletShots)], in: sceneView) }
                    self.isProcessing = false
                }
            } catch { await MainActor.run { self.status = "Error: \(error.localizedDescription)"; self.isProcessing = false } }
        }
    }

    private func medianDetection(_ dets: [Detection3D]) -> Detection3D {
        let n = dets.count
        let xs = dets.map { $0.size.x }.sorted(), ys = dets.map { $0.size.y }.sorted(), zs = dets.map { $0.size.z }.sorted()
        let cx = dets.map { $0.center.x }.sorted()[n/2], cy = dets.map { $0.center.y }.sorted()[n/2], cz = dets.map { $0.center.z }.sorted()[n/2]
        let center = simd_float3(cx,cy,cz), medYaw = circularMedianAngle(dets.map { $0.yaw })
        let cosY = cos(medYaw), sinY = sin(medYaw)
        let wt = simd_float4x4(simd_float4(cosY,0,sinY,0), simd_float4(0,1,0,0), simd_float4(-sinY,0,cosY,0), simd_float4(cx,cy,cz,1))
        return Detection3D(center: center, size: simd_float3(xs[n/2],ys[n/2],zs[n/2]), yaw: medYaw,
                           confidence: dets.map { $0.confidence }.reduce(0,+)/Float(n), worldTransform: wt, label: dets.first?.label)
    }

    private func circularMedianAngle(_ angles: [Float]) -> Float {
        guard angles.count > 1 else { return angles.first ?? 0 }
        let s = angles.map { sin($0) }.reduce(0,+)/Float(angles.count)
        let c = angles.map { cos($0) }.reduce(0,+)/Float(angles.count)
        let mean = atan2(s,c)
        return angles.min(by: { abs($0-mean) < abs($1-mean) }) ?? mean
    }

    // MARK: - Undo / Clear

    func undoLast() { borrarMedicion() }

    func clearBoxes() { boxNodes.forEach { $0.removeFromParentNode() }; boxNodes.removeAll(); detections.removeAll() }

    func clearAll() {
        clearBoxes(); lastDetections3D.removeAll(); debugBBoxes.removeAll()
        clearCurrentMarkers()
        savedAncho = nil; savedLargo = nil; savedAlto = nil
        dimPhase = .ancho; tapPhase = .waitingFirst
        lastTapScreen = nil
        status = currentInstruction
    }

    // MARK: - Frame capture

    func captureCurrentFrame() -> Data? {
        guard let frame = sceneView?.session.currentFrame else { return nil }
        let ci = CIImage(cvPixelBuffer: frame.capturedImage).oriented(.right)
        let ctx = CIContext()
        guard let cg = ctx.createCGImage(ci, from: ci.extent) else { return nil }
        return UIImage(cgImage: cg).jpegData(compressionQuality: 0.75)
    }

    func captureAndUpload() {
        guard let frame = sceneView?.session.currentFrame else { status = "Sin frame AR"; return }
        guard !isProcessing else { return }
        isProcessing = true; status = "Subiendo foto..."
        Task.detached {
            do {
                let ci = CIImage(cvPixelBuffer: frame.capturedImage).oriented(.right)
                let ctx = CIContext()
                guard let cg = ctx.createCGImage(ci, from: ci.extent) else { throw UploadError.imageConversionFailed }
                guard let jpeg = UIImage(cgImage: cg).jpegData(compressionQuality: 0.85) else { throw UploadError.imageConversionFailed }
                try await DriveUploader.shared.upload(imageData: jpeg, mode: "CAJA")
                await MainActor.run { self.status = "Foto subida ✓"; self.isProcessing = false }
            } catch { await MainActor.run { self.status = "Error: \(error.localizedDescription)"; self.isProcessing = false } }
        }
    }

    enum UploadError: LocalizedError {
        case imageConversionFailed
        var errorDescription: String? { "No se pudo convertir el frame a JPEG" }
    }
}

// MARK: - Helpers

func pixelBufferToFloatArray(_ pb: CVPixelBuffer, targetSize: Int = 640) -> ([Float], Int, Int) {
    var ci = CIImage(cvPixelBuffer: pb); let ctx = CIContext()
    let w = ci.extent.width, h = ci.extent.height, s = min(w,h)
    ci = ci.cropped(to: CGRect(x:(w-s)/2, y:(h-s)/2, width:s, height:s))
    let r = ci.transformed(by: CGAffineTransform(scaleX: CGFloat(targetSize)/s, y: CGFloat(targetSize)/s))
    var rgba = [UInt8](repeating: 0, count: targetSize*targetSize*4)
    ctx.render(r, toBitmap: &rgba, rowBytes: targetSize*4,
               bounds: CGRect(x:r.extent.origin.x, y:r.extent.origin.y, width:CGFloat(targetSize), height:CGFloat(targetSize)),
               format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())
    let n = targetSize*targetSize; var out = [Float](repeating: 0, count: 3*n)
    for i in 0..<n { out[i]=Float(rgba[i*4])/255; out[n+i]=Float(rgba[i*4+1])/255; out[2*n+i]=Float(rgba[i*4+2])/255 }
    return (out, targetSize, targetSize)
}

func maskToUIImage(_ mask: CVPixelBuffer) -> UIImage? {
    let ci = CIImage(cvPixelBuffer: mask)
    guard let colorized = CIFilter(name: "CIColorMatrix", parameters: [
        kCIInputImageKey: ci,
        "inputRVector": CIVector(x: 1, y: 0, z: 0, w: 0),
        "inputGVector": CIVector(x: 1, y: 0, z: 0, w: 0),
        "inputBVector": CIVector(x: 0, y: 0, z: 0, w: 0),
        "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 0.55),
        "inputBiasVector": CIVector(x: 0, y: 0, z: 0, w: 0)
    ])?.outputImage else { return nil }
    return UIImage(ciImage: colorized.oriented(.right))
}

func extractDepthMap(_ buf: CVPixelBuffer) -> [[Float]] {
    CVPixelBufferLockBaseAddress(buf, .readOnly); defer { CVPixelBufferUnlockBaseAddress(buf, .readOnly) }
    let h=CVPixelBufferGetHeight(buf), w=CVPixelBufferGetWidth(buf), bpr=CVPixelBufferGetBytesPerRow(buf)
    let base=CVPixelBufferGetBaseAddress(buf)!
    var out = [[Float]](repeating:[Float](repeating:0,count:w),count:h)
    for y in 0..<h { let row=base.advanced(by:y*bpr).assumingMemoryBound(to:Float32.self); for x in 0..<w { out[y][x]=row[x] } }
    return out
}
