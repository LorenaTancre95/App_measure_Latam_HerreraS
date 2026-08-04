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
    @Published var debugInfo: String = ""
    @Published var instruction: String = ""
    @Published var crosshairHit: Bool = false
    @Published var liveAimPoint: simd_float3? = nil

    // MARK: - TAP mode: rectangle detection + 1 tap para LARGO

    enum RectMeasureState { case detecting, largoTap, done }

    /// Estado actual del flujo de medición
    @Published var rectState: RectMeasureState = .detecting
    /// Esquinas [TL, TR, BL, BR] del rectángulo detectado en coordenadas de pantalla (para overlay)
    @Published var liveRectScreen: [CGPoint]? = nil
    /// Dimensiones confirmadas tras CONFIRMAR
    @Published var confirmedAncho: Float? = nil
    @Published var confirmedAlto: Float? = nil
    @Published var confirmedLargo: Float? = nil

    private var liveRectCorners3D: [simd_float3]? = nil
    private var confirmedCorners3D: [simd_float3]? = nil

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
    private var samSegmenter: SAMSegmenter?
    private var boxNodes: [SCNNode] = []
    private var markerNodes: [SCNNode] = []
    private var lastDetections3D: [Detection3D] = []

    func setup(sceneView: ARSCNView) {
        self.sceneView = sceneView
        Task.detached { await self.loadTAPModels() }
    }

    nonisolated private func loadTAPModels() async {
        let pallet = try? PalletDetector()
        await MainActor.run {
            self.palletDetector = pallet
            self.status = "Apuntá la cara frontal de la caja"
        }
    }

    nonisolated private func loadYOLO() async {
        await MainActor.run { self.status = "Cargando detector CAJA..." }
        guard let yoloPath = Bundle.main.path(forResource: "best", ofType: "onnx") else {
            await MainActor.run { self.status = "best.onnx no encontrado" }
            return
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

        isProcessing = true
        status = "Detectando..."
        Task.detached {
            do {
                let (img, _, _) = pixelBufferToFloatArray(frame.capturedImage, targetSize: 640)
                let conf = await MainActor.run { self.confidenceThreshold }
                let yoloBoxes = try await MainActor.run {
                    try yoloDetector.detect(image: img, imageWidth: 640, imageHeight: 640, confThreshold: conf)
                }
                guard !yoloBoxes.isEmpty else {
                    await MainActor.run { self.status = "No cajas detectadas"; self.isProcessing = false }
                    return
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
                    return (CGRect(x: min(tl.x, br.x), y: min(tl.y, br.y),
                                  width: abs(br.x-tl.x), height: abs(br.y-tl.y)), box.score)
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
                let nShots = 5
                var shots: [Detection3D] = []
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
                    if finalShots.isEmpty {
                        self.status = "Sin geometría — apuntá más de frente o acercate"
                    } else if let sv = self.sceneView {
                        self.placeBoxes([self.medianDetection(finalShots)], in: sv)
                    }
                    self.isProcessing = false
                }
            } catch {
                await MainActor.run { self.status = "Error: \(error.localizedDescription)"; self.isProcessing = false }
            }
        }
    }

    private func medianDetection(_ dets: [Detection3D]) -> Detection3D {
        let n = dets.count
        let xs = dets.map { $0.size.x }.sorted()
        let ys = dets.map { $0.size.y }.sorted()
        let zs = dets.map { $0.size.z }.sorted()
        let medSize = simd_float3(xs[n/2], ys[n/2], zs[n/2])
        let cx = dets.map { $0.center.x }.sorted()[n/2]
        let cy = dets.map { $0.center.y }.sorted()[n/2]
        let cz = dets.map { $0.center.z }.sorted()[n/2]
        let center = simd_float3(cx, cy, cz)
        let medYaw = circularMedianAngle(dets.map { $0.yaw })
        let cosY = cos(medYaw), sinY = sin(medYaw)
        let worldTransform = simd_float4x4(
            simd_float4( cosY, 0, sinY, 0),
            simd_float4(    0, 1,    0, 0),
            simd_float4(-sinY, 0, cosY, 0),
            simd_float4(center.x, center.y, center.z, 1)
        )
        return Detection3D(center: center, size: medSize, yaw: medYaw,
                           confidence: dets.map { $0.confidence }.reduce(0,+) / Float(n),
                           worldTransform: worldTransform, label: dets.first?.label)
    }

    private func circularMedianAngle(_ angles: [Float]) -> Float {
        guard angles.count > 1 else { return angles.first ?? 0 }
        let sinMean = angles.map { sin($0) }.reduce(0,+) / Float(angles.count)
        let cosMean = angles.map { cos($0) }.reduce(0,+) / Float(angles.count)
        let mean = atan2(sinMean, cosMean)
        return angles.min(by: { abs($0 - mean) < abs($1 - mean) }) ?? mean
    }

    // MARK: - Rendering

    private func placeBoxes(_ detections: [Detection3D], in sceneView: ARSCNView) {
        lastDetections3D = detections
        clearBoxes()
        let colors: [UIColor] = [.systemGreen, .systemRed, .systemBlue]
        for (i, det) in detections.enumerated() {
            let color = colors[i % colors.count]
            let box = SCNBox(width: CGFloat(det.size.x), height: CGFloat(det.size.y), length: CGFloat(det.size.z), chamferRadius: 0)
            let mat = SCNMaterial(); mat.diffuse.contents = color.withAlphaComponent(0.15); mat.isDoubleSided = true
            box.materials = [mat]
            let node = SCNNode(geometry: box)
            node.simdWorldTransform = det.worldTransform
            addWireframe(to: node, size: det.size, color: color, radius: 0.004)
            let label = det.label ?? "caja"
            let sz = measureUnit.formatBox(det.size.x, det.size.y, det.size.z)
            addLabel("\(label)\n\(sz)", to: node, offset: det.size.y/2+0.04)
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
            let cyl = SCNCylinder(radius: CGFloat(radius), height: CGFloat(simd_distance(a,b)))
            cyl.materials = [mat]
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

    // MARK: - Pallet mode

    private func detectPallet(frame: ARFrame, sceneView: ARSCNView) {
        guard let detector = palletDetector else {
            status = "PalletDetector no disponible"; return
        }
        isProcessing = true; status = "Detectando carga..."
        Task.detached {
            do {
                let conf = await MainActor.run { self.confidenceThreshold }
                let optMask = try await MainActor.run {
                    try detector.detect(pixelBuffer: frame.capturedImage, confThreshold: conf)
                }
                guard let mask = optMask else {
                    await MainActor.run { self.status = "No se detectó carga"; self.isProcessing = false }
                    return
                }
                let nShots = 3; var shots: [Detection3D] = []
                for i in 1...nShots {
                    await MainActor.run { self.status = "Midiendo \(i)/\(nShots)..." }
                    let f = await MainActor.run { self.sceneView?.session.currentFrame }
                    if let f, f.sceneDepth != nil {
                        let det = await MainActor.run { PalletMeasurer.measure(frame: f, mask: mask) }
                        if let det { shots.append(det) }
                    }
                    if i < nShots { try? await Task.sleep(nanoseconds: 300_000_000) }
                }
                await MainActor.run {
                    if shots.isEmpty { self.status = "Sin geometría — acercate más" }
                    else { self.placeBoxes([self.medianDetection(shots)], in: sceneView) }
                    self.isProcessing = false
                }
            } catch {
                await MainActor.run { self.status = "Error: \(error.localizedDescription)"; self.isProcessing = false }
            }
        }
    }

    // MARK: - Rectangle measurement (TAP mode)
    //
    // Flujo:
    // 1. ARViewContainer corre VNDetectRectanglesRequest cada 0.3s → llama updateLiveRect()
    // 2. Usuario presiona CONFIRMAR → confirmFaceRect() → ANCHO+ALTO desde las 4 esquinas LiDAR
    // 3. Usuario toca 1 punto en el borde lejano del techo → measureLargoAt() → LARGO
    //
    // ANCHO = promedio de dist(TL,TR) y dist(BL,BR)
    // ALTO  = promedio de dist(TL,BL) y dist(TR,BR)
    // LARGO = proyección del punto lejano sobre la normal del plano frontal

    /// Llamado por el coordinator con el rectángulo detectado en cada frame.
    /// corners3D: [TL, TR, BL, BR] en espacio mundial · cornersScreen: mismas en píxeles pantalla.
    func updateLiveRect(corners3D: [simd_float3], cornersScreen: [CGPoint]) {
        guard rectState == .detecting else { return }
        liveRectCorners3D = corners3D
        liveRectScreen = cornersScreen
    }

    func clearLiveRect() {
        guard rectState == .detecting else { return }
        liveRectCorners3D = nil
        liveRectScreen = nil
    }

    /// Bloquea el rectángulo actual y calcula ANCHO + ALTO desde las esquinas 3D.
    func confirmFaceRect() {
        guard let corners = liveRectCorners3D, corners.count == 4 else { return }
        confirmedCorners3D = corners
        let tl = corners[0], tr = corners[1], bl = corners[2], br = corners[3]
        confirmedAncho = (simd_distance(tl, tr) + simd_distance(bl, br)) / 2
        confirmedAlto  = (simd_distance(tl, bl) + simd_distance(tr, br)) / 2
        confirmedLargo = nil
        liveRectScreen = nil
        rectState = .largoTap
        status = "Tocá el borde lejano del techo"
    }

    /// 1 tap en el borde lejano del techo → LARGO por proyección sobre la normal del plano frontal.
    func measureLargoAt(point: CGPoint) {
        guard rectState == .largoTap,
              let corners = confirmedCorners3D, corners.count == 4 else { return }
        guard let pt3D = tapTo3D(point: point) else {
            status = "Sin superficie — apuntá directamente sobre la caja"
            return
        }
        if let sv = sceneView { placeMarker(at: pt3D, in: sv) }

        let tl = corners[0], tr = corners[1], bl = corners[2]
        let topCenter = (tl + tr) / 2
        // Normal del plano frontal (producto cruz de los dos bordes)
        let normal = simd_normalize(simd_cross(tr - tl, bl - tl))
        let largo = max(0.03, abs(simd_dot(pt3D - topCenter, normal)))
        confirmedLargo = largo

        let ancho = confirmedAncho ?? 0.3
        let alto  = confirmedAlto  ?? 0.2
        let det = DetectionInfo(label: "caja",
                                size: simd_float3(largo, alto, ancho),
                                confidence: 1.0)
        detections.append(det)
        status = "✓ \(measureUnit.formatBox(det.size.x, det.size.y, det.size.z))"
        rectState = .done
    }

    // MARK: - Utilities LiDAR / AR

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
        let centerRow = base.advanced(by: tapDY * rb).assumingMemoryBound(to: Float32.self)
        let centerD   = centerRow[tapDX]
        var chosenD   = Float.infinity
        var chosenDX  = tapDX, chosenDY = tapDY
        if centerD > 0.05, centerD < 8.0 {
            chosenD = centerD
        } else {
            for dy in -1...1 {
                let py  = max(0, min(dH - 1, tapDY + dy))
                let row = base.advanced(by: py * rb).assumingMemoryBound(to: Float32.self)
                for dx in -1...1 {
                    let px = max(0, min(dW - 1, tapDX + dx))
                    let val = row[px]
                    if val > 0.05, val < 8.0, val < chosenD { chosenD = val; chosenDX = px; chosenDY = py }
                }
            }
        }
        guard chosenD < Float.infinity else { return nil }
        let intr = frame.camera.intrinsics
        let fx = intr[0][0], fy = intr[1][1], cx = intr[2][0], cy = intr[2][1]
        let ix  = Float(chosenDX) / Float(dW) * bufW
        let iy  = Float(chosenDY) / Float(dH) * bufH
        let cam = simd_float4((ix - cx) / fx * chosenD, (iy - cy) / fy * chosenD, -chosenD, 1)
        let w   = frame.camera.transform * cam
        return simd_float3(w.x, w.y, w.z) / w.w
    }

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

    private func placeMarker(at position: simd_float3, in sceneView: ARSCNView) {
        let sphere = SCNSphere(radius: 0.006)
        let mat = SCNMaterial(); mat.diffuse.contents = UIColor.systemCyan
        sphere.materials = [mat]
        let node = SCNNode(geometry: sphere); node.simdPosition = position
        sceneView.scene.rootNode.addChildNode(node); markerNodes.append(node)
    }

    func clearBoxes() {
        boxNodes.forEach { $0.removeFromParentNode() }
        boxNodes.removeAll()
        detections.removeAll()
    }

    func undoLast() {
        switch rectState {
        case .done:
            clearBoxes()
            markerNodes.forEach { $0.removeFromParentNode() }
            markerNodes.removeAll()
            confirmedLargo = nil
            rectState = .largoTap
            status = "Tocá el borde lejano del techo"
        case .largoTap:
            confirmedCorners3D = nil
            confirmedAncho = nil
            confirmedAlto = nil
            rectState = .detecting
            status = "Apuntá la cara frontal de la caja"
        case .detecting:
            clearAll()
        }
    }

    func clearAll() {
        clearBoxes()
        lastDetections3D.removeAll()
        debugBBoxes.removeAll()
        debugInfo = ""
        markerNodes.forEach { $0.removeFromParentNode() }
        markerNodes.removeAll()
        liveRectCorners3D = nil
        liveRectScreen = nil
        confirmedCorners3D = nil
        confirmedAncho = nil
        confirmedAlto = nil
        confirmedLargo = nil
        rectState = .detecting
        status = "Apuntá la cara frontal de la caja"
    }

    // MARK: - Frame capture

    func captureCurrentFrame() -> Data? {
        guard let frame = sceneView?.session.currentFrame else { return nil }
        let ci = CIImage(cvPixelBuffer: frame.capturedImage).oriented(.right)
        let ctx = CIContext()
        guard let cg = ctx.createCGImage(ci, from: ci.extent) else { return nil }
        return UIImage(cgImage: cg).jpegData(compressionQuality: 0.75)
    }

    // MARK: - Dataset capture

    func captureAndUpload() {
        guard let frame = sceneView?.session.currentFrame else { status = "Sin frame AR"; return }
        guard !isProcessing else { return }
        let mode = measureMode == .box ? "CAJA" : "OVERSIZE"
        isProcessing = true; status = "Subiendo foto..."
        Task.detached {
            do {
                let ci = CIImage(cvPixelBuffer: frame.capturedImage).oriented(.right)
                let ctx = CIContext()
                guard let cg = ctx.createCGImage(ci, from: ci.extent) else { throw UploadError.imageConversionFailed }
                let uiImg = UIImage(cgImage: cg)
                guard let jpeg = uiImg.jpegData(compressionQuality: 0.85) else { throw UploadError.imageConversionFailed }
                try await DriveUploader.shared.upload(imageData: jpeg, mode: mode)
                await MainActor.run { self.status = "Foto subida a Drive ✓"; self.isProcessing = false }
            } catch {
                await MainActor.run { self.status = "Error: \(error.localizedDescription)"; self.isProcessing = false }
            }
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
