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
    @Published var segmentationOverlay: UIImage? = nil   // shown during TAP mode preview
    @Published var debugInfo: String = ""               // visible debug panel (no Xcode needed)
    @Published var cornerInstruction: String = "1/4 — Tocá la esquina superior IZQUIERDA (frente)"
    @Published var frozenFrameImage: UIImage? = nil   // frozen photo shown while user taps corners

    // 0 = nothing tapped yet, 1 = P0 done, 2 = P0+P1 done
    var cornerStep: Int {
        if cornerInstruction.hasPrefix("2/") { return 1 }
        if cornerInstruction.hasPrefix("3/") { return 2 }
        return 0
    }

    enum MeasureMode { case box, oversize, tap }

    var sceneView: ARSCNView?
    var viewportSize: CGSize = UIScreen.main.bounds.size
    let viewfinderNorm = CGRect(x: 0.1, y: 0.18, width: 0.8, height: 0.64)
    private var yoloDetector: YOLODetector?
    private var palletDetector: PalletDetector?
    private var samSegmenter: SAMSegmenter?
    private var boxNodes: [SCNNode] = []
    private var cornerPts: [simd_float3] = []
    private var markerNodes: [SCNNode] = []
    // Frozen-frame data saved at captureFrame() time
    private var frozenDepthMap: CVPixelBuffer? = nil
    private var frozenCameraTransform: simd_float4x4 = matrix_identity_float4x4
    private var frozenIntrinsics: simd_float3x3 = matrix_identity_float3x3
    private var frozenDisplayTransform: CGAffineTransform = .identity
    private var frozenCapturedSize: CGSize = .zero

    func setup(sceneView: ARSCNView) {
        self.sceneView = sceneView
        Task.detached { await self.loadTAPModels() }
    }

    nonisolated private func loadTAPModels() async {
        let pallet = try? PalletDetector()
        await MainActor.run {
            self.palletDetector = pallet
            self.status = "Encuadrá la caja y presioná CAPTURAR"
        }
    }

    // Loads YOLO for CAJA mode. Frees SAM first to make room.
    nonisolated private func loadYOLO() async {
        await MainActor.run {
            self.samSegmenter = nil   // free SAM memory before ONNX runtime loads
            self.status = "Cargando detector CAJA..."
        }
        guard let yoloPath = Bundle.main.path(forResource: "best", ofType: "onnx") else {
            await MainActor.run { self.status = "best.onnx no encontrado" }
            return
        }
        do {
            let yolo = try YOLODetector(modelPath: yoloPath)
            await MainActor.run {
                self.yoloDetector = yolo
                self.status = "Apuntá la caja al visor"
            }
        } catch {
            await MainActor.run { self.status = "YOLO falló: \(error.localizedDescription)" }
        }
    }

    func detectNow() {
        guard let sceneView, let frame = sceneView.session.currentFrame else { status = "Not ready"; return }
        guard frame.sceneDepth != nil else { status = "No LiDAR depth"; return }

        if measureMode == .oversize {
            detectPallet(frame: frame, sceneView: sceneView); return
        }

        // Lazy-load YOLO on first CAJA use (frees SAM first)
        guard let yoloDetector else {
            Task.detached { await self.loadYOLO() }
            return
        }
        let hasFloor = frame.anchors.contains { ($0 as? ARPlaneAnchor)?.alignment == .horizontal }
        isProcessing = true
        status = hasFloor ? "Detectando (piso ✓)..." : "Detectando (sin piso aún)..."
        Task.detached {
            do {
                // ── YOLO: una sola vez sobre el frame actual ───────────
                let (img, _, _) = pixelBufferToFloatArray(frame.capturedImage, targetSize: 640)
                let conf = await MainActor.run { self.confidenceThreshold }
                let yoloBoxes = try yoloDetector.detect(image: img, imageWidth: 640, imageHeight: 640, confThreshold: conf)
                guard !yoloBoxes.isEmpty else {
                    await MainActor.run { self.status = "No cajas detectadas"; self.isProcessing = false }
                    return
                }
                let topBoxes = Array(yoloBoxes.sorted { $0.score > $1.score }.prefix(5))

                // ── Overlay 2D ─────────────────────────────────────────
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

                // ── Viewfinder filter ──────────────────────────────────
                let vfN = await MainActor.run { self.viewfinderNorm }
                let vfR = CGRect(x: vfN.minX*vp.width, y: vfN.minY*vp.height,
                                 width: vfN.width*vp.width, height: vfN.height*vp.height)
                let vfPassed: [YOLOBox] = zip(topBoxes, screenBoxes).compactMap { box, scr in
                    vfR.contains(CGPoint(x: scr.rect.midX, y: scr.rect.midY)) ? box : nil
                }
                guard let best = vfPassed.first ?? topBoxes.first else {
                    await MainActor.run { self.isProcessing = false }; return
                }
                if vfPassed.isEmpty { await MainActor.run { self.status = "Apuntá la caja al viewfinder" } }

                // ── Multi-shot: BoxMeasurer2 × 5 frames, mediana ───────
                // YOLO ya localizó la caja; ahora medimos 5 veces con LiDAR
                // en frames consecutivos para promediar el ruido.
                let nShots = 5
                var shots: [Detection3D] = []
                for i in 1...nShots {
                    await MainActor.run { self.status = "Midiendo \(i)/\(nShots)..." }
                    let f = await MainActor.run { self.sceneView?.session.currentFrame }
                    if let f, f.sceneDepth != nil,
                       let det = BoxMeasurer2.measure(frame: f, yoloBox: best) {
                        shots.append(det)
                    }
                    if i < nShots { try? await Task.sleep(nanoseconds: 250_000_000) }
                }

                await MainActor.run {
                    if shots.isEmpty {
                        self.status = "Sin geometría — apuntá más de frente o acercate"
                    } else if let sv = self.sceneView {
                        self.placeBoxes([self.medianDetection(shots)], in: sv)
                    }
                    self.isProcessing = false
                }
            } catch {
                await MainActor.run { self.status = "Error: \(error.localizedDescription)"; self.isProcessing = false }
            }
        }
    }

    // Combina N mediciones tomando la mediana de cada dimensión y del ángulo.
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

    // Mediana circular: promedia sin/cos, devuelve la muestra más cercana a ese ángulo medio.
    private func circularMedianAngle(_ angles: [Float]) -> Float {
        guard angles.count > 1 else { return angles.first ?? 0 }
        let sinMean = angles.map { sin($0) }.reduce(0,+) / Float(angles.count)
        let cosMean = angles.map { cos($0) }.reduce(0,+) / Float(angles.count)
        let mean = atan2(sinMean, cosMean)
        return angles.min(by: { abs($0 - mean) < abs($1 - mean) }) ?? mean
    }

    // MARK: - Rendering

    private func placeBoxes(_ detections: [Detection3D], in sceneView: ARSCNView) {
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
            let sz = String(format: "%.0fx%.0fx%.0f cm", det.size.x*100, det.size.y*100, det.size.z*100)
            addLabel("\(label)\n\(sz)", to: node, offset: det.size.y/2+0.04)
            sceneView.scene.rootNode.addChildNode(node); boxNodes.append(node)
        }
        let summary = detections.map { String(format:"%.0fx%.0fx%.0f cm",$0.size.x*100,$0.size.y*100,$0.size.z*100) }.joined(separator:" | ")
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

    // MARK: - Pallet mode (SAM segmentation + LiDAR bbox)

    private func detectPallet(frame: ARFrame, sceneView: ARSCNView) {
        guard let detector = palletDetector else {
            status = "PalletDetector no disponible — revisá pallet_seg.mlpackage"; return
        }
        isProcessing = true
        status = "Detectando carga..."

        Task.detached {
            do {
                let conf = await MainActor.run { self.confidenceThreshold }
                guard let mask = try detector.detect(pixelBuffer: frame.capturedImage, confThreshold: conf) else {
                    await MainActor.run { self.status = "No se detectó carga (bajá el umbral de confianza)"; self.isProcessing = false }
                    return
                }

                // Multi-shot: medir 3 veces con LiDAR y tomar mediana
                let nShots = 3
                var shots: [Detection3D] = []
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
                    if shots.isEmpty {
                        self.status = "Sin geometría — apuntá más cerca o acercate"
                    } else {
                        self.placeBoxes([self.medianDetection(shots)], in: sceneView)
                    }
                    self.isProcessing = false
                }
            } catch {
                await MainActor.run {
                    self.status = "Error SAM: \(error.localizedDescription)"
                    self.isProcessing = false
                }
            }
        }
    }

    // MARK: - Freeze-frame capture

    /// Freezes the current ARKit frame (RGB + depth + camera data).
    /// After calling this, the user taps corners on the frozen photo; measureAtTap uses the saved data.
    func captureFrame() {
        guard let sceneView, let frame = sceneView.session.currentFrame else {
            status = "Sin frame AR"; return
        }
        guard let depthMap = frame.sceneDepth?.depthMap else {
            status = "Sin LiDAR depth — esperá que se calibre"; return
        }

        clearAll()  // remove overlays before snapshot so they don't appear on the frozen image

        // sceneView.snapshot() captures exactly what ARKit renders on screen.
        // This guarantees that a tap at (tx, ty) on the frozen image maps to the same
        // screen coordinate as displayTransform.inverted() expects — no aspect-ratio mismatch.
        let screenShot = sceneView.snapshot()

        // ARKit recycles pixel buffers between frames — copy the depth map so
        // the background LiDAR scan can access it safely after the frame is gone.
        frozenDepthMap         = ARViewModel.copyPixelBuffer(depthMap)
        frozenCameraTransform  = frame.camera.transform
        frozenIntrinsics       = frame.camera.intrinsics
        frozenDisplayTransform = frame.displayTransform(for: .portrait, viewportSize: viewportSize)
        frozenCapturedSize     = CGSize(width:  CVPixelBufferGetWidth(frame.capturedImage),
                                        height: CVPixelBufferGetHeight(frame.capturedImage))
        frozenFrameImage       = screenShot

        status             = "Foto congelada ✓"
        cornerInstruction  = "1/3 — Esquina sup. IZQUIERDA"
    }

    // MARK: - Corner tap measurement (3 taps on frozen photo → OBB with auto depth)
    //
    // Tap order (tapped on the frozen image displayed on screen):
    //   P0: front-top-LEFT
    //   P1: front-top-RIGHT  → width = |P1−P0|
    //   P2: front-bottom-RIGHT → height = |P2−P1|
    //
    // After 3 taps, depth is computed automatically by scanning the frozen LiDAR map
    // for points behind the front face within its width×height footprint.

    func measureAtTap(at point: CGPoint) {
        guard !isProcessing else { return }
        guard let sceneView else { status = "Sin AR view"; return }

        guard frozenDepthMap != nil else {
            status = "Primero presioná CAPTURAR"
            return
        }

        guard let pt3D = tapTo3DFrozen(point: point) else {
            status = "Sin profundidad — tocá sobre la caja (no el fondo)"
            return
        }

        let n = cornerPts.count
        cornerPts.append(pt3D)
        placeMarker(at: pt3D, in: sceneView)
        debugInfo += "P\(n): (\(String(format:"%.2f",pt3D.x)),\(String(format:"%.2f",pt3D.y)),\(String(format:"%.2f",pt3D.z))m)\n"

        switch cornerPts.count {
        case 1:
            debugInfo = "P0: (\(String(format:"%.2f",pt3D.x)),\(String(format:"%.2f",pt3D.y)),\(String(format:"%.2f",pt3D.z))m)\n"
            cornerInstruction = "2/3 — Esquina sup. DERECHA"
            status = "Esquina 1 marcada ✓"
        case 2:
            cornerInstruction = "3/3 — Esquina inf. DERECHA"
            status = "Esquina 2 marcada ✓"
        case 3:
            cornerInstruction = ""
            isProcessing = true
            status = "Calculando profundidad..."
            // Capture all data on main thread before going to background —
            // avoids actor-isolation issues and ensures safe buffer access.
            let pts  = cornerPts
            let sv   = sceneView
            let dBuf = frozenDepthMap          // already a copy made in captureFrame()
            let bSz  = frozenCapturedSize
            let intr = frozenIntrinsics
            let camT = frozenCameraTransform
            Task.detached { [weak self] in
                let depth = ARViewModel.scanDepth(
                    cornerPts: pts, depthBuffer: dBuf,
                    bufSize: bSz, intrinsics: intr, cameraT: camT)
                let det = ARViewModel.buildBox(cornerPts: pts, depth: depth)
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    if let det {
                        self.placeBoxes([det], in: sv)
                        self.frozenFrameImage = nil
                        self.frozenDepthMap   = nil
                    } else {
                        self.status = "Sin datos LiDAR — acercate más e intentá de nuevo"
                        self.cornerInstruction = "1/3 — Esquina sup. IZQUIERDA"
                    }
                    self.cornerPts.removeAll()
                    self.markerNodes.forEach { $0.removeFromParentNode() }
                    self.markerNodes.removeAll()
                    self.isProcessing = false
                }
            }
        default:
            break
        }
    }

    // Scans the copied LiDAR depth buffer and returns the max depth projection
    // in the face-normal direction within the front face's width×height footprint.
    // Pure static function — no actor isolation, safe to call from Task.detached.
    nonisolated private static func scanDepth(
        cornerPts: [simd_float3],
        depthBuffer: CVPixelBuffer?,
        bufSize: CGSize,
        intrinsics: simd_float3x3,
        cameraT: simd_float4x4
    ) -> Float {
        guard let dBuf = depthBuffer, cornerPts.count >= 3 else { return 0 }

        let P0 = cornerPts[0], P1 = cornerPts[1], P2 = cornerPts[2]
        let widthVec  = P1 - P0
        let heightVec = P2 - P1
        let width  = simd_length(widthVec)
        let height = simd_length(heightVec)
        guard width > 0.01, height > 0.01 else { return 0 }

        let widthAxis   = simd_normalize(widthVec)
        let heightAxis  = simd_normalize(heightVec)
        let depthAxis   = simd_normalize(simd_cross(widthAxis, heightAxis))
        let frontCenter = (P0 + P2) * 0.5

        let bufW = Float(bufSize.width)
        let bufH = Float(bufSize.height)
        let dW   = CVPixelBufferGetWidth(dBuf)
        let dH   = CVPixelBufferGetHeight(dBuf)
        let fx   = intrinsics[0][0], fy = intrinsics[1][1]
        let cx   = intrinsics[2][0], cy = intrinsics[2][1]
        // Allow 1.5× face extents to catch back corners even with slight tap error
        let halfW = width  * 0.75
        let halfH = height * 0.75

        CVPixelBufferLockBaseAddress(dBuf, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(dBuf, .readOnly) }
        let rb   = CVPixelBufferGetBytesPerRow(dBuf)
        guard let base = CVPixelBufferGetBaseAddress(dBuf) else { return 0 }

        var maxProj: Float = 0
        for py in 0..<dH {
            let row = base.advanced(by: py * rb).assumingMemoryBound(to: Float32.self)
            for px in 0..<dW {
                let d = row[px]
                guard d > 0.05, d < 8.0 else { continue }
                let ix  = Float(px) / Float(dW) * bufW
                let iy  = Float(py) / Float(dH) * bufH
                let cam = simd_float4((ix - cx) / fx * d, (iy - cy) / fy * d, -d, 1)
                let ww  = cameraT * cam
                let p3d = simd_float3(ww.x, ww.y, ww.z) / ww.w
                let dp  = p3d - frontCenter
                let wP  = simd_dot(dp, widthAxis)
                let hP  = simd_dot(dp, heightAxis)
                let dP  = simd_dot(dp, depthAxis)
                guard abs(wP) < halfW, abs(hP) < halfH,
                      dP > 0.01, dP < 3.0 else { continue }
                if dP > maxProj { maxProj = dP }
            }
        }
        return maxProj
    }

    nonisolated private static func buildBox(cornerPts: [simd_float3], depth: Float) -> Detection3D? {
        guard cornerPts.count >= 3, depth > 0.01 else { return nil }
        let P0 = cornerPts[0], P1 = cornerPts[1], P2 = cornerPts[2]
        let widthVec  = P1 - P0
        let heightVec = P2 - P1
        let width  = simd_length(widthVec)
        let height = simd_length(heightVec)
        guard width > 0.01, height > 0.01 else { return nil }
        let widthAxis  = simd_normalize(widthVec)
        let heightAxis = simd_normalize(heightVec)
        let depthAxis  = simd_normalize(simd_cross(widthAxis, heightAxis))
        let frontCenter = (P0 + P2) * 0.5
        let center = frontCenter + depthAxis * depth * 0.5
        let yaw  = atan2(widthAxis.z, widthAxis.x)
        let cosY = cos(yaw), sinY = sin(yaw)
        let worldTransform = simd_float4x4(
            simd_float4( cosY, 0, sinY, 0),
            simd_float4(    0, 1,    0, 0),
            simd_float4(-sinY, 0, cosY, 0),
            simd_float4(center.x, center.y, center.z, 1)
        )
        return Detection3D(center: center,
                           size: simd_float3(width, height, depth),
                           yaw: yaw, confidence: 1.0,
                           worldTransform: worldTransform, label: "caja")
    }

    // Deep-copies a CVPixelBuffer so ARKit's buffer recycling cannot corrupt it.
    nonisolated private static func copyPixelBuffer(_ src: CVPixelBuffer) -> CVPixelBuffer? {
        let fmt = CVPixelBufferGetPixelFormatType(src)
        let w   = CVPixelBufferGetWidth(src)
        let h   = CVPixelBufferGetHeight(src)
        var dst: CVPixelBuffer?
        guard CVPixelBufferCreate(kCFAllocatorDefault, w, h, fmt, nil, &dst) == kCVReturnSuccess,
              let dst else { return nil }
        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(dst, [])
        memcpy(CVPixelBufferGetBaseAddress(dst)!,
               CVPixelBufferGetBaseAddress(src)!,
               CVPixelBufferGetBytesPerRow(src) * h)
        CVPixelBufferUnlockBaseAddress(dst, [])
        CVPixelBufferUnlockBaseAddress(src, .readOnly)
        return dst
    }

    private func tapTo3DFrozen(point: CGPoint) -> simd_float3? {
        guard let depthBuffer = frozenDepthMap else { return nil }

        let bufW = Float(frozenCapturedSize.width)
        let bufH = Float(frozenCapturedSize.height)
        let dW   = CVPixelBufferGetWidth(depthBuffer)
        let dH   = CVPixelBufferGetHeight(depthBuffer)

        let invertedT = frozenDisplayTransform.inverted()
        let normTap   = CGPoint(x: point.x / viewportSize.width, y: point.y / viewportSize.height)
        let normImg   = normTap.applying(invertedT)
        let tapDX     = max(0, min(dW - 1, Int(Float(normImg.x) * Float(dW))))
        let tapDY     = max(0, min(dH - 1, Int(Float(normImg.y) * Float(dH))))

        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }
        let rb   = CVPixelBufferGetBytesPerRow(depthBuffer)
        guard let base = CVPixelBufferGetBaseAddress(depthBuffer) else { return nil }

        // 5×5 neighbourhood: minimum valid depth = closest surface (the box, not the background)
        var bestD  = Float.infinity
        var bestDX = tapDX, bestDY = tapDY
        for dy in -2...2 {
            let py = max(0, min(dH - 1, tapDY + dy))
            let row = base.advanced(by: py * rb).assumingMemoryBound(to: Float32.self)
            for dx in -2...2 {
                let px = max(0, min(dW - 1, tapDX + dx))
                let val = row[px]
                if val > 0.05, val < 8.0, val < bestD { bestD = val; bestDX = px; bestDY = py }
            }
        }
        guard bestD < Float.infinity else { return nil }

        let intr = frozenIntrinsics
        let fx = intr[0][0], fy = intr[1][1], cx = intr[2][0], cy = intr[2][1]
        let ix  = Float(bestDX) / Float(dW) * bufW
        let iy  = Float(bestDY) / Float(dH) * bufH
        let cam = simd_float4((ix - cx) / fx * bestD, (iy - cy) / fy * bestD, -bestD, 1)
        let w   = frozenCameraTransform * cam
        return simd_float3(w.x, w.y, w.z) / w.w
    }

    // Live tapTo3D (kept for reference, not used in frozen flow)
    private func tapTo3D(point: CGPoint, frame: ARFrame) -> simd_float3? {
        guard let depthBuffer = frame.sceneDepth?.depthMap else { return nil }

        let bufW = Float(CVPixelBufferGetWidth(frame.capturedImage))
        let bufH = Float(CVPixelBufferGetHeight(frame.capturedImage))
        let dW   = CVPixelBufferGetWidth(depthBuffer)
        let dH   = CVPixelBufferGetHeight(depthBuffer)

        let displayT  = frame.displayTransform(for: .portrait, viewportSize: viewportSize)
        let invertedT = displayT.inverted()
        let normTap   = CGPoint(x: point.x / viewportSize.width, y: point.y / viewportSize.height)
        let normImg   = normTap.applying(invertedT)
        let tapDX     = max(0, min(dW - 1, Int(Float(normImg.x) * Float(dW))))
        let tapDY     = max(0, min(dH - 1, Int(Float(normImg.y) * Float(dH))))

        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }
        let rb   = CVPixelBufferGetBytesPerRow(depthBuffer)
        guard let base = CVPixelBufferGetBaseAddress(depthBuffer) else { return nil }

        // Sample 5×5 neighbourhood around tap; use the CLOSEST valid depth.
        // LiDAR at box edges/corners reads the background (wall, floor) in mixed pixels.
        // The minimum valid depth selects the foreground surface (the box itself).
        var bestD  = Float.infinity
        var bestDX = tapDX, bestDY = tapDY
        for dy in -2...2 {
            let py = max(0, min(dH - 1, tapDY + dy))
            let row = base.advanced(by: py * rb).assumingMemoryBound(to: Float32.self)
            for dx in -2...2 {
                let px = max(0, min(dW - 1, tapDX + dx))
                let val = row[px]
                if val > 0.05, val < 8.0, val < bestD { bestD = val; bestDX = px; bestDY = py }
            }
        }
        guard bestD < Float.infinity else { return nil }

        let intr = frame.camera.intrinsics
        let fx = intr[0][0], fy = intr[1][1], cx = intr[2][0], cy = intr[2][1]
        let ix  = Float(bestDX) / Float(dW) * bufW
        let iy  = Float(bestDY) / Float(dH) * bufH
        let cam = simd_float4((ix - cx) / fx * bestD, (iy - cy) / fy * bestD, -bestD, 1)
        let w   = frame.camera.transform * cam
        return simd_float3(w.x, w.y, w.z) / w.w
    }


    private func placeMarker(at position: simd_float3, in sceneView: ARSCNView) {
        let sphere = SCNSphere(radius: 0.012)
        let mat = SCNMaterial()
        mat.diffuse.contents = UIColor.systemGreen
        sphere.materials = [mat]
        let node = SCNNode(geometry: sphere)
        node.simdPosition = position
        sceneView.scene.rootNode.addChildNode(node)
        markerNodes.append(node)
    }

    func floorDetected() {
        guard !isCalibrated else { return }
        isCalibrated = true
        if !isProcessing { status = "Piso calibrado ✓ — apuntá la caja al visor" }
    }

    func clearBoxes() { boxNodes.forEach { $0.removeFromParentNode() }; boxNodes.removeAll(); detections.removeAll() }
    func clearAll() {
        clearBoxes()
        debugBBoxes.removeAll()
        debugInfo = ""
        cornerPts.removeAll()
        markerNodes.forEach { $0.removeFromParentNode() }
        markerNodes.removeAll()
        frozenFrameImage = nil
        frozenDepthMap   = nil
        cornerInstruction = "1/3 — Esquina sup. IZQUIERDA"
    }

    // MARK: - Dataset capture

    func captureAndUpload() {
        guard let frame = sceneView?.session.currentFrame else { status = "Sin frame AR"; return }
        guard !isProcessing else { return }
        let mode = measureMode == .box ? "CAJA" : "OVERSIZE"
        isProcessing = true
        status = "Subiendo foto..."

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
                await MainActor.run { self.status = "Error subida: \(error.localizedDescription)"; self.isProcessing = false }
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

/// Renders a Vision segmentation mask as a yellow-tinted UIImage for debug overlay.
func maskToUIImage(_ mask: CVPixelBuffer) -> UIImage? {
    let ci = CIImage(cvPixelBuffer: mask)
    // Tint yellow: multiply R and G channels, zero B
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
