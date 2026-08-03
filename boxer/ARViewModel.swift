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
    @Published var cornerInstruction: String = "1/4 — Esquina sup. IZQUIERDA"
    @Published var crosshairHit: Bool = false          // surface detected at center crosshair
    @Published var isSnapping: Bool = false            // crosshair snapped to a feature point
    @Published var liveAimPoint: simd_float3? = nil   // current 3D position under crosshair
    @Published var lastCornerScreen: CGPoint? = nil   // last placed corner projected to 2D screen

    // Real-time distance from last placed corner to current crosshair aim
    var liveDistance: Float? {
        guard let last = cornerPts.last, let aim = liveAimPoint else { return nil }
        return simd_distance(last, aim)
    }

    // 0=none, 1=P0, 2=P0+P1, 3=P0..P2, 4=all 4 placed (awaiting confirm)
    var cornerStep: Int {
        if cornerInstruction.hasPrefix("2/") { return 1 }
        if cornerInstruction.hasPrefix("3/") { return 2 }
        if cornerInstruction.hasPrefix("4/") { return 3 }
        if cornerInstruction.hasPrefix("✓")  { return 4 }
        return 0
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
            // Rebuild 3D labels and status when unit changes
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
    private(set) var cornerPts: [simd_float3] = []
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
            self.status = "Apuntá a la caja y tocá las esquinas"
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
                let yoloBoxes = try await MainActor.run {
                    try yoloDetector.detect(image: img, imageWidth: 640, imageHeight: 640, confThreshold: conf)
                }
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
                let optMask = try await MainActor.run {
                    try detector.detect(pixelBuffer: frame.capturedImage, confThreshold: conf)
                }
                guard let mask = optMask else {
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

    // MARK: - Corner tap measurement (4 taps on live view → exact OBB)
    //
    // Like the iPhone Measure app: tap on the live AR view, ARKit raycasts the surface
    // and returns a 3D world-space point. Points are stable across phone movements
    // because they're in ARKit's world coordinate frame.
    //
    // Tap order:
    //   P0: front-top-LEFT
    //   P1: front-top-RIGHT  → width  = |P1−P0|
    //   P2: front-bottom-RIGHT → height = |P2−P1|
    //   P3: any point on side or top face → depth = |dot(P3−P0, cross(widthAxis,heightAxis))|

    // Captures the 3D point at the screen center (crosshair aim-and-capture flow).
    func captureCenter() {
        let center = CGPoint(x: viewportSize.width / 2, y: viewportSize.height / 2)
        measureAtTap(at: center)
    }

    func measureAtTap(at point: CGPoint) {
        guard !isProcessing else { return }
        guard let sceneView else { status = "Sin AR view"; return }

        guard let pt3D = tapTo3D(point: point) else {
            status = "Sin superficie — tocá directamente sobre la caja"
            return
        }

        let n = cornerPts.count
        cornerPts.append(pt3D)
        placeMarker(at: pt3D, in: sceneView)
        debugInfo += "P\(n): (\(String(format:"%.2f",pt3D.x)),\(String(format:"%.2f",pt3D.y)),\(String(format:"%.2f",pt3D.z))m)\n"

        switch cornerPts.count {
        case 1:
            debugInfo = "P0: (\(String(format:"%.2f",pt3D.x)),\(String(format:"%.2f",pt3D.y)),\(String(format:"%.2f",pt3D.z))m)\n"
            cornerInstruction = "2/4 — Esquina sup. DERECHA"
            status = "Esquina 1 ✓"
        case 2:
            cornerInstruction = "3/4 — Esquina inf. DERECHA"
            status = "Esquina 2 ✓"
        case 3:
            cornerInstruction = "4/4 — Tocar la ESQUINA TRASERA superior"
            status = "Esquina 3 ✓ — Moverse para ver la esquina de atrás"
        case 4:
            // P3 placed — show marker so user can verify before computing
            cornerInstruction = "✓ Verificá la esquina y tocá MEDIR"
            status = "P3 colocado — ¿está bien? Tocá MEDIR o UNDO"
        default:
            break
        }
    }

    // Called by the MEDIR button after the user verifies all 4 markers.
    func confirmBox() {
        guard cornerPts.count == 4, let sceneView else { return }
        isProcessing = true
        if let det = computeBox() {
            placeBoxes([det], in: sceneView)
        } else {
            status = "Geometría inválida — intentá de nuevo"
            cornerInstruction = "1/4 — Esquina sup. IZQUIERDA"
        }
        cornerPts.removeAll()
        markerNodes.forEach { $0.removeFromParentNode() }
        markerNodes.removeAll()
        isProcessing = false
    }

    // Exact OBB from 4 tapped corners.
    // widthAxis = normalize(P1−P0), heightAxis = normalize(P2−P1)
    // depthAxis = normalize(cross(widthAxis, heightAxis))
    // depth = |dot(P3−P0, depthAxis)|
    private func computeBox() -> Detection3D? {
        guard cornerPts.count == 4 else { return nil }
        let P0 = cornerPts[0], P1 = cornerPts[1], P2 = cornerPts[2], P3 = cornerPts[3]
        let widthVec  = P1 - P0
        let heightVec = P2 - P1
        let width  = simd_length(widthVec)
        let height = simd_length(heightVec)
        guard width > 0.01, height > 0.01 else { return nil }
        let widthAxis  = simd_normalize(widthVec)
        let heightAxis = simd_normalize(heightVec)
        let depthAxis  = simd_normalize(simd_cross(widthAxis, heightAxis))
        let depthProj  = simd_dot(P3 - P0, depthAxis)
        let depth      = abs(depthProj)
        guard depth > 0.01 else { return nil }
        let frontCenter = P0 + widthVec * 0.5 + heightVec * 0.5
        let depthSign: Float = depthProj >= 0 ? 1 : -1
        let center = frontCenter + depthAxis * depthSign * depth * 0.5
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

    // Finds the ARKit feature point closest to the ray through `screenPoint`.
    // Returns nil if no point is within `snapRadius` meters of the ray.
    // nonisolated + static → safe to call from SceneKit render thread.
    nonisolated static func nearestFeaturePoint(
        frame: ARFrame,
        screenPoint: CGPoint,
        viewportSize: CGSize,
        snapRadius: Float = 0.035
    ) -> simd_float3? {
        guard let pts = frame.rawFeaturePoints, !pts.points.isEmpty else { return nil }

        // Ray origin = camera position in world space
        let camT = frame.camera.transform
        let origin = simd_float3(camT.columns.3.x, camT.columns.3.y, camT.columns.3.z)

        // Map portrait screen point → normalized image coords via displayTransform
        let displayT = frame.displayTransform(for: .portrait, viewportSize: viewportSize)
        let norm = CGPoint(x: screenPoint.x / viewportSize.width,
                           y: screenPoint.y / viewportSize.height)
        let normImg = norm.applying(displayT.inverted())

        // Ray direction in world space using camera intrinsics
        let intr = frame.camera.intrinsics
        let res  = frame.camera.imageResolution
        let imgX = Float(normImg.x) * Float(res.width)
        let imgY = Float(normImg.y) * Float(res.height)
        let fx = intr[0][0], fy = intr[1][1], cx = intr[2][0], cy = intr[2][1]
        // In ARKit camera space: X right, Y up, looking in -Z
        let dirCam  = simd_float4((imgX - cx) / fx, -(imgY - cy) / fy, -1, 0)
        let dirWorld4 = camT * dirCam
        let dir = simd_normalize(simd_float3(dirWorld4.x, dirWorld4.y, dirWorld4.z))

        // Find feature point with smallest perpendicular distance to ray
        var bestPerp = snapRadius
        var bestPt: simd_float3? = nil
        for pt in pts.points {
            let v = pt - origin
            let proj = simd_dot(v, dir)
            guard proj > 0.08, proj < 6.0 else { continue }  // must be in front, within 6m
            let perp = simd_distance(origin + dir * proj, pt)
            if perp < bestPerp { bestPerp = perp; bestPt = pt }
        }
        return bestPt
    }

    // Maps a screen tap to a 3D world-space point using ARKit raycasting + LiDAR depth fallback.
    // Points are in ARKit world space → stable across phone movements between taps.
    private func tapTo3D(point: CGPoint) -> simd_float3? {
        guard let sceneView, let frame = sceneView.session.currentFrame else { return nil }

        // Tier 1: ARKit mesh/plane raycast — most accurate when mesh is built up.
        for target: ARRaycastQuery.Target in [.existingPlaneGeometry, .estimatedPlane] {
            if let q = sceneView.raycastQuery(from: point, allowing: target, alignment: .any),
               let r = sceneView.session.raycast(q).first {
                isSnapping = false
                let col = r.worldTransform.columns.3
                return simd_float3(col.x, col.y, col.z)
            }
        }

        // Tier 2: Feature point snap — snaps to ARKit-tracked visual features near the ray.
        // Great for box edges and corners that ARKit hasn't built into planes yet.
        if let snapped = ARViewModel.nearestFeaturePoint(frame: frame, screenPoint: point,
                                                          viewportSize: viewportSize) {
            isSnapping = true
            return snapped
        }
        isSnapping = false

        // Tier 3: Direct LiDAR depth — prefer smoothed (temporal filter) over raw.
        guard let depthBuffer = (frame.smoothedSceneDepth ?? frame.sceneDepth)?.depthMap else { return nil }
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
        var bestD = Float.infinity, bestDX = tapDX, bestDY = tapDY
        for dy in -2...2 {
            let py  = max(0, min(dH - 1, tapDY + dy))
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
    func undoLastCorner() {
        guard !cornerPts.isEmpty else { clearAll(); return }
        cornerPts.removeLast()
        if let last = markerNodes.last {
            last.removeFromParentNode()
            markerNodes.removeLast()
        }
        switch cornerPts.count {
        case 0: cornerInstruction = "1/4 — Esquina sup. IZQUIERDA"
        case 1: cornerInstruction = "2/4 — Esquina sup. DERECHA"
        case 2: cornerInstruction = "3/4 — Esquina inf. DERECHA"
        case 3: cornerInstruction = "4/4 — Tocar la ESQUINA TRASERA superior"
        default: break
        }
        debugInfo = cornerPts.isEmpty ? "" : cornerPts.enumerated().map { i, p in
            "P\(i): (\(String(format:"%.2f",p.x)),\(String(format:"%.2f",p.y)),\(String(format:"%.2f",p.z))m)\n"
        }.joined()
    }

    func clearAll() {
        clearBoxes()
        lastDetections3D.removeAll()
        debugBBoxes.removeAll()
        debugInfo = ""
        cornerPts.removeAll()
        markerNodes.forEach { $0.removeFromParentNode() }
        markerNodes.removeAll()
        cornerInstruction = "1/4 — Esquina sup. IZQUIERDA"
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
