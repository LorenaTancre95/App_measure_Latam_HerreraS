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
    @Published var measureMode: MeasureMode = .box
    @Published var segmentationOverlay: UIImage? = nil   // shown during TAP mode preview

    enum MeasureMode { case box, oversize, tap }

    var sceneView: ARSCNView?
    var viewportSize: CGSize = UIScreen.main.bounds.size
    let viewfinderNorm = CGRect(x: 0.1, y: 0.18, width: 0.8, height: 0.64)
    private var yoloDetector: YOLODetector?
    private var palletDetector: PalletDetector?
    private var samSegmenter: SAMSegmenter?
    private var boxNodes: [SCNNode] = []

    func setup(sceneView: ARSCNView) {
        self.sceneView = sceneView
        Task.detached { await self.loadModelsInBackground() }
    }

    nonisolated private func loadModelsInBackground() async {
        let yoloPath = Bundle.main.path(forResource: "best", ofType: "onnx")
        await MainActor.run { self.status = "Loading YOLO..." }
        guard let yoloPath else {
            await MainActor.run { self.status = "best.onnx not found" }
            return
        }
        let yolo: YOLODetector
        do { yolo = try YOLODetector(modelPath: yoloPath) }
        catch {
            await MainActor.run { self.status = "YOLO failed: \(error.localizedDescription)" }
            return
        }

        // Load PalletDetector (non-fatal)
        let pallet = try? PalletDetector()

        // SAMSegmenter is NOT loaded here — it's loaded lazily on first TAP use.
        // Loading YOLO + Pallet + SAM simultaneously causes an OOM kill on device
        // because the SAM encoder (1024×1024 cpuOnly) spikes memory by ~300 MB.

        await MainActor.run {
            self.yoloDetector = yolo
            self.palletDetector = pallet
            self.status = "Apuntá al piso para calibrar..."
        }
    }

    func detectNow() {
        guard let sceneView, let frame = sceneView.session.currentFrame else { status = "Not ready"; return }
        guard frame.sceneDepth != nil else { status = "No LiDAR depth"; return }

        if measureMode == .oversize {
            detectPallet(frame: frame, sceneView: sceneView); return
        }

        guard let yoloDetector else { status = "YOLO no cargado"; return }
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

    // MARK: - Tap-to-select

    func measureAtTap(at point: CGPoint) {
        guard let sceneView, let frame = sceneView.session.currentFrame else {
            status = "Sin frame AR"; return
        }
        guard frame.sceneDepth != nil else { status = "No LiDAR depth"; return }
        guard !isProcessing else { return }

        isProcessing = true
        clearAll()
        status = "Segmentando..."

        let vp = viewportSize
        Task.detached {
            // Primary: depth flood fill — works even when box and floor have same color.
            // Flood fill follows depth discontinuities, stopping at box edges.
            if let det = DepthBoxMeasurer.measure(frame: frame, tapPoint: point, viewportSize: vp) {
                await MainActor.run {
                    if let sv = self.sceneView { self.placeBoxes([det], in: sv) }
                    self.isProcessing = false
                }
                return
            }

            await MainActor.run {
                self.status = "Sin geometría — acercate más o tocá el centro de la caja"
                self.isProcessing = false
            }
        }
    }

    func floorDetected() {
        guard !isCalibrated else { return }
        isCalibrated = true
        if !isProcessing { status = "Piso calibrado ✓ — apuntá la caja al visor" }
    }

    func clearBoxes() { boxNodes.forEach { $0.removeFromParentNode() }; boxNodes.removeAll(); detections.removeAll() }
    func clearAll() { clearBoxes(); debugBBoxes.removeAll() }

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

func extractDepthMap(_ buf: CVPixelBuffer) -> [[Float]] {
    CVPixelBufferLockBaseAddress(buf, .readOnly); defer { CVPixelBufferUnlockBaseAddress(buf, .readOnly) }
    let h=CVPixelBufferGetHeight(buf), w=CVPixelBufferGetWidth(buf), bpr=CVPixelBufferGetBytesPerRow(buf)
    let base=CVPixelBufferGetBaseAddress(buf)!
    var out = [[Float]](repeating:[Float](repeating:0,count:w),count:h)
    for y in 0..<h { let row=base.advanced(by:y*bpr).assumingMemoryBound(to:Float32.self); for x in 0..<w { out[y][x]=row[x] } }
    return out
}
