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

    var sceneView: ARSCNView?
    var viewportSize: CGSize = UIScreen.main.bounds.size
    /// Viewfinder in normalized screen coords [0,1]. UI and filter share this.
    let viewfinderNorm = CGRect(x: 0.1, y: 0.18, width: 0.8, height: 0.64)
    private var yoloDetector: YOLODetector?
    private var boxNodes: [SCNNode] = []

    func setup(sceneView: ARSCNView) {
        self.sceneView = sceneView
        Task.detached { await self.loadModelsInBackground() }
    }

    // MARK: - Model Loading

    nonisolated private func loadModelsInBackground() async {
        let modelPath = Bundle.main.path(forResource: "best", ofType: "onnx")

        await MainActor.run { self.status = "Loading YOLO..." }
        guard let modelPath else {
            await MainActor.run { self.status = "best.onnx not found in bundle" }
            return
        }
        let yolo: YOLODetector
        do { yolo = try YOLODetector(modelPath: modelPath) }
        catch {
            await MainActor.run { self.status = "YOLO failed: \(error.localizedDescription)" }
            return
        }
        await MainActor.run {
            self.yoloDetector = yolo
            self.status = "Ready — center box in viewfinder, tap Detect"
        }
    }

    // MARK: - Detection

    func detectNow() {
        guard let sceneView, let frame = sceneView.session.currentFrame,
              let yoloDetector else {
            status = "Not ready"; return
        }
        guard frame.sceneDepth != nil else {
            status = "No LiDAR depth — needs iPhone Pro"; return
        }
        isProcessing = true
        status = "Detecting..."

        Task.detached {
            do {
                let results = try await self.runPipeline(frame: frame, yolo: yoloDetector)
                await MainActor.run {
                    self.placeBoxes(results, in: sceneView)
                    self.isProcessing = false
                }
            } catch {
                await MainActor.run {
                    self.status = "Error: \(error.localizedDescription)"
                    self.isProcessing = false
                }
            }
        }
    }

    nonisolated private func runPipeline(
        frame: ARFrame, yolo: YOLODetector
    ) async throws -> [Detection3D] {

        // 1. YOLO 2D detection on 640×640 square crop.
        let (yoloImage, _, _) = pixelBufferToFloatArray(frame.capturedImage, targetSize: 640)
        let confThreshold = await MainActor.run { self.confidenceThreshold }
        let yoloBoxes = try yolo.detect(
            image: yoloImage, imageWidth: 640, imageHeight: 640,
            confThreshold: confThreshold
        )
        guard !yoloBoxes.isEmpty else {
            await MainActor.run { self.status = "No cajas detectadas" }
            return []
        }
        let topBoxes = Array(yoloBoxes.sorted { $0.score > $1.score }.prefix(3))

        // 2. Project YOLO boxes to screen space (portrait viewport) for overlay + filter.
        let vp = await MainActor.run { self.viewportSize }
        let displayT = frame.displayTransform(for: .portrait, viewportSize: vp)
        let res = frame.camera.imageResolution
        let imgW = Float(res.width), imgH = Float(res.height)
        let side = min(imgW, imgH)
        let ox = (imgW - side) / 2, oy = (imgH - side) / 2

        let screenBoxes: [(rect: CGRect, score: Float)] = topBoxes.map { box in
            func pt(_ x: Float, _ y: Float) -> CGPoint {
                CGPoint(x: CGFloat((x / 640 * side + ox) / imgW),
                        y: CGFloat((y / 640 * side + oy) / imgH)).applying(displayT)
            }
            let tl = pt(box.xmin, box.ymin), br = pt(box.xmax, box.ymax)
            return (CGRect(x: min(tl.x, br.x), y: min(tl.y, br.y),
                           width: abs(br.x - tl.x), height: abs(br.y - tl.y)), box.score)
        }
        await MainActor.run { self.debugBBoxes = screenBoxes }

        // 3. Viewfinder filter — only process the box whose center is in the viewfinder.
        let vfNorm = await MainActor.run { self.viewfinderNorm }
        let vfRect = CGRect(x: vfNorm.minX * vp.width,  y: vfNorm.minY * vp.height,
                            width: vfNorm.width * vp.width, height: vfNorm.height * vp.height)
        let vfBoxes: [YOLOBox] = zip(topBoxes, screenBoxes).compactMap { (box, screen) in
            vfRect.contains(CGPoint(x: screen.rect.midX, y: screen.rect.midY)) ? box : nil
        }
        let candidates = vfBoxes.isEmpty ? topBoxes : vfBoxes

        if vfBoxes.isEmpty {
            await MainActor.run { self.status = "Apuntá la caja al viewfinder" }
        }

        // 4. Floor-plane filter — discard detections not 0–1.8 m above ground.
        let bufW = Float(CVPixelBufferGetWidth(frame.capturedImage))
        let bufH = Float(CVPixelBufferGetHeight(frame.capturedImage))
        let depthBuf = frame.sceneDepth!.depthMap
        let depthW = CVPixelBufferGetWidth(depthBuf)
        let depthH = CVPixelBufferGetHeight(depthBuf)
        let rawDepth = extractDepthMap(depthBuf)

        let floorY: Float? = frame.anchors
            .compactMap { $0 as? ARPlaneAnchor }
            .filter { $0.alignment == .horizontal }
            .map { Float($0.transform.columns.3.y) }
            .min()

        let filtered: [YOLOBox] = candidates.filter { box in
            let bSide = min(bufW, bufH)
            let bOx = (bufW - bSide) / 2, bOy = (bufH - bSide) / 2
            let imgX = (box.xmin + box.xmax) / 2 / 640 * bSide + bOx
            let imgY = (box.ymin + box.ymax) / 2 / 640 * bSide + bOy
            let dx = Int(imgX / bufW * Float(depthW))
            let dy = Int(imgY / bufH * Float(depthH))
            guard dx >= 0, dx < depthW, dy >= 0, dy < depthH else { return true }
            let depth = rawDepth[dy][dx]
            guard depth > 0.1, depth < 8.0 else { return true }
            let intr = frame.camera.intrinsics
            let camPt = simd_float4((imgX - intr[2][0]) / intr[0][0] * depth,
                                     (imgY - intr[2][1]) / intr[1][1] * depth,
                                     -depth, 1)
            let worldY = (frame.camera.transform * camPt).y
            if let floor = floorY { return (worldY - floor) >= -0.05 && (worldY - floor) <= 1.8 }
            return true
        }
        let toMeasure = filtered.isEmpty ? candidates : filtered

        // 5. Measure each box with LiDAR depth map + PCA.
        var results: [Detection3D] = []
        for box in toMeasure {
            await MainActor.run { self.status = "Midiendo \(box.label)..." }
            if let det = DepthMeasurer.measure(frame: frame, yoloBox: box) {
                results.append(det)
            }
        }

        if results.isEmpty {
            await MainActor.run {
                self.status = "No se pudo medir — acercate más o girá la caja"
            }
        }
        return results
    }

    // MARK: - 3D Box Rendering

    private func placeBoxes(_ detections: [Detection3D], in sceneView: ARSCNView) {
        clearBoxes()
        let colors: [UIColor] = [.systemGreen, .systemRed, .systemBlue]

        for (i, det) in detections.enumerated() {
            let color = colors[i % colors.count]

            let box = SCNBox(width: CGFloat(det.size.x), height: CGFloat(det.size.y),
                             length: CGFloat(det.size.z), chamferRadius: 0)
            let mat = SCNMaterial()
            mat.diffuse.contents = color.withAlphaComponent(0.15)
            mat.isDoubleSided = true
            box.materials = [mat]

            let node = SCNNode(geometry: box)
            node.simdWorldTransform = det.worldTransform

            addWireframe(to: node, size: det.size, color: color, radius: 0.004)

            let label = det.label ?? "caja"
            let sizeStr = String(format: "%.0fx%.0fx%.0f cm",
                                 det.size.x * 100, det.size.y * 100, det.size.z * 100)
            addLabel("\(label)\n\(sizeStr)", to: node, offset: det.size.y / 2 + 0.04)

            sceneView.scene.rootNode.addChildNode(node)
            boxNodes.append(node)
        }

        let summary = detections.map {
            String(format: "%.0fx%.0fx%.0f cm", $0.size.x*100, $0.size.y*100, $0.size.z*100)
        }.joined(separator: " | ")
        status = detections.isEmpty ? "Sin detecciones" : summary

        detections.forEach {
            self.detections.append(DetectionInfo(
                label: $0.label ?? "caja", size: $0.size, confidence: $0.confidence
            ))
        }
    }

    private func addWireframe(to parent: SCNNode, size: simd_float3, color: UIColor, radius: Float) {
        let hw = size.x / 2, hh = size.y / 2, hd = size.z / 2
        let mat = SCNMaterial()
        mat.diffuse.contents = color

        let edges: [(simd_float3, simd_float3)] = [
            ([-hw,-hh,-hd], [ hw,-hh,-hd]),
            ([ hw,-hh,-hd], [ hw,-hh, hd]),
            ([ hw,-hh, hd], [-hw,-hh, hd]),
            ([-hw,-hh, hd], [-hw,-hh,-hd]),
            ([-hw, hh,-hd], [ hw, hh,-hd]),
            ([ hw, hh,-hd], [ hw, hh, hd]),
            ([ hw, hh, hd], [-hw, hh, hd]),
            ([-hw, hh, hd], [-hw, hh,-hd]),
            ([-hw,-hh,-hd], [-hw, hh,-hd]),
            ([ hw,-hh,-hd], [ hw, hh,-hd]),
            ([ hw,-hh, hd], [ hw, hh, hd]),
            ([-hw,-hh, hd], [-hw, hh, hd]),
        ].map { (simd_float3($0.0), simd_float3($0.1)) }

        for (a, b) in edges {
            let cyl = SCNCylinder(radius: CGFloat(radius), height: CGFloat(simd_distance(a, b)))
            cyl.materials = [mat]
            let n = SCNNode(geometry: cyl)
            n.simdPosition = (a + b) / 2
            let dir = simd_normalize(b - a)
            let dot = simd_dot(simd_float3(0,1,0), dir)
            if abs(dot) < 0.999 {
                let axis = simd_normalize(simd_cross(simd_float3(0,1,0), dir))
                n.simdRotation = simd_float4(axis, acos(dot))
            }
            parent.addChildNode(n)
        }
    }

    private func addLabel(_ text: String, to parent: SCNNode, offset: Float) {
        let scnText = SCNText(string: text, extrusionDepth: 0.005)
        scnText.font = UIFont.systemFont(ofSize: 0.04, weight: .bold)
        scnText.firstMaterial?.diffuse.contents = UIColor.white
        scnText.flatness = 0.1
        let node = SCNNode(geometry: scnText)
        node.position = SCNVector3(-0.06, offset, 0)
        node.constraints = [SCNBillboardConstraint()]
        parent.addChildNode(node)
    }

    func clearBoxes() {
        boxNodes.forEach { $0.removeFromParentNode() }
        boxNodes.removeAll()
        detections.removeAll()
    }

    func clearAll() {
        clearBoxes()
        debugBBoxes.removeAll()
    }
}

// MARK: - Image helpers

func pixelBufferToFloatArray(
    _ pixelBuffer: CVPixelBuffer,
    targetSize: Int = 640
) -> ([Float], Int, Int) {
    var ciImage = CIImage(cvPixelBuffer: pixelBuffer)
    let context = CIContext()
    let w = ciImage.extent.width, h = ciImage.extent.height
    let side = min(w, h)
    ciImage = ciImage.cropped(to: CGRect(x: (w-side)/2, y: (h-side)/2, width: side, height: side))
    let scale = CGFloat(targetSize) / side
    let resized = ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
    var rgba = [UInt8](repeating: 0, count: targetSize * targetSize * 4)
    context.render(resized, toBitmap: &rgba, rowBytes: targetSize * 4,
                   bounds: CGRect(x: resized.extent.origin.x, y: resized.extent.origin.y,
                                  width: CGFloat(targetSize), height: CGFloat(targetSize)),
                   format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())
    let n = targetSize * targetSize
    var result = [Float](repeating: 0, count: 3 * n)
    for i in 0..<n {
        result[i]         = Float(rgba[i*4])     / 255.0
        result[n + i]     = Float(rgba[i*4 + 1]) / 255.0
        result[2*n + i]   = Float(rgba[i*4 + 2]) / 255.0
    }
    return (result, targetSize, targetSize)
}

func extractDepthMap(_ depthBuffer: CVPixelBuffer) -> [[Float]] {
    CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }
    let h = CVPixelBufferGetHeight(depthBuffer)
    let w = CVPixelBufferGetWidth(depthBuffer)
    let bpr = CVPixelBufferGetBytesPerRow(depthBuffer)
    let base = CVPixelBufferGetBaseAddress(depthBuffer)!
    var result = [[Float]](repeating: [Float](repeating: 0, count: w), count: h)
    for y in 0..<h {
        let row = base.advanced(by: y * bpr).assumingMemoryBound(to: Float32.self)
        for x in 0..<w { result[y][x] = row[x] }
    }
    return result
}
