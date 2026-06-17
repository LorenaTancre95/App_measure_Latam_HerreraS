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
    private var boxerNet: BoxerNet?
    private var yoloDetector: YOLODetector?
    private var boxNodes: [SCNNode] = []

    func setup(sceneView: ARSCNView) {
        self.sceneView = sceneView
        Task.detached { await self.loadModelsInBackground() }
    }

    // MARK: - Model Loading

    nonisolated private func loadModelsInBackground() async {
        let yoloPath = Bundle.main.path(forResource: "yolo11n", ofType: "onnx")
        let boxerPath = Bundle.main.path(forResource: "BoxerNet", ofType: "onnx")

        await MainActor.run { self.status = "Loading YOLO..." }
        guard let yoloPath else {
            await MainActor.run { self.status = "yolo11n.onnx not found" }
            return
        }
        let yolo: YOLODetector
        do { yolo = try YOLODetector(modelPath: yoloPath) }
        catch {
            await MainActor.run { self.status = "YOLO failed: \(error.localizedDescription)" }
            return
        }

        await MainActor.run { self.status = "Loading BoxerNet..." }

        var boxer: BoxerNet? = nil
        if let boxerPath {
            do { boxer = try BoxerNet(modelPath: boxerPath) }
            catch {
                // BoxerNet failed (model too large or unsupported device).
                // Fall back to YOLO-only mode — 3D lifting will be unavailable.
                await MainActor.run {
                    self.yoloDetector = yolo
                    self.status = "YOLO only (BoxerNet no disponible: \(error.localizedDescription))"
                }
                return
            }
        }

        await MainActor.run {
            self.yoloDetector = yolo
            self.boxerNet = boxer
            self.status = boxer != nil ? "Ready — tap Detect 3D" : "Ready — YOLO only (sin BoxerNet.onnx)"
        }
    }

    // MARK: - Detection

    func detectNow() {
        guard let sceneView, let frame = sceneView.session.currentFrame,
              let yoloDetector else {
            status = "Not ready"; return
        }
        guard frame.sceneDepth != nil else {
            status = "No LiDAR depth"; return
        }

        isProcessing = true
        status = "Detecting..."

        let capturedBoxerNet = boxerNet
        Task.detached {
            do {
                let results = try await self.runPipeline(frame: frame, boxer: capturedBoxerNet, yolo: yoloDetector)
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
        frame: ARFrame, boxer: BoxerNet?, yolo: YOLODetector
    ) async throws -> [Detection3D] {
        // 1. YOLO detection (640×640).
        let (yoloImage, _, _) = pixelBufferToFloatArray(frame.capturedImage, targetSize: 640)
        let yoloBoxes = try yolo.detect(image: yoloImage, imageWidth: 640, imageHeight: 640)
        guard !yoloBoxes.isEmpty else {
            await MainActor.run { self.status = "No objects detected" }
            return []
        }
        let topBoxes = Array(yoloBoxes.sorted { $0.score > $1.score }.prefix(3))

        // Show YOLO 2D bboxes as screen-space overlay for debugging.
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
        await MainActor.run {
            self.debugBBoxes = screenBoxes
            if let f = screenBoxes.first {
                self.status = String(format: "screen (%.0f,%.0f) %.0fx%.0f",
                                     f.rect.origin.x, f.rect.origin.y,
                                     f.rect.width, f.rect.height)
            }
        }

        // 2. If BoxerNet is unavailable, report detections without 3D lifting.
        guard let boxer else {
            await MainActor.run {
                self.status = "\(topBoxes.count) objeto(s) — BoxerNet no disponible"
            }
            return []
        }

        // 3. Scale YOLO boxes (640 → 960) for BoxerNet.
        let (boxerImage, _, _) = pixelBufferToFloatArray(frame.capturedImage, targetSize: BoxerNet.imageSize)
        let scale = Float(BoxerNet.imageSize) / 640.0
        let boxes2D = topBoxes.map { box in
            Box2D(xmin: box.xmin * scale, ymin: box.ymin * scale,
                  xmax: box.xmax * scale, ymax: box.ymax * scale,
                  label: box.label, score: box.score)
        }

        // 4. Extract LiDAR depth + scale intrinsics.
        let depthMap = extractDepthMap(frame.sceneDepth!.depthMap)
        let intrinsics = scaleIntrinsicsWithCrop(
            frame.camera.intrinsics,
            from: frame.camera.imageResolution,
            toSize: BoxerNet.imageSize
        )

        // 5. BoxerNet 3D lifting.
        let conf = await MainActor.run { self.confidenceThreshold }
        let detections = try boxer.predict(
            image: boxerImage, depthMap: depthMap, intrinsics: intrinsics,
            imageResolution: frame.camera.imageResolution,
            cameraTransform: frame.camera.transform, boxes2D: boxes2D,
            confidenceThreshold: conf
        )

        // Debug: show YOLO 2D bbox in status (640×640 coords)
        if let b = topBoxes.first {
            let bw = b.xmax - b.xmin
            let bh = b.ymax - b.ymin
            await MainActor.run {
                self.status = String(format: "bbox %.0f×%.0f (%.2f:1) conf %.2f",
                                     bw, bh, bw/bh, b.score)
            }
        }
        return detections
    }

    // MARK: - 3D Box Rendering

    private func placeBoxes(_ detections: [Detection3D], in sceneView: ARSCNView) {
        clearBoxes()
        let colors: [UIColor] = [.systemRed, .systemGreen, .systemBlue]

        for (i, det) in detections.enumerated() {
            let color = colors[i % colors.count]

            // Semi-transparent fill.
            let box = SCNBox(width: CGFloat(det.size.x), height: CGFloat(det.size.y),
                             length: CGFloat(det.size.z), chamferRadius: 0)
            let mat = SCNMaterial()
            mat.diffuse.contents = color.withAlphaComponent(0.3)
            mat.isDoubleSided = true
            box.materials = [mat]

            let node = SCNNode(geometry: box)
            node.simdWorldTransform = det.worldTransform

            // Thick wireframe edges (12 cylinders).
            addWireframe(to: node, size: det.size, color: color, radius: 0.003)

            // Floating label.
            let label = det.label ?? "object"
            let sizeStr = String(format: "%.0fx%.0fx%.0f cm",
                                 det.size.x * 100, det.size.y * 100, det.size.z * 100)
            addLabel("\(label)\n\(sizeStr)", to: node, offset: det.size.y / 2 + 0.03)

            sceneView.scene.rootNode.addChildNode(node)
            boxNodes.append(node)
        }

        detections.forEach { det in
            self.detections.append(DetectionInfo(
                label: det.label ?? "object", size: det.size, confidence: det.confidence
            ))
        }
    }

    private func addWireframe(to parent: SCNNode, size: simd_float3, color: UIColor, radius: Float) {
        let hw = size.x / 2, hh = size.y / 2, hd = size.z / 2
        let edgeMat = SCNMaterial()
        edgeMat.diffuse.contents = color

        let edges: [(simd_float3, simd_float3)] = [
            (simd_float3(-hw, -hh, -hd), simd_float3( hw, -hh, -hd)),
            (simd_float3( hw, -hh, -hd), simd_float3( hw, -hh,  hd)),
            (simd_float3( hw, -hh,  hd), simd_float3(-hw, -hh,  hd)),
            (simd_float3(-hw, -hh,  hd), simd_float3(-hw, -hh, -hd)),
            (simd_float3(-hw,  hh, -hd), simd_float3( hw,  hh, -hd)),
            (simd_float3( hw,  hh, -hd), simd_float3( hw,  hh,  hd)),
            (simd_float3( hw,  hh,  hd), simd_float3(-hw,  hh,  hd)),
            (simd_float3(-hw,  hh,  hd), simd_float3(-hw,  hh, -hd)),
            (simd_float3(-hw, -hh, -hd), simd_float3(-hw,  hh, -hd)),
            (simd_float3( hw, -hh, -hd), simd_float3( hw,  hh, -hd)),
            (simd_float3( hw, -hh,  hd), simd_float3( hw,  hh,  hd)),
            (simd_float3(-hw, -hh,  hd), simd_float3(-hw,  hh,  hd)),
        ]

        for (a, b) in edges {
            let cyl = SCNCylinder(radius: CGFloat(radius), height: CGFloat(simd_distance(a, b)))
            cyl.materials = [edgeMat]
            let node = SCNNode(geometry: cyl)
            node.simdPosition = (a + b) / 2
            let dir = simd_normalize(b - a)
            let dot = simd_dot(simd_float3(0, 1, 0), dir)
            if abs(dot) < 0.999 {
                let axis = simd_normalize(simd_cross(simd_float3(0, 1, 0), dir))
                node.simdRotation = simd_float4(axis, acos(dot))
            }
            parent.addChildNode(node)
        }
    }

    private func addLabel(_ text: String, to parent: SCNNode, offset: Float) {
        let scnText = SCNText(string: text, extrusionDepth: 0.005)
        scnText.font = UIFont.systemFont(ofSize: 0.03, weight: .bold)
        scnText.firstMaterial?.diffuse.contents = UIColor.white
        scnText.flatness = 0.1
        let node = SCNNode(geometry: scnText)
        node.position = SCNVector3(-0.05, offset, 0)
        node.constraints = [SCNBillboardConstraint()]
        parent.addChildNode(node)
    }

    func clearBoxes() {
        boxNodes.forEach { $0.removeFromParentNode() }
        boxNodes.removeAll()
        detections.removeAll()
        // debugBBoxes kept intentionally so 2D overlay stays visible alongside 3D box
    }

    func clearAll() {
        clearBoxes()
        debugBBoxes.removeAll()
    }
}

// MARK: - Image Helpers

func pixelBufferToFloatArray(
    _ pixelBuffer: CVPixelBuffer,
    targetSize: Int = BoxerNet.imageSize
) -> ([Float], Int, Int) {
    var ciImage = CIImage(cvPixelBuffer: pixelBuffer)
    let context = CIContext()

    // Center-crop to square.
    let w = ciImage.extent.width, h = ciImage.extent.height
    let side = min(w, h)
    ciImage = ciImage.cropped(to: CGRect(x: (w - side) / 2, y: (h - side) / 2,
                                          width: side, height: side))

    // Resize to target.
    let scale = CGFloat(targetSize) / side
    let resized = ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))

    // Render to RGBA.
    var rgba = [UInt8](repeating: 0, count: targetSize * targetSize * 4)
    context.render(resized, toBitmap: &rgba, rowBytes: targetSize * 4,
                   bounds: CGRect(x: resized.extent.origin.x, y: resized.extent.origin.y,
                                  width: CGFloat(targetSize), height: CGFloat(targetSize)),
                   format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())

    // RGBA → CHW float32.
    let n = targetSize * targetSize
    var result = [Float](repeating: 0, count: 3 * n)
    for i in 0..<n {
        result[i]         = Float(rgba[i * 4])     / 255.0
        result[n + i]     = Float(rgba[i * 4 + 1]) / 255.0
        result[2 * n + i] = Float(rgba[i * 4 + 2]) / 255.0
    }
    return (result, targetSize, targetSize)
}

func extractDepthMap(_ depthBuffer: CVPixelBuffer) -> [[Float]] {
    CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }

    let h = CVPixelBufferGetHeight(depthBuffer)
    let w = CVPixelBufferGetWidth(depthBuffer)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(depthBuffer)
    let base = CVPixelBufferGetBaseAddress(depthBuffer)!

    var result = [[Float]](repeating: [Float](repeating: 0, count: w), count: h)
    for y in 0..<h {
        let row = base.advanced(by: y * bytesPerRow).assumingMemoryBound(to: Float32.self)
        for x in 0..<w { result[y][x] = row[x] }
    }
    return result
}

func scaleIntrinsicsWithCrop(
    _ intrinsics: simd_float3x3, from: CGSize, toSize: Int
) -> simd_float3x3 {
    let w = Float(from.width), h = Float(from.height)
    let side = min(w, h)
    let scale = Float(toSize) / side

    var s = intrinsics
    s[0][0] *= scale                                     // fx
    s[1][1] *= scale                                     // fy
    s[2][0] = (intrinsics[2][0] - (w - side) / 2) * scale  // cx
    s[2][1] = (intrinsics[2][1] - (h - side) / 2) * scale  // cy
    return s
}
