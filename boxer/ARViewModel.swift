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
    private var yoloDetector: YOLODetector?
    private var boxNodes: [SCNNode] = []

    func setup(sceneView: ARSCNView) {
        self.sceneView = sceneView
        Task.detached { await self.loadModelsInBackground() }
    }

    // MARK: - Model Loading

    nonisolated private func loadModelsInBackground() async {
        // best.onnx bundled as "best" (1-class caja detector, 100 epochs).
        let modelPath = Bundle.main.path(forResource: "best", ofType: "onnx")
            ?? Bundle.main.path(forResource: "yolo11n", ofType: "onnx")

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
            self.status = "Ready — tap Detect 3D"
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
        // 1. YOLO detection (640×640 square crop).
        let (yoloImage, _, _) = pixelBufferToFloatArray(frame.capturedImage, targetSize: 640)
        let confThreshold = await MainActor.run { self.confidenceThreshold }
        let yoloBoxes = try yolo.detect(
            image: yoloImage, imageWidth: 640, imageHeight: 640,
            confThreshold: confThreshold
        )
        guard !yoloBoxes.isEmpty else {
            await MainActor.run { self.status = "No cajas detected" }
            return []
        }
        let topBoxes = Array(yoloBoxes.sorted { $0.score > $1.score }.prefix(3))

        // 2. Show YOLO 2D bboxes as screen-space overlay for debugging.
        let vp = await MainActor.run { self.viewportSize }
        let displayT = frame.displayTransform(for: .portrait, viewportSize: vp)
        let bufW = Float(CVPixelBufferGetWidth(frame.capturedImage))
        let bufH = Float(CVPixelBufferGetHeight(frame.capturedImage))
        let side = min(bufW, bufH)
        let ox = (bufW - side) / 2, oy = (bufH - side) / 2
        let screenBoxes: [(rect: CGRect, score: Float)] = topBoxes.map { box in
            func pt(_ x: Float, _ y: Float) -> CGPoint {
                CGPoint(x: CGFloat((x / 640 * side + ox) / bufW),
                        y: CGFloat((y / 640 * side + oy) / bufH)).applying(displayT)
            }
            let tl = pt(box.xmin, box.ymin), br = pt(box.xmax, box.ymax)
            return (CGRect(x: min(tl.x, br.x), y: min(tl.y, br.y),
                           width: abs(br.x - tl.x), height: abs(br.y - tl.y)), box.score)
        }
        await MainActor.run { self.debugBBoxes = screenBoxes }

        // 3. Measure each box with ARMesh + PCA.
        var results: [Detection3D] = []
        for box in topBoxes {
            await MainActor.run {
                self.status = "Measuring \(box.label)..."
            }
            if let det = BoxMeasurer.measure(frame: frame, yoloBox: box) {
                results.append(det)
            }
        }

        if results.isEmpty {
            await MainActor.run {
                self.status = "Mesh vacío — mueve el teléfono para escanear primero"
            }
        }
        return results
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
            let label = det.label ?? "caja"
            let sizeStr = String(format: "%.0fx%.0fx%.0f cm",
                                 det.size.x * 100, det.size.y * 100, det.size.z * 100)
            addLabel("\(label)\n\(sizeStr)", to: node, offset: det.size.y / 2 + 0.03)

            sceneView.scene.rootNode.addChildNode(node)
            boxNodes.append(node)
        }

        let summary = detections.map { det in
            String(format: "%.0fx%.0fx%.0f cm",
                   det.size.x * 100, det.size.y * 100, det.size.z * 100)
        }.joined(separator: " | ")
        status = detections.isEmpty ? "Sin detecciones" : summary

        detections.forEach { det in
            self.detections.append(DetectionInfo(
                label: det.label ?? "caja", size: det.size, confidence: det.confidence
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
    }

    func clearAll() {
        clearBoxes()
        debugBBoxes.removeAll()
    }
}

// MARK: - Image Helpers

func pixelBufferToFloatArray(
    _ pixelBuffer: CVPixelBuffer,
    targetSize: Int = 640
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
