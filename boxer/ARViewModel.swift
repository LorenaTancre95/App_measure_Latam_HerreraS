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
    /// Viewfinder in normalized screen coords [0,1]. UI and filter use this.
    let viewfinderNorm = CGRect(x: 0.1, y: 0.18, width: 0.8, height: 0.64)
    private var boxerNet: BoxerNet?
    private var yoloDetector: YOLODetector?
    private var boxNodes: [SCNNode] = []

    func setup(sceneView: ARSCNView) {
        self.sceneView = sceneView
        Task.detached { await self.loadModelsInBackground() }
    }

    // MARK: - Model Loading

    nonisolated private func loadModelsInBackground() async {
        // best.onnx = custom 1-class "caja" detector (models_yolo/best.onnx, 100 epochs).
        let yoloPath = Bundle.main.path(forResource: "best", ofType: "onnx")
        let boxerPath = Bundle.main.path(forResource: "BoxerNet", ofType: "onnx")

        await MainActor.run { self.status = "Loading YOLO..." }
        guard let yoloPath else {
            await MainActor.run { self.status = "best.onnx not found in bundle" }
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
        await MainActor.run { self.debugBBoxes = screenBoxes }

        // 2. Viewfinder filter — only keep boxes whose projected center
        //    falls inside the viewfinder rectangle.
        let vfNorm = await MainActor.run { self.viewfinderNorm }
        let vfRect = CGRect(x: vfNorm.minX * vp.width,  y: vfNorm.minY * vp.height,
                            width: vfNorm.width * vp.width, height: vfNorm.height * vp.height)
        let afterVF: [YOLOBox] = zip(topBoxes, screenBoxes).compactMap { (box, screen) in
            vfRect.contains(CGPoint(x: screen.rect.midX, y: screen.rect.midY)) ? box : nil
        }
        let vfBoxes = afterVF.isEmpty ? topBoxes : afterVF  // fallback: all boxes if none in VF

        await MainActor.run {
            if afterVF.isEmpty {
                self.status = "Apuntá la caja al viewfinder"
            }
        }

        // 3. Floor ROI filter — keep only detections whose LiDAR center is
        //    0–1.5 m above the lowest ARKit horizontal plane (the floor).
        //    If no floor has been detected yet, keep all boxes.
        let bufW = CVPixelBufferGetWidth(frame.capturedImage)
        let bufH = CVPixelBufferGetHeight(frame.capturedImage)
        let depthPixelBuffer = frame.sceneDepth!.depthMap
        let depthW = CVPixelBufferGetWidth(depthPixelBuffer)
        let depthH = CVPixelBufferGetHeight(depthPixelBuffer)
        let rawDepth = extractDepthMap(depthPixelBuffer)

        let floorY: Float? = frame.anchors
            .compactMap { $0 as? ARPlaneAnchor }
            .filter { $0.alignment == .horizontal }
            .map { Float($0.transform.columns.3.y) }
            .min()

        let roiBoxes: [YOLOBox] = vfBoxes.filter { box in
            // Map YOLO center (640×640) → landscape buffer pixels.
            let bside = min(Float(bufW), Float(bufH))
            let bOx = (Float(bufW) - bside) / 2
            let bOy = (Float(bufH) - bside) / 2
            let imgX = (box.xmin + box.xmax) / 2 / 640.0 * bside + bOx
            let imgY = (box.ymin + box.ymax) / 2 / 640.0 * bside + bOy

            // Sample LiDAR depth at that pixel.
            let dx = Int(imgX / Float(bufW) * Float(depthW))
            let dy = Int(imgY / Float(bufH) * Float(depthH))
            guard dx >= 0, dx < depthW, dy >= 0, dy < depthH else { return true }
            let depth = rawDepth[dy][dx]
            guard depth > 0.1, depth < 8.0 else { return true }

            // Unproject to world space using camera intrinsics (landscape buffer coords).
            let intr = frame.camera.intrinsics
            let camPt = simd_float4(
                (imgX - intr[2][0]) / intr[0][0] * depth,
                (imgY - intr[2][1]) / intr[1][1] * depth,
                -depth,   // ARKit camera: –Z forward
                1
            )
            let worldPt = frame.camera.transform * camPt
            let worldY = worldPt.y / worldPt.w

            // If floor is known, filter by height above floor.
            if let floor = floorY {
                let h = worldY - floor
                return h >= -0.05 && h <= 1.8   // −5 cm tolerance … 1.8 m max box height
            }
            return true   // no floor detected yet — allow everything
        }

        let filteredBoxes = roiBoxes.isEmpty ? vfBoxes : roiBoxes

        await MainActor.run {
            if let floor = floorY {
                let kept = filteredBoxes.count
                let total = vfBoxes.count
                if kept < total {
                    self.status = "ROI: \(kept)/\(total) cajas sobre el piso"
                }
            }
        }

        // 3. If BoxerNet is unavailable, report detections without 3D lifting.
        guard let boxer else {
            await MainActor.run {
                self.status = "\(filteredBoxes.count) objeto(s) — BoxerNet no disponible"
            }
            return []
        }

        // 4. Scale filtered boxes (640 → 960) for BoxerNet.
        let (boxerImage, _, _) = pixelBufferToFloatArray(frame.capturedImage, targetSize: BoxerNet.imageSize)
        let scale = Float(BoxerNet.imageSize) / 640.0
        let boxes2D = filteredBoxes.map { box in
            Box2D(xmin: box.xmin * scale, ymin: box.ymin * scale,
                  xmax: box.xmax * scale, ymax: box.ymax * scale,
                  label: box.label, score: box.score)
        }

        // 5. Scale intrinsics for BoxerNet (reuses bufW/bufH/rawDepth from ROI step).
        let bufferSize = CGSize(width: bufW, height: bufH)
        let intrinsics = scaleIntrinsicsWithCrop(
            frame.camera.intrinsics,
            from: bufferSize,
            toSize: BoxerNet.imageSize
        )

        // 6. BoxerNet 3D lifting.
        let conf = await MainActor.run { self.confidenceThreshold }
        let detections = try boxer.predict(
            image: boxerImage, depthMap: rawDepth, intrinsics: intrinsics,
            imageResolution: bufferSize,
            cameraTransform: frame.camera.transform, boxes2D: boxes2D,
            confidenceThreshold: conf
        )

        // Debug: show bbox of first kept detection.
        if let b = filteredBoxes.first {
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
