import ARKit
import Vision
import SceneKit

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

    // MARK: - Vision
    private var visionRequest: VNDetectRectanglesRequest!
    private var lastVisionTime: TimeInterval = 0
    private let visionInterval: TimeInterval = 0.15

    // MARK: - Stability
    private var buffer: [NativeMeasurement] = []
    private let stabilityWindow = 5
    private let thresholdCm = 3.0

    // MARK: - Overlay
    private var overlayNodes: [SCNNode] = []

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
        setupVision()
    }

    // MARK: - Vision setup
    private func setupVision() {
        visionRequest = VNDetectRectanglesRequest()
        visionRequest.minimumAspectRatio = 0.15
        visionRequest.maximumAspectRatio = 1.0
        visionRequest.minimumSize        = 0.15
        visionRequest.maximumObservations = 3
        visionRequest.minimumConfidence  = 0.65
    }

    // MARK: - Frame processing (background-thread safe)
    // (processing moved to ARSessionDelegate extension below)

    // MARK: - LiDAR measurement (must run on main thread)
    private func measureOnMain(snapshot: ARFrameSnapshot) {
        measure(rectangle: snapshot.observation, frame: snapshot.frame, depth: snapshot.depth)
    }

    private func measure(rectangle obs: VNRectangleObservation,
                         frame: ARFrame, depth: ARDepthData) {
        guard let sv = sceneView else { return }
        let size = sv.bounds.size

        func toScreen(_ p: CGPoint) -> CGPoint {
            CGPoint(x: p.x * size.width, y: (1 - p.y) * size.height)
        }

        let sBL = toScreen(obs.bottomLeft)
        let sBR = toScreen(obs.bottomRight)
        let sTL = toScreen(obs.topLeft)
        let sTR = toScreen(obs.topRight)

        guard
            let p3BL = worldPoint(sBL, frame: frame, depth: depth),
            let p3BR = worldPoint(sBR, frame: frame, depth: depth),
            let p3TL = worldPoint(sTL, frame: frame, depth: depth),
            let p3TR = worldPoint(sTR, frame: frame, depth: depth)
        else { return }

        let c = Double(simd_distance(p3BL, p3BR)) * 100
        let a = Double(simd_distance(p3BL, p3TL)) * 100
        let l = estimateDepth(frame: frame, p3BL: p3BL, p3BR: p3BR) ?? (c * 0.5)

        guard c > 5, c < 300, a > 5, a < 300, l > 2 else { return }

        let m = NativeMeasurement(comprimento: c, largura: l, altura: a)
        addToBuffer(m)
        updateOverlay(bl: p3BL, br: p3BR, tl: p3TL, tr: p3TR)
    }

    // MARK: - LiDAR unproject
    private func worldPoint(_ screenPt: CGPoint,
                            frame: ARFrame,
                            depth: ARDepthData) -> SIMD3<Float>? {
        guard let sv = sceneView else { return nil }

        let depthMap = depth.depthMap
        let dW = CVPixelBufferGetWidth(depthMap)
        let dH = CVPixelBufferGetHeight(depthMap)

        let dx = Int((screenPt.y / sv.bounds.height) * CGFloat(dW))
        let dy = Int((1 - screenPt.x / sv.bounds.width) * CGFloat(dH))
        let sx = max(0, min(dW - 1, dx))
        let sy = max(0, min(dH - 1, dy))

        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(depthMap) else { return nil }

        let depthVal = base.assumingMemoryBound(to: Float32.self)[sy * dW + sx]
        guard depthVal > 0.1, depthVal < 10 else { return nil }

        let intr = frame.camera.intrinsics
        let iW   = Float(CVPixelBufferGetWidth(frame.capturedImage))
        let iH   = Float(CVPixelBufferGetHeight(frame.capturedImage))

        let imgX = Float(screenPt.y / sv.bounds.height) * iW
        let imgY = Float(1 - screenPt.x / sv.bounds.width) * iH

        let xCam = (imgX - intr[2][0]) / intr[0][0] * depthVal
        let yCam = (imgY - intr[2][1]) / intr[1][1] * depthVal
        let pt   = frame.camera.transform * SIMD4<Float>(xCam, yCam, depthVal, 1)

        return SIMD3<Float>(pt.x, pt.y, pt.z)
    }

    // MARK: - Depth estimation
    private func estimateDepth(frame: ARFrame,
                               p3BL: SIMD3<Float>,
                               p3BR: SIMD3<Float>) -> Double? {
        guard let sv = sceneView else { return nil }

        let center = (p3BL + p3BR) / 2
        let projected = sv.projectPoint(SCNVector3(center.x, center.y, center.z))
        let screenPt = CGPoint(x: CGFloat(projected.x), y: CGFloat(projected.y))

        guard
            let query = sv.raycastQuery(from: screenPt,
                                        allowing: .existingPlaneGeometry,
                                        alignment: .horizontal),
            let result = sv.session.raycast(query).first
        else { return nil }

        let back = result.worldTransform.columns.3
        return Double(simd_distance(center,
                                    SIMD3<Float>(back.x, back.y, back.z))) * 100
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
              range(\.largura)     < thresholdCm,
              range(\.altura)      < thresholdCm else { return }

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

    // MARK: - Manual points
    func addManualPoint(_ p: SIMD3<Float>) {
        manualPoints.append(p)

        let sphere = SCNSphere(radius: 0.01)
        sphere.firstMaterial?.diffuse.contents  = UIColor.yellow
        sphere.firstMaterial?.lightingModel     = .constant
        let node = SCNNode(geometry: sphere)
        node.position = SCNVector3(p.x, p.y, p.z)
        sceneView?.scene.rootNode.addChildNode(node)
        overlayNodes.append(node)

        if manualPoints.count == 3 {
            let p0 = manualPoints[0], p1 = manualPoints[1], p2 = manualPoints[2]
            let c = Double(simd_distance(p0, p1)) * 100
            let a = Double(simd_distance(p0, p2)) * 100
            let m = NativeMeasurement(comprimento: c, largura: c * 0.5, altura: a)
            lastMeasurement = m
            DispatchQueue.main.async { [weak self] in self?.onUpdate(m) }
        }
    }

    // MARK: - Overlay
    func updateOverlay(bl: SIMD3<Float>, br: SIMD3<Float>,
                       tl: SIMD3<Float>, tr: SIMD3<Float>) {
        guard let sv = sceneView else { return }
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.clearOverlay()

            let color = UIColor(red: 0, green: 0.85, blue: 0.95, alpha: 1)
            let edges: [(SIMD3<Float>, SIMD3<Float>)] = [
                (bl, br), (br, tr), (tr, tl), (tl, bl)
            ]
            for (s, e) in edges {
                let node = self.makeLine(from: s, to: e, color: color)
                sv.scene.rootNode.addChildNode(node)
                self.overlayNodes.append(node)
            }
        }
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
    }

    private func makeLine(from s: SIMD3<Float>,
                          to e: SIMD3<Float>,
                          color: UIColor) -> SCNNode {
        let v = e - s
        let len = simd_length(v)
        let cyl = SCNCylinder(radius: 0.002, height: CGFloat(len))
        cyl.firstMaterial?.diffuse.contents = color
        cyl.firstMaterial?.lightingModel    = .constant

        let node = SCNNode(geometry: cyl)
        let mid  = (s + e) / 2
        node.position = SCNVector3(mid.x, mid.y, mid.z)

        let dir  = simd_normalize(v)
        let up   = SIMD3<Float>(0, 1, 0)
        let axis = simd_cross(up, dir)
        node.rotation = SCNVector4(axis.x, axis.y, axis.z, acos(simd_dot(up, dir)))
        return node
    }
}

// MARK: - Delegates
extension BoxDetectionCoordinator: ARSCNViewDelegate {
    func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
        guard anchor is ARPlaneAnchor else { return }
        DispatchQueue.main.async { [weak self] in self?.onPlaneFound() }
    }
}

extension BoxDetectionCoordinator: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // Vision runs on ARKit background thread — UIKit calls dispatched to main
        guard
            mode == .auto,
            frame.timestamp - lastVisionTime > visionInterval,
            let depthData = frame.sceneDepth
        else { return }
        lastVisionTime = frame.timestamp

        let handler = VNImageRequestHandler(
            cvPixelBuffer: frame.capturedImage,
            orientation: .right,
            options: [:]
        )
        try? handler.perform([visionRequest])

        guard
            let results = visionRequest.results as? [VNRectangleObservation],
            let best = results.max(by: { $0.boundingBox.area < $1.boundingBox.area }),
            best.boundingBox.area > 0.04
        else { return }

        let snapshot = ARFrameSnapshot(frame: frame, depth: depthData, observation: best)
        DispatchQueue.main.async { [weak self] in
            self?.measureOnMain(snapshot: snapshot)
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        print("[ARKit] session failed: \(error.localizedDescription)")
    }

    func sessionWasInterrupted(_ session: ARSession) {
        print("[ARKit] session interrupted")
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        print("[ARKit] interruption ended")
    }
}

private extension CGRect {
    var area: CGFloat { width * height }
}

private struct ARFrameSnapshot {
    let frame: ARFrame
    let depth: ARDepthData
    let observation: VNRectangleObservation
}
