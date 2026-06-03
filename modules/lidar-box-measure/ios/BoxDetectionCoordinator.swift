import ARKit
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

    // MARK: - Timing
    private var lastScanTime: TimeInterval = 0
    private let scanInterval: TimeInterval = 0.15

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
    }

    // MARK: - Depth-scan measurement (replaces Vision rectangle detection)
    // Scans outward from the screen center (reticle) to find LiDAR depth
    // discontinuities that mark the physical edges of the box.
    // Works regardless of the box's visual appearance.
    private func measureFromCenter(frame: ARFrame, depth: ARDepthData) {
        guard let sv = sceneView else { return }
        let vp = sv.bounds.size
        let cx = vp.width / 2
        let cy = vp.height / 2

        guard let centerD = sampleDepth(at: CGPoint(x: cx, y: cy), frame: frame, depth: depth),
              centerD > 0.15, centerD < 4.0
        else { return }

        let edgeThreshold: Float = 0.10  // 10 cm depth jump = box edge
        let step: CGFloat = 5
        // Scan stops at 88% of screen to avoid picking up walls/floor beyond a box
        let limL = vp.width  * 0.12, limR = vp.width  * 0.88
        let limT = vp.height * 0.12, limB = vp.height * 0.88

        func scanEdge(dx: CGFloat, dy: CGFloat) -> CGPoint {
            var x = cx + dx, y = cy + dy
            while x >= limL, x <= limR, y >= limT, y <= limB {
                if let d = sampleDepth(at: CGPoint(x: x, y: y), frame: frame, depth: depth) {
                    if abs(d - centerD) > edgeThreshold {
                        return CGPoint(x: x - dx, y: y - dy)
                    }
                } else {
                    return CGPoint(x: x - dx, y: y - dy)
                }
                x += dx; y += dy
            }
            return CGPoint(x: x - dx, y: y - dy)
        }

        let lPt = scanEdge(dx: -step, dy: 0)
        let rPt = scanEdge(dx: +step, dy: 0)
        let tPt = scanEdge(dx: 0, dy: -step)
        let bPt = scanEdge(dx: 0, dy: +step)

        let spanX = rPt.x - lPt.x
        let spanY = bPt.y - tPt.y

        // Require minimum face size; reject regions that hit the scan limit on ALL sides
        // (that means we're measuring the floor/wall, not a box face)
        guard spanX > 50, spanY > 50 else { return }
        let hitLimitX = lPt.x <= limL + step && rPt.x >= limR - step
        let hitLimitY = tPt.y <= limT + step && bPt.y >= limB - step
        guard !(hitLimitX && hitLimitY) else { return }

        let bl = CGPoint(x: lPt.x, y: bPt.y)
        let br = CGPoint(x: rPt.x, y: bPt.y)
        let tl = CGPoint(x: lPt.x, y: tPt.y)
        let tr = CGPoint(x: rPt.x, y: tPt.y)

        guard
            let p3BL = worldPoint(bl, frame: frame, depth: depth),
            let p3BR = worldPoint(br, frame: frame, depth: depth),
            let p3TL = worldPoint(tl, frame: frame, depth: depth),
            let p3TR = worldPoint(tr, frame: frame, depth: depth)
        else { return }

        // Reject horizontal surfaces (floor, table-top): their face normal points up (Y ≈ 1)
        let right      = simd_normalize(p3BR - p3BL)
        let up         = simd_normalize(p3TL - p3BL)
        let faceNormal = simd_normalize(simd_cross(right, up))
        guard abs(faceNormal.y) < 0.65 else { return }

        let c = Double(simd_distance(p3BL, p3BR)) * 100
        let a = Double(simd_distance(p3BL, p3TL)) * 100
        let l = estimateDepth(frame: frame,
                              p3BL: p3BL, p3BR: p3BR,
                              p3TL: p3TL, p3TR: p3TR,
                              depth: depth) ?? min(c, a) * 0.6

        guard c > 5, c < 300, a > 5, a < 300, l > 2 else { return }

        let m = NativeMeasurement(comprimento: c, largura: l, altura: a)
        addToBuffer(m)
        updateOverlay(bl: p3BL, br: p3BR, tl: p3TL, tr: p3TR, measurement: m)
    }

    // Reads a single depth sample from the LiDAR depth map at a screen point.
    private func sampleDepth(at screenPt: CGPoint,
                             frame: ARFrame,
                             depth: ARDepthData) -> Float? {
        guard let sv = sceneView else { return nil }
        let vp = sv.bounds.size
        let invDisplay = frame.displayTransform(for: .portrait, viewportSize: vp).inverted()
        let normCam = CGPoint(x: screenPt.x / vp.width,
                              y: screenPt.y / vp.height).applying(invDisplay)
        let dm = depth.depthMap
        let dW = CVPixelBufferGetWidth(dm)
        let dH = CVPixelBufferGetHeight(dm)
        let sx = max(0, min(dW - 1, Int(normCam.x * CGFloat(dW))))
        let sy = max(0, min(dH - 1, Int(normCam.y * CGFloat(dH))))
        CVPixelBufferLockBaseAddress(dm, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(dm, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(dm) else { return nil }
        let v = base.assumingMemoryBound(to: Float32.self)[sy * dW + sx]
        return v > 0.02 && v < 8 ? v : nil
    }

    // MARK: - LiDAR unproject
    private func worldPoint(_ screenPt: CGPoint,
                            frame: ARFrame,
                            depth: ARDepthData) -> SIMD3<Float>? {
        guard let sv = sceneView else { return nil }
        let viewportSize = sv.bounds.size

        let invDisplay = frame.displayTransform(for: .portrait,
                                               viewportSize: viewportSize).inverted()
        let normVP  = CGPoint(x: screenPt.x / viewportSize.width,
                              y: screenPt.y / viewportSize.height)
        let normCam = normVP.applying(invDisplay)

        let depthMap = depth.depthMap
        let dW = CVPixelBufferGetWidth(depthMap)
        let dH = CVPixelBufferGetHeight(depthMap)
        let sx = max(0, min(dW - 1, Int(normCam.x * CGFloat(dW))))
        let sy = max(0, min(dH - 1, Int(normCam.y * CGFloat(dH))))

        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(depthMap) else { return nil }

        let depthVal = base.assumingMemoryBound(to: Float32.self)[sy * dW + sx]
        guard depthVal > 0.05, depthVal < 8 else { return nil }

        let intr = frame.camera.intrinsics
        let iW   = Float(frame.camera.imageResolution.width)
        let iH   = Float(frame.camera.imageResolution.height)

        let imgX = Float(normCam.x) * iW
        let imgY = Float(normCam.y) * iH

        let xCam = (imgX - intr[2][0]) / intr[0][0] * depthVal
        let yCam = (imgY - intr[2][1]) / intr[1][1] * depthVal
        // ARKit camera looks in -Z; LiDAR depth is a positive distance → negate Z
        let pt   = frame.camera.transform * SIMD4<Float>(xCam, yCam, -depthVal, 1)

        return SIMD3<Float>(pt.x, pt.y, pt.z)
    }

    // MARK: - Depth estimation via LiDAR depth map
    private func estimateDepth(frame: ARFrame,
                               p3BL: SIMD3<Float>,
                               p3BR: SIMD3<Float>,
                               p3TL: SIMD3<Float>,
                               p3TR: SIMD3<Float>,
                               depth: ARDepthData) -> Double? {
        guard let sv = sceneView else { return nil }
        let viewportSize = sv.bounds.size

        let right      = simd_normalize(p3BR - p3BL)
        let up         = simd_normalize(p3TL - p3BL)
        let faceNormal = simd_normalize(simd_cross(right, up))

        let frontCenter = (p3BL + p3BR + p3TL + p3TR) / 4
        guard let frontDepth = depthAtWorldPoint(frontCenter,
                                                  frame: frame,
                                                  depthData: depth,
                                                  viewportSize: viewportSize)
        else { return nil }

        let probeOffsets: [Float] = [0.05, 0.10, 0.20, 0.35, 0.55]
        for offset in probeOffsets {
            let probe = frontCenter - faceNormal * offset
            guard let probeDepth = depthAtWorldPoint(probe,
                                                      frame: frame,
                                                      depthData: depth,
                                                      viewportSize: viewportSize)
            else { continue }
            let gap = probeDepth - frontDepth
            if gap > 0.02 { return Double(gap) * 100 }
        }
        return nil
    }

    private func depthAtWorldPoint(_ worldPt: SIMD3<Float>,
                                   frame: ARFrame,
                                   depthData: ARDepthData,
                                   viewportSize: CGSize) -> Float? {
        guard let sv = sceneView else { return nil }

        let projected = sv.projectPoint(SCNVector3(worldPt.x, worldPt.y, worldPt.z))
        let screenPt  = CGPoint(x: CGFloat(projected.x), y: CGFloat(projected.y))

        guard screenPt.x >= 0, screenPt.x < viewportSize.width,
              screenPt.y >= 0, screenPt.y < viewportSize.height
        else { return nil }

        let invDisplay = frame.displayTransform(for: .portrait,
                                               viewportSize: viewportSize).inverted()
        let normCam = CGPoint(x: screenPt.x / viewportSize.width,
                              y: screenPt.y / viewportSize.height).applying(invDisplay)

        let depthMap = depthData.depthMap
        let dW = CVPixelBufferGetWidth(depthMap)
        let dH = CVPixelBufferGetHeight(depthMap)
        let sx = max(0, min(dW - 1, Int(normCam.x * CGFloat(dW))))
        let sy = max(0, min(dH - 1, Int(normCam.y * CGFloat(dH))))

        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(depthMap) else { return nil }

        let val = base.assumingMemoryBound(to: Float32.self)[sy * dW + sx]
        return val > 0.02 && val < 8 ? val : nil
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

        // comprimento and altura are measured directly from LiDAR edge scans → stable.
        // largura comes from the noisier depth probe → just take its median.
        guard range(\.comprimento) < thresholdCm,
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

    // MARK: - Manual points (4-tap: base-left, base-right, base-back, top)
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
            let l = Double(simd_distance(p0, p2)) * 100
            let m = NativeMeasurement(comprimento: c, largura: l, altura: l)
            lastMeasurement = m
            DispatchQueue.main.async { [weak self] in self?.onUpdate(m) }
        } else if manualPoints.count == 4 {
            let p0 = manualPoints[0], p1 = manualPoints[1]
            let p2 = manualPoints[2], p3 = manualPoints[3]
            let c = Double(simd_distance(p0, p1)) * 100
            let l = Double(simd_distance(p0, p2)) * 100
            let a = Double(simd_distance(p0, p3)) * 100
            let m = NativeMeasurement(comprimento: c, largura: l, altura: a)
            lastMeasurement = m
            manualPoints.removeAll()
            overlayNodes.forEach { $0.removeFromParentNode() }
            overlayNodes.removeAll()
            DispatchQueue.main.async { [weak self] in self?.onUpdate(m) }
        }
    }

    // MARK: - Overlay
    func updateOverlay(bl: SIMD3<Float>, br: SIMD3<Float>,
                       tl: SIMD3<Float>, tr: SIMD3<Float>,
                       measurement: NativeMeasurement) {
        guard let sv = sceneView else { return }
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.clearOverlay()

            let yellow = UIColor(red: 1.0, green: 0.82, blue: 0.0, alpha: 1)

            let right      = simd_normalize(br - bl)
            let up         = simd_normalize(tl - bl)
            let faceNormal = simd_normalize(simd_cross(right, up))
            let depthM     = Float(max(measurement.largura, 2) / 100.0)
            let extrudeDir = -faceNormal

            let bbl = bl + extrudeDir * depthM
            let bbr = br + extrudeDir * depthM
            let btl = tl + extrudeDir * depthM
            let btr = tr + extrudeDir * depthM

            let edges: [(SIMD3<Float>, SIMD3<Float>)] = [
                (bl, br), (br, tr), (tr, tl), (tl, bl),
                (bbl, bbr), (bbr, btr), (btr, btl), (btl, bbl),
                (bl, bbl), (br, bbr), (tl, btl), (tr, btr),
            ]
            for (s, e) in edges {
                let node = self.makeLine(from: s, to: e, color: yellow)
                sv.scene.rootNode.addChildNode(node)
                self.overlayNodes.append(node)
            }

            let labelData: [(String, SIMD3<Float>)] = [
                ("\(Int(measurement.comprimento.rounded())) cm", (bl + br) / 2 + up * (-0.055)),
                ("\(Int(measurement.altura.rounded())) cm",      (bl + tl) / 2 + right * (-0.065)),
                ("\(Int(measurement.largura.rounded())) cm",     (br + bbr) / 2 + right * 0.065),
            ]
            for (text, pos) in labelData {
                let node = self.makeTextNode(text, color: yellow)
                node.position = SCNVector3(pos.x, pos.y, pos.z)
                sv.scene.rootNode.addChildNode(node)
                self.overlayNodes.append(node)
            }
        }
    }

    private func makeTextNode(_ text: String, color: UIColor) -> SCNNode {
        let geo = SCNText(string: text, extrusionDepth: 0)
        geo.font = UIFont.boldSystemFont(ofSize: 48)
        geo.flatness = 0.1
        geo.firstMaterial?.diffuse.contents  = color
        geo.firstMaterial?.lightingModel     = .constant
        geo.firstMaterial?.isDoubleSided     = true

        let node  = SCNNode(geometry: geo)
        let scale: Float = 0.001
        node.scale = SCNVector3(scale, scale, scale)

        let (minB, maxB) = node.boundingBox
        node.pivot = SCNMatrix4MakeTranslation(
            (maxB.x - minB.x) / 2 + minB.x,
            (maxB.y - minB.y) / 2 + minB.y,
            0
        )
        let constraint = SCNBillboardConstraint()
        constraint.freeAxes = .all
        node.constraints = [constraint]
        return node
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
        let v   = e - s
        let len = simd_length(v)
        let cyl = SCNCylinder(radius: 0.002, height: CGFloat(len))
        cyl.firstMaterial?.diffuse.contents = color
        cyl.firstMaterial?.lightingModel    = .constant

        let node = SCNNode(geometry: cyl)
        let mid  = (s + e) / 2
        node.position = SCNVector3(mid.x, mid.y, mid.z)

        let dir  = simd_normalize(v)
        let up   = SIMD3<Float>(0, 1, 0)
        let dot  = simd_dot(up, dir)

        if abs(dot) > 0.9999 {
            if dot < 0 { node.rotation = SCNVector4(1, 0, 0, Float.pi) }
        } else {
            let axis  = simd_normalize(simd_cross(up, dir))
            let angle = acos(max(-1, min(1, dot)))
            node.rotation = SCNVector4(axis.x, axis.y, axis.z, angle)
        }
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
        guard
            mode == .auto,
            frame.timestamp - lastScanTime > scanInterval,
            let depthData = frame.sceneDepth
        else { return }
        lastScanTime = frame.timestamp

        let capturedFrame = frame
        let capturedDepth = depthData
        DispatchQueue.main.async { [weak self] in
            self?.measureFromCenter(frame: capturedFrame, depth: capturedDepth)
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
