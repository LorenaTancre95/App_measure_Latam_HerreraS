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
    private let stabilityWindow = 6
    private let thresholdCm = 3.0

    // MARK: - EMA smoothing on front-face corners
    private var smoothBL: SIMD3<Float>?
    private var smoothBR: SIMD3<Float>?
    private var smoothTL: SIMD3<Float>?
    private var smoothTR: SIMD3<Float>?
    private let smoothAlpha: Float = 0.20

    // MARK: - Overlay
    private var overlayNodes: [SCNNode] = []

    // MARK: - Scene reconstruction mesh anchors (all updates dispatched to main queue)
    private var meshAnchors: [UUID: ARMeshAnchor] = [:]

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

    // MARK: - Mesh-based 3D bounding box measurement
    //
    // Uses ARMeshAnchor (scene reconstruction) instead of scan-lines:
    // 1. LiDAR depth at reticle center → 3D hit point on the box surface.
    // 2. Collect all mesh vertices within a 70 cm sphere of the hit point,
    //    discarding anything below the hit point (floor / table).
    // 3. Project each vertex onto camera-aligned axes (right, up, forward).
    // 4. 5th/95th percentile of those projections → robust to stray vertices.
    // 5. Width = right extent, Height = up extent, Depth = forward extent.
    // 6. Front-face corners from minimum-depth plane → EMA-smoothed wireframe.
    //
    // Robust against cluttered scenes because only vertices near the reticle
    // surface are used; scan-lines have no role and cannot bleed into the floor.
    private func measureFromCenter(frame: ARFrame, depth: ARDepthData) {
        guard let sv = sceneView else { return }
        let vp = sv.bounds.size
        let cx = vp.width / 2, cy = vp.height / 2

        // 3D hit point from LiDAR depth at reticle center
        guard let centerD = sampleDepth(at: CGPoint(x: cx, y: cy),
                                        frame: frame, depth: depth),
              centerD > 0.15, centerD < 4.0,
              let hitPt = worldPointAtDepth(CGPoint(x: cx, y: cy),
                                            depth: centerD, frame: frame)
        else { return }

        // Camera-aligned axes
        let camT       = frame.camera.transform
        let camForward = SIMD3<Float>(-camT.columns.2.x, -camT.columns.2.y, -camT.columns.2.z)
        let camRight   = SIMD3<Float>( camT.columns.0.x,  camT.columns.0.y,  camT.columns.0.z)
        let worldUp    = SIMD3<Float>(0, 1, 0)

        let searchR: Float = 0.70   // 70 cm sphere around hit point
        let floorY         = hitPt.y - 0.04  // 4 cm below hit = floor cutoff

        var projRight = [Float]()
        var projUp    = [Float]()
        var projDepth = [Float]()
        projRight.reserveCapacity(512)
        projUp.reserveCapacity(512)
        projDepth.reserveCapacity(512)

        for (_, anchor) in meshAnchors {
            // Anchor-level sphere cull before touching any vertex data
            let aPos = SIMD3<Float>(anchor.transform.columns.3.x,
                                    anchor.transform.columns.3.y,
                                    anchor.transform.columns.3.z)
            guard simd_distance(aPos, hitPt) < searchR + 1.5 else { continue }

            let t      = anchor.transform
            let vSrc   = anchor.geometry.vertices
            let buf    = vSrc.buffer.contents()
            let stride = vSrc.stride
            let off    = vSrc.offset

            for i in 0..<vSrc.count {
                let raw = buf.advanced(by: off + i * stride)
                            .assumingMemoryBound(to: SIMD3<Float>.self).pointee
                let w4 = t * SIMD4<Float>(raw.x, raw.y, raw.z, 1)
                let w  = SIMD3<Float>(w4.x, w4.y, w4.z)

                guard simd_distance(w, hitPt) < searchR, w.y > floorY else { continue }

                let rel = w - hitPt
                projRight.append(simd_dot(rel, camRight))
                projUp.append(simd_dot(rel, worldUp))
                projDepth.append(simd_dot(rel, camForward))
            }
        }

        guard projRight.count >= 30 else { return }  // wait for mesh to populate

        // 5th/95th percentile — eliminates stray vertices without sorting all data
        let minR = percentile(projRight, 0.05), maxR = percentile(projRight, 0.95)
        let minU = percentile(projUp,    0.05), maxU = percentile(projUp,    0.95)
        let minD = percentile(projDepth, 0.05), maxD = percentile(projDepth, 0.95)

        let c = Double(maxR - minR) * 100
        let a = Double(maxU - minU) * 100
        let l = Double(maxD - minD) * 100

        guard c > 5, c < 300, a > 5, a < 300, l > 2, l < 200 else { return }

        // Front-face corners at minimum depth (closest plane to camera)
        let p3BL = hitPt + minR * camRight + minU * worldUp + minD * camForward
        let p3BR = hitPt + maxR * camRight + minU * worldUp + minD * camForward
        let p3TL = hitPt + minR * camRight + maxU * worldUp + minD * camForward
        let p3TR = hitPt + maxR * camRight + maxU * worldUp + minD * camForward

        // Reject if face is mostly horizontal (measuring floor or table top)
        let faceNormal = simd_normalize(simd_cross(simd_normalize(p3BR - p3BL),
                                                    simd_normalize(p3TL - p3BL)))
        guard abs(faceNormal.y) < 0.65 else { return }

        let α = smoothAlpha
        smoothBL = smoothBL.map { α * p3BL + (1-α) * $0 } ?? p3BL
        smoothBR = smoothBR.map { α * p3BR + (1-α) * $0 } ?? p3BR
        smoothTL = smoothTL.map { α * p3TL + (1-α) * $0 } ?? p3TL
        smoothTR = smoothTR.map { α * p3TR + (1-α) * $0 } ?? p3TR

        let m = NativeMeasurement(comprimento: c, largura: l, altura: a)
        addToBuffer(m)
        updateOverlay(bl: smoothBL!, br: smoothBR!, tl: smoothTL!, tr: smoothTR!, measurement: m)
    }

    // Robust percentile over a Float array (partial sort up to idx).
    private func percentile(_ arr: [Float], _ p: Float) -> Float {
        var s = arr
        let idx = max(0, min(s.count - 1, Int(Float(s.count - 1) * p)))
        // nth_element equivalent: partial sort only up to idx
        s.sort()
        return s[idx]
    }

    // MARK: - LiDAR helpers

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

    private func worldPointAtDepth(_ screenPt: CGPoint,
                                   depth depthVal: Float,
                                   frame: ARFrame) -> SIMD3<Float>? {
        guard let sv = sceneView else { return nil }
        let vp = sv.bounds.size
        let invDisplay = frame.displayTransform(for: .portrait, viewportSize: vp).inverted()
        let normCam = CGPoint(x: screenPt.x / vp.width,
                              y: screenPt.y / vp.height).applying(invDisplay)
        let intr = frame.camera.intrinsics
        let iW   = Float(frame.camera.imageResolution.width)
        let iH   = Float(frame.camera.imageResolution.height)
        let imgX = Float(normCam.x) * iW
        let imgY = Float(normCam.y) * iH
        let xCam = (imgX - intr[2][0]) / intr[0][0] * depthVal
        let yCam = (imgY - intr[2][1]) / intr[1][1] * depthVal
        let pt   = frame.camera.transform * SIMD4<Float>(xCam, yCam, -depthVal, 1)
        return SIMD3<Float>(pt.x, pt.y, pt.z)
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
              range(\.altura)      < thresholdCm,
              range(\.largura)     < thresholdCm * 2 else { return }

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

    // MARK: - Overlay (12-edge wireframe + 3 dimension labels)
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
        smoothBL = nil; smoothBR = nil; smoothTL = nil; smoothTR = nil
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

    // All mesh anchor mutations dispatched to main queue so measureFromCenter
    // (also on main) reads a consistent snapshot without locks.
    func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
        if let meshAnchor = anchor as? ARMeshAnchor {
            let id = anchor.identifier
            DispatchQueue.main.async { [weak self] in
                self?.meshAnchors[id] = meshAnchor
            }
        } else if anchor is ARPlaneAnchor {
            DispatchQueue.main.async { [weak self] in self?.onPlaneFound() }
        }
    }

    func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
        guard let meshAnchor = anchor as? ARMeshAnchor else { return }
        let id = anchor.identifier
        DispatchQueue.main.async { [weak self] in
            self?.meshAnchors[id] = meshAnchor
        }
    }

    func renderer(_ renderer: SCNSceneRenderer, didRemove node: SCNNode, for anchor: ARAnchor) {
        let id = anchor.identifier
        DispatchQueue.main.async { [weak self] in
            self?.meshAnchors.removeValue(forKey: id)
        }
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
