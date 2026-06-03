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

    // MARK: - Overlay smoothing (exponential moving average on corner positions)
    private var smoothBL: SIMD3<Float>?
    private var smoothBR: SIMD3<Float>?
    private var smoothTL: SIMD3<Float>?
    private var smoothTR: SIMD3<Float>?
    private let smoothAlpha: Float = 0.25

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

    // MARK: - Depth-scan measurement
    // Scans 5 horizontal + 5 vertical lines from the reticle center.
    // Uses median edge positions (robust to outlier scan lines).
    // Unprojects corners using the known face depth (centerD) so edge-pixel depth
    // noise doesn't corrupt the 3D corner positions.
    // Applies exponential smoothing to corner positions to reduce wireframe jitter.
    private func measureFromCenter(frame: ARFrame, depth: ARDepthData) {
        guard let sv = sceneView else { return }
        let vp = sv.bounds.size
        let cx = vp.width / 2, cy = vp.height / 2

        guard let centerD = sampleDepth(at: CGPoint(x: cx, y: cy), frame: frame, depth: depth),
              centerD > 0.15, centerD < 4.0
        else { return }

        let threshold: Float = 0.05   // 5 cm depth jump = box edge
        let step: CGFloat    = 4
        let limL = vp.width  * 0.08, limR = vp.width  * 0.92
        let limT = vp.height * 0.06, limB = vp.height * 0.94

        func edgeX(fromX: CGFloat, y: CGFloat, dx: CGFloat) -> CGFloat? {
            var x = fromX + dx
            while (dx < 0 ? x >= limL : x <= limR) {
                if let d = sampleDepth(at: CGPoint(x: x, y: y), frame: frame, depth: depth) {
                    if abs(d - centerD) > threshold { return x - dx }
                } else {
                    return x - dx
                }
                x += dx
            }
            return nil
        }

        func edgeY(x: CGFloat, fromY: CGFloat, dy: CGFloat) -> CGFloat? {
            var y = fromY + dy
            while (dy < 0 ? y >= limT : y <= limB) {
                if let d = sampleDepth(at: CGPoint(x: x, y: y), frame: frame, depth: depth) {
                    if abs(d - centerD) > threshold { return y - dy }
                } else {
                    return y - dy
                }
                y += dy
            }
            return nil
        }

        // 5 horizontal lines: validate each starts on the same surface as center
        let yOffs: [CGFloat] = [-0.15, -0.07, 0, 0.07, 0.15].map { $0 * vp.height }
        var lefts: [CGFloat] = [], rights: [CGFloat] = []
        for yOff in yOffs {
            let scanY = cy + yOff
            guard scanY >= limT, scanY <= limB,
                  let d = sampleDepth(at: CGPoint(x: cx, y: scanY), frame: frame, depth: depth),
                  abs(d - centerD) < threshold
            else { continue }
            if let lx = edgeX(fromX: cx, y: scanY, dx: -step) { lefts.append(lx) }
            if let rx = edgeX(fromX: cx, y: scanY, dx: +step) { rights.append(rx) }
        }

        // 5 vertical lines
        let xOffs: [CGFloat] = [-0.15, -0.07, 0, 0.07, 0.15].map { $0 * vp.width }
        var tops: [CGFloat] = [], bottoms: [CGFloat] = []
        for xOff in xOffs {
            let scanX = cx + xOff
            guard scanX >= limL, scanX <= limR,
                  let d = sampleDepth(at: CGPoint(x: scanX, y: cy), frame: frame, depth: depth),
                  abs(d - centerD) < threshold
            else { continue }
            if let ty = edgeY(x: scanX, fromY: cy, dy: -step) { tops.append(ty) }
            if let by = edgeY(x: scanX, fromY: cy, dy: +step) { bottoms.append(by) }
        }

        // Need at least 2 valid readings per direction
        guard lefts.count >= 2, rights.count >= 2,
              tops.count  >= 2, bottoms.count >= 2 else { return }

        // Median edge positions — robust to outlier scan lines
        let finalLeft   = medianCG(lefts)
        let finalRight  = medianCG(rights)
        let finalTop    = medianCG(tops)
        let finalBottom = medianCG(bottoms)

        guard finalRight - finalLeft > 30, finalBottom - finalTop > 30 else { return }

        let bl = CGPoint(x: finalLeft,  y: finalBottom)
        let br = CGPoint(x: finalRight, y: finalBottom)
        let tl = CGPoint(x: finalLeft,  y: finalTop)
        let tr = CGPoint(x: finalRight, y: finalTop)

        // Use centerD for all corners — avoids noisy mixed-depth readings at edges
        guard
            let p3BL = worldPointAtDepth(bl, depth: centerD, frame: frame),
            let p3BR = worldPointAtDepth(br, depth: centerD, frame: frame),
            let p3TL = worldPointAtDepth(tl, depth: centerD, frame: frame),
            let p3TR = worldPointAtDepth(tr, depth: centerD, frame: frame)
        else { return }

        // Reject horizontal surfaces (floor, table)
        let fRight     = simd_normalize(p3BR - p3BL)
        let fUp        = simd_normalize(p3TL - p3BL)
        let faceNormal = simd_normalize(simd_cross(fRight, fUp))
        guard abs(faceNormal.y) < 0.65 else { return }

        let c = Double(simd_distance(p3BL, p3BR)) * 100
        let a = Double(simd_distance(p3BL, p3TL)) * 100
        // Estimate depth from the top face (if the user tilts slightly, the top is visible)
        let l = estimateDepthFromTopFace(frame: frame,
                                         depth: depth,
                                         topLeft: tl, topRight: tr,
                                         centerD: centerD,
                                         faceNormal: faceNormal) ?? min(c, a) * 0.6

        guard c > 5, c < 300, a > 5, a < 300, l > 2 else { return }

        // Exponential moving average to smooth corner jitter
        let α = smoothAlpha
        smoothBL = smoothBL.map { α * p3BL + (1-α) * $0 } ?? p3BL
        smoothBR = smoothBR.map { α * p3BR + (1-α) * $0 } ?? p3BR
        smoothTL = smoothTL.map { α * p3TL + (1-α) * $0 } ?? p3TL
        smoothTR = smoothTR.map { α * p3TR + (1-α) * $0 } ?? p3TR

        let m = NativeMeasurement(comprimento: c, largura: l, altura: a)
        addToBuffer(m)
        updateOverlay(bl: smoothBL!, br: smoothBR!, tl: smoothTL!, tr: smoothTR!, measurement: m)
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

    // MARK: - LiDAR unproject (using provided depth — avoids noisy edge-pixel readings)
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

    // MARK: - Depth estimation via top-face scan
    // Scans upward from the top edge of the detected front face.
    // If the top face of the box is visible (camera tilted slightly down), its extent
    // from the front-top edge to the back-top edge gives the box depth (largura).
    private func estimateDepthFromTopFace(frame: ARFrame,
                                           depth: ARDepthData,
                                           topLeft: CGPoint,
                                           topRight: CGPoint,
                                           centerD: Float,
                                           faceNormal: SIMD3<Float>) -> Double? {
        guard let sv = sceneView else { return nil }
        let vp   = sv.bounds.size
        let topX = (topLeft.x + topRight.x) / 2
        let topY = topLeft.y
        let step: CGFloat = 4
        let limT = vp.height * 0.03

        // Walk upward from top edge; collect depth samples on the top face
        var y = topY - step
        var backY: CGFloat? = nil
        var lastD: Float = centerD

        while y >= limT {
            guard let d = sampleDepth(at: CGPoint(x: topX, y: y),
                                      frame: frame, depth: depth) else {
                backY = y + step; break
            }
            // Top face depth is usually similar to or slightly larger than centerD.
            // A jump > 12 cm indicates we left the box.
            if abs(d - centerD) > 0.12 {
                backY = y + step; break
            }
            lastD = d
            y -= step
        }

        guard let beY = backY else { return nil }

        let frontPt3 = worldPointAtDepth(CGPoint(x: topX, y: topY),
                                          depth: centerD, frame: frame)
        guard let backD = sampleDepth(at: CGPoint(x: topX, y: beY),
                                      frame: frame, depth: depth),
              let backPt3 = worldPointAtDepth(CGPoint(x: topX, y: beY),
                                               depth: backD, frame: frame),
              let fp3 = frontPt3
        else { return nil }

        let dist = Double(simd_distance(fp3, backPt3)) * 100
        return dist > 2 && dist < 200 ? dist : nil
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

    private func medianCG(_ arr: [CGFloat]) -> CGFloat {
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
