import ARKit
import SceneKit
import CoreML
import Vision

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
    private let scanInterval: TimeInterval = 0.20

    // MARK: - Stability
    private var buffer: [NativeMeasurement] = []
    private let stabilityWindow = 6
    private let thresholdCm = 3.0

    // MARK: - EMA smoothing on front-face corners
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

    // MARK: - Vision rectangle detection + LiDAR measurement
    private func measureFromCenter(frame: ARFrame, depth: ARDepthData) {
        guard let sv = sceneView else { return }
        let vp = sv.bounds.size
        let cx = vp.width / 2, cy = vp.height / 2

        guard let centerD = sampleDepth(at: CGPoint(x: cx, y: cy), frame: frame, depth: depth),
              centerD > 0.15, centerD < 4.0
        else { return }

        // ── Detectar rectángulos con Vision ────────────────────────────────────
        let request = VNDetectRectanglesRequest()
        request.minimumConfidence    = 0.4
        request.minimumAspectRatio   = VNAspectRatio(0.15)
        request.maximumAspectRatio   = VNAspectRatio(6.0)
        request.minimumSize          = Float(0.07)
        request.maximumObservations  = 10
        request.quadratureTolerance  = VNDegrees(45)

        // orientation: .right → el sensor landscape se trata como portrait
        let handler = VNImageRequestHandler(
            cvPixelBuffer: frame.capturedImage,
            orientation:   .right,
            options:       [:]
        )
        guard (try? handler.perform([request])) != nil,
              let results = request.results,
              !results.isEmpty
        else { return }

        // Convierte coords Vision (portrait, origen bottom-left, Y↑)
        // → UIKit screen (portrait, origen top-left, Y↓)
        func vToS(_ p: CGPoint) -> CGPoint {
            CGPoint(x: p.x * vp.width, y: (1 - p.y) * vp.height)
        }

        // Elegir el rectángulo cuyo centro esté más cerca del reticle
        let maxDist = min(vp.width, vp.height) * 0.42
        guard let obs = results.min(by: { a, b in
            let ca = vToS(CGPoint(x: a.boundingBox.midX, y: a.boundingBox.midY))
            let cb = vToS(CGPoint(x: b.boundingBox.midX, y: b.boundingBox.midY))
            return hypot(ca.x - cx, ca.y - cy) < hypot(cb.x - cx, cb.y - cy)
        }) else { return }

        let obsCenter = vToS(CGPoint(x: obs.boundingBox.midX, y: obs.boundingBox.midY))
        guard hypot(obsCenter.x - cx, obsCenter.y - cy) < maxDist else { return }

        // Esquinas del cuadrilátero detectado por Vision
        let tl = vToS(obs.topLeft)
        let tr = vToS(obs.topRight)
        let bl = vToS(obs.bottomLeft)
        let br = vToS(obs.bottomRight)

        // Ancho y alto mínimos en pantalla
        let faceW = hypot(br.x - bl.x, br.y - bl.y)
        let faceH = hypot(tl.y - bl.y, tl.x - bl.x)
        guard faceW > 30, faceH > 30 else { return }

        // ── Proyectar a 3D con LiDAR ───────────────────────────────────────────
        let gs: CGFloat = 30
        let dxL = sampleDepth(at: CGPoint(x: cx - gs, y: cy), frame: frame, depth: depth)
        let dxR = sampleDepth(at: CGPoint(x: cx + gs, y: cy), frame: frame, depth: depth)
        let dyT = sampleDepth(at: CGPoint(x: cx, y: cy - gs), frame: frame, depth: depth)
        let dyB = sampleDepth(at: CGPoint(x: cx, y: cy + gs), frame: frame, depth: depth)
        let gx: Float = (dxL != nil && dxR != nil && abs(dxR! - dxL!) < 0.20)
                        ? (dxR! - dxL!) / Float(2 * gs) : 0
        let gy: Float = (dyT != nil && dyB != nil && abs(dyB! - dyT!) < 0.20)
                        ? (dyB! - dyT!) / Float(2 * gs) : 0
        func expD(_ p: CGPoint) -> Float {
            centerD + gx * Float(p.x - cx) + gy * Float(p.y - cy)
        }

        guard
            let p3BL = worldPointAtDepth(bl, depth: expD(bl), frame: frame),
            let p3BR = worldPointAtDepth(br, depth: expD(br), frame: frame),
            let p3TL = worldPointAtDepth(tl, depth: expD(tl), frame: frame),
            let p3TR = worldPointAtDepth(tr, depth: expD(tr), frame: frame)
        else { return }

        let fRight     = simd_normalize(p3BR - p3BL)
        let fUp        = simd_normalize(p3TL - p3BL)
        let faceNormal = simd_normalize(simd_cross(fRight, fUp))
        guard abs(faceNormal.y) < 0.65 else { return }

        let c = Double(simd_distance(p3BL, p3BR)) * 100
        let a = Double(simd_distance(p3BL, p3TL)) * 100
        let l = estimateDepthFromTopFace(frame: frame, depth: depth,
                                         topLeft: tl, topRight: tr,
                                         centerD: centerD) ?? min(c, a) * 0.6

        guard c > 5, c < 300, a > 5, a < 300, l > 2 else { return }

        let α = smoothAlpha
        smoothBL = smoothBL.map { α * p3BL + (1-α) * $0 } ?? p3BL
        smoothBR = smoothBR.map { α * p3BR + (1-α) * $0 } ?? p3BR
        smoothTL = smoothTL.map { α * p3TL + (1-α) * $0 } ?? p3TL
        smoothTR = smoothTR.map { α * p3TR + (1-α) * $0 } ?? p3TR

        let m = NativeMeasurement(comprimento: c, largura: l, altura: a)
        addToBuffer(m)
        updateOverlay(bl: smoothBL!, br: smoothBR!, tl: smoothTL!, tr: smoothTR!, measurement: m)
    }

    // MARK: - Estimación de profundidad (largura) escaneando la cara superior
    private func estimateDepthFromTopFace(frame: ARFrame, depth: ARDepthData,
                                          topLeft: CGPoint, topRight: CGPoint,
                                          centerD: Float) -> Double? {
        var estimates = [Double]()
        for frac in [0.25, 0.50, 0.75] as [CGFloat] {
            let sx = topLeft.x + frac * (topRight.x - topLeft.x)
            var sy = topLeft.y
            guard let startD = sampleDepth(at: CGPoint(x: sx, y: sy), frame: frame, depth: depth),
                  let topPt  = worldPointAtDepth(CGPoint(x: sx, y: sy), depth: startD, frame: frame)
            else { continue }
            let topFaceY = topPt.y
            var prevPt   = topPt
            while sy > 8 {
                sy -= 5
                guard let d   = sampleDepth(at: CGPoint(x: sx, y: sy), frame: frame, depth: depth),
                      let wPt = worldPointAtDepth(CGPoint(x: sx, y: sy), depth: d, frame: frame)
                else { break }
                if wPt.y < topFaceY - 0.07 { break }
                if d - startD > 0.22        { break }
                prevPt = wPt
            }
            let boxDepth = Double(simd_distance(topPt, prevPt)) * 100
            if boxDepth > 3, boxDepth < 200 { estimates.append(boxDepth) }
        }
        return estimates.isEmpty ? nil : median(estimates)
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

    // MARK: - Manual points (3-4 toques)
    func addManualPoint(_ p: SIMD3<Float>) {
        manualPoints.append(p)

        let sphere = SCNSphere(radius: 0.01)
        sphere.firstMaterial?.diffuse.contents = UIColor.yellow
        sphere.firstMaterial?.lightingModel    = .constant
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

    // MARK: - Overlay (12 aristas + 3 etiquetas)
    func updateOverlay(bl: SIMD3<Float>, br: SIMD3<Float>,
                       tl: SIMD3<Float>, tr: SIMD3<Float>,
                       measurement: NativeMeasurement) {
        guard let sv = sceneView else { return }
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.clearOverlay()

            let green = UIColor(red: 0.0, green: 0.90, blue: 0.3, alpha: 1)

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
                let node = self.makeLine(from: s, to: e, color: green)
                sv.scene.rootNode.addChildNode(node)
                self.overlayNodes.append(node)
            }

            let labelData: [(String, SIMD3<Float>)] = [
                ("\(Int(measurement.comprimento.rounded())) cm", (bl + br) / 2 + up * (-0.055)),
                ("\(Int(measurement.altura.rounded())) cm",      (bl + tl) / 2 + right * (-0.065)),
                ("\(Int(measurement.largura.rounded())) cm",     (br + bbr) / 2 + right * 0.065),
            ]
            for (text, pos) in labelData {
                let node = self.makeTextNode(text, color: green)
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
        geo.firstMaterial?.diffuse.contents = color
        geo.firstMaterial?.lightingModel    = .constant
        geo.firstMaterial?.isDoubleSided    = true

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

        let dir = simd_normalize(v)
        let up  = SIMD3<Float>(0, 1, 0)
        let dot = simd_dot(up, dir)

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
        if anchor is ARPlaneAnchor {
            DispatchQueue.main.async { [weak self] in self?.onPlaneFound() }
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
    func sessionWasInterrupted(_ session: ARSession)    { print("[ARKit] interrupted") }
    func sessionInterruptionEnded(_ session: ARSession) { print("[ARKit] resumido") }
}
