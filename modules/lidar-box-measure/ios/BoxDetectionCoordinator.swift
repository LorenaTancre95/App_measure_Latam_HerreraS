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

    // MARK: - Depth-map region growing measurement
    // BFS flood-fill sobre el depth map: todos los píxeles conectados cuya
    // profundidad esté dentro de ±6 cm de centerD forman la región de la caja.
    // El bounding box de esa región da las esquinas de la cara frontal sin
    // necesidad de scan-lines manuales, gradiente, ni múltiples validaciones.
    private func measureFromCenter(frame: ARFrame, depth: ARDepthData) {
        guard let sv = sceneView else { return }
        let vp = sv.bounds.size
        let cx = vp.width / 2, cy = vp.height / 2

        guard let centerD = sampleDepth(at: CGPoint(x: cx, y: cy), frame: frame, depth: depth),
              centerD > 0.15, centerD < 4.0
        else { return }

        let dm  = depth.depthMap
        let dW  = CVPixelBufferGetWidth(dm)
        let dH  = CVPixelBufferGetHeight(dm)

        // Screen center → depth map pixel
        let invDisplay = frame.displayTransform(for: .portrait, viewportSize: vp).inverted()
        let normCam    = CGPoint(x: cx / vp.width, y: cy / vp.height).applying(invDisplay)
        let cDX        = max(0, min(dW - 1, Int(normCam.x * CGFloat(dW))))
        let cDY        = max(0, min(dH - 1, Int(normCam.y * CGFloat(dH))))

        guard let region = growRegion(depthMap: dm, cx: cDX, cy: cDY,
                                       centerD: centerD, dW: dW, dH: dH)
        else { return }

        // Depth map bounding box → screen coordinates.
        // The displayTransform includes a 90° rotation (landscape camera → portrait screen),
        // so depth map corner (minX,minY) does NOT map to screen top-left. We compute all 4
        // corners and sort by screen Y/X to get correctly labelled tl/tr/bl/br.
        let displayTx = frame.displayTransform(for: .portrait, viewportSize: vp)
        func depthToScreen(_ dx: Int, _ dy: Int) -> CGPoint {
            let nc = CGPoint(x: CGFloat(dx) / CGFloat(dW), y: CGFloat(dy) / CGFloat(dH))
            let ns = nc.applying(displayTx)
            return CGPoint(x: ns.x * vp.width, y: ns.y * vp.height)
        }

        let corners = [
            depthToScreen(region.minX, region.minY),
            depthToScreen(region.minX, region.maxY),
            depthToScreen(region.maxX, region.minY),
            depthToScreen(region.maxX, region.maxY),
        ]
        let byY    = corners.sorted { $0.y < $1.y }
        let topTwo = Array(byY.prefix(2)).sorted { $0.x < $1.x }
        let botTwo = Array(byY.suffix(2)).sorted { $0.x < $1.x }
        let tl = topTwo[0], tr = topTwo[1]
        let bl = botTwo[0], br = botTwo[1]

        guard br.x - bl.x > 20, bl.y - tl.y > 20 else { return }

        // Gradient correction at center — handles angled faces for corner unprojection
        let gs: CGFloat = 30
        let dxL = sampleDepth(at: CGPoint(x: cx - gs, y: cy), frame: frame, depth: depth)
        let dxR = sampleDepth(at: CGPoint(x: cx + gs, y: cy), frame: frame, depth: depth)
        let dyT = sampleDepth(at: CGPoint(x: cx, y: cy - gs), frame: frame, depth: depth)
        let dyB = sampleDepth(at: CGPoint(x: cx, y: cy + gs), frame: frame, depth: depth)
        let gx: Float = (dxL != nil && dxR != nil && abs(dxR! - dxL!) < 0.20)
                        ? (dxR! - dxL!) / Float(2 * gs) : 0
        let gy: Float = (dyT != nil && dyB != nil && abs(dyB! - dyT!) < 0.20)
                        ? (dyB! - dyT!) / Float(2 * gs) : 0
        func expD(_ px: CGFloat, _ py: CGFloat) -> Float {
            centerD + gx * Float(px - cx) + gy * Float(py - cy)
        }

        guard
            let p3BL = worldPointAtDepth(bl, depth: expD(bl.x, bl.y), frame: frame),
            let p3BR = worldPointAtDepth(br, depth: expD(br.x, br.y), frame: frame),
            let p3TL = worldPointAtDepth(tl, depth: expD(tl.x, tl.y), frame: frame),
            let p3TR = worldPointAtDepth(tr, depth: expD(tr.x, tr.y), frame: frame)
        else { return }

        let fRight     = simd_normalize(p3BR - p3BL)
        let fUp        = simd_normalize(p3TL - p3BL)
        let faceNormal = simd_normalize(simd_cross(fRight, fUp))
        guard abs(faceNormal.y) < 0.65 else { return }

        let c = Double(simd_distance(p3BL, p3BR)) * 100
        let a = Double(simd_distance(p3BL, p3TL)) * 100
        let l = estimateDepthFromTopFace(frame: frame, depth: depth,
                                         topLeft: tl, topRight: tr,
                                         centerD: centerD,
                                         faceNormal: faceNormal) ?? min(c, a) * 0.6

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

    // MARK: - BFS flood-fill sobre depth map
    // Expande desde el píxel central a todos los píxeles conectados cuya profundidad
    // esté dentro de `threshold` de `centerD`. Cap de 40 % del mapa evita que el
    // piso o paredes al mismo depth llenen toda la imagen.
    private func growRegion(depthMap: CVPixelBuffer, cx: Int, cy: Int,
                             centerD: Float, dW: Int, dH: Int)
        -> (minX: Int, maxX: Int, minY: Int, maxY: Int)?
    {
        let threshold: Float = 0.08   // 8 cm: cardboard has higher LiDAR variance than 6 cm
        let maxPixels        = dW * dH * 2 / 5

        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(depthMap) else { return nil }
        let ptr = base.assumingMemoryBound(to: Float32.self)

        var visited = [Bool](repeating: false, count: dW * dH)
        var queue   = [(Int, Int)]()
        queue.reserveCapacity(min(dW * dH / 4, 8192))

        visited[cy * dW + cx] = true
        queue.append((cx, cy))

        var minX = cx, maxX = cx, minY = cy, maxY = cy
        var count = 0, qi = 0

        while qi < queue.count, count < maxPixels {
            let (x, y) = queue[qi]; qi += 1; count += 1
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] as [(Int, Int)] {
                let nx = x + dx, ny = y + dy
                guard nx >= 0, nx < dW, ny >= 0, ny < dH else { continue }
                let idx = ny * dW + nx
                guard !visited[idx] else { continue }
                visited[idx] = true
                let v = ptr[idx]
                guard v > 0.02, v < 8.0, abs(v - centerD) < threshold else { continue }
                queue.append((nx, ny))
                if nx < minX { minX = nx }; if nx > maxX { maxX = nx }
                if ny < minY { minY = ny }; if ny > maxY { maxY = ny }
            }
        }

        guard (maxX - minX) > 5, (maxY - minY) > 5, count > 50 else { return nil }
        return (minX, maxX, minY, maxY)
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

    // MARK: - Depth estimation via top-face scan (world-Y plane anchor)
    // The box top face is a horizontal plane in world space — all points on it share
    // approximately the same world Y. Scanning upward in screen space follows that plane.
    // When we cross the back edge the depth map hits the floor or background, whose
    // world Y drops clearly below the top-face plane. That drop is the stopping criterion.
    // This replaces the previous heuristic (jump from previous sample) which failed when
    // the floor behind the box had a gradual depth gradient matching the top face.
    // Three columns (25 / 50 / 75 % across the top edge) are scanned; median is returned.
    private func estimateDepthFromTopFace(frame: ARFrame,
                                           depth: ARDepthData,
                                           topLeft: CGPoint,
                                           topRight: CGPoint,
                                           centerD: Float,
                                           faceNormal: SIMD3<Float>) -> Double? {
        guard let sv = sceneView else { return nil }
        let vp         = sv.bounds.size
        let step: CGFloat = 3
        let limT       = vp.height * 0.03
        let topCenterX = (topLeft.x + topRight.x) / 2
        let topY       = topLeft.y

        // Anchor: actual LiDAR depth at the top-edge center (more accurate than centerD).
        let topEdgeD = sampleDepth(at: CGPoint(x: topCenterX, y: topY),
                                   frame: frame, depth: depth) ?? centerD
        guard let frontPt3 = worldPointAtDepth(CGPoint(x: topCenterX, y: topY),
                                                depth: topEdgeD,
                                                frame: frame)
        else { return nil }

        let topFaceY: Float   = frontPt3.y
        let yTolerance: Float = 0.07   // 7 cm allows LiDAR noise + slight box flex

        // Three columns across the top edge for robustness against one bad column.
        let scanXs: [CGFloat] = [
            topLeft.x + (topRight.x - topLeft.x) * 0.25,
            topCenterX,
            topLeft.x + (topRight.x - topLeft.x) * 0.75,
        ]

        var depthResults: [Double] = []

        for scanX in scanXs {
            var y         = topY - step
            var lastGoodY = topY
            var lastD     = topEdgeD

            while y >= limT {
                guard let d = sampleDepth(at: CGPoint(x: scanX, y: y),
                                          frame: frame, depth: depth),
                      let wPt = worldPointAtDepth(CGPoint(x: scanX, y: y),
                                                  depth: d, frame: frame)
                else { break }

                // Still on the top face if world Y stays within tolerance of the anchor.
                // Floor/background behind the box sits lower → wPt.y drops below threshold.
                if wPt.y < topFaceY - yTolerance { break }
                if wPt.y > topFaceY + yTolerance { break }

                lastGoodY = y
                lastD     = d
                y        -= step
            }

            guard topY - lastGoodY > step,
                  let backPt3 = worldPointAtDepth(CGPoint(x: scanX, y: lastGoodY),
                                                  depth: lastD, frame: frame)
            else { continue }

            let vec              = backPt3 - frontPt3
            let depthAlongNormal = abs(simd_dot(vec, faceNormal))
            let distDirect       = Double(simd_length(vec)) * 100
            let result           = depthAlongNormal > 0.01
                                   ? Double(depthAlongNormal) * 100
                                   : distDirect
            if result > 2 && result < 200 { depthResults.append(result) }
        }

        guard depthResults.count >= 2 else { return nil }
        return median(depthResults)
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
