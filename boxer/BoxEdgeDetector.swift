import ARKit
import simd

/// Detecta los bordes izquierdo, derecho y superior de una caja centrada en pantalla
/// escaneando el depth map LiDAR en 4 direcciones desde el centro.
/// Un salto de profundidad > threshold en al menos 2 píxeles consecutivos = borde físico.
struct BoxEdgeDetector {

    struct Result {
        /// Puntos de pantalla donde termina la superficie de la caja (último pixel foreground).
        let leftScreen:  CGPoint
        let rightScreen: CGPoint
        let topScreen:   CGPoint
    }

    static func detect(
        frame: ARFrame,
        viewportSize: CGSize,
        depthThreshold: Float = 0.06   // 6 cm de salto = borde
    ) -> Result? {

        guard let depthBuf = (frame.smoothedSceneDepth ?? frame.sceneDepth)?.depthMap
        else { return nil }

        let dW = CVPixelBufferGetWidth(depthBuf)
        let dH = CVPixelBufferGetHeight(depthBuf)
        let rb = CVPixelBufferGetBytesPerRow(depthBuf)

        CVPixelBufferLockBaseAddress(depthBuf, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuf, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(depthBuf) else { return nil }

        // Transformación pantalla → depth map (maneja la rotación portrait/landscape)
        let inv = frame.displayTransform(for: .portrait, viewportSize: viewportSize).inverted()

        func depthAt(_ screenPt: CGPoint) -> Float {
            let norm = CGPoint(x: screenPt.x / viewportSize.width,
                               y: screenPt.y / viewportSize.height).applying(inv)
            let dx = max(0, min(dW - 1, Int(norm.x * CGFloat(dW))))
            let dy = max(0, min(dH - 1, Int(norm.y * CGFloat(dH))))
            return base.advanced(by: dy * rb).assumingMemoryBound(to: Float32.self)[dx]
        }

        // Profundidad mediana del centro (referencia = cara de la caja)
        let center = CGPoint(x: viewportSize.width / 2, y: viewportSize.height / 2)
        var centerSamples: [Float] = []
        for d: CGFloat in [-8, -4, 0, 4, 8] {
            for e: CGFloat in [-8, -4, 0, 4, 8] {
                let v = depthAt(CGPoint(x: center.x + d, y: center.y + e))
                if v > 0.05, v < 8.0 { centerSamples.append(v) }
            }
        }
        guard centerSamples.count >= 5 else { return nil }
        centerSamples.sort()
        let ref = centerSamples[centerSamples.count / 2]
        guard ref > 0.15, ref < 5.0 else { return nil }

        let step: CGFloat = 4   // paso en píxeles de pantalla

        // Escanea desde el centro en la dirección dada.
        // Devuelve el último punto foreground antes del salto de profundidad.
        func scan(dx: CGFloat, dy: CGFloat) -> CGPoint? {
            var consecJumps = 0
            var edgePt: CGPoint? = nil

            let maxSteps = Int(max(viewportSize.width, viewportSize.height) / step) + 1
            for i in 1...maxSteps {
                let pt = CGPoint(x: center.x + dx * CGFloat(i) * step,
                                 y: center.y + dy * CGFloat(i) * step)
                guard pt.x >= 2, pt.x <= viewportSize.width  - 2,
                      pt.y >= 2, pt.y <= viewportSize.height - 2 else { break }

                let d = depthAt(pt)
                let invalid = d < 0.05 || d > 8.0
                let jumped  = !invalid && abs(d - ref) > depthThreshold

                if jumped || invalid {
                    if edgePt == nil {
                        // Último punto bueno = el anterior al salto
                        edgePt = CGPoint(x: center.x + dx * CGFloat(i - 1) * step,
                                         y: center.y + dy * CGFloat(i - 1) * step)
                    }
                    consecJumps += 1
                    if consecJumps >= 2 { return edgePt }
                } else {
                    consecJumps = 0
                    edgePt = nil
                }
            }
            return nil
        }

        guard let leftPt  = scan(dx: -1, dy:  0),
              let rightPt = scan(dx:  1, dy:  0),
              let topPt   = scan(dx:  0, dy: -1)
        else { return nil }

        // Verificar sanidad: el ancho detectado debe ser razonable (> 5 cm en pantalla)
        let aparentWidth = rightPt.x - leftPt.x
        guard aparentWidth > 20 else { return nil }   // < 20 px → falso positivo

        return Result(leftScreen: leftPt, rightScreen: rightPt, topScreen: topPt)
    }
}
