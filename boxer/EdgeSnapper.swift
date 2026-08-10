import ARKit
import CoreGraphics

/// Detecta el borde de objeto más cercano al centro del crosshair usando
/// el gradiente del depth map LiDAR. Un salto brusco de profundidad entre
/// píxeles adyacentes = arista de objeto. Igual al comportamiento de Measure.
enum EdgeSnapper {

    /// Busca la arista más cercana a `center` dentro de `searchRadius` puntos de pantalla.
    /// Devuelve el punto de pantalla de la arista, o nil si no hay ninguna cerca.
    static func nearest(
        frame: ARFrame,
        near center: CGPoint,
        viewportSize: CGSize,
        searchRadius: CGFloat = 44,
        threshold: Float = 0.04         // 4 cm de salto de profundidad por píxel = arista
    ) -> CGPoint? {
        guard let depthBuf = (frame.smoothedSceneDepth ?? frame.sceneDepth)?.depthMap else { return nil }
        let dW = CVPixelBufferGetWidth(depthBuf)
        let dH = CVPixelBufferGetHeight(depthBuf)

        // Convertir centro de pantalla → píxel en el depth map
        let inv  = frame.displayTransform(for: .portrait, viewportSize: viewportSize).inverted()
        let norm = CGPoint(x: center.x / viewportSize.width,
                           y: center.y / viewportSize.height).applying(inv)
        let cDX  = Int(norm.x * CGFloat(dW))
        let cDY  = Int(norm.y * CGFloat(dH))

        // Radio de búsqueda en coordenadas del depth map
        let rX = Int(searchRadius * CGFloat(dW) / viewportSize.width)
        let rY = Int(searchRadius * CGFloat(dH) / viewportSize.height)

        CVPixelBufferLockBaseAddress(depthBuf, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuf, .readOnly) }
        let rb = CVPixelBufferGetBytesPerRow(depthBuf)
        guard let base = CVPixelBufferGetBaseAddress(depthBuf) else { return nil }

        // Acceso seguro al depth map
        func d(_ x: Int, _ y: Int) -> Float {
            let px = max(0, min(dW - 1, x))
            let py = max(0, min(dH - 1, y))
            return base.advanced(by: py * rb).assumingMemoryBound(to: Float32.self)[px]
        }

        var bestSq = Int.max
        var bestX  = -1
        var bestY  = -1

        for dy in -rY...rY {
            for dx in -rX...rX {
                let px = cDX + dx
                let py = cDY + dy
                guard px > 1, px < dW - 2, py > 1, py < dH - 2 else { continue }

                // Sólo considero este píxel si está más cerca del centro que el mejor actual
                let sq = dx * dx + dy * dy
                guard sq < bestSq else { continue }

                // Filtrar lecturas inválidas
                let v = d(px, py)
                guard v > 0.05, v < 8.0 else { continue }

                // Gradiente de profundidad en X e Y (central difference de 2 píxeles)
                let gx = abs(d(px + 1, py) - d(px - 1, py))
                let gy = abs(d(px, py + 1) - d(px, py - 1))

                if max(gx, gy) > threshold {
                    bestSq = sq
                    bestX  = px
                    bestY  = py
                }
            }
        }

        guard bestX >= 0 else { return nil }

        // Convertir píxel del depth map → punto de pantalla
        let normSnap = CGPoint(x: CGFloat(bestX) / CGFloat(dW),
                               y: CGFloat(bestY) / CGFloat(dH))
        let fwd = frame.displayTransform(for: .portrait, viewportSize: viewportSize)
        let s   = normSnap.applying(fwd)
        return CGPoint(x: s.x * viewportSize.width, y: s.y * viewportSize.height)
    }
}
