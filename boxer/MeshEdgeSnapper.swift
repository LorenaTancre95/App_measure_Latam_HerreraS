import ARKit
import simd

/// Detecta aristas geométricas reales usando la malla 3D reconstruida por ARKit (LiDAR).
/// Una arista es un borde entre dos caras del mesh con ángulo diedro > umbral.
/// Reconstruye la lista de aristas cada 0.4s y retorna la más cercana al crosshair.
final class MeshEdgeSnapper {

    private var sharpEdges: [(simd_float3, simd_float3)] = []
    private var lastRebuild: TimeInterval = -1
    private let rebuildInterval: TimeInterval = 0.4
    private let minCos: Float   // cos(minAngleDeg) — por debajo de esto = arista

    init(minAngleDeg: Float = 25) {
        minCos = cos(minAngleDeg * .pi / 180)
    }

    /// Llamar desde el render thread cada 0.1s.
    /// Devuelve el punto de pantalla de la arista más cercana dentro de `searchRadius` px.
    func nearest(
        frame: ARFrame,
        near center: CGPoint,
        viewportSize: CGSize,
        searchRadius: CGFloat = 44,
        time: TimeInterval
    ) -> CGPoint? {
        if time - lastRebuild > rebuildInterval {
            rebuild(frame: frame)
            lastRebuild = time
        }
        guard !sharpEdges.isEmpty else { return nil }

        var bestSq = searchRadius * searchRadius
        var bestPt: CGPoint? = nil
        let cam = frame.camera

        for (v1, v2) in sharpEdges {
            // Muestreo en 3 puntos del segmento de arista
            for t: Float in [0, 0.5, 1] {
                let wp = v1 + t * (v2 - v1)
                guard let sp = project(wp, camera: cam, size: viewportSize) else { continue }
                let dx = sp.x - center.x; let dy = sp.y - center.y
                let d2 = dx*dx + dy*dy
                if d2 < bestSq { bestSq = d2; bestPt = sp }
            }
        }
        return bestPt
    }

    // MARK: - Reconstrucción de aristas

    private func rebuild(frame: ARFrame) {
        let camPos = simd_float3(frame.camera.transform.columns.3.x,
                                  frame.camera.transform.columns.3.y,
                                  frame.camera.transform.columns.3.z)
        // Solo procesar anchors dentro de 2.5m (evita procesar toda la habitación)
        let nearAnchors = frame.anchors
            .compactMap { $0 as? ARMeshAnchor }
            .filter {
                let p = $0.transform.columns.3
                return simd_distance(camPos, simd_float3(p.x, p.y, p.z)) < 2.5
            }

        var edges: [(simd_float3, simd_float3)] = []
        for anchor in nearAnchors {
            processAnchor(anchor, into: &edges)
            if edges.count > 3000 { break }   // cap para no saturar
        }
        sharpEdges = edges
    }

    private func processAnchor(_ anchor: ARMeshAnchor,
                                into edges: inout [(simd_float3, simd_float3)]) {
        let mesh      = anchor.geometry
        let T         = anchor.transform
        let vertCount = mesh.vertices.count
        let faceCount = mesh.faces.count
        guard vertCount > 0, faceCount > 0 else { return }

        // ── Leer vértices en world-space ──────────────────────────────────
        var wv = [simd_float3]()
        wv.reserveCapacity(vertCount)
        let vs   = mesh.vertices
        let vBase = vs.buffer.contents()
        for i in 0..<vertCount {
            let ptr = vBase.advanced(by: vs.offset + vs.stride * i)
                          .assumingMemoryBound(to: (Float, Float, Float).self)
            let p = ptr.pointee
            let w = T * simd_float4(p.0, p.1, p.2, 1)
            wv.append(simd_float3(w.x, w.y, w.z) / w.w)
        }

        // ── Leer caras y calcular normales ────────────────────────────────
        let fe  = mesh.faces
        let bpi = fe.bytesPerIndex
        let icp = fe.indexCountPerPrimitive   // 3 para triángulos
        let fBase = fe.buffer.contents()

        struct Tri { var i0, i1, i2: UInt32 }
        var tris = [Tri](); var fn = [simd_float3]()
        tris.reserveCapacity(faceCount); fn.reserveCapacity(faceCount)

        for i in 0..<faceCount {
            let base = fe.offset + i * icp * bpi
            func idx(_ j: Int) -> UInt32 {
                let p = fBase.advanced(by: base + j * bpi)
                return bpi == 2
                    ? UInt32(p.assumingMemoryBound(to: UInt16.self).pointee)
                    : p.assumingMemoryBound(to: UInt32.self).pointee
            }
            let i0 = idx(0), i1 = idx(1), i2 = idx(2)
            tris.append(Tri(i0: i0, i1: i1, i2: i2))
            let a = wv[Int(i0)], b = wv[Int(i1)], c = wv[Int(i2)]
            fn.append(simd_normalize(simd_cross(b - a, c - a)))
        }

        // ── Adjacencia arista → caras ─────────────────────────────────────
        struct EK: Hashable {
            let lo, hi: UInt32
            init(_ a: UInt32, _ b: UInt32) { lo = min(a,b); hi = max(a,b) }
        }
        var f1Map = [EK: Int]()
        var f2Map = [EK: Int]()
        f1Map.reserveCapacity(faceCount * 3)

        for (fi, t) in tris.enumerated() {
            for ek in [EK(t.i0,t.i1), EK(t.i1,t.i2), EK(t.i2,t.i0)] {
                if      f1Map[ek] == nil { f1Map[ek] = fi }
                else if f2Map[ek] == nil { f2Map[ek] = fi }
            }
        }

        // ── Emitir aristas con ángulo diedro > umbral ─────────────────────
        for (ek, fa) in f1Map {
            guard let fb = f2Map[ek] else { continue }
            if simd_dot(fn[fa], fn[fb]) < minCos {
                edges.append((wv[Int(ek.lo)], wv[Int(ek.hi)]))
            }
        }
    }

    // MARK: - Proyección world → pantalla

    private func project(_ wp: simd_float3, camera: ARCamera, size: CGSize) -> CGPoint? {
        let view = simd_inverse(camera.transform)
        let cam  = view * simd_float4(wp.x, wp.y, wp.z, 1)
        guard cam.z < -0.05 else { return nil }   // detrás de la cámara
        let proj = camera.projectionMatrix(for: .portrait, viewportSize: size,
                                           zNear: 0.001, zFar: 100)
        let clip = proj * cam
        let ndc  = simd_float2(clip.x, clip.y) / clip.w
        guard abs(ndc.x) <= 1.3, abs(ndc.y) <= 1.3 else { return nil }
        return CGPoint(
            x: CGFloat((ndc.x + 1) * 0.5) * size.width,
            y: CGFloat((1 - ndc.y) * 0.5) * size.height
        )
    }
}
