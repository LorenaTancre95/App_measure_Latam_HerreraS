import ARKit
import simd

/// Detecta aristas físicas del mesh LiDAR y las estabiliza con tracking temporal.
///
/// Flujo por frame:
///   1. Lee ARMeshAnchor.geometry → calcula normales por cara → detecta aristas con ángulo diedro > umbral
///   2. Asocia cada arista nueva con la arista trackeada más cercana (por midpoint 3D)
///   3. Suaviza posición con EMA y cuenta frames consecutivos
///   4. Publica solo las aristas confirmadas (≥ N frames) → snap estable sin saltos
final class MeshEdgeTracker {

    // MARK: - Types

    private struct TrackedEdge {
        var a: simd_float3
        var b: simd_float3
        var frames: Int
        var lastSeen: Int
        var midpoint: simd_float3 { (a + b) / 2 }
        var length: Float { simd_distance(a, b) }
    }

    // MARK: - Config

    /// cos del umbral de ángulo diedro; detecta si dot(n1,n2) < valor → ángulo > ~70°
    private let dihedralDotThreshold: Float = 0.35
    /// distancia máxima (m) entre midpoints para considerar la misma arista entre frames
    private let matchRadius:          Float = 0.06
    /// distancia máxima (m) al crosshair para activar el snap
    private let snapRadius:           Float = 0.15
    /// frames consecutivos necesarios antes de publicar la arista
    private let confirmThreshold:     Int   = 5
    /// frames sin detectarse antes de eliminar la arista
    private let maxMissedFrames:      Int   = 12
    /// factor EMA: 0.18 = suavizado pesado (posición se mueve lento → sin saltos)
    private let emaAlpha:             Float = 0.18
    /// descarta aristas más cortas que esto (m) — filtra ruido del mesh
    private let minEdgeLength:        Float = 0.04

    // MARK: - State (protegido por lock)

    private let lock = NSLock()
    private var tracked: [TrackedEdge] = []
    private var frameIdx = 0
    private var _confirmed: [(simd_float3, simd_float3)] = []

    /// Snapshot thread-safe de aristas confirmadas (world space).
    var confirmedEdges: [(simd_float3, simd_float3)] {
        lock.lock(); defer { lock.unlock() }
        return _confirmed
    }

    // MARK: - Update (llamar desde background thread)

    func update(meshAnchors: [ARMeshAnchor]) {
        let raw = meshAnchors.flatMap { detectEdges(anchor: $0) }

        lock.lock()
        defer { lock.unlock() }
        frameIdx += 1

        var matched = Set<Int>()

        for r in raw {
            let rMid = (r.0 + r.1) / 2
            var bestIdx: Int? = nil
            var bestDist = matchRadius

            for (i, t) in tracked.enumerated() {
                guard !matched.contains(i) else { continue }
                let d = simd_distance(rMid, t.midpoint)
                if d < bestDist { bestDist = d; bestIdx = i }
            }

            if let idx = bestIdx {
                // Actualizar posición con EMA
                tracked[idx].a = tracked[idx].a + (r.0 - tracked[idx].a) * emaAlpha
                tracked[idx].b = tracked[idx].b + (r.1 - tracked[idx].b) * emaAlpha
                tracked[idx].frames  += 1
                tracked[idx].lastSeen = frameIdx
                matched.insert(idx)
            } else {
                tracked.append(TrackedEdge(a: r.0, b: r.1, frames: 1, lastSeen: frameIdx))
            }
        }

        // Eliminar aristas que no se vieron hace demasiado
        tracked = tracked.filter { frameIdx - $0.lastSeen < maxMissedFrames }

        // Publicar solo las confirmadas
        _confirmed = tracked
            .filter { $0.frames >= confirmThreshold && $0.length >= minEdgeLength }
            .map    { ($0.a, $0.b) }
    }

    // MARK: - Snap query (cualquier thread)

    /// Retorna el punto más cercano sobre cualquier arista confirmada a `aimPoint`,
    /// si está dentro de `snapRadius`. Nil si no hay arista cercana.
    func snap(to aimPoint: simd_float3) -> simd_float3? {
        let edges = confirmedEdges
        var best: (dist: Float, pt: simd_float3)? = nil
        for (a, b) in edges {
            let pt = closestOnSegment(aimPoint, a: a, b: b)
            let d  = simd_distance(pt, aimPoint)
            if d < snapRadius, best == nil || d < best!.dist {
                best = (d, pt)
            }
        }
        return best?.pt
    }

    // MARK: - Edge detection

    private func detectEdges(anchor: ARMeshAnchor) -> [(simd_float3, simd_float3)] {
        let geo   = anchor.geometry
        let vSrc  = geo.vertices
        let fElem = geo.faces
        let vCount = vSrc.count
        let fCount = fElem.count
        guard vCount > 3, fCount > 0 else { return [] }

        // Leer vértices en world space
        let vBase = vSrc.buffer.contents().advanced(by: vSrc.offset)
        var verts = [simd_float3](repeating: .zero, count: vCount)
        for i in 0..<vCount {
            let p = vBase.advanced(by: i * vSrc.stride)
                         .assumingMemoryBound(to: simd_float3.self).pointee
            let w = anchor.transform * simd_float4(p.x, p.y, p.z, 1)
            verts[i] = simd_float3(w.x, w.y, w.z)
        }

        // Leer índices de caras
        let fBase = fElem.buffer.contents()
        let bpi   = fElem.bytesPerIndex          // 2 (UInt16) o 4 (UInt32)
        let icp   = fElem.indexCountPerPrimitive // siempre 3

        func faceIndices(_ f: Int) -> (Int, Int, Int) {
            let off = fBase.advanced(by: f * icp * bpi)
            if bpi == 4 {
                let p = off.assumingMemoryBound(to: UInt32.self)
                return (Int(p[0]), Int(p[1]), Int(p[2]))
            }
            let p = off.assumingMemoryBound(to: UInt16.self)
            return (Int(p[0]), Int(p[1]), Int(p[2]))
        }

        // Normales por cara calculadas desde posiciones (más precisas que las interpoladas de ARKit)
        var fNormals = [simd_float3](repeating: .zero, count: fCount)
        for f in 0..<fCount {
            let (i0, i1, i2) = faceIndices(f)
            guard i0 < vCount, i1 < vCount, i2 < vCount else { continue }
            let n = simd_cross(verts[i1] - verts[i0], verts[i2] - verts[i0])
            let len = simd_length(n)
            if len > 1e-8 { fNormals[f] = n / len }
        }

        // Adyacencia: clave de arista → [índice de cara]
        var adj = [Int64: [Int]]()
        adj.reserveCapacity(fCount * 3)
        for f in 0..<fCount {
            let (i0, i1, i2) = faceIndices(f)
            for (a, b) in [(i0,i1),(i1,i2),(i0,i2)] {
                adj[edgeKey(a, b), default: []].append(f)
            }
        }

        // Detectar aristas con ángulo diedro > umbral
        var result: [(simd_float3, simd_float3)] = []
        for (key, faces) in adj {
            guard faces.count == 2 else { continue }
            let n0 = fNormals[faces[0]], n1 = fNormals[faces[1]]
            guard simd_length(n0) > 0.5, simd_length(n1) > 0.5 else { continue }
            guard simd_dot(n0, n1) < dihedralDotThreshold else { continue }
            let (lo, hi) = decodeKey(key)
            guard lo < vCount, hi < vCount else { continue }
            result.append((verts[lo], verts[hi]))
        }
        return result
    }

    // MARK: - Helpers

    private func edgeKey(_ a: Int, _ b: Int) -> Int64 {
        let lo = UInt32(min(a, b)), hi = UInt32(max(a, b))
        return Int64(bitPattern: UInt64(lo) << 32 | UInt64(hi))
    }

    private func decodeKey(_ key: Int64) -> (Int, Int) {
        let u = UInt64(bitPattern: key)
        return (Int(u >> 32), Int(u & 0xFFFF_FFFF))
    }

    private func closestOnSegment(_ p: simd_float3, a: simd_float3, b: simd_float3) -> simd_float3 {
        let ab = b - a
        let lenSq = simd_dot(ab, ab)
        guard lenSq > 1e-8 else { return a }
        return a + ab * simd_clamp(simd_dot(p - a, ab) / lenSq, 0, 1)
    }
}
