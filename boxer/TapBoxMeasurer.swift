import ARKit
import UIKit
import simd

/// Tap-to-select box measurement using MobileSAM.
///
/// Flow:
///   1. User taps a portrait screen point
///   2. ARKit displayTransform (inverted) maps tap → normalized image coords → landscape pixel
///   3. Landscape pixel → SAM center-crop → 1024×1024 prompt
///   4. SAMSegmenter encoder+decoder → [[Bool]] mask at decoder resolution
///   5. Yellow preview UIImage built from mask so user can verify
///   6. PalletMeasurer filters LiDAR depth through mask → OBB → Detection3D
struct TapBoxMeasurer {

    // MARK: - Public API

    /// Segments the object at `tapPoint` and returns the mask + a yellow preview image.
    /// Requires `frame` so we can use `displayTransform` for correct coordinate mapping.
    static func segmentWithPreview(
        frame: ARFrame,
        tapPoint: CGPoint,
        viewportSize: CGSize,
        samSegmenter: SAMSegmenter
    ) -> (mask: [[Bool]], preview: UIImage)? {

        let pixelBuffer = frame.capturedImage
        let samPoint = portraitTapToSAMCoords(
            frame: frame,
            tapPoint: tapPoint,
            viewportSize: viewportSize,
            pixelBuffer: pixelBuffer
        )

        print("[TAP-DBG] SAM prompt: (\(Int(samPoint.x)), \(Int(samPoint.y)))")

        guard let mask = try? samSegmenter.segment(pixelBuffer: pixelBuffer,
                                                    promptPoint: samPoint)
        else { print("[TAP-DBG] SAM returned nil"); return nil }

        let trueCount = mask.joined().filter { $0 }.count
        print("[TAP-DBG] mask \(mask.count)×\(mask.first?.count ?? 0), true pixels: \(trueCount)")

        // bounding box of true pixels
        var minR = mask.count, maxR = 0, minC = mask.first?.count ?? 0, maxC = 0
        for (r, row) in mask.enumerated() {
            for (c, v) in row.enumerated() where v {
                minR = min(minR, r); maxR = max(maxR, r)
                minC = min(minC, c); maxC = max(maxC, c)
            }
        }
        print("[TAP-DBG] mask bbox rows \(minR)-\(maxR) cols \(minC)-\(maxC)")

        guard trueCount > 50 else { return nil }

        let preview = buildPreviewImage(mask: mask)
        return (mask, preview)
    }

    /// Measures the 3D bounding box of the object covered by `mask`.
    static func measure(frame: ARFrame, mask: [[Bool]]) -> Detection3D? {
        PalletMeasurer.measure(frame: frame, mask: mask)
    }

    // MARK: - Coordinate mapping

    /// Maps a portrait-screen tap to SAM 1024×1024 image coordinates.
    ///
    /// Uses `frame.displayTransform(for: .portrait, viewportSize:)` — the authoritative
    /// ARKit mapping from normalized image coords to normalized viewport coords.
    /// Inverting it gives viewport → image, no hand-rolled rotation guesswork needed.
    static func portraitTapToSAMCoords(
        frame: ARFrame,
        tapPoint: CGPoint,
        viewportSize: CGSize,
        pixelBuffer: CVPixelBuffer
    ) -> CGPoint {

        let bufW = Float(CVPixelBufferGetWidth(pixelBuffer))
        let bufH = Float(CVPixelBufferGetHeight(pixelBuffer))

        // ARKit: image normalized [0,1] → viewport normalized [0,1]
        // Inverse: viewport normalized → image normalized
        let displayT  = frame.displayTransform(for: .portrait, viewportSize: viewportSize)
        let invertedT = displayT.inverted()

        let normTap = CGPoint(x: tapPoint.x / viewportSize.width,
                              y: tapPoint.y / viewportSize.height)
        let normImg = normTap.applying(invertedT)

        // Normalized image → landscape pixel
        let imgX = Float(normImg.x) * bufW
        let imgY = Float(normImg.y) * bufH

        // Center-crop to square (mirrors what SAMSegmenter.encodeImage does)
        let side = min(bufW, bufH)
        let ox   = (bufW - side) / 2
        let oy   = (bufH - side) / 2

        let cropX = imgX - ox
        let cropY = imgY - oy

        // Scale crop to 1024×1024
        let S = Float(SAMSegmenter.imageSize)
        let samX = max(0, min(S - 1, cropX / side * S))
        let samY = max(0, min(S - 1, cropY / side * S))

        return CGPoint(x: CGFloat(samX), y: CGFloat(samY))
    }

    // MARK: - Preview image

    /// Yellow semi-transparent UIImage from a [[Bool]] mask.
    /// Mask is in landscape orientation; .right orientation rotates 90° for portrait display.
    private static func buildPreviewImage(mask: [[Bool]]) -> UIImage {
        let mH = mask.count
        guard mH > 0 else { return UIImage() }
        let mW = mask[0].count

        var rgba = [UInt8](repeating: 0, count: mW * mH * 4)
        for y in 0..<mH {
            for x in 0..<mW {
                if mask[y][x] {
                    let i = (y * mW + x) * 4
                    rgba[i + 0] = 255   // R
                    rgba[i + 1] = 220   // G
                    rgba[i + 2] = 0     // B
                    rgba[i + 3] = 140   // A ~55%
                }
            }
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: &rgba,
            width: mW, height: mH,
            bitsPerComponent: 8,
            bytesPerRow: mW * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let cgImage = ctx.makeImage() else { return UIImage() }

        return UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)
    }
}
