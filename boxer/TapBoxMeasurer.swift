import ARKit
import UIKit
import simd

/// Tap-to-select box measurement using MobileSAM.
///
/// Flow:
///   1. User taps a portrait screen point
///   2. Tap coords mapped → SAM 1024×1024 prompt (center-crop of landscape capturedImage)
///   3. SAMSegmenter encoder+decoder → [[Bool]] mask at decoder resolution
///   4. Yellow preview UIImage built from mask so user can verify
///   5. PalletMeasurer filters LiDAR depth through mask → OBB → Detection3D
struct TapBoxMeasurer {

    // MARK: - Public API

    /// Segments the object at `tapPoint` and returns the mask + a yellow preview image.
    /// Returns nil if SAM produces no meaningful result at the tapped location.
    static func segmentWithPreview(
        pixelBuffer: CVPixelBuffer,
        tapPoint: CGPoint,
        viewportSize: CGSize,
        samSegmenter: SAMSegmenter
    ) -> (mask: [[Bool]], preview: UIImage)? {

        let samPoint = portraitTapToSAMCoords(
            tapPoint: tapPoint,
            viewportSize: viewportSize,
            pixelBuffer: pixelBuffer
        )

        guard let mask = try? samSegmenter.segment(pixelBuffer: pixelBuffer,
                                                    promptPoint: samPoint)
        else { return nil }

        let trueCount = mask.joined().filter { $0 }.count
        guard trueCount > 50 else { return nil }

        let preview = buildPreviewImage(mask: mask)
        return (mask, preview)
    }

    /// Measures the 3D bounding box of the object covered by `mask`.
    /// Delegates directly to PalletMeasurer which handles
    /// LiDAR → crop-space → mask-space mapping and floor removal.
    static func measure(frame: ARFrame, mask: [[Bool]]) -> Detection3D? {
        PalletMeasurer.measure(frame: frame, mask: mask)
    }

    // MARK: - Coordinate mapping

    /// Maps a portrait-screen tap to SAM 1024×1024 image coordinates.
    ///
    /// capturedImage is landscape (width > height, e.g. 1920×1440).
    /// Screen is portrait. SAM encoder center-crops to a square then scales to 1024.
    static func portraitTapToSAMCoords(
        tapPoint: CGPoint,
        viewportSize: CGSize,
        pixelBuffer: CVPixelBuffer
    ) -> CGPoint {

        let bufW = Float(CVPixelBufferGetWidth(pixelBuffer))   // e.g. 1920
        let bufH = Float(CVPixelBufferGetHeight(pixelBuffer))  // e.g. 1440

        // Portrait screen (tapX, tapY) → landscape image (imgX, imgY)
        let imgX = Float(tapPoint.y / viewportSize.height) * bufW
        let imgY = (1.0 - Float(tapPoint.x / viewportSize.width)) * bufH

        // Center-crop to square (matches SAM encoder crop in SAMSegmenter.swift)
        let side = min(bufW, bufH)
        let ox = (bufW - side) / 2
        let oy = (bufH - side) / 2

        let cropX = imgX - ox
        let cropY = imgY - oy

        // Scale crop coords to 1024×1024
        let S = Float(SAMSegmenter.imageSize)
        let samX = max(0, min(S - 1, cropX / side * S))
        let samY = max(0, min(S - 1, cropY / side * S))

        return CGPoint(x: CGFloat(samX), y: CGFloat(samY))
    }

    // MARK: - Preview image

    /// Yellow semi-transparent UIImage from a [[Bool]] mask.
    /// Mask rows are in landscape orientation; .right orientation rotates to portrait for display.
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
