import Vision
import ARKit
import CoreImage

// Runs VNGenerateForegroundInstanceMaskRequest on a captured frame and returns
// a binary mask (UInt8 CVPixelBuffer, 1 = selected object, 0 = background)
// sized to match the LiDAR depth map dimensions.
final class SegmentationProcessor {

    // Returns a binary mask aligned to depthMap dimensions, or nil if nothing
    // was found at the tapped screen point.
    func segment(frame: ARFrame,
                 tapPoint: CGPoint,
                 viewSize: CGSize) -> CVPixelBuffer? {

        let pixelBuffer = frame.capturedImage

        // Vision processes capturedImage in landscape-right; orientation .right
        // makes it internally handle the 90° rotation so results are in portrait space.
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: .right,
                                            options: [:])

        let request = VNGenerateForegroundInstanceMaskRequest()
        do {
            try handler.perform([request])
        } catch {
            return nil
        }

        guard let obs = request.results?.first as? VNInstanceMaskObservation else {
            return nil
        }

        // --- Find which instance the user tapped ---
        // Map portrait-screen tap → normalized landscape-image coordinates
        // (same convention used in worldPoint() in BoxDetectionCoordinator)
        let normImgX = Float(tapPoint.y / viewSize.height)
        let normImgY = Float(1.0 - tapPoint.x / viewSize.width)

        guard let instanceMask = extractInstanceMaskBuffer(from: obs) else { return nil }

        let maskW = CVPixelBufferGetWidth(instanceMask)
        let maskH = CVPixelBufferGetHeight(instanceMask)

        let px = max(0, min(maskW - 1, Int(normImgX * Float(maskW))))
        let py = max(0, min(maskH - 1, Int(normImgY * Float(maskH))))

        CVPixelBufferLockBaseAddress(instanceMask, .readOnly)
        let ptr = CVPixelBufferGetBaseAddress(instanceMask)!
            .assumingMemoryBound(to: UInt8.self)
        let selectedLabel = ptr[py * maskW + px]
        CVPixelBufferUnlockBaseAddress(instanceMask, .readOnly)

        // label 0 = background
        guard selectedLabel != 0 else { return nil }

        // --- Build binary mask at depth map resolution ---
        guard let depthData = frame.sceneDepth else { return nil }
        let depthMap = depthData.depthMap
        let depthW = CVPixelBufferGetWidth(depthMap)
        let depthH = CVPixelBufferGetHeight(depthMap)

        return buildBinaryMask(instanceMask: instanceMask,
                               selectedLabel: selectedLabel,
                               outputWidth: depthW,
                               outputHeight: depthH)
    }

    // MARK: - Private

    // VNInstanceMaskObservation.instanceMask is a CVPixelBuffer with one-byte-per-pixel
    // instance labels. Extract it directly.
    private func extractInstanceMaskBuffer(from obs: VNInstanceMaskObservation) -> CVPixelBuffer? {
        // instanceMask property gives us the raw label map
        return obs.instanceMask
    }

    // Creates a new CVPixelBuffer (kCVPixelFormatType_OneComponent8) at the given
    // output size where pixels matching selectedLabel are 255, everything else 0.
    private func buildBinaryMask(instanceMask: CVPixelBuffer,
                                 selectedLabel: UInt8,
                                 outputWidth: Int,
                                 outputHeight: Int) -> CVPixelBuffer? {

        var binaryBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            outputWidth, outputHeight,
            kCVPixelFormatType_OneComponent8,
            nil,
            &binaryBuffer
        )
        guard status == kCVReturnSuccess, let binary = binaryBuffer else { return nil }

        let maskW = CVPixelBufferGetWidth(instanceMask)
        let maskH = CVPixelBufferGetHeight(instanceMask)

        CVPixelBufferLockBaseAddress(instanceMask, .readOnly)
        CVPixelBufferLockBaseAddress(binary, [])
        defer {
            CVPixelBufferUnlockBaseAddress(instanceMask, .readOnly)
            CVPixelBufferUnlockBaseAddress(binary, [])
        }

        let srcPtr = CVPixelBufferGetBaseAddress(instanceMask)!
            .assumingMemoryBound(to: UInt8.self)
        let dstPtr = CVPixelBufferGetBaseAddress(binary)!
            .assumingMemoryBound(to: UInt8.self)

        for dy in 0 ..< outputHeight {
            for dx in 0 ..< outputWidth {
                // Scale depth coords → mask coords
                let mx = max(0, min(maskW - 1, Int(Double(dx) / Double(outputWidth)  * Double(maskW))))
                let my = max(0, min(maskH - 1, Int(Double(dy) / Double(outputHeight) * Double(maskH))))
                dstPtr[dy * outputWidth + dx] = srcPtr[my * maskW + mx] == selectedLabel ? 255 : 0
            }
        }

        return binary
    }
}
