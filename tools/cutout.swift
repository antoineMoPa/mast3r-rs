// Usage: swiftc -O tools/cutout.swift -o cutout && ./cutout <input.jpg> <output.png>  (macOS 14+)
// Writes an RGBA PNG (EXIF orientation applied) where pixels outside the largest
// foreground instance found by Vision are transparent.
import Foundation
import Vision
import CoreImage
import ImageIO

let args = CommandLine.arguments
let input = URL(fileURLWithPath: args[1])
let output = URL(fileURLWithPath: args[2])
let source = CIImage(contentsOf: input, options: [.applyOrientationProperty: true])!
let handler = VNImageRequestHandler(ciImage: source)
let request = VNGenerateForegroundInstanceMaskRequest()
try handler.perform([request])
let observation = request.results!.first!
// Pick the instance with the most pixels (the mannequin, not a hand at the edge).
var best = observation.allInstances.first!
var bestCount = -1
for instance in observation.allInstances {
    let mask = try observation.generateScaledMaskForImage(forInstances: [instance], from: handler)
    CVPixelBufferLockBaseAddress(mask, .readOnly)
    let w = CVPixelBufferGetWidth(mask), h = CVPixelBufferGetHeight(mask)
    let stride = CVPixelBufferGetBytesPerRow(mask) / 4
    let p = CVPixelBufferGetBaseAddress(mask)!.assumingMemoryBound(to: Float32.self)
    var count = 0
    for y in 0..<h { for x in 0..<w where p[y * stride + x] > 0.5 { count += 1 } }
    CVPixelBufferUnlockBaseAddress(mask, .readOnly)
    if count > bestCount { bestCount = count; best = instance }
}
let mask = try observation.generateScaledMaskForImage(forInstances: [best], from: handler)
let maskImage = CIImage(cvPixelBuffer: mask)
let clear = CIImage(color: .clear).cropped(to: source.extent)
let cut = source.applyingFilter("CIBlendWithMask", parameters: [
    kCIInputBackgroundImageKey: clear, kCIInputMaskImageKey: maskImage])
let context = CIContext()
try context.writePNGRepresentation(of: cut, to: output, format: .RGBA8,
                                   colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
print("\(args[1]): \(observation.allInstances.count) instances, kept \(bestCount) px")
