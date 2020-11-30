import Flutter
import UIKit
import MLKitImageLabelingAutoML
import MLKitVision

public class SwiftAutomlImageLabelingPlugin: NSObject, FlutterPlugin {
    private let registrar: FlutterPluginRegistrar
    
    private var lastLabelerId = 0
    private var labelers = Dictionary<Int, ImageLabeler>()
    private var labelerBitmapSizes = Dictionary<Int, CGSize>()
    private var labelerBitmaps = Dictionary<Int, UnsafeMutableBufferPointer<UInt8>>()
    private var labelerQueues = Dictionary<Int, DispatchQueue>()
    
    init(registrar: FlutterPluginRegistrar) {
        self.registrar = registrar
        super.init()
    }
    
    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(name: "dev.sonerik.automl_image_labeling", binaryMessenger: registrar.messenger())
        let instance = SwiftAutomlImageLabelingPlugin(registrar: registrar)
        registrar.addMethodCallDelegate(instance, channel: channel)
    }
    
    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "prepareLabeler":
            let args = call.arguments as! Dictionary<String, Any?>
            let manifestFileAssetPathParam = args["manifestFileAssetPath"] as! String
            let manifestFileAssetPackageParam = args["manifestFileAssetPackage"] as? String
            let manifestFileAssetPath: String
            if let manifestFileAssetPackageParam = manifestFileAssetPackageParam {
                let key = registrar.lookupKey(
                    forAsset: manifestFileAssetPathParam,
                    fromPackage: manifestFileAssetPackageParam
                )
                manifestFileAssetPath = Bundle.main.path(forResource: key, ofType: nil)!
            } else {
                let key = registrar.lookupKey(forAsset: manifestFileAssetPathParam)
                manifestFileAssetPath = Bundle.main.path(forResource: key, ofType: nil)!
            }
            
            let confidenceThreshold = Float(args["confidenceThreshold"] as! Double)
            let bitmapWidth = args["bitmapWidth"] as! Int
            let bitmapHeight = args["bitmapHeight"] as! Int
            
            let id = initLabeler(
                manifestFileAssetPath: manifestFileAssetPath,
                confidenceThreshold: confidenceThreshold,
                bitmapSize: CGSize(width: bitmapWidth, height: bitmapHeight)
            );
            
            result(id)
        case "disposeLabeler":
            let args = call.arguments as! Dictionary<String, Any?>
            let id = args["id"] as! Int
            disposeLabeler(id: id)
            result(nil)
        case "processImage":
            let args = call.arguments as! Dictionary<String, Any?>
            let id = args["labelerId"] as! Int
            let imageRgbBytes = args["rgbBytes"] as! FlutterStandardTypedData
            let size = labelerBitmapSizes[id]!
            let labeler = labelers[id]!
            var pixels = labelerBitmaps[id]!
            labelerQueues[id]!.async {
                imageRgbBytes.data.withUnsafeBytes {
                    var bytes = $0
                    writeRgbByteArrayToBitmap(rgbBytes: &bytes, argbBitmap: &pixels)
                }
                
                let uiImage = imageFromARGB32Bitmap(pixels: &pixels, width: Int(size.width), height: Int(size.height))!
                let image = VisionImage(image: uiImage)
                labeler.process(image) { labels, error in
                    guard error == nil, let labels = labels, !labels.isEmpty else {
                        DispatchQueue.main.async {
                            result(FlutterError(code: "", message: error?.localizedDescription, details: nil))
                        }
                        return
                    }
                    
                    var resultData = [Dictionary<String, Any>]()
                    for label in labels {
                        resultData.append([
                            "label": label.text,
                            "confidence": label.confidence
                        ])
                    }
                    
                    DispatchQueue.main.async {
                        result(resultData)
                    }
                }
            }
        default:
            result(FlutterMethodNotImplemented)
        }
    }
    
    private func initLabeler(
        manifestFileAssetPath: String,
        confidenceThreshold: Float,
        bitmapSize: CGSize
    ) -> Int {
        let id = lastLabelerId
        lastLabelerId += 1
        
        let queue = DispatchQueue(label: "AutoML Labeler #\(id)")
        let localModel = AutoMLImageLabelerLocalModel(manifestPath: manifestFileAssetPath)
        let options = AutoMLImageLabelerOptions(localModel: localModel)
        options.confidenceThreshold = NSNumber(value: confidenceThreshold)
        let imageLabeler = ImageLabeler.imageLabeler(options: options)
        
        labelers[id] = imageLabeler
        labelerQueues[id] = queue
        labelerBitmapSizes[id] = bitmapSize
        labelerBitmaps[id] = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: Int(bitmapSize.width * bitmapSize.height * 4))
        
        return id
    }
    
    private func disposeLabeler(id: Int) {
        labelers.removeValue(forKey: id)
        labelerBitmapSizes.removeValue(forKey: id)
        labelerBitmaps[id]?.deallocate()
        labelerBitmaps.removeValue(forKey: id)
        labelerQueues.removeValue(forKey: id)
    }
}

private func imageFromARGB32Bitmap(pixels: inout UnsafeMutableBufferPointer<UInt8>, width: Int, height: Int) -> UIImage? {
    guard width > 0 && height > 0 else { return nil }
    guard pixels.count == width * height * 4 else { return nil }
    
    let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue)
    let bitsPerComponent = 8
    let bitsPerPixel = 32
    
    guard let providerRef = CGDataProvider(
        data: NSData(bytesNoCopy: pixels.baseAddress!, length: pixels.count, deallocator: { (UnsafeMutableRawPointer, Int) in })
    )
    else { return nil }
    
    guard let cgim = CGImage(
        width: width,
        height: height,
        bitsPerComponent: bitsPerComponent,
        bitsPerPixel: bitsPerPixel,
        bytesPerRow: width * 4,
        space: rgbColorSpace,
        bitmapInfo: bitmapInfo,
        provider: providerRef,
        decode: nil,
        shouldInterpolate: true,
        intent: .defaultIntent
    )
    else { return nil }
    
    return UIImage(cgImage: cgim)
}

private func writeRgbByteArrayToBitmap(rgbBytes: inout UnsafeRawBufferPointer, argbBitmap: inout UnsafeMutableBufferPointer<UInt8>) {
    let nrOfPixels: Int = rgbBytes.count / 3 // Three bytes per pixel
    if (nrOfPixels > argbBitmap.count) {
        return
    }
    var colorIndex = 0
    for i in (0..<nrOfPixels) {
        let r = 0xFF & rgbBytes[3 * i]
        let g = 0xFF & rgbBytes[3 * i + 1]
        let b = 0xFF & rgbBytes[3 * i + 2]
        
        argbBitmap[colorIndex] = 255
        argbBitmap[colorIndex + 1] = r
        argbBitmap[colorIndex + 2] = g
        argbBitmap[colorIndex + 3] = b
        
        colorIndex += 4
    }
}
