import Flutter
import UIKit
import MLKitImageLabelingAutoML
import MLKitVision

public class SwiftAutomlImageLabelingPlugin: NSObject, FlutterPlugin {
    private let registrar: FlutterPluginRegistrar
    
    private var lastLabelerId = 0
    private var labelers = Dictionary<Int, ImageLabeler>()
    private var labelerBitmapSizes = Dictionary<Int, CGSize>()
//    private var labelerBitmaps = Dictionary<Int, UnsafeMutableBufferPointer<UInt8>>()
    private var labelerBitmaps = Dictionary<Int, Array<PixelData>>()
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
                //                let cgImage = writeRgbByteArrayToBitmap(bytes: imageRgbBytes.data, width: Int(size.width), height: Int(size.height))
                //                let uiImage = UIImage(cgImage: cgImage)
                var bytes = [UInt8](imageRgbBytes.data)
                writeRgbByteArrayToBitmap(bytes: &bytes, pixels: &pixels)
                
                let uiImage = imageFromARGB32Bitmap(pixels: &pixels, width: Int(size.width), height: Int(size.height))!
                let image = VisionImage(image: uiImage)
//                image.provideImageData(<#T##data: UnsafeMutableRawPointer##UnsafeMutableRawPointer#>, bytesPerRow: <#T##Int#>, origin: <#T##Int#>, <#T##y: Int##Int#>, size: <#T##Int#>, <#T##height: Int##Int#>, userInfo: <#T##Any?#>)
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
//        labelerBitmaps[id] = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: Int(bitmapSize.width * bitmapSize.height * 4))
        labelerBitmaps[id] = Array<PixelData>(
            repeating: PixelData(a: 0, r: 0, g: 0, b: 0),
            count: Int(bitmapSize.width * bitmapSize.height)
        )
        
        return id
    }
    
    private func disposeLabeler(id: Int) {
        labelers.removeValue(forKey: id)
        labelerBitmapSizes.removeValue(forKey: id)
        labelerBitmaps.removeValue(forKey: id)
        labelerQueues.removeValue(forKey: id)
    }
}

//private func writeRgbByteArrayToBitmap(bytes: Data, width: Int, height: Int) -> CGImage {
//    var data = bytes
//    return data.withUnsafeMutableBytes { (arg: UnsafeMutableRawBufferPointer) -> CGImage in
////            let dataArray = [UInt8](bytes)
////            let imageDataPointer = UnsafeMutablePointer<UInt8>(mutating: dataArray)
//
//        let imageDataPointer = arg.baseAddress!
//
//        let colorSpaceRef = CGColorSpaceCreateDeviceGray()
//
//        let bitsPerComponent = 8
//        let bytesPerPixel = 1
//        let bitsPerPixel = bytesPerPixel * bitsPerComponent
//        let bytesPerRow = bytesPerPixel * width
//        let totalBytes = height * bytesPerRow
//
//        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
////            .union(.ByteOrderDefault)
//
//        let releaseMaskImagePixelData: CGDataProviderReleaseDataCallback = { (info: UnsafeMutableRawPointer?, data: UnsafeRawPointer, size: Int) -> () in
//            // https://developer.apple.com/reference/coregraphics/cgdataproviderreleasedatacallback
//            // N.B. 'CGDataProviderRelease' is unavailable: Core Foundation objects are automatically memory managed
//            return
//        }
//        let providerRef = CGDataProvider(dataInfo: nil, data: imageDataPointer, size: totalBytes, releaseData: releaseMaskImagePixelData)
//        return CGImage(
//            width: width,
//            height: height,
//            bitsPerComponent: bitsPerComponent,
//            bitsPerPixel: bitsPerPixel,
//            bytesPerRow: bytesPerRow,
//            space: colorSpaceRef,
//            bitmapInfo: bitmapInfo,
//            provider: providerRef!,
//            decode: nil,
//            shouldInterpolate: false,
//            intent: CGColorRenderingIntent.defaultIntent
//        )!
//    }
//}
//
//private func writeRgbByteArrayToBitmap(bytes: Data) {
////        let array = [UInt8](bytes)
//
//
////        val nrOfPixels: Int = bytes.size / 3 // Three bytes per pixel
////        for (i in 0 until nrOfPixels) {
////            val r = 0xFF and bytes[3 * i].toInt()
////            val g = 0xFF and bytes[3 * i + 1].toInt()
////            val b = 0xFF and bytes[3 * i + 2].toInt()
////
////            pixels[i] = Color.rgb(r, g, b)
////        }
////        bitmap.setPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
//}

private struct PixelData {
    var a: UInt8
    var r: UInt8
    var g: UInt8
    var b: UInt8
}

private func imageFromARGB32Bitmap(pixels: inout [PixelData], width: Int, height: Int) -> UIImage? {
    guard width > 0 && height > 0 else { return nil }
    guard pixels.count == width * height else { return nil }
    
    let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue)
    let bitsPerComponent = 8
    let bitsPerPixel = 32
    
    var data = pixels // Copy to mutable []
    guard let providerRef = CGDataProvider(
        data: NSData(bytes: &data,
                     length: data.count * MemoryLayout<PixelData>.size
        )
    )
    else { return nil }
    
    guard let cgim = CGImage(
        width: width,
        height: height,
        bitsPerComponent: bitsPerComponent,
        bitsPerPixel: bitsPerPixel,
        bytesPerRow: width * MemoryLayout<PixelData>.size,
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

private func writeRgbByteArrayToBitmap(bytes: inout [UInt8], pixels: inout [PixelData]) {
    let nrOfPixels: Int = bytes.count / 3 // Three bytes per pixel
    if (nrOfPixels < pixels.count) {
        return
    }
    for i in (0...nrOfPixels-1) {
        let r = 0xFF & bytes[3 * i]
        let g = 0xFF & bytes[3 * i + 1]
        let b = 0xFF & bytes[3 * i + 2]
        
        pixels[i] = PixelData(a: 255, r: r, g: g, b: b)
    }
}
