package dev.sonerik.automl_image_labeling

import android.graphics.Bitmap
import android.graphics.Color
import android.os.Handler
import android.os.Looper
import android.util.Size
import androidx.annotation.NonNull
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.label.ImageLabeler
import com.google.mlkit.vision.label.ImageLabeling
import com.google.mlkit.common.model.LocalModel
import com.google.mlkit.vision.label.custom.CustomImageLabelerOptions
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.embedding.engine.plugins.activity.ActivityAware
import io.flutter.embedding.engine.plugins.activity.ActivityPluginBinding
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import java.util.concurrent.*


/** AutomlImageLabelingPlugin */
class AutomlImageLabelingPlugin : FlutterPlugin, MethodCallHandler, ActivityAware {
    /// The MethodChannel that will the communication between Flutter and native Android
    ///
    /// This local reference serves to register the plugin with the Flutter Engine and unregister it
    /// when the Flutter Engine is detached from the Activity
    private lateinit var channel: MethodChannel
    private var lifecycle: Lifecycle? = null

    private var lastLabelerId = 0
    private val labelers = mutableMapOf<Int, ImageLabeler>()
    private val labelerBitmaps = mutableMapOf<Int, Bitmap>()
    private val labelerBitmapBuffers = mutableMapOf<Int, IntArray>()
    private val executors = mutableMapOf<Int, Executor>()

    private lateinit var binding: FlutterPlugin.FlutterPluginBinding

    override fun onAttachedToEngine(@NonNull flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
        binding = flutterPluginBinding
        channel = MethodChannel(flutterPluginBinding.binaryMessenger, "dev.sonerik.automl_image_labeling")
        channel.setMethodCallHandler(this)
    }

    override fun onMethodCall(@NonNull call: MethodCall, @NonNull result: Result) {
        when (call.method) {
            "prepareLabeler" -> {
                val modelFileAssetPathParam = call.argument<String>("modelFileAssetPath")!!
                val modelFileAssetPackageParam = call.argument<String?>("modelFileAssetPackage")
                val modelFileAssetPath = if (modelFileAssetPackageParam != null)
                    binding.flutterAssets.getAssetFilePathBySubpath(modelFileAssetPathParam, modelFileAssetPackageParam)
                else
                    binding.flutterAssets.getAssetFilePathBySubpath(modelFileAssetPathParam)

                val confidenceThreshold = call.argument<Double>("confidenceThreshold")!!
                val bitmapWidth = call.argument<Int>("bitmapWidth")!!
                val bitmapHeight = call.argument<Int>("bitmapHeight")!!
                val id = initLabeler(
                        assetFilePath = modelFileAssetPath,
                        confidenceThreshold = confidenceThreshold.toFloat(),
                        bitmapSize = Size(bitmapWidth, bitmapHeight)
                )
                result.success(id)
            }
            "disposeLabeler" -> {
                val id = call.argument<Int>("id")!!
                disposeLabeler(id)
                result.success(null)
            }
            "processImage" -> {
                val id = call.argument<Int>("labelerId")!!
                val imageRgbBytes = call.argument<ByteArray>("rgbBytes")!!
                executors[id]!!.execute {
                    val bitmap = labelerBitmaps[id]!!
                    writeRgbByteArrayToBitmap(imageRgbBytes, labelerBitmapBuffers[id]!!, bitmap)
                    val inputImage = InputImage.fromBitmap(bitmap, 0)
                    labelers[id]!!.process(inputImage).addOnSuccessListener { labels ->
                        val results = labels.map {
                            mapOf(
                                    "label" to it.text,
                                    "confidence" to it.confidence
                            )
                        }
                        Handler(Looper.getMainLooper()).post {
                            result.success(results)
                        }
                    }.addOnFailureListener {
                        Handler(Looper.getMainLooper()).post {
                            result.error("", it.localizedMessage, null)
                        }
                    }
                }
            }
            else -> result.notImplemented()
        }
    }

    private fun initLabeler(
            assetFilePath: String,
            confidenceThreshold: Float,
            bitmapSize: Size
    ): Int {
        val id = lastLabelerId++

        val executor = Executors.newSingleThreadExecutor()
        executors[id] = executor

        val localModel = LocalModel.Builder()
                .setAssetFilePath(assetFilePath)
                .build()
        val localModelOptions = CustomImageLabelerOptions.Builder(localModel)
                .setConfidenceThreshold(confidenceThreshold)
                .build()
        val imageLabeler = ImageLabeling.getClient(localModelOptions)
        lifecycle?.addObserver(imageLabeler)

        labelers[id] = imageLabeler
        labelerBitmaps[id] = Bitmap.createBitmap(bitmapSize.width, bitmapSize.height, Bitmap.Config.ARGB_8888)
        labelerBitmapBuffers[id] = IntArray(bitmapSize.width * bitmapSize.height)

        return id
    }

    private fun disposeLabeler(id: Int) {
        labelers[id]?.let {
            lifecycle?.removeObserver(it)
            it.close()
        }
        labelers.remove(id)

        labelerBitmaps[id]?.recycle()
        labelerBitmaps.remove(id)

        labelerBitmapBuffers.remove(id)
    }

    override fun onDetachedFromEngine(@NonNull binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
    }

    override fun onAttachedToActivity(binding: ActivityPluginBinding) {
        lifecycle = (binding.activity as LifecycleOwner).lifecycle
    }

    override fun onDetachedFromActivity() {
        lifecycle = null
    }

    override fun onDetachedFromActivityForConfigChanges() {}

    override fun onReattachedToActivityForConfigChanges(binding: ActivityPluginBinding) {}
}

private fun writeRgbByteArrayToBitmap(bytes: ByteArray, pixels: IntArray, bitmap: Bitmap) {
    val nrOfPixels: Int = bytes.size / 3 // Three bytes per pixel
    for (i in 0 until nrOfPixels) {
        val r = 0xFF and bytes[3 * i].toInt()
        val g = 0xFF and bytes[3 * i + 1].toInt()
        val b = 0xFF and bytes[3 * i + 2].toInt()

        pixels[i] = Color.rgb(r, g, b)
    }
    bitmap.setPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
}