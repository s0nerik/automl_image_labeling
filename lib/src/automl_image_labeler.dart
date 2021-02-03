import 'dart:async';
import 'dart:typed_data';
import 'dart:ui';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

const _channel = MethodChannel('dev.sonerik.automl_image_labeling');

@immutable
class AutoMlImageLabel {
  const AutoMlImageLabel._({
    @required this.label,
    @required this.confidence,
  });

  final String label;
  final double confidence;

  @override
  String toString() {
    return '($label: ${confidence.toStringAsFixed(2)})';
  }
}

class AutoMlImageLabeler {
  AutoMlImageLabeler({
    @required this.modelFileAssetPath,
    this.modelFileAssetPackage,
    this.confidenceThreshold = 0.5,
    this.bitmapSize = const Size(224, 224),
  })  : assert(modelFileAssetPath != null),
        assert(confidenceThreshold != null),
        assert(bitmapSize != null);

  final String modelFileAssetPath;
  final String modelFileAssetPackage;
  final double confidenceThreshold;
  final Size bitmapSize;

  final _idCompleter = Completer<int>();

  Future<void> init() async {
    try {
      final id = await _channel.invokeMethod('prepareLabeler', {
        'modelFileAssetPath': modelFileAssetPath,
        if (modelFileAssetPackage != null)
          'modelFileAssetPackage': modelFileAssetPackage,
        'confidenceThreshold': confidenceThreshold,
        'bitmapWidth': bitmapSize.width.toInt(),
        'bitmapHeight': bitmapSize.height.toInt(),
      });
      _idCompleter.complete(id);
    } catch (e, stackTrace) {
      _idCompleter.completeError(e, stackTrace);
      rethrow;
    }
  }

  Future<List<AutoMlImageLabel>> process(Uint8List rgbBytes) async {
    final id = await _idCompleter.future;
    final results = await _channel.invokeMethod<List<dynamic>>('processImage', {
      'labelerId': id,
      'rgbBytes': rgbBytes,
    });

    return results
        .cast<Map<dynamic, dynamic>>()
        .map((r) => AutoMlImageLabel._(
              label: r['label'] as String,
              confidence: r['confidence'] as double,
            ))
        .toList();
  }

  Future<void> dispose() async {
    final id = await _idCompleter.future;
    await _channel.invokeMethod('disposeLabeler', {
      'id': id,
    });
  }
}
