import 'package:automl_image_labeling/automl_image_labeling.dart';
import 'package:camera2/camera2.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

class AnalysisScreen extends StatelessWidget {
  const AnalysisScreen({Key key}) : super(key: key);

  static const path = '/analysis_screen';

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      body: _Body(),
    );
  }
}

class _Body extends StatefulWidget {
  const _Body({
    Key key,
  }) : super(key: key);

  @override
  __BodyState createState() => __BodyState();
}

class __BodyState extends State<_Body> {
  CameraPreviewController _ctrl;
  var _analysisResult = '';
  var _allResults = '';

  var _hasPermission = false;

  var _fps = 0.0;
  var _requestImageDurationMs = 0.0;
  var _processImageDurationMs = 0.0;

  static const _centerCropAspectRatio = 16.0 / 10.0;
  static const _centerCropWidthPercent = 0.8;

  final _labeler = AutoMlImageLabeler(
    manifestFileAssetPath: 'assets/test_model/manifest.json',
    confidenceThreshold: 0,
  );

  @override
  void initState() {
    super.initState();
    _labeler.init();
    _runLabeling();
  }

  @override
  void dispose() {
    _labeler.dispose();
    super.dispose();
  }

  Future<void> _runLabeling() async {
    final permissionStatus = await Permission.camera.request();
    if (permissionStatus == PermissionStatus.granted) {
      _hasPermission = true;
      if (mounted) {
        setState(() {});
      }
    }

    final stopwatch = Stopwatch();
    var totalPasses = 0;

    final reqImageStopwatch = Stopwatch();
    var totalRequests = 0;

    final processImageStopwatch = Stopwatch();
    var totalProcesses = 0;

    while (mounted) {
      if (_ctrl == null) {
        await Future<void>.delayed(const Duration(milliseconds: 100));
        continue;
      }
      stopwatch.start();

      reqImageStopwatch.start();
      final imageBytes = await _ctrl.requestImageForAnalysis();
      reqImageStopwatch.stop();
      totalRequests++;
      _requestImageDurationMs =
          reqImageStopwatch.elapsedMilliseconds / totalRequests;

      if (imageBytes != null) {
        try {
          processImageStopwatch.start();
          final results = await _labeler.process(imageBytes);
          processImageStopwatch.stop();
          totalProcesses++;
          _processImageDurationMs =
              processImageStopwatch.elapsedMilliseconds / totalProcesses;

          results.sort((a, b) => a.label.compareTo(b.label));
          _allResults = results.join('\n');
        } catch (e) {
          debugPrint(e.toString());
        }
      } else {
        totalPasses = 0;
        stopwatch.stop();
        stopwatch.reset();
      }

      stopwatch.stop();
      totalPasses += 1;
      _fps = totalPasses / stopwatch.elapsedMilliseconds * 1000;

      if (mounted) {
        setState(() {});
      }
    }
    stopwatch.stop();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Analysis'),
      ),
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Container(
            height: 200,
            alignment: Alignment.topCenter,
            child: Column(
              children: [
                Text(
                  'FPS: ${_fps.toStringAsFixed(1)}, '
                  'REQUEST: ${_requestImageDurationMs.toStringAsFixed(1)}, '
                  'PROCESS: ${_processImageDurationMs.toStringAsFixed(1)}',
                  textAlign: TextAlign.center,
                ),
                Text(
                  _analysisResult,
                  textAlign: TextAlign.center,
                ),
                Text(_allResults),
              ],
            ),
          ),
          Expanded(
            child: _hasPermission ? _buildPreview() : const SizedBox.shrink(),
          ),
        ],
      ),
    );
  }

  Widget _buildPreview() {
    return Stack(
      children: [
        Positioned.fill(
          child: Camera2Preview(
            analysisOptions: const Camera2AnalysisOptions(
              colorOrder: ColorOrder.rgb,
              normalization: Normalization.ubyte,
              centerCropWidthPercent: _centerCropWidthPercent,
              centerCropAspectRatio: _centerCropAspectRatio,
            ),
            onPlatformViewCreated: (ctrl) => _ctrl = ctrl,
          ),
        ),
        Center(
          child: SizedBox(
            width: MediaQuery.of(context).size.width * _centerCropWidthPercent,
            child: AspectRatio(
              aspectRatio: _centerCropAspectRatio,
              child: Container(
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.white),
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }
}
