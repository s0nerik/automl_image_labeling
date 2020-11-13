#import "AutomlImageLabelingPlugin.h"
#if __has_include(<automl_image_labeling/automl_image_labeling-Swift.h>)
#import <automl_image_labeling/automl_image_labeling-Swift.h>
#else
// Support project import fallback if the generated compatibility header
// is not copied when this plugin is created as a library.
// https://forums.swift.org/t/swift-static-libraries-dont-copy-generated-objective-c-header/19816
#import "automl_image_labeling-Swift.h"
#endif

@implementation AutomlImageLabelingPlugin
+ (void)registerWithRegistrar:(NSObject<FlutterPluginRegistrar>*)registrar {
  [SwiftAutomlImageLabelingPlugin registerWithRegistrar:registrar];
}
@end
