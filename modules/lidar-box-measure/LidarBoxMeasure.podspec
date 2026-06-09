require 'json'

package = JSON.parse(File.read(File.join(__dir__, 'package.json')))

Pod::Spec.new do |s|
  s.name           = 'LidarBoxMeasure'
  s.version        = package['version']
  s.summary        = 'LiDAR box measurement module for MedidorCaixas'
  s.homepage       = 'https://github.com/coki0291/App_measure_Latam'
  s.license        = { :type => 'MIT' }
  s.authors        = { 'coki0291' => 'slherrera91@gmail.com' }
  s.platform       = :ios, '15.0'
  s.source         = { :path => '.' }
  s.static_framework = true

  s.dependency 'ExpoModulesCore'
  s.dependency 'onnxruntime-objc', '>= 1.18.0'

  s.source_files = 'ios/**/*.{swift,m,mm}'
  s.resources    = ['ios/box_detector.mlpackage',
                    'ios/sam_encoder.mlpackage',
                    'ios/sam_decoder.mlpackage',
                    'ios/*.onnx']
  s.frameworks   = ['ARKit', 'SceneKit', 'AVFoundation', 'CoreML', 'Vision']

  s.pod_target_xcconfig = {
    'SWIFT_VERSION' => '5.0',
  }
end
