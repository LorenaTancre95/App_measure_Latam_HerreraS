require 'json'

package = JSON.parse(File.read(File.join(__dir__, '../../package.json')))

Pod::Spec.new do |s|
  s.name           = 'LidarBoxMeasure'
  s.version        = package['version']
  s.summary        = 'LiDAR box measurement module for MedidorCaixas'
  s.homepage       = 'https://github.com/ecargo/medidor-caixas'
  s.license        = package['license']
  s.authors        = package['author']
  s.platform       = :ios, '17.0'
  s.source         = { git: '' }
  s.static_framework = true

  s.dependency 'ExpoModulesCore'

  s.source_files = 'ios/**/*.{swift,m,mm}'

  s.frameworks = ['ARKit', 'Vision', 'SceneKit', 'AVFoundation']

  s.pod_target_xcconfig = {
    'SWIFT_VERSION' => '5.9'
  }
end
