# MedidorCaixas iOS — Setup Xcode

## Requisitos
- Xcode 16+
- iPhone 17 Pro Max (LiDAR obrigatório para modo AUTO)
- iOS 17.0+ deployment target
- macOS para compilar

## Passos para criar o projeto Xcode

1. **Novo projeto**
   - Xcode → File → New → Project
   - Escolha: iOS → App
   - Product Name: `MedidorCaixas`
   - Interface: SwiftUI
   - Language: Swift
   - Desmarque "Include Tests" (opcional)

2. **Adicionar os arquivos**
   - Delete o `ContentView.swift` gerado pelo Xcode
   - Arraste a pasta `MedidorCaixas/` (com subpastas Models/, ViewModels/, Views/, AR/) para dentro do projeto no Xcode
   - Marque "Copy items if needed" e "Create groups"

3. **Info.plist — Permissões**
   Adicione em Info.plist (ou no Target → Info tab):
   ```
   NSCameraUsageDescription  →  "Necessário para medir caixas com realidade aumentada"
   ```

4. **Deployment Target**
   - Target → General → Minimum Deployments: iOS 17.0

5. **Capabilities**
   - Target → Signing & Capabilities
   - Selecione seu Team (Apple Developer Account)

6. **Compilar e testar**
   - Conecte iPhone 17 Pro Max via cabo ou TrustWifi
   - Selecione o dispositivo no Xcode
   - Cmd+R para compilar e rodar

## Estrutura de arquivos

```
MedidorCaixas/
├── MedidorCaixasApp.swift          ← Ponto de entrada
├── Models/
│   ├── BoxMeasurement.swift        ← Resultado de medição AR
│   ├── Volume.swift                ← Item da lista (C/L/A + peso)
│   └── Theme.swift                 ← Cores e estilos
├── ViewModels/
│   ├── MeditionFormViewModel.swift ← Lógica do formulário
│   └── ARMeasurementViewModel.swift← Estado da câmera AR
├── Views/
│   ├── MeditionFormView.swift      ← Tela principal (formulário)
│   ├── ARMeasurementView.swift     ← Tela AR (câmera + overlay)
│   └── Components/
│       ├── FormFieldView.swift     ← Campos reutilizáveis
│       └── VolumeRowView.swift     ← Linha da tabela + totais
└── AR/
    ├── ARViewContainer.swift       ← UIViewRepresentable do ARSCNView
    └── BoxDetectionCoordinator.swift ← LiDAR + Vision + SceneKit
```

## Como funciona a medição com LiDAR

1. ARKit detecta plano horizontal (chão) — "1/2 Calibrar"
2. Vision detecta o retângulo da caixa no frame da câmera — "2/2 Calibrar"  
3. Para cada canto do retângulo detectado, amostramos a profundidade real do
   mapa de profundidade LiDAR (ARFrame.sceneDepth.depthMap)
4. Reprojetamos os pontos 2D + profundidade → coordenadas 3D no mundo AR
5. Calculamos distâncias reais: C = largura face frontal, A = altura face frontal
6. L (profundidade) = estimada via raycast lateral ao plano do chão
7. Buffer de estabilidade: 5 frames consecutivos dentro de ±3 cm → medição confirmada

## Próximos passos (integração API)

Em `MeditionFormViewModel.swift`, método `finalizar()`:
- Substitua o `DispatchQueue.main.asyncAfter` por uma chamada URLSession real
- O payload a enviar:
  ```json
  {
    "minuta": "957-00000156",
    "volumes": [
      { "c": 40, "l": 20, "a": 33, "quantidade": 12, "pesoUnit": 52 }
    ]
  }
  ```
