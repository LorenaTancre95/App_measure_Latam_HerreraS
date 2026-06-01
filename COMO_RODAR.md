# Como rodar no iPhone 17 Pro Max — desde Windows

## O que é EAS Build?
O serviço de build do Expo compila o app iOS nos **servidores deles** (que têm Mac).
Você escreve código no Windows, eles compilam, você instala no iPhone.

---

## Passo 1 — Instalar ferramentas (uma vez só)

Abra o **Terminal / PowerShell** no Windows:

```bash
# Instalar Node.js (se não tiver): https://nodejs.org  → versão LTS

# Instalar Expo e EAS CLI globalmente
npm install -g expo-cli eas-cli

# Verificar
eas --version
```

---

## Passo 2 — Criar conta Expo (grátis)

Acesse: https://expo.dev → Create account

```bash
# Fazer login no terminal
eas login
```

---

## Passo 3 — Instalar dependências do projeto

No terminal, dentro da pasta `MedidorCaixasRN/`:

```bash
cd "C:\Users\silvi\OneDrive\Desktop\App_e_Cargo_BR\MedidorCaixasRN"
npm install
```

---

## Passo 4 — Configurar EAS no projeto

```bash
eas init --id medidor-caixas
```

Isso vai perguntar algumas coisas — responda com os defaults.

---

## Passo 5 — Conta Apple Developer (necessário para instalar no iPhone)

- Acesse: https://developer.apple.com/programs
- Plano Individual: USD 99/ano
- Ou use **Ad Hoc distribution** (grátis para até 100 dispositivos no mesmo account)

```bash
# Configurar credenciais iOS no EAS
eas credentials --platform ios
```

---

## Passo 6 — Compilar para iOS

```bash
# Build de preview (instala direto no iPhone via TestFlight ou link)
eas build --platform ios --profile preview
```

O EAS vai:
1. Fazer upload do seu código para os servidores deles
2. Compilar numa Mac virtual (leva ~15-20 min na primeira vez)
3. Gerar um arquivo `.ipa`
4. Te mandar o link para download

---

## Passo 7 — Instalar no iPhone

### Opção A: Via link QR (mais simples)
- O EAS te manda um link e QR code
- Escaneie com o iPhone → instala pelo Safari

### Opção B: Via TestFlight
```bash
eas submit --platform ios
```
Distribui pelo TestFlight (mais profissional, aceita até 10.000 testadores)

---

## Ciclo de desenvolvimento

```
Edita código no Windows
    ↓
git push (opcional, mas recomendado)
    ↓
eas build --platform ios --profile preview
    ↓
Instala no iPhone pelo link
    ↓
Testa → volta pro início
```

---

## Estrutura do projeto

```
MedidorCaixasRN/
├── App.tsx                        ← Navegação principal
├── src/
│   ├── screens/
│   │   ├── FormScreen.tsx         ← Formulário (tela 1 e 4)
│   │   └── ARCameraScreen.tsx     ← Câmera AR (tela 2 e 3)
│   ├── components/
│   │   ├── FormField.tsx
│   │   └── VolumeRow.tsx
│   ├── models/
│   │   ├── BoxMeasurement.ts
│   │   └── Volume.ts
│   └── theme.ts                   ← Cores navy/purple
└── modules/
    └── lidar-box-measure/
        ├── index.ts               ← Interface JS
        └── ios/
            ├── LidarBoxMeasureModule.swift   ← Expo Module
            ├── ARBoxView.swift               ← View ARKit
            └── BoxDetectionCoordinator.swift ← LiDAR + Vision
```

## Integrar a API (quando estiver pronta)

Em `src/screens/FormScreen.tsx`, função `finalizar()`:

```typescript
const finalizar = async () => {
  setFinal(true);
  try {
    await fetch('https://sua-api.com/medicoes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        minuta,
        volumes: volumes.map(v => ({
          c: v.comprimento, l: v.largura, a: v.altura,
          quantidade: v.quantidade, pesoUnit: v.pesoUnit,
        })),
      }),
    });
  } finally {
    setFinal(false);
  }
};
```
