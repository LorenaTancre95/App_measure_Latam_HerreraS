//
//  ContentView.swift
//  boxer
//

import SwiftUI
import UIKit
import simd

struct ContentView: View {
    @StateObject private var viewModel = ARViewModel()
    @StateObject private var signInMgr = GoogleSignInManager.shared
    /// Cuando se pasa este callback, ContentView muestra USAR/REMEDIAR en lugar del modo standalone.
    /// El tercer parámetro es la foto AR como JPEG (nil si no pudo capturarse).
    var onConfirm: ((DetectionInfo, ARViewModel.MeasureUnit, Data?) -> Void)?

    var body: some View {
        ZStack {
            ARViewContainer(viewModel: viewModel)

            // YOLO 2D detection overlay (debug)
            ZStack(alignment: .topLeading) {
                ForEach(Array(viewModel.debugBBoxes.enumerated()), id: \.offset) { _, item in
                    Rectangle()
                        .stroke(Color.yellow, lineWidth: 2)
                        .frame(width: item.rect.width, height: item.rect.height)
                        .offset(x: item.rect.origin.x, y: item.rect.origin.y)
                }
            }
            .ignoresSafeArea()
            .allowsHitTesting(false)

            // Viewfinder overlay
            GeometryReader { geo in
                let vf = viewModel.viewfinderNorm
                let rect = CGRect(
                    x: vf.minX * geo.size.width,
                    y: vf.minY * geo.size.height,
                    width: vf.width * geo.size.width,
                    height: vf.height * geo.size.height
                )
                ViewfinderOverlay(rect: rect)
            }
            .ignoresSafeArea()
            .allowsHitTesting(false)

            // Segmentation mask overlay
            if let overlay = viewModel.segmentationOverlay {
                Image(uiImage: overlay)
                    .resizable()
                    .scaledToFill()
                    .ignoresSafeArea()
                    .allowsHitTesting(false)
                    .transition(.opacity)
                    .overlay(alignment: .top) {
                        Label("Verificá que cubra solo la caja", systemImage: "eye")
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundColor(.black)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 8)
                            .background(Color.yellow.opacity(0.9))
                            .cornerRadius(12)
                            .padding(.top, 60)
                    }
            }

            // Debug panel (top-left, TAP mode)
            if viewModel.measureMode == .tap, !viewModel.debugInfo.isEmpty {
                VStack {
                    Text(viewModel.debugInfo)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(.white)
                        .padding(8)
                        .background(.black.opacity(0.75))
                        .cornerRadius(8)
                        .padding(.top, 60)
                        .padding(.leading, 12)
                    Spacer()
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .allowsHitTesting(false)
            }

            // TAP mode: crosshair + guía de dimensiones + botón CAPTURAR
            if viewModel.measureMode == .tap,
               !viewModel.isProcessing,
               viewModel.detections.isEmpty {

                // Crosshair con línea en vivo al primer punto
                AimingCrosshairView(
                    hit:              viewModel.crosshairHit,
                    isSnapping:       viewModel.isSnapping,
                    step:             viewModel.tapStep,
                    liveDistance:     viewModel.liveDistance,
                    lastCornerScreen: viewModel.lastCornerScreen,
                    unit:             viewModel.measureUnit
                )
                .allowsHitTesting(false)

                VStack {
                    Spacer()
                    HStack(alignment: .bottom) {
                        VStack(alignment: .leading, spacing: 8) {
                            BoxGuideView(tapStep: viewModel.tapStep)
                            DimMeasureView(
                                tapPoints: viewModel.tapPoints,
                                floorY: viewModel.floorY,
                                unit: viewModel.measureUnit
                            )
                        }
                        .padding(.leading, 16)
                        Spacer()
                    }

                    // Botón CAPTURAR (visible mientras no están las 3 dimensiones completas)
                    Button(action: { viewModel.captureCenter() }) {
                        HStack(spacing: 8) {
                            Image(systemName: "scope")
                                .font(.system(size: 18, weight: .bold))
                            Text(viewModel.tapStep < 2 ? "ANCHO — PUNTO \(viewModel.tapStep + 1)/2" : "LARGO — PUNTO \((viewModel.tapStep - 2) + 1)/2")
                                .font(.system(size: 16, weight: .heavy))
                        }
                        .foregroundColor(.black)
                        .frame(maxWidth: .infinity)
                        .frame(height: 54)
                        .background(viewModel.crosshairHit ? Color.yellow : Color.white.opacity(0.85))
                        .cornerRadius(16)
                        .padding(.horizontal, 60)
                    }
                    .padding(.bottom, 24)
                }
            }

            // Top spacer
            VStack { Spacer() }

            // Modo integrado: USAR / REMEDIAR
            if let confirm = onConfirm, !viewModel.detections.isEmpty {
                VStack {
                    Spacer()
                    VStack(spacing: 10) {
                        HStack(spacing: 8) {
                            Image(systemName: "checkmark.circle.fill").foregroundColor(.green)
                            Text("Medición confirmada")
                                .font(.system(size: 14, weight: .semibold)).foregroundColor(.white)
                            Spacer()
                            let d = viewModel.detections[0]
                            Text(viewModel.measureUnit.formatBox(d.size.x, d.size.y, d.size.z))
                                .font(.system(size: 12, design: .monospaced)).foregroundColor(.white.opacity(0.8))
                        }
                        .padding(.horizontal, 14).padding(.vertical, 10)
                        .background(.black.opacity(0.65)).cornerRadius(10)

                        HStack(spacing: 12) {
                            Button(action: { viewModel.clearAll() }) {
                                Text("REMEDIAR")
                                    .font(.system(size: 14, weight: .bold)).foregroundColor(.white)
                                    .frame(maxWidth: .infinity).frame(height: 46)
                                    .background(.white.opacity(0.15)).cornerRadius(10)
                                    .overlay(RoundedRectangle(cornerRadius: 10).stroke(.white.opacity(0.3), lineWidth: 1))
                            }
                            Button(action: {
                                let photo = viewModel.captureCurrentFrame()
                                confirm(viewModel.detections[0], viewModel.measureUnit, photo)
                            }) {
                                Text("USAR")
                                    .font(.system(size: 14, weight: .heavy)).foregroundColor(.white)
                                    .frame(maxWidth: .infinity).frame(height: 46)
                                    .background(Color.green).cornerRadius(10)
                            }
                        }
                    }
                    .padding(.horizontal, 16).padding(.bottom, 24)
                }
            }

            // Modo standalone: tarjetas de detección
            if onConfirm == nil, !viewModel.detections.isEmpty {
                VStack {
                    Spacer()
                    HStack {
                        VStack(alignment: .leading, spacing: 6) {
                            ForEach(Array(viewModel.detections.enumerated()), id: \.element.id) { i, det in
                                DetectionCard(detection: det, color: boxColor(i), unit: viewModel.measureUnit)
                            }
                            Button(action: { viewModel.clearAll() }) {
                                HStack(spacing: 6) {
                                    Image(systemName: "trash").font(.system(size: 12))
                                    Text("Limpiar").font(.system(size: 13, weight: .medium))
                                }
                                .foregroundColor(.white)
                                .padding(.horizontal, 10).padding(.vertical, 6)
                                .background(.red.opacity(0.7)).cornerRadius(6)
                            }
                        }
                        .padding(.leading, 16)
                        Spacer()
                    }
                    .padding(.bottom, 80)
                }
            }

            // Controles de la derecha: selector de unidad + UNDO/BORRAR + Google
            HStack {
                Spacer()
                Text(viewModel.status)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.white.opacity(0.8))
                    .padding(.trailing, 12)
                VStack(spacing: 8) {
                    // Selector de unidad
                    HStack(spacing: 0) {
                        ForEach(ARViewModel.MeasureUnit.allCases, id: \.self) { unit in
                            unitButton(unit)
                        }
                    }
                    .background(.black.opacity(0.45))
                    .cornerRadius(8)

                    // UNDO / BORRAR
                    Button(action: { viewModel.undoLast() }) {
                        ZStack {
                            Circle().fill(.white).frame(width: 70, height: 70)
                            Circle()
                                .fill(viewModel.isProcessing ? Color.gray : Color.orange)
                                .frame(width: 60, height: 60)
                            if viewModel.isProcessing {
                                ProgressView().tint(.white)
                            } else {
                                VStack(spacing: 1) {
                                    Image(systemName: viewModel.tapStep > 0 ? "arrow.uturn.backward" : "xmark")
                                        .font(.system(size: 18, weight: .bold))
                                        .foregroundColor(.white)
                                    Text(viewModel.tapStep > 0 ? "UNDO" : "BORRAR")
                                        .font(.system(size: 8, weight: .bold))
                                        .foregroundColor(.white)
                                }
                            }
                        }
                    }
                    .disabled(viewModel.isProcessing)

                    // Google sign-in / Drive scope
                    if !signInMgr.isSignedIn || !signInMgr.hasDriveScope {
                        Button(action: {
                            if let vc = UIApplication.shared.connectedScenes
                                .compactMap({ $0 as? UIWindowScene })
                                .first?.windows.first?.rootViewController {
                                if signInMgr.isSignedIn {
                                    signInMgr.grantDriveScope(presenting: vc)
                                } else {
                                    signInMgr.signIn(presenting: vc)
                                }
                            }
                        }) {
                            HStack(spacing: 4) {
                                Image(systemName: signInMgr.isSignedIn ? "folder.badge.plus" : "person.crop.circle.badge.plus")
                                Text(signInMgr.isSignedIn ? "Drive" : "Google")
                                    .font(.system(size: 10, weight: .semibold))
                            }
                            .foregroundColor(.white)
                            .padding(.horizontal, 8).padding(.vertical, 5)
                            .background(signInMgr.isSignedIn ? .orange.opacity(0.8) : .black.opacity(0.55))
                            .cornerRadius(8)
                            .overlay(RoundedRectangle(cornerRadius: 8).stroke(.white.opacity(0.3), lineWidth: 1))
                        }
                    }
                }
                .padding(.trailing, 20)
            }
        }
    }

    @ViewBuilder
    private func unitButton(_ unit: ARViewModel.MeasureUnit) -> some View {
        let active = viewModel.measureUnit == unit
        Button(action: { viewModel.measureUnit = unit }) {
            Text(unit.rawValue)
                .font(.system(size: 11, weight: .bold))
                .foregroundColor(active ? .black : .white.opacity(0.6))
                .padding(.horizontal, 10).padding(.vertical, 5)
                .background(active ? Color.yellow : Color.clear)
                .cornerRadius(7)
        }
    }
}

// MARK: - Diagrama visual de la caja

/// Muestra la caja en proyección oblicua con los 4 puntos de tap resaltados según el paso actual.
struct BoxGuideView: View {
    let tapStep: Int  // 0..4

    var body: some View {
        Canvas { ctx, size in
            let w = size.width, h = size.height
            func p(_ nx: CGFloat, _ ny: CGFloat) -> CGPoint { CGPoint(x: nx*w, y: ny*h) }

            // Proyección cabinet: cara frontal + cara superior + cara derecha
            let fBL = p(0.06, 0.90), fBR = p(0.60, 0.90)
            let fTL = p(0.06, 0.40), fTR = p(0.60, 0.40)
            let dvx = w * 0.28, dvy = h * -0.26
            let bTL = CGPoint(x: fTL.x+dvx, y: fTL.y+dvy)
            let bTR = CGPoint(x: fTR.x+dvx, y: fTR.y+dvy)
            let bBR = CGPoint(x: fBR.x+dvx, y: fBR.y+dvy)
            let bBL = CGPoint(x: fBL.x+dvx, y: fBL.y+dvy)

            func fillFace(_ pts: [CGPoint], _ c: Color) {
                var path = Path(); path.move(to: pts[0])
                pts.dropFirst().forEach { path.addLine(to: $0) }; path.closeSubpath()
                ctx.fill(path, with: .color(c))
                ctx.stroke(path, with: .color(.white.opacity(0.35)), lineWidth: 1.5)
            }
            func dashedLine(_ a: CGPoint, _ b: CGPoint) {
                var path = Path(); path.move(to: a); path.addLine(to: b)
                ctx.stroke(path, with: .color(.white.opacity(0.18)), style: StrokeStyle(lineWidth: 1, dash: [4,3]))
            }
            func dimArrow(_ a: CGPoint, _ b: CGPoint, _ c: Color, done: Bool) {
                var path = Path(); path.move(to: a); path.addLine(to: b)
                ctx.stroke(path, with: .color(c.opacity(done ? 0.5 : 0.95)),
                           style: StrokeStyle(lineWidth: 2.5, lineCap: .round, dash: [7,4]))
            }
            func dot(_ pt: CGPoint, _ c: Color, active: Bool) {
                let r: CGFloat = active ? 8 : 5.5
                let rect = CGRect(x: pt.x-r, y: pt.y-r, width: r*2, height: r*2)
                ctx.fill(Path(ellipseIn: rect), with: .color(c))
                ctx.stroke(Path(ellipseIn: rect), with: .color(.white.opacity(0.9)), lineWidth: 1.3)
                if active {
                    let big = CGRect(x: pt.x-13, y: pt.y-13, width: 26, height: 26)
                    ctx.stroke(Path(ellipseIn: big), with: .color(c.opacity(0.35)), lineWidth: 1.5)
                }
            }

            let anchoActive = tapStep < 2
            let largoActive = tapStep >= 2 && tapStep < 4

            fillFace([fBL, fBR, fTR, fTL], anchoActive ? .yellow.opacity(0.14) : .white.opacity(0.04))
            fillFace([fTL, fTR, bTR, bTL], largoActive ? .cyan.opacity(0.16)   : .white.opacity(0.07))
            fillFace([fBR, fTR, bTR, bBR], .white.opacity(0.02))
            dashedLine(bBL, bTL); dashedLine(bBL, bBR); dashedLine(bBL, fBL)

            // Puntos de tap
            let ptL = CGPoint(x: fTL.x, y: (fTL.y+fBL.y)/2)           // ANCHO izquierdo
            let ptR = CGPoint(x: fTR.x, y: (fTR.y+fBR.y)/2)           // ANCHO derecho
            let ptN = CGPoint(x: (fTL.x+fTR.x)/2, y: fTL.y)           // LARGO cercano
            let ptF = CGPoint(x: (bTL.x+bTR.x)/2, y: bTL.y)           // LARGO lejano

            dimArrow(ptL, ptR, .yellow, done: tapStep >= 2)
            if tapStep >= 2 { dimArrow(ptN, ptF, .cyan, done: tapStep >= 4) }

            // ALTO: flecha vertical derecha
            let altX = min(bBR.x + 10, w - 8)
            do {
                var arr = Path()
                arr.move(to: CGPoint(x: altX, y: fBR.y))
                arr.addLine(to: CGPoint(x: altX, y: fTR.y))
                ctx.stroke(arr, with: .color(.orange.opacity(0.75)),
                           style: StrokeStyle(lineWidth: 2, lineCap: .round, dash: [4,3]))
            }

            dot(ptL, tapStep > 0 ? .green : (tapStep == 0 ? .yellow : .gray),     active: tapStep == 0)
            dot(ptR, tapStep > 1 ? .green : (tapStep == 1 ? .yellow : .white.opacity(0.3)), active: tapStep == 1)
            dot(ptN, tapStep > 2 ? .green : (tapStep == 2 ? .cyan   : .white.opacity(0.25)), active: tapStep == 2)
            dot(ptF, tapStep > 3 ? .green : (tapStep == 3 ? .cyan   : .white.opacity(0.25)), active: tapStep == 3)
        }
        .frame(width: 180, height: 128)
        .background(.black.opacity(0.82))
        .cornerRadius(14)
        .overlay(alignment: .topLeading) {
            HStack(spacing: 6) {
                Text(tapStep < 2 ? "ANCHO" : tapStep < 4 ? "LARGO" : "✓ LISTO")
                    .font(.system(size: 10, weight: .heavy))
                    .foregroundColor(tapStep < 2 ? .yellow : tapStep < 4 ? .cyan : .green)
                Spacer()
                Text("ALTO auto")
                    .font(.system(size: 9))
                    .foregroundColor(.orange.opacity(0.8))
            }
            .padding(.horizontal, 8).padding(.top, 6)
        }
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(Color.white.opacity(0.1), lineWidth: 1))
    }
}

// MARK: - Panel de valores medidos

/// Muestra ANCHO / LARGO / ALTO con sus valores según el progreso de los 4 taps.
struct DimMeasureView: View {
    let tapPoints: [simd_float3]
    let floorY: Float?
    let unit: ARViewModel.MeasureUnit

    var body: some View {
        VStack(spacing: 0) {
            dimRow(label: "ANCHO", color: .yellow,
                   value: tapPoints.count >= 2 ? unit.format(simd_distance(tapPoints[0], tapPoints[1])) + " " + unit.rawValue : nil,
                   hint: tapPoints.count < 2 ? (tapPoints.count == 0 ? "izq → der" : "→ 2° tap") : nil,
                   done: tapPoints.count >= 2)
            Divider().background(Color.white.opacity(0.1))
            dimRow(label: "LARGO", color: .cyan,
                   value: tapPoints.count >= 4 ? unit.format(simd_distance(tapPoints[2], tapPoints[3])) + " " + unit.rawValue : nil,
                   hint: tapPoints.count >= 2 && tapPoints.count < 4 ? (tapPoints.count == 2 ? "cerca → lejos" : "→ 2° tap") : nil,
                   done: tapPoints.count >= 4)
            Divider().background(Color.white.opacity(0.1))
            altoRow
        }
        .background(.black.opacity(0.82))
        .cornerRadius(14)
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(Color.white.opacity(0.1), lineWidth: 1))
    }

    private func dimRow(label: String, color: Color, value: String?, hint: String?, done: Bool) -> some View {
        HStack(spacing: 10) {
            ZStack {
                Circle().fill(done ? color : color.opacity(0.15)).frame(width: 26, height: 26)
                if done { Image(systemName: "checkmark").font(.system(size: 10, weight: .heavy)).foregroundColor(.black) }
            }
            Text(label).font(.system(size: 13, weight: .heavy))
                .foregroundColor(done ? color : (hint != nil ? color : .white.opacity(0.3)))
                .frame(width: 52, alignment: .leading)
            if let v = value {
                Text(v).font(.system(size: 14, weight: .bold, design: .monospaced)).foregroundColor(color)
            } else if let h = hint {
                Text(h).font(.system(size: 11)).foregroundColor(color.opacity(0.7))
            } else {
                Text("—").font(.system(size: 13)).foregroundColor(.white.opacity(0.2))
            }
            Spacer()
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
    }

    private var altoRow: some View {
        HStack(spacing: 10) {
            ZStack {
                Circle().fill(tapPoints.count >= 4 && floorY != nil ? Color.orange : Color.white.opacity(0.1))
                    .frame(width: 26, height: 26)
                Image(systemName: "arrow.up.and.down").font(.system(size: 9, weight: .heavy))
                    .foregroundColor(tapPoints.count >= 4 && floorY != nil ? .black : .white.opacity(0.3))
            }
            Text("ALTO").font(.system(size: 13, weight: .heavy))
                .foregroundColor(tapPoints.count >= 4 && floorY != nil ? .orange : .white.opacity(0.3))
                .frame(width: 52, alignment: .leading)
            if tapPoints.count >= 4, let fy = floorY {
                let avgY = (tapPoints[2].y + tapPoints[3].y) / 2.0
                Text(unit.format(max(0.03, avgY - fy)) + " " + unit.rawValue)
                    .font(.system(size: 14, weight: .bold, design: .monospaced)).foregroundColor(.orange)
            } else {
                Text("automático").font(.system(size: 11)).foregroundColor(.white.opacity(0.2))
            }
            Spacer()
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
    }
}

// MARK: - Detection Card

struct DetectionCard: View {
    let detection: DetectionInfo
    let color: Color
    let unit: ARViewModel.MeasureUnit

    var body: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(color)
                .frame(width: 10, height: 10)
            Text(detection.label)
                .font(.system(size: 13, weight: .bold))
                .foregroundColor(.white)
            Text(unit.formatBox(detection.size.x, detection.size.y, detection.size.z))
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(.white.opacity(0.8))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.black.opacity(0.6))
        .cornerRadius(6)
    }
}

func boxColor(_ index: Int) -> Color {
    let colors: [Color] = [.red, .green, .blue]
    return colors[index % colors.count]
}

// MARK: - Aiming Crosshair

/// Crosshair estilo app Measure: línea punteada desde el último punto al crosshair, distancia en vivo.
struct AimingCrosshairView: View {
    let hit: Bool
    let isSnapping: Bool
    let step: Int          // 0..6 (2 taps por dimensión)
    let liveDistance: Float?
    let lastCornerScreen: CGPoint?
    let unit: ARViewModel.MeasureUnit

    private let ringSize: CGFloat = 52
    private let lineLen: CGFloat  = 14

    var body: some View {
        GeometryReader { geo in
            let cx = geo.size.width  / 2
            let cy = geo.size.height / 2
            ZStack {
                Canvas { ctx, _ in
                    let color: Color = isSnapping ? .orange : (hit ? .yellow : .white)

                    // Línea punteada desde el 1° punto al crosshair
                    if let lcs = lastCornerScreen {
                        var lp = Path()
                        lp.move(to: lcs)
                        lp.addLine(to: CGPoint(x: cx, y: cy))
                        ctx.stroke(lp, with: .color(.white.opacity(0.85)),
                                   style: StrokeStyle(lineWidth: 1.8, lineCap: .round, dash: [8, 5]))
                        // Punto rojo en el 1° punto ya colocado
                        ctx.fill(Path(ellipseIn: CGRect(x: lcs.x-9, y: lcs.y-9, width: 18, height: 18)),
                                 with: .color(.red.opacity(0.95)))
                        ctx.stroke(Path(ellipseIn: CGRect(x: lcs.x-9, y: lcs.y-9, width: 18, height: 18)),
                                   with: .color(.white), style: StrokeStyle(lineWidth: 1.5))
                    }

                    // Ring del crosshair
                    let ringRect = CGRect(x: cx - ringSize/2, y: cy - ringSize/2,
                                          width: ringSize, height: ringSize)
                    ctx.stroke(Path(ellipseIn: ringRect), with: .color(color.opacity(0.9)),
                               style: StrokeStyle(lineWidth: 2))
                    // Líneas cruzadas
                    func seg(_ ax: CGFloat, _ ay: CGFloat, _ bx: CGFloat, _ by: CGFloat) {
                        var p = Path(); p.move(to: CGPoint(x: ax, y: ay)); p.addLine(to: CGPoint(x: bx, y: by))
                        ctx.stroke(p, with: .color(color.opacity(0.9)),
                                   style: StrokeStyle(lineWidth: 2, lineCap: .round))
                    }
                    seg(cx - ringSize/2 - lineLen, cy, cx - ringSize/2, cy)
                    seg(cx + ringSize/2, cy, cx + ringSize/2 + lineLen, cy)
                    seg(cx, cy - ringSize/2 - lineLen, cx, cy - ringSize/2)
                    seg(cx, cy + ringSize/2, cx, cy + ringSize/2 + lineLen)
                    // Punto central
                    ctx.fill(Path(ellipseIn: CGRect(x: cx-3, y: cy-3, width: 6, height: 6)),
                             with: .color(color))
                }

                // Distancia en vivo durante el 2° tap de cada par
                if let dist = liveDistance {
                    Text(unit.format(dist) + " " + unit.rawValue)
                        .font(.system(size: 20, weight: .bold, design: .rounded))
                        .foregroundColor(.white)
                        .padding(.horizontal, 14).padding(.vertical, 7)
                        .background(.black.opacity(0.65))
                        .cornerRadius(12)
                        .position(x: cx, y: cy - ringSize/2 - 36)
                }

                // Indicador de tap bajo el crosshair (1/2 o 2/2 dentro de cada par)
                if step < 4 {
                    Text("\(step % 2 + 1)/2")
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .foregroundColor(hit ? .yellow : .white.opacity(0.7))
                        .position(x: cx, y: cy + ringSize/2 + 18)
                }
            }
        }
        .ignoresSafeArea()
    }
}

// MARK: - Viewfinder

struct ViewfinderOverlay: View {
    let rect: CGRect

    var body: some View {
        Canvas { ctx, _ in
            let corner: CGFloat = 28
            let lw: CGFloat = 3

            var outer = Path()
            outer.addRect(CGRect(x: 0, y: 0, width: 9999, height: 9999))
            outer.addRoundedRect(in: rect, cornerSize: CGSize(width: 6, height: 6))
            ctx.fill(outer, with: .color(.black.opacity(0.35)))

            var p = Path()
            let corners: [(CGPoint, CGFloat, CGFloat)] = [
                (CGPoint(x: rect.minX, y: rect.minY),  1,  1),
                (CGPoint(x: rect.maxX, y: rect.minY), -1,  1),
                (CGPoint(x: rect.maxX, y: rect.maxY), -1, -1),
                (CGPoint(x: rect.minX, y: rect.maxY),  1, -1),
            ]
            for (origin, dx, dy) in corners {
                p.move(to: CGPoint(x: origin.x + dx * corner, y: origin.y))
                p.addLine(to: origin)
                p.addLine(to: CGPoint(x: origin.x, y: origin.y + dy * corner))
            }
            ctx.stroke(p, with: .color(.white), style: StrokeStyle(lineWidth: lw, lineCap: .round))

            let cx = rect.midX, cy = rect.midY
            let dot = Path(ellipseIn: CGRect(x: cx - 3, y: cy - 3, width: 6, height: 6))
            ctx.fill(dot, with: .color(.white.opacity(0.6)))
        }
    }
}
