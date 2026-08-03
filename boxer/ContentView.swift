//
//  ContentView.swift
//  boxer
//

import SwiftUI
import UIKit

struct ContentView: View {
    @StateObject private var viewModel = ARViewModel()
    @StateObject private var signInMgr = GoogleSignInManager.shared
    /// Cuando se pasa este callback, ContentView muestra USAR/REMEDIAR en lugar del modo standalone.
    var onConfirm: ((DetectionInfo, ARViewModel.MeasureUnit) -> Void)?

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
                        // Panel de progreso de dimensiones
                        DimMeasureView(
                            measurements: viewModel.measurements,
                            hasFirstPoint: viewModel.firstPoint != nil,
                            unit: viewModel.measureUnit
                        )
                        .padding(.leading, 16)
                        Spacer()
                    }

                    // Botón CAPTURAR (visible mientras no están las 3 dimensiones completas)
                    Button(action: { viewModel.captureCenter() }) {
                        HStack(spacing: 8) {
                            Image(systemName: "scope")
                                .font(.system(size: 18, weight: .bold))
                            Text(viewModel.firstPoint == nil ? "CAPTURAR 1° PUNTO" : "CAPTURAR 2° PUNTO")
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
                            Button(action: { confirm(viewModel.detections[0], viewModel.measureUnit) }) {
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

// MARK: - Panel de progreso de dimensiones

/// Muestra las 3 dimensiones (ANCHO / LARGO / ALTO) con su estado:
/// verde + valor cuando está completa, amarillo cuando es la actual, gris cuando está pendiente.
struct DimMeasureView: View {
    let measurements: [Float]
    let hasFirstPoint: Bool
    let unit: ARViewModel.MeasureUnit

    private let labels = ARViewModel.dimLabels

    var body: some View {
        VStack(spacing: 0) {
            ForEach(0..<3, id: \.self) { i in
                let done    = i < measurements.count
                let current = i == measurements.count
                HStack(spacing: 10) {
                    // Indicador de estado
                    ZStack {
                        Circle()
                            .fill(done ? Color.green : (current ? Color.yellow : Color.white.opacity(0.15)))
                            .frame(width: 26, height: 26)
                        if done {
                            Image(systemName: "checkmark")
                                .font(.system(size: 10, weight: .heavy))
                                .foregroundColor(.black)
                        } else {
                            Text("\(i + 1)")
                                .font(.system(size: 11, weight: .heavy))
                                .foregroundColor(current ? .black : .white.opacity(0.4))
                        }
                    }

                    // Nombre de la dimensión
                    Text(labels[i])
                        .font(.system(size: 13, weight: .heavy))
                        .foregroundColor(done ? .green : (current ? .yellow : .white.opacity(0.35)))
                        .frame(width: 52, alignment: .leading)

                    // Valor o estado
                    if done {
                        Text(unit.format(measurements[i]) + " " + unit.rawValue)
                            .font(.system(size: 14, weight: .bold, design: .monospaced))
                            .foregroundColor(.green)
                    } else if current {
                        Text(hasFirstPoint ? "→ 2° punto" : "→ 1° punto")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundColor(.yellow)
                    } else {
                        Text("—")
                            .font(.system(size: 13))
                            .foregroundColor(.white.opacity(0.2))
                    }

                    Spacer()
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .background(current ? Color.yellow.opacity(0.08) : Color.clear)

                if i < 2 {
                    Divider().background(Color.white.opacity(0.1))
                }
            }
        }
        .background(.black.opacity(0.82))
        .cornerRadius(14)
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(Color.white.opacity(0.1), lineWidth: 1))
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

                // Distancia en vivo sobre el crosshair (solo cuando hay 1° punto colocado)
                if let dist = liveDistance, step % 2 == 1 {
                    Text(unit.format(dist) + " " + unit.rawValue)
                        .font(.system(size: 20, weight: .bold, design: .rounded))
                        .foregroundColor(.white)
                        .padding(.horizontal, 14).padding(.vertical, 7)
                        .background(.black.opacity(0.65))
                        .cornerRadius(12)
                        .position(x: cx, y: cy - ringSize/2 - 36)
                }

                // Indicador de punto (1° o 2°) bajo el crosshair
                if step < 6 {
                    let ptNum = (step % 2) + 1
                    Text("\(ptNum)/2")
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
