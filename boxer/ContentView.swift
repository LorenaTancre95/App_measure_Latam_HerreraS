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
    var onConfirm: ((DetectionInfo, ARViewModel.MeasureUnit, Data?) -> Void)?

    var body: some View {
        ZStack {
            ARViewContainer(viewModel: viewModel)

            // YOLO debug bboxes
            ZStack(alignment: .topLeading) {
                ForEach(Array(viewModel.debugBBoxes.enumerated()), id: \.offset) { _, item in
                    Rectangle().stroke(Color.yellow, lineWidth: 2)
                        .frame(width: item.rect.width, height: item.rect.height)
                        .offset(x: item.rect.origin.x, y: item.rect.origin.y)
                }
            }
            .ignoresSafeArea().allowsHitTesting(false)

            // Viewfinder
            GeometryReader { geo in
                let vf = viewModel.viewfinderNorm
                ViewfinderOverlay(rect: CGRect(x: vf.minX*geo.size.width, y: vf.minY*geo.size.height,
                                               width: vf.width*geo.size.width, height: vf.height*geo.size.height))
            }
            .ignoresSafeArea().allowsHitTesting(false)

            // ── TAP MODE ─────────────────────────────────────────────────────
            if viewModel.measureMode == .tap, !viewModel.isProcessing, viewModel.detections.isEmpty {

                // Crosshair: para ANCHO, LARGO y ALTO (todas las dimensiones)
                if viewModel.tapPhase != .preview {
                    MeasureCrosshairView(
                        hit:           viewModel.crosshairHit,
                        isStable:      viewModel.isAimStable,
                        step:          viewModel.tapStep,
                        liveDistance:  viewModel.liveDistance,
                        lastTapScreen: viewModel.lastTapScreen,
                        unit:          viewModel.measureUnit,
                        activeDim:     viewModel.activeDim
                    )
                    .ignoresSafeArea().allowsHitTesting(false)
                }

                // Panel inferior: siempre visible (muestra GUARDAR/BORRAR en preview)
                VStack {
                    Spacer()
                    tapPanel
                }
            }

            // ── RESULTADO integrado (USAR / REMEDIAR) ─────────────────────
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

            // ── RESULTADO standalone ───────────────────────────────────────
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
                                .foregroundColor(.white).padding(.horizontal, 10).padding(.vertical, 6)
                                .background(.red.opacity(0.7)).cornerRadius(6)
                            }
                        }
                        .padding(.leading, 16); Spacer()
                    }.padding(.bottom, 80)
                }
            }

            // ── CONTROLES DERECHA ──────────────────────────────────────────
            HStack {
                Spacer()
                VStack(spacing: 8) {
                    // Selector unidad
                    HStack(spacing: 0) {
                        ForEach(ARViewModel.MeasureUnit.allCases, id: \.self) { unit in
                            unitBtn(unit)
                        }
                    }
                    .background(.black.opacity(0.45)).cornerRadius(8)

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
                                        .font(.system(size: 18, weight: .bold)).foregroundColor(.white)
                                    Text(viewModel.tapStep > 0 ? "UNDO" : "BORRAR")
                                        .font(.system(size: 8, weight: .bold)).foregroundColor(.white)
                                }
                            }
                        }
                    }.disabled(viewModel.isProcessing)

                    if !signInMgr.isSignedIn || !signInMgr.hasDriveScope {
                        Button(action: {
                            if let vc = UIApplication.shared.connectedScenes
                                .compactMap({ $0 as? UIWindowScene })
                                .first?.windows.first?.rootViewController {
                                if signInMgr.isSignedIn { signInMgr.grantDriveScope(presenting: vc) }
                                else { signInMgr.signIn(presenting: vc) }
                            }
                        }) {
                            HStack(spacing: 4) {
                                Image(systemName: signInMgr.isSignedIn ? "folder.badge.plus" : "person.crop.circle.badge.plus")
                                Text(signInMgr.isSignedIn ? "Drive" : "Google")
                                    .font(.system(size: 10, weight: .semibold))
                            }
                            .foregroundColor(.white).padding(.horizontal, 8).padding(.vertical, 5)
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

    // MARK: - Panel TAP

    private var tapPanel: some View {
        VStack(spacing: 12) {

            // ── 3 chips de progreso ──────────────────────────────────────────
            HStack(spacing: 8) {
                dimChip("ANCHO", val: viewModel.savedAncho, phase: .ancho)
                dimChip("LARGO", val: viewModel.savedLargo, phase: .largo)
                dimChip("ALTO",  val: viewModel.savedAlto,  phase: .alto)
            }

            if viewModel.tapPhase == .preview {
                // ── Preview: distancia grande + GUARDAR / BORRAR ─────────────
                if let dist = viewModel.measuredDistance {
                    Text(viewModel.measureUnit.format(dist) + " " + viewModel.measureUnit.rawValue)
                        .font(.system(size: 42, weight: .heavy, design: .rounded))
                        .foregroundColor(dimColor(viewModel.dimPhase))
                        .padding(.vertical, 2)
                }
                HStack(spacing: 10) {
                    Button(action: { viewModel.borrarMedicion() }) {
                        Text("BORRAR")
                            .font(.system(size: 15, weight: .bold)).foregroundColor(.white)
                            .frame(maxWidth: .infinity).frame(height: 50)
                            .background(Color.white.opacity(0.12)).cornerRadius(12)
                            .overlay(RoundedRectangle(cornerRadius: 12).stroke(.white.opacity(0.25), lineWidth: 1))
                    }
                    Button(action: { viewModel.guardarMedicion() }) {
                        Text("GUARDAR")
                            .font(.system(size: 15, weight: .heavy)).foregroundColor(.black)
                            .frame(maxWidth: .infinity).frame(height: 50)
                            .background(dimColor(viewModel.dimPhase)).cornerRadius(12)
                    }
                }

            } else {
                // ── ANCHO / LARGO / ALTO: crosshair + CAPTURAR ──────────────
                Text(viewModel.currentInstruction)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(dimColor(viewModel.dimPhase))
                    .multilineTextAlignment(.center)
                    .frame(minHeight: 18)

                Button(action: { viewModel.captureCenter() }) {
                    HStack(spacing: 8) {
                        Image(systemName: viewModel.isAimStable ? "checkmark.circle.fill" : "scope")
                            .font(.system(size: 18, weight: .bold))
                        Text(viewModel.isAimStable ? "CAPTURAR" : "Estabilizando...")
                            .font(.system(size: 16, weight: .heavy))
                    }
                    .foregroundColor(viewModel.isAimStable ? .black : .white.opacity(0.45))
                    .frame(maxWidth: .infinity).frame(height: 54)
                    .background(viewModel.isAimStable ? dimColor(viewModel.dimPhase) : Color.white.opacity(0.10))
                    .cornerRadius(16)
                }
                .disabled(!viewModel.isAimStable)
            }
        }
        .padding(.horizontal, 20).padding(.vertical, 14)
        .background(.black.opacity(0.55))
        .cornerRadius(20)
        .padding(.horizontal, 12).padding(.bottom, 16)
    }

    // Chip de progreso para cada dimensión
    @ViewBuilder
    private func dimChip(_ name: String, val: Float?, phase: ARViewModel.DimPhase) -> some View {
        let isActive = viewModel.dimPhase == phase
        let color    = dimColor(phase)
        Group {
            if let v = val {
                HStack(spacing: 3) {
                    Image(systemName: "checkmark").font(.system(size: 9, weight: .bold))
                    Text(viewModel.measureUnit.format(v) + viewModel.measureUnit.rawValue)
                        .font(.system(size: 11, weight: .bold))
                }
                .foregroundColor(.white)
                .padding(.horizontal, 10).padding(.vertical, 6)
                .background(color.opacity(0.28)).cornerRadius(8)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(color.opacity(0.85), lineWidth: 1))
            } else {
                Text(name)
                    .font(.system(size: 11, weight: .bold))
                    .foregroundColor(isActive ? color : color.opacity(0.35))
                    .padding(.horizontal, 10).padding(.vertical, 6)
                    .background(isActive ? Color.white.opacity(0.12) : Color.white.opacity(0.04))
                    .cornerRadius(8)
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(color.opacity(isActive ? 0.6 : 0.2), lineWidth: 1))
            }
        }
        .frame(maxWidth: .infinity)
    }

    private func dimColor(_ phase: ARViewModel.DimPhase) -> Color {
        switch phase {
        case .ancho: return .yellow
        case .largo: return .cyan
        case .alto:  return .orange
        case .done:  return .green
        }
    }

    // Overload para MeasureCrosshairView que sigue usando Int
    private func dimColor(_ step: Int) -> Color {
        switch step {
        case 0, 1: return .yellow
        case 2, 3: return .cyan
        default:   return .orange
        }
    }

    @ViewBuilder
    private func unitBtn(_ unit: ARViewModel.MeasureUnit) -> some View {
        let active = viewModel.measureUnit == unit
        Button(action: { viewModel.measureUnit = unit }) {
            Text(unit.rawValue)
                .font(.system(size: 11, weight: .bold))
                .foregroundColor(active ? .black : .white.opacity(0.6))
                .padding(.horizontal, 10).padding(.vertical, 5)
                .background(active ? Color.yellow : Color.clear).cornerRadius(7)
        }
    }
}

// MARK: - Crosshair con preview

/// Crosshair estilo Measure: muestra dónde va a caer el tap, línea al punto anterior, distancia viva.
struct MeasureCrosshairView: View {
    let hit: Bool
    let isStable: Bool
    let step: Int
    let liveDistance: Float?
    let lastTapScreen: CGPoint?
    let unit: ARViewModel.MeasureUnit
    let activeDim: String

    private let ringSize: CGFloat = 48
    private let lineLen:  CGFloat = 12

    var body: some View {
        GeometryReader { geo in
            let cx = geo.size.width / 2
            let cy = geo.size.height / 2
            ZStack {
                Canvas { ctx, _ in
                    let color: Color = hit ? dimColor : .white.opacity(0.5)

                    // Línea punteada desde el último tap al crosshair
                    if let lts = lastTapScreen, step % 2 == 1 {
                        var lp = Path(); lp.move(to: lts); lp.addLine(to: CGPoint(x: cx, y: cy))
                        ctx.stroke(lp, with: .color(dimColor.opacity(0.85)),
                                   style: StrokeStyle(lineWidth: 2, lineCap: .round, dash: [8, 5]))
                        // Punto del tap anterior
                        let r: CGFloat = 8
                        ctx.fill(Path(ellipseIn: CGRect(x: lts.x-r, y: lts.y-r, width: r*2, height: r*2)),
                                 with: .color(dimColor))
                        ctx.stroke(Path(ellipseIn: CGRect(x: lts.x-r, y: lts.y-r, width: r*2, height: r*2)),
                                   with: .color(.white.opacity(0.9)), style: StrokeStyle(lineWidth: 1.5))
                    }

                    // Ring externo: más grueso y brillante cuando está estabilizado
                    let rr = CGRect(x: cx-ringSize/2, y: cy-ringSize/2, width: ringSize, height: ringSize)
                    ctx.stroke(Path(ellipseIn: rr), with: .color(color.opacity(isStable ? 1.0 : 0.6)),
                               style: StrokeStyle(lineWidth: isStable ? 3 : 1.5))
                    // Ring interior cuando estable (doble anillo = "bloqueado")
                    if isStable {
                        let inner: CGFloat = ringSize * 0.6
                        let ri = CGRect(x: cx-inner/2, y: cy-inner/2, width: inner, height: inner)
                        ctx.stroke(Path(ellipseIn: ri), with: .color(color.opacity(0.5)),
                                   style: StrokeStyle(lineWidth: 1))
                    }

                    // Cruz
                    func seg(_ ax: CGFloat, _ ay: CGFloat, _ bx: CGFloat, _ by: CGFloat) {
                        var p = Path(); p.move(to: CGPoint(x: ax, y: ay)); p.addLine(to: CGPoint(x: bx, y: by))
                        ctx.stroke(p, with: .color(color.opacity(0.9)), style: StrokeStyle(lineWidth: 2, lineCap: .round))
                    }
                    seg(cx-ringSize/2-lineLen, cy, cx-ringSize/2, cy)
                    seg(cx+ringSize/2, cy, cx+ringSize/2+lineLen, cy)
                    seg(cx, cy-ringSize/2-lineLen, cx, cy-ringSize/2)
                    seg(cx, cy+ringSize/2, cx, cy+ringSize/2+lineLen)

                    // Punto central: relleno sólido cuando estable
                    let dotR: CGFloat = isStable ? 6 : 3
                    ctx.fill(Path(ellipseIn: CGRect(x: cx-dotR, y: cy-dotR, width: dotR*2, height: dotR*2)),
                             with: .color(color))
                }

                // Distancia en vivo (2° tap del par)
                if let dist = liveDistance {
                    Text(unit.format(dist) + " " + unit.rawValue)
                        .font(.system(size: 22, weight: .bold, design: .rounded))
                        .foregroundColor(.white)
                        .padding(.horizontal, 14).padding(.vertical, 7)
                        .background(.black.opacity(0.65)).cornerRadius(12)
                        .position(x: cx, y: cy - ringSize/2 - 36)
                }

                // Indicador "tap X/2" bajo el crosshair
                if step < 6 {
                    Text("\(step % 2 + 1) / 2")
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .foregroundColor(hit ? dimColor : .white.opacity(0.5))
                        .position(x: cx, y: cy + ringSize/2 + 18)
                }
            }
        }
        .ignoresSafeArea()
    }

    private var dimColor: Color {
        switch step {
        case 0, 1: return .yellow
        case 2, 3: return .cyan
        default:   return .orange
        }
    }
}

// MARK: - Detection Card

struct DetectionCard: View {
    let detection: DetectionInfo
    let color: Color
    let unit: ARViewModel.MeasureUnit
    var body: some View {
        HStack(spacing: 8) {
            Circle().fill(color).frame(width: 10, height: 10)
            Text(detection.label).font(.system(size: 13, weight: .bold)).foregroundColor(.white)
            Text(unit.formatBox(detection.size.x, detection.size.y, detection.size.z))
                .font(.system(size: 11, design: .monospaced)).foregroundColor(.white.opacity(0.8))
        }
        .padding(.horizontal, 10).padding(.vertical, 6)
        .background(.black.opacity(0.6)).cornerRadius(6)
    }
}

func boxColor(_ index: Int) -> Color { [Color.red, .green, .blue][index % 3] }

// MARK: - Viewfinder

struct ViewfinderOverlay: View {
    let rect: CGRect
    var body: some View {
        Canvas { ctx, _ in
            let corner: CGFloat = 28, lw: CGFloat = 3
            var outer = Path()
            outer.addRect(CGRect(x: 0, y: 0, width: 9999, height: 9999))
            outer.addRoundedRect(in: rect, cornerSize: CGSize(width: 6, height: 6))
            ctx.fill(outer, with: .color(.black.opacity(0.35)))
            var p = Path()
            for (origin, dx, dy) in [(CGPoint(x:rect.minX,y:rect.minY),CGFloat(1),CGFloat(1)),
                                      (CGPoint(x:rect.maxX,y:rect.minY),-1,CGFloat(1)),
                                      (CGPoint(x:rect.maxX,y:rect.maxY),-1,CGFloat(-1)),
                                      (CGPoint(x:rect.minX,y:rect.maxY),CGFloat(1),-1)] {
                p.move(to: CGPoint(x: origin.x + dx*corner, y: origin.y))
                p.addLine(to: origin)
                p.addLine(to: CGPoint(x: origin.x, y: origin.y + dy*corner))
            }
            ctx.stroke(p, with: .color(.white), style: StrokeStyle(lineWidth: lw, lineCap: .round))
        }
    }
}
