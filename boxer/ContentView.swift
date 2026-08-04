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
    /// Callback en modo integrado (MedicionView): devuelve medición + foto.
    var onConfirm: ((DetectionInfo, ARViewModel.MeasureUnit, Data?) -> Void)?

    var body: some View {
        ZStack {
            ARViewContainer(viewModel: viewModel)

            // YOLO debug bboxes
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
                    .resizable().scaledToFill().ignoresSafeArea()
                    .allowsHitTesting(false).transition(.opacity)
            }

            // ── TAP MODE ──────────────────────────────────────────────────────
            if viewModel.measureMode == .tap, !viewModel.isProcessing, viewModel.detections.isEmpty {

                // Overlay del rectángulo detectado en vivo
                if viewModel.rectState == .detecting,
                   let corners = viewModel.liveRectScreen {
                    RectOverlayView(corners: corners)
                        .ignoresSafeArea()
                        .allowsHitTesting(false)
                }

                // Crosshair (solo en paso LARGO para guiar el tap)
                if viewModel.rectState == .largoTap {
                    SimpleCrosshairView(hit: viewModel.crosshairHit)
                        .ignoresSafeArea()
                        .allowsHitTesting(false)
                }

                // Panel inferior según estado
                VStack {
                    Spacer()
                    tapModePanel
                }
            }

            // ── RESULTADO: modo integrado (USAR / REMEDIAR) ───────────────────
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

            // ── RESULTADO: modo standalone ────────────────────────────────────
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

            // ── CONTROLES DERECHA: unidad + UNDO + Google ─────────────────────
            HStack {
                Spacer()
                Text(viewModel.status)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(.white.opacity(0.75))
                    .padding(.trailing, 8)
                    .lineLimit(2)
                    .multilineTextAlignment(.trailing)
                VStack(spacing: 8) {
                    HStack(spacing: 0) {
                        ForEach(ARViewModel.MeasureUnit.allCases, id: \.self) { unit in
                            unitButton(unit)
                        }
                    }
                    .background(.black.opacity(0.45)).cornerRadius(8)

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
                                    Image(systemName: viewModel.rectState == .detecting ? "xmark" : "arrow.uturn.backward")
                                        .font(.system(size: 18, weight: .bold)).foregroundColor(.white)
                                    Text(viewModel.rectState == .detecting ? "BORRAR" : "UNDO")
                                        .font(.system(size: 8, weight: .bold)).foregroundColor(.white)
                                }
                            }
                        }
                    }
                    .disabled(viewModel.isProcessing)

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

    // MARK: - Panel TAP mode

    @ViewBuilder
    private var tapModePanel: some View {
        switch viewModel.rectState {
        case .detecting:
            detectingPanel
        case .largoTap:
            largoPanel
        case .done:
            EmptyView()
        }
    }

    /// Panel "DETECTING": instrucción + botón CONFIRMAR (verde cuando hay rectángulo)
    private var detectingPanel: some View {
        VStack(spacing: 12) {
            // Mini panel de estado
            HStack(spacing: 10) {
                Image(systemName: viewModel.liveRectScreen != nil ? "rectangle.inset.filled" : "viewfinder")
                    .font(.system(size: 20))
                    .foregroundColor(viewModel.liveRectScreen != nil ? .yellow : .white.opacity(0.5))
                VStack(alignment: .leading, spacing: 2) {
                    Text(viewModel.liveRectScreen != nil ? "Cara detectada" : "Buscando cara frontal...")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundColor(viewModel.liveRectScreen != nil ? .yellow : .white.opacity(0.7))
                    Text("Apuntá de frente a la caja")
                        .font(.system(size: 11))
                        .foregroundColor(.white.opacity(0.45))
                }
                Spacer()
            }
            .padding(.horizontal, 16).padding(.vertical, 10)
            .background(.black.opacity(0.7)).cornerRadius(12)
            .overlay(RoundedRectangle(cornerRadius: 12)
                .stroke(viewModel.liveRectScreen != nil ? Color.yellow.opacity(0.6) : Color.white.opacity(0.1), lineWidth: 1.5))

            // CONFIRMAR
            Button(action: { viewModel.confirmFaceRect() }) {
                HStack(spacing: 8) {
                    Image(systemName: "checkmark.rectangle.fill")
                        .font(.system(size: 18, weight: .bold))
                    Text("CONFIRMAR CARA")
                        .font(.system(size: 16, weight: .heavy))
                }
                .foregroundColor(viewModel.liveRectScreen != nil ? .black : .white.opacity(0.4))
                .frame(maxWidth: .infinity).frame(height: 54)
                .background(viewModel.liveRectScreen != nil ? Color.yellow : Color.white.opacity(0.12))
                .cornerRadius(16)
            }
            .disabled(viewModel.liveRectScreen == nil)
        }
        .padding(.horizontal, 20).padding(.bottom, 24)
    }

    /// Panel "LARGO TAP": muestra ANCHO+ALTO confirmados + instrucción para tap del LARGO
    private var largoPanel: some View {
        VStack(spacing: 12) {
            // Dimensiones confirmadas
            HStack(spacing: 0) {
                confirmedDim(label: "ANCHO",
                             value: viewModel.confirmedAncho.map { viewModel.measureUnit.format($0) + " " + viewModel.measureUnit.rawValue },
                             color: .yellow)
                Divider().background(.white.opacity(0.15)).frame(height: 36)
                confirmedDim(label: "ALTO",
                             value: viewModel.confirmedAlto.map { viewModel.measureUnit.format($0) + " " + viewModel.measureUnit.rawValue },
                             color: .orange)
                Divider().background(.white.opacity(0.15)).frame(height: 36)
                confirmedDim(label: "LARGO",
                             value: nil,
                             color: .cyan,
                             pending: true)
            }
            .padding(.vertical, 10)
            .background(.black.opacity(0.75)).cornerRadius(12)
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(.white.opacity(0.1), lineWidth: 1))

            // Instrucción + botón captura crosshair
            Button(action: {
                let center = CGPoint(x: viewModel.viewportSize.width / 2,
                                     y: viewModel.viewportSize.height / 2)
                viewModel.measureLargoAt(point: center)
            }) {
                HStack(spacing: 8) {
                    Image(systemName: "scope").font(.system(size: 18, weight: .bold))
                    Text("LARGO — tocá borde lejano")
                        .font(.system(size: 15, weight: .heavy))
                }
                .foregroundColor(.black)
                .frame(maxWidth: .infinity).frame(height: 54)
                .background(viewModel.crosshairHit ? Color.cyan : Color.white.opacity(0.85))
                .cornerRadius(16)
            }
        }
        .padding(.horizontal, 20).padding(.bottom, 24)
    }

    private func confirmedDim(label: String, value: String?, color: Color, pending: Bool = false) -> some View {
        VStack(spacing: 3) {
            Text(label).font(.system(size: 9, weight: .bold)).foregroundColor(color.opacity(0.7))
            if let v = value {
                Text(v).font(.system(size: 14, weight: .heavy, design: .monospaced)).foregroundColor(color)
            } else if pending {
                Text("—").font(.system(size: 16)).foregroundColor(color.opacity(0.3))
                    .overlay(alignment: .center) {
                        ProgressView().scaleEffect(0.6).tint(color.opacity(0.5))
                    }
            }
        }
        .frame(maxWidth: .infinity)
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

// MARK: - Rectangle overlay (cara detectada)

/// Dibuja el cuadrilátero del rectángulo detectado sobre la pantalla.
struct RectOverlayView: View {
    let corners: [CGPoint]  // [TL, TR, BL, BR]

    var body: some View {
        Canvas { ctx, _ in
            guard corners.count == 4 else { return }
            let tl = corners[0], tr = corners[1], bl = corners[2], br = corners[3]

            var fill = Path()
            fill.move(to: tl); fill.addLine(to: tr); fill.addLine(to: br); fill.addLine(to: bl)
            fill.closeSubpath()
            ctx.fill(fill, with: .color(.yellow.opacity(0.12)))
            ctx.stroke(fill, with: .color(.yellow.opacity(0.9)),
                       style: StrokeStyle(lineWidth: 2.5, lineCap: .round, lineJoin: .round, dash: [10, 5]))

            // Esquinas marcadas
            for pt in [tl, tr, bl, br] {
                let r: CGFloat = 7
                ctx.fill(Path(ellipseIn: CGRect(x: pt.x-r, y: pt.y-r, width: r*2, height: r*2)),
                         with: .color(.yellow))
                ctx.stroke(Path(ellipseIn: CGRect(x: pt.x-r, y: pt.y-r, width: r*2, height: r*2)),
                           with: .color(.black.opacity(0.5)), style: StrokeStyle(lineWidth: 1))
            }
        }
    }
}

// MARK: - Simple Crosshair para LARGO tap

struct SimpleCrosshairView: View {
    let hit: Bool
    private let ringSize: CGFloat = 52
    private let lineLen: CGFloat  = 14

    var body: some View {
        GeometryReader { geo in
            let cx = geo.size.width / 2, cy = geo.size.height / 2
            Canvas { ctx, _ in
                let color: Color = hit ? .cyan : .white
                let ringRect = CGRect(x: cx-ringSize/2, y: cy-ringSize/2, width: ringSize, height: ringSize)
                ctx.stroke(Path(ellipseIn: ringRect), with: .color(color.opacity(0.9)),
                           style: StrokeStyle(lineWidth: 2))
                func seg(_ ax: CGFloat, _ ay: CGFloat, _ bx: CGFloat, _ by: CGFloat) {
                    var p = Path(); p.move(to: CGPoint(x: ax, y: ay)); p.addLine(to: CGPoint(x: bx, y: by))
                    ctx.stroke(p, with: .color(color.opacity(0.9)),
                               style: StrokeStyle(lineWidth: 2, lineCap: .round))
                }
                seg(cx-ringSize/2-lineLen, cy, cx-ringSize/2, cy)
                seg(cx+ringSize/2, cy, cx+ringSize/2+lineLen, cy)
                seg(cx, cy-ringSize/2-lineLen, cx, cy-ringSize/2)
                seg(cx, cy+ringSize/2, cx, cy+ringSize/2+lineLen)
                ctx.fill(Path(ellipseIn: CGRect(x: cx-3, y: cy-3, width: 6, height: 6)),
                         with: .color(color))
            }
        }
        .ignoresSafeArea()
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
            Text(detection.label)
                .font(.system(size: 13, weight: .bold)).foregroundColor(.white)
            Text(unit.formatBox(detection.size.x, detection.size.y, detection.size.z))
                .font(.system(size: 11, design: .monospaced)).foregroundColor(.white.opacity(0.8))
        }
        .padding(.horizontal, 10).padding(.vertical, 6)
        .background(.black.opacity(0.6)).cornerRadius(6)
    }
}

func boxColor(_ index: Int) -> Color {
    let colors: [Color] = [.red, .green, .blue]
    return colors[index % colors.count]
}

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
        }
    }
}
