//
//  ContentView.swift
//  boxer
//
//  Created by Bharath Kumar Adinarayan on 09.04.26.
//

import SwiftUI
import UIKit

struct ContentView: View {
    @StateObject private var viewModel = ARViewModel()
    @StateObject private var signInMgr = GoogleSignInManager.shared
    /// Cuando se pasa este callback, ContentView muestra USAR/REMEDIAR en lugar del modo standalone.
    var onConfirm: ((DetectionInfo) -> Void)?

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

            // Segmentation mask overlay (TAP mode — shown for 1.5 s before OBB)
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

            // Debug panel (top-left, always visible in TAP mode)
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

            // TAP mode: aiming crosshair + guide diagram + CAPTURAR button
            if viewModel.measureMode == .tap,
               !viewModel.isProcessing,
               viewModel.detections.isEmpty {

                // Crosshair in the absolute center of the screen
                AimingCrosshairView(hit: viewModel.crosshairHit,
                                    step: viewModel.cornerStep)
                    .allowsHitTesting(false)

                VStack {
                    Spacer()
                    HStack(alignment: .bottom) {
                        BoxGuideView(
                            tappedCount: viewModel.cornerStep,
                            instruction: viewModel.cornerInstruction
                        )
                        .padding(.leading, 16)
                        Spacer()
                    }
                    // CAPTURAR / MEDIR button — bottom center, thumb-reachable
                    if viewModel.cornerStep < 4 {
                        Button(action: { viewModel.captureCenter() }) {
                            HStack(spacing: 8) {
                                Image(systemName: "scope")
                                    .font(.system(size: 18, weight: .bold))
                                Text("CAPTURAR")
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
                    } else {
                        HStack(spacing: 12) {
                            Button(action: { viewModel.undoLastCorner() }) {
                                Label("UNDO P3", systemImage: "arrow.uturn.backward")
                                    .font(.system(size: 13, weight: .bold))
                                    .foregroundColor(.white)
                                    .frame(height: 50)
                                    .frame(maxWidth: .infinity)
                                    .background(Color.orange.opacity(0.9))
                                    .cornerRadius(14)
                            }
                            Button(action: { viewModel.confirmBox() }) {
                                Label("MEDIR", systemImage: "checkmark.circle.fill")
                                    .font(.system(size: 15, weight: .heavy))
                                    .foregroundColor(.black)
                                    .frame(height: 50)
                                    .frame(maxWidth: .infinity)
                                    .background(Color.green)
                                    .cornerRadius(14)
                            }
                        }
                        .padding(.horizontal, 24)
                        .padding(.bottom, 24)
                    }
                }
            }

            // Top spacer (removed status bar)
            VStack { Spacer() }

            // Modo integrado: USAR / REMEDIAR
            if let confirm = onConfirm, !viewModel.detections.isEmpty {
                VStack {
                    Spacer()
                    VStack(spacing: 10) {
                        // Tarjeta "Medición confirmada"
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
                            Button(action: { confirm(viewModel.detections[0]) }) {
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

            // Confidence slider bottom right
            VStack {
                Spacer()
                HStack {
                    Spacer()
                    VStack(spacing: 2) {
                        Text(String(format: "conf: %.1f", viewModel.confidenceThreshold))
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundColor(.white.opacity(0.7))
                        Slider(value: $viewModel.confidenceThreshold, in: 0.1...0.9, step: 0.1)
                            .frame(width: 120)
                            .tint(.white)
                    }
                }
                .padding(.trailing, 20)
                .padding(.bottom, 30)
            }

            // Capture button right centre + mode toggle + calibration state
            HStack {
                Spacer()
                Text(viewModel.status)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.white.opacity(0.8))
                    .padding(.trailing, 12)
                VStack(spacing: 8) {
                    // CAJA / OVERSIZE / TAP toggle
                    HStack(spacing: 0) {
                        modeButton("CAJA", mode: .box)
                        modeButton("OVERSIZE", mode: .oversize)
                        modeButton("TAP", mode: .tap)
                    }
                    .background(.black.opacity(0.45))
                    .cornerRadius(8)

                    // Unit selector: cm | m | in
                    HStack(spacing: 0) {
                        ForEach(ARViewModel.MeasureUnit.allCases, id: \.self) { unit in
                            unitButton(unit)
                        }
                    }
                    .background(.black.opacity(0.45))
                    .cornerRadius(8)

                    // Scan button (hidden in TAP mode — user taps directly on screen)
                    if viewModel.measureMode != .tap {
                        Button(action: { viewModel.detectNow() }) {
                            ZStack {
                                Circle()
                                    .fill(.white)
                                    .frame(width: 70, height: 70)
                                Circle()
                                    .fill(viewModel.isProcessing ? .gray : (viewModel.isCalibrated ? .blue : .orange))
                                    .frame(width: 60, height: 60)
                                if viewModel.isProcessing {
                                    ProgressView().tint(.white)
                                } else if !viewModel.isCalibrated {
                                    Image(systemName: "arrow.down.to.line")
                                        .font(.system(size: 22))
                                        .foregroundColor(.white)
                                } else {
                                    Image(systemName: viewModel.measureMode == .oversize ? "shippingbox.fill" : "cube.transparent.fill")
                                        .font(.system(size: 24))
                                        .foregroundColor(.white)
                                }
                            }
                        }
                        .disabled(viewModel.isProcessing || !viewModel.isCalibrated)
                    } else {
                        // TAP mode: CAPTURAR is at the bottom center; this button just resets/undoes
                        Button(action: {
                            if viewModel.cornerStep > 0 { viewModel.undoLastCorner() }
                            else { viewModel.clearAll() }
                        }) {
                            ZStack {
                                Circle().fill(.white).frame(width: 70, height: 70)
                                Circle()
                                    .fill(viewModel.isProcessing ? Color.gray : Color.orange)
                                    .frame(width: 60, height: 60)
                                if viewModel.isProcessing {
                                    ProgressView().tint(.white)
                                } else {
                                    VStack(spacing: 1) {
                                        Image(systemName: viewModel.cornerStep > 0 ? "arrow.uturn.backward" : "xmark")
                                            .font(.system(size: 18, weight: .bold))
                                            .foregroundColor(.white)
                                        Text(viewModel.cornerStep > 0 ? "UNDO" : "BORRAR")
                                            .font(.system(size: 8, weight: .bold))
                                            .foregroundColor(.white)
                                    }
                                }
                            }
                        }
                        .disabled(viewModel.isProcessing)
                    }

                    // Botón de captura de foto para dataset
                    Button(action: { viewModel.captureAndUpload() }) {
                        Image(systemName: "camera.fill")
                            .font(.system(size: 18))
                            .foregroundColor(.white)
                            .frame(width: 44, height: 44)
                            .background(.black.opacity(0.5))
                            .clipShape(Circle())
                            .overlay(Circle().stroke(.white.opacity(0.3), lineWidth: 1))
                    }
                    .disabled(viewModel.isProcessing)

                    // Botón Google: sign-in si no hay sesión, o pedir Drive scope si falta
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
    private func modeButton(_ title: String, mode: ARViewModel.MeasureMode) -> some View {
        let active = viewModel.measureMode == mode
        Button(action: { viewModel.measureMode = mode }) {
            Text(title)
                .font(.system(size: 11, weight: .bold))
                .foregroundColor(active ? .black : .white.opacity(0.6))
                .padding(.horizontal, 10).padding(.vertical, 5)
                .background(active ? Color.white : Color.clear)
                .cornerRadius(7)
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

// Must match colors in ARViewModel.placeBoxes
func boxColor(_ index: Int) -> Color {
    let colors: [Color] = [.red, .green, .blue]
    return colors[index % colors.count]
}

// MARK: - Aiming Crosshair

/// Full-screen overlay with a crosshair at the center for aim-and-capture flow.
/// Ring turns yellow when the surface under the crosshair is detected.
struct AimingCrosshairView: View {
    let hit: Bool    // surface detected at center
    let step: Int    // 0-3: which corner is next

    private let ringSize: CGFloat = 52
    private let lineLen: CGFloat  = 14

    var body: some View {
        GeometryReader { geo in
            let cx = geo.size.width  / 2
            let cy = geo.size.height / 2
            Canvas { ctx, _ in
                let color: Color = hit ? .yellow : .white
                // Outer ring
                let ringRect = CGRect(x: cx - ringSize/2, y: cy - ringSize/2,
                                      width: ringSize, height: ringSize)
                ctx.stroke(Path(ellipseIn: ringRect), with: .color(color.opacity(0.9)),
                           style: StrokeStyle(lineWidth: 2))
                // Cross lines
                func line(_ ax: CGFloat, _ ay: CGFloat, _ bx: CGFloat, _ by: CGFloat) {
                    var p = Path(); p.move(to: CGPoint(x: ax, y: ay)); p.addLine(to: CGPoint(x: bx, y: by))
                    ctx.stroke(p, with: .color(color.opacity(0.9)),
                               style: StrokeStyle(lineWidth: 2, lineCap: .round))
                }
                line(cx - ringSize/2 - lineLen, cy, cx - ringSize/2, cy)
                line(cx + ringSize/2, cy, cx + ringSize/2 + lineLen, cy)
                line(cx, cy - ringSize/2 - lineLen, cx, cy - ringSize/2)
                line(cx, cy + ringSize/2, cx, cy + ringSize/2 + lineLen)
                // Center dot
                ctx.fill(Path(ellipseIn: CGRect(x: cx-3, y: cy-3, width: 6, height: 6)),
                         with: .color(color))
            }
            // Step label inside the ring
            Text("\(step + 1)/4")
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundColor(hit ? .yellow : .white.opacity(0.7))
                .position(x: cx, y: cy + ringSize/2 + 18)
        }
        .ignoresSafeArea()
    }
}

// MARK: - Box Guide

/// Perspective wireframe diagram with 4 numbered tap-point markers.
/// Tap order: 1=front-top-left, 2=front-top-right, 3=front-bottom-right, 4=back-top-right corner.
struct BoxGuideView: View {
    let tappedCount: Int   // 0..3: how many corners have been recorded
    let instruction: String

    private static let W: CGFloat = 168
    private static let H: CGFloat = 118

    // Front face vertices
    private static let fTL = CGPoint(x: 14,  y: 36)
    private static let fTR = CGPoint(x: 104, y: 36)
    private static let fBR = CGPoint(x: 104, y: 110)
    private static let fBL = CGPoint(x: 14,  y: 110)
    // Perspective back face
    private static let tTL = CGPoint(x: 64,  y: 16)
    private static let tTR = CGPoint(x: 154, y: 16)
    private static let tBR = CGPoint(x: 154, y: 90)
    private static let tBL = CGPoint(x: 64,  y: 90)

    // 4 tap positions: 3 front corners + back-top-right corner (gives full depth via dot product)
    private static let tapPositions: [CGPoint] = [
        fTL,   // 1 — front top left
        fTR,   // 2 — front top right
        fBR,   // 3 — front bottom right
        tTR,   // 4 — back top right corner (depth = |dot(P3−P0, depthAxis)|)
    ]

    private func markerColor(_ idx: Int) -> Color {
        if idx < tappedCount { return .green }
        if idx == tappedCount { return .yellow }
        return .white.opacity(0.25)
    }

    var body: some View {
        VStack(spacing: 6) {
            ZStack {
                Canvas { ctx, _ in
                    func edge(_ a: CGPoint, _ b: CGPoint, dashed: Bool = false) {
                        var p = Path(); p.move(to: a); p.addLine(to: b)
                        let style = dashed
                            ? StrokeStyle(lineWidth: 1.2, dash: [4, 3])
                            : StrokeStyle(lineWidth: 1.8, lineCap: .round, lineJoin: .round)
                        ctx.stroke(p, with: .color(.white.opacity(0.75)), style: style)
                    }
                    let fTL = BoxGuideView.fTL, fTR = BoxGuideView.fTR
                    let fBR = BoxGuideView.fBR, fBL = BoxGuideView.fBL
                    let tTL = BoxGuideView.tTL, tTR = BoxGuideView.tTR
                    let tBR = BoxGuideView.tBR, tBL = BoxGuideView.tBL

                    // Visible edges (solid)
                    edge(fTL, fTR); edge(fTR, fBR); edge(fBR, fBL); edge(fBL, fTL)
                    edge(fTL, tTL); edge(fTR, tTR); edge(tTL, tTR)
                    edge(fBR, tBR); edge(tTR, tBR)
                    // Hidden edges (dashed)
                    edge(fBL, tBL, dashed: true)
                    edge(tTL, tBL, dashed: true)
                    edge(tBL, tBR, dashed: true)

                    // Dashed ring around tTR when waiting for the 4th tap
                    if tappedCount == 3 {
                        let target = BoxGuideView.tTR
                        var ring = Path()
                        ring.addEllipse(in: CGRect(x: target.x - 16, y: target.y - 16, width: 32, height: 32))
                        ctx.stroke(ring, with: .color(.yellow.opacity(0.85)),
                                   style: StrokeStyle(lineWidth: 2, dash: [4, 3]))
                    }
                }
                .frame(width: BoxGuideView.W, height: BoxGuideView.H)

                ForEach(0..<4, id: \.self) { i in
                    ZStack {
                        Circle()
                            .fill(markerColor(i))
                            .frame(width: 22, height: 22)
                        if i < tappedCount {
                            Image(systemName: "checkmark")
                                .font(.system(size: 9, weight: .heavy))
                                .foregroundColor(.black)
                        } else {
                            Text("\(i + 1)")
                                .font(.system(size: 11, weight: .heavy))
                                .foregroundColor(i == tappedCount ? .black : .white.opacity(0.7))
                        }
                    }
                    .position(BoxGuideView.tapPositions[i])
                }
            }
            .frame(width: BoxGuideView.W, height: BoxGuideView.H)

            if !shortInstruction.isEmpty {
                Text(shortInstruction)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(tappedCount == 3 ? .yellow : .white)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: BoxGuideView.W)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(.black.opacity(0.80))
        .cornerRadius(14)
    }

    // Strip "X/4 — " prefix so only the action text is shown below the diagram
    private var shortInstruction: String {
        if let r = instruction.range(of: " — ") {
            return String(instruction[r.upperBound...])
        }
        return instruction
    }
}

// MARK: - Viewfinder

struct ViewfinderOverlay: View {
    let rect: CGRect

    var body: some View {
        Canvas { ctx, _ in
            let corner: CGFloat = 28
            let lw: CGFloat = 3

            // Dim area outside the viewfinder.
            var outer = Path()
            outer.addRect(CGRect(x: 0, y: 0, width: 9999, height: 9999))
            outer.addRoundedRect(in: rect, cornerSize: CGSize(width: 6, height: 6))
            ctx.fill(outer, with: .color(.black.opacity(0.35)))

            // Corner brackets.
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

            // Crosshair dot in center.
            let cx = rect.midX, cy = rect.midY
            let dot = Path(ellipseIn: CGRect(x: cx - 3, y: cy - 3, width: 6, height: 6))
            ctx.fill(dot, with: .color(.white.opacity(0.6)))
        }
    }
}
