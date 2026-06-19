//
//  ContentView.swift
//  boxer
//
//  Created by Bharath Kumar Adinarayan on 09.04.26.
//

import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = ARViewModel()

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

            // Top spacer (removed status bar)
            VStack { Spacer() }

            // Detection cards at bottom left
            if !viewModel.detections.isEmpty {
                VStack {
                    Spacer()
                    HStack {
                        VStack(alignment: .leading, spacing: 6) {
                            ForEach(Array(viewModel.detections.enumerated()), id: \.element.id) { i, det in
                                DetectionCard(detection: det, color: boxColor(i))
                            }
                            Button(action: { viewModel.clearAll() }) {
                                HStack(spacing: 6) {
                                    Image(systemName: "trash")
                                        .font(.system(size: 12))
                                    Text("Clear all")
                                        .font(.system(size: 13, weight: .medium))
                                }
                                .foregroundColor(.white)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(.red.opacity(0.7))
                                .cornerRadius(6)
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

            // Capture button right centre + calibration state
            HStack {
                Spacer()
                Text(viewModel.status)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.white.opacity(0.8))
                    .padding(.trailing, 12)
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
                            Image(systemName: "cube.transparent.fill")
                                .font(.system(size: 24))
                                .foregroundColor(.white)
                        }
                    }
                }
                .disabled(viewModel.isProcessing || !viewModel.isCalibrated)
                .padding(.trailing, 20)
            }
        }
    }
}

struct DetectionCard: View {
    let detection: DetectionInfo
    let color: Color

    var body: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(color)
                .frame(width: 10, height: 10)
            Text(detection.label)
                .font(.system(size: 13, weight: .bold))
                .foregroundColor(.white)
            Text(String(format: "%.0fx%.0fx%.0f",
                        detection.size.x * 100,
                        detection.size.y * 100,
                        detection.size.z * 100))
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(.white.opacity(0.7))
            Text("cm")
                .font(.system(size: 11))
                .foregroundColor(.white.opacity(0.5))
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
