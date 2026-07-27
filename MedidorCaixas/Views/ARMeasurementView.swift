import SwiftUI
import ARKit

struct ARMeasurementView: View {
    @StateObject private var arViewModel = ARMeasurementViewModel()
    @Environment(\.dismiss) private var dismiss

    var onUsar: (BoxMeasurement) -> Void

    var body: some View {
        ZStack {
            // MARK: AR Camera
            ARViewContainer(viewModel: arViewModel)
                .ignoresSafeArea()

            // MARK: Scan-frame corners
            ScanFrameView(isConfirmed: arViewModel.isConfirmed)

            // MARK: State-driven instruction / spinner
            VStack {
                Spacer()
                instructionOverlay
                Spacer()
            }

            // MARK: Error toast
            if let msg = arViewModel.errorMessage {
                VStack {
                    ErrorToastView(message: msg)
                        .padding(.top, 60)
                    Spacer()
                }
            }

            // MARK: Bottom panel
            VStack(spacing: 0) {
                Spacer()
                bottomPanel
            }
        }
        .navigationBarHidden(true)
    }

    // MARK: - Instruction overlay (centre of screen)
    @ViewBuilder
    private var instructionOverlay: some View {
        switch arViewModel.state {
        case .searchingPlane:
            InstructionBanner(
                icon: "arrow.down.to.line",
                text: "Aponte para o chão para calibrar"
            )

        case .readyToSelect:
            InstructionBanner(
                icon: "hand.tap",
                text: "Toque na caixa que deseja medir"
            )

        case .processing:
            ProcessingIndicatorView()

        case .measuring, .confirmed:
            EmptyView()
        }
    }

    // MARK: - Bottom panel
    @ViewBuilder
    private var bottomPanel: some View {
        if arViewModel.isConfirmed, let m = arViewModel.measurement {
            ConfirmedPanelView(
                measurement: m,
                onRemedir: { arViewModel.remedir() },
                onUsar: {
                    onUsar(m)
                    dismiss()
                }
            )
        } else if arViewModel.state == .measuring, let m = arViewModel.measurement {
            MeasuringPanelView(
                measurement: m,
                statusText: m.displayText,
                canConfirm: true,
                onConfirm: { arViewModel.confirm() }
            )
        } else {
            EmptyView()
        }
    }
}

// MARK: - Subviews

struct InstructionBanner: View {
    let icon: String
    let text: String

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: icon)
                .font(.system(size: 18))
                .foregroundColor(AppTheme.yellow)
            Text(text)
                .font(.system(size: 15, weight: .semibold))
                .foregroundColor(.white)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(Color.black.opacity(0.65))
        .cornerRadius(20)
        .padding(.bottom, 180)
    }
}

struct ProcessingIndicatorView: View {
    @State private var rotating = false

    var body: some View {
        VStack(spacing: 12) {
            Circle()
                .trim(from: 0, to: 0.75)
                .stroke(AppTheme.yellow, lineWidth: 3)
                .frame(width: 40, height: 40)
                .rotationEffect(.degrees(rotating ? 360 : 0))
                .animation(.linear(duration: 0.8).repeatForever(autoreverses: false),
                           value: rotating)
                .onAppear { rotating = true }

            Text("Analisando...")
                .font(.system(size: 14, weight: .semibold))
                .foregroundColor(.white)
        }
        .padding(20)
        .background(Color.black.opacity(0.65))
        .cornerRadius(16)
        .padding(.bottom, 180)
    }
}

struct ErrorToastView: View {
    let message: String

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.orange)
            Text(message)
                .font(.system(size: 13, weight: .medium))
                .foregroundColor(.white)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(Color.black.opacity(0.75))
        .cornerRadius(12)
        .padding(.horizontal, 24)
    }
}

struct ScanFrameView: View {
    let isConfirmed: Bool
    private let cornerSize: CGFloat = 28

    var color: Color { isConfirmed ? AppTheme.arConfirmed : AppTheme.arScanBorder }

    var body: some View {
        GeometryReader { geo in
            let inset: CGFloat = 40
            let rect = CGRect(x: inset, y: inset * 2,
                              width: geo.size.width - inset * 2,
                              height: geo.size.height - inset * 4)
            ZStack {
                cornerMark(at: CGPoint(x: rect.minX, y: rect.minY), corner: .topLeft)
                cornerMark(at: CGPoint(x: rect.maxX, y: rect.minY), corner: .topRight)
                cornerMark(at: CGPoint(x: rect.minX, y: rect.maxY), corner: .bottomLeft)
                cornerMark(at: CGPoint(x: rect.maxX, y: rect.maxY), corner: .bottomRight)
            }
        }
    }

    private enum Corner { case topLeft, topRight, bottomLeft, bottomRight }

    @ViewBuilder
    private func cornerMark(at point: CGPoint, corner: Corner) -> some View {
        Path { path in
            switch corner {
            case .topLeft:
                path.move(to: CGPoint(x: point.x, y: point.y + cornerSize))
                path.addLine(to: point)
                path.addLine(to: CGPoint(x: point.x + cornerSize, y: point.y))
            case .topRight:
                path.move(to: CGPoint(x: point.x - cornerSize, y: point.y))
                path.addLine(to: point)
                path.addLine(to: CGPoint(x: point.x, y: point.y + cornerSize))
            case .bottomLeft:
                path.move(to: CGPoint(x: point.x, y: point.y - cornerSize))
                path.addLine(to: point)
                path.addLine(to: CGPoint(x: point.x + cornerSize, y: point.y))
            case .bottomRight:
                path.move(to: CGPoint(x: point.x - cornerSize, y: point.y))
                path.addLine(to: point)
                path.addLine(to: CGPoint(x: point.x, y: point.y - cornerSize))
            }
        }
        .stroke(color, lineWidth: 3)
    }
}

struct MeasuringPanelView: View {
    let measurement: BoxMeasurement
    let statusText: String
    let canConfirm: Bool
    let onConfirm: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            Text(statusText)
                .font(.system(size: 16, weight: .semibold))
                .foregroundColor(.white)
                .padding(.vertical, 12)
                .frame(maxWidth: .infinity)
                .background(Color.black.opacity(0.55))

            HStack {
                Circle()
                    .fill(AppTheme.yellow)
                    .frame(width: 10, height: 10)
                Text("Medição pronta")
                    .font(.caption)
                    .foregroundColor(.white)
                Spacer()
                if canConfirm {
                    Button(action: onConfirm) {
                        Text("Confirmar")
                            .font(.system(size: 13, weight: .bold))
                            .foregroundColor(.black)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 6)
                            .background(AppTheme.yellow)
                            .cornerRadius(20)
                    }
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(AppTheme.yellow.opacity(0.15))
        }
    }
}

struct ConfirmedPanelView: View {
    let measurement: BoxMeasurement
    let onRemedir: () -> Void
    let onUsar: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Image(systemName: "checkmark.square.fill")
                    .foregroundColor(AppTheme.green)
                Text("Medição confirmada")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundColor(AppTheme.green)
            }
            .padding(.top, 12)

            Text(String(format: "Vol: %.4f m³  •  Peso Cubado: %.1f kg",
                        measurement.volumeM3, measurement.pesoCubado))
                .font(.system(size: 13))
                .foregroundColor(AppTheme.textSecondary)
                .padding(.top, 4)

            HStack(spacing: 12) {
                Button(action: onRemedir) {
                    HStack {
                        Image(systemName: "arrow.clockwise")
                        Text("Remedir")
                    }
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundColor(AppTheme.textPrimary)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(AppTheme.card)
                    .cornerRadius(12)
                    .overlay(RoundedRectangle(cornerRadius: 12)
                        .stroke(AppTheme.fieldBorder, lineWidth: 1))
                }

                Button(action: onUsar) {
                    HStack {
                        Image(systemName: "checkmark")
                        Text("Usar")
                    }
                    .font(.system(size: 15, weight: .bold))
                    .foregroundColor(.black)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(AppTheme.green)
                    .cornerRadius(12)
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
        }
        .frame(maxWidth: .infinity)
        .background(Color.black.opacity(0.80))
    }
}
