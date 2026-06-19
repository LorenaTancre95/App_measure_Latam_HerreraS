import SwiftUI

struct MinutaView: View {
    @EnvironmentObject var appState: AppState
    @State private var minutaText = ""
    @FocusState private var focused: Bool

    var body: some View {
        ZStack {
            Color.latamBlue.ignoresSafeArea()

            VStack(spacing: 0) {
                // Título
                Text("REGISTRAR MEDICIÓN")
                    .font(.system(size: 14, weight: .heavy))
                    .foregroundColor(.white)
                    .padding(.top, 16)
                    .padding(.bottom, 36)

                // Campo de minuta
                HStack(spacing: 12) {
                    ZStack {
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(Color.green, lineWidth: 2)
                            .background(Color.latamCard.cornerRadius(10))
                        HStack {
                            TextField("", text: $minutaText,
                                      prompt: Text("Número de minuta").foregroundColor(.white.opacity(0.35)))
                                .foregroundColor(.white)
                                .keyboardType(.numbersAndPunctuation)
                                .focused($focused)
                                .padding(.leading, 14)
                            Spacer()
                            Image(systemName: "barcode.viewfinder")
                                .foregroundColor(.white.opacity(0.45))
                                .padding(.trailing, 14)
                        }
                    }
                    .frame(height: 52)

                    Button(action: confirmar) {
                        Image(systemName: "arrow.right.circle.fill")
                            .font(.system(size: 38))
                            .foregroundColor(minutaText.isEmpty ? .white.opacity(0.25) : .white)
                    }
                    .disabled(minutaText.isEmpty)
                }
                .padding(.horizontal, 20)

                Spacer()
            }
        }
        .navigationBarHidden(true)
        .overlay(alignment: .topLeading) {
            Button(action: { appState.path.removeLast() }) {
                HStack(spacing: 4) {
                    Image(systemName: "chevron.left")
                    Text("Volver")
                }
                .foregroundColor(.white)
                .padding(16)
            }
        }
        .onAppear { focused = true }
    }

    private func confirmar() {
        guard !minutaText.isEmpty else { return }
        appState.path.append(.medicion(minutaText))
    }
}
