import SwiftUI

struct MinutaView: View {
    @EnvironmentObject var appState: AppState
    @StateObject private var speech = SpeechRecognizer()
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
                            .stroke(speech.isListening ? Color.red : Color.green, lineWidth: 2)
                            .background(Color.latamCard.cornerRadius(10))
                            .animation(.easeInOut(duration: 0.2), value: speech.isListening)
                        HStack {
                            TextField("", text: $minutaText,
                                      prompt: Text("Número de minuta").foregroundColor(.white.opacity(0.35)))
                                .foregroundColor(.white)
                                .keyboardType(.numbersAndPunctuation)
                                .focused($focused)
                                .padding(.leading, 14)
                            Spacer()
                            Button(action: {
                                focused = false
                                speech.toggle()
                            }) {
                                Image(systemName: speech.isListening ? "waveform" : "mic")
                                    .foregroundColor(speech.isListening ? .red : .white.opacity(0.6))
                                    .symbolEffect(.pulse, isActive: speech.isListening)
                                    .font(.system(size: 20))
                            }
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

                if let err = speech.errorMsg {
                    Text(err)
                        .font(.caption)
                        .foregroundColor(.red.opacity(0.85))
                        .padding(.top, 8)
                }

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
        .onChange(of: speech.transcript) { _, val in
            let cleaned = val.trimmingCharacters(in: .whitespacesAndNewlines)
            if !cleaned.isEmpty {
                minutaText = cleaned
            }
        }
    }

    private func confirmar() {
        if speech.isListening { speech.toggle() }
        guard !minutaText.isEmpty else { return }
        appState.path.append(.medicion(minutaText))
    }
}
