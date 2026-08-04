import Speech
import AVFoundation

/// Graba voz y publica el texto reconocido en `transcript`.
/// Uso: llamar `toggle()` para iniciar/detener. Bindear `transcript` al TextField.
@MainActor
final class SpeechRecognizer: ObservableObject {
    @Published var transcript  = ""
    @Published var isListening = false
    @Published var errorMsg:   String? = nil

    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "es-419"))
    private var request:   SFSpeechAudioBufferRecognitionRequest?
    private var task:      SFSpeechRecognitionTask?
    private let engine   = AVAudioEngine()

    func toggle() {
        isListening ? stopListening() : requestAndStart()
    }

    // MARK: - Private

    private func requestAndStart() {
        SFSpeechRecognizer.requestAuthorization { [weak self] authStatus in
            guard authStatus == .authorized else {
                Task { @MainActor in self?.errorMsg = "Permiso de voz denegado" }
                return
            }
            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                guard granted else {
                    Task { @MainActor in self?.errorMsg = "Permiso de micrófono denegado" }
                    return
                }
                Task { @MainActor in self?.startListening() }
            }
        }
    }

    private func startListening() {
        errorMsg = nil
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.record, mode: .measurement, options: .duckOthers)
            try session.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            errorMsg = "Audio no disponible"; return
        }

        request = SFSpeechAudioBufferRecognitionRequest()
        guard let request else { return }
        request.shouldReportPartialResults = true

        let node   = engine.inputNode
        let format = node.outputFormat(forBus: 0)
        node.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buf, _ in
            self?.request?.append(buf)
        }
        do { try engine.start() } catch {
            errorMsg = "No se pudo iniciar el audio"; return
        }
        isListening = true

        task = recognizer?.recognitionTask(with: request) { [weak self] result, _ in
            guard let self else { return }
            if let r = result {
                Task { @MainActor in
                    self.transcript = r.bestTranscription.formattedString
                    if r.isFinal { self.stopListening() }
                }
            }
        }
    }

    func stopListening() {
        guard isListening else { return }
        engine.stop()
        engine.inputNode.removeTap(onBus: 0)
        request?.endAudio()
        task?.cancel()
        request = nil; task = nil
        isListening = false
        try? AVAudioSession.sharedInstance().setActive(false,
                                                       options: .notifyOthersOnDeactivation)
    }
}
