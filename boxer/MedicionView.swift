import SwiftUI

// MARK: - Pantalla principal de medición
struct MedicionView: View {
    let minuta: String
    @EnvironmentObject var appState: AppState

    // Campos del formulario
    @State private var cVal = ""     // Largo (m)
    @State private var lVal = ""     // Ancho (m)
    @State private var aVal = ""     // Alto  (m)
    @State private var volsVal = ""
    @State private var pesoUnitVal = ""

    // Unidad usada en la última medición AR (determina labels y conversión)
    @State private var measureUnit: ARViewModel.MeasureUnit = .cm

    // Ítems confirmados
    @State private var items: [MedicionItem] = []

    // Modales
    @State private var showSheet = false
    @State private var showCamera = false
    @State private var camTrigger: CamTrigger = .manual

    enum CamTrigger { case manual, unitario, lote }

    var body: some View {
        ZStack {
            Color.latamBlue.ignoresSafeArea()

            VStack(spacing: 0) {
                Text("REGISTRAR MEDICIÓN")
                    .font(.system(size: 14, weight: .heavy))
                    .foregroundColor(.white)
                    .padding(.top, 16)
                    .padding(.bottom, 14)

                ScrollView {
                    VStack(spacing: 14) {
                        minutaRow
                        dimensionesRow
                        pesosRow
                        agregarBtn
                        if !items.isEmpty { tablaItems }
                    }
                    .padding(.bottom, 24)
                }

                if !items.isEmpty { botonesFinales }
            }
        }
        .navigationBarHidden(true)
        .overlay(alignment: .topLeading) {
            Button(action: { appState.path.removeLast() }) {
                HStack(spacing: 4) {
                    Image(systemName: "chevron.left")
                    Text("Volver")
                }
                .foregroundColor(.white).padding(16)
            }
        }
        .sheet(isPresented: $showSheet) { tipoSheet }
        .fullScreenCover(isPresented: $showCamera) {
            ARCameraWrapper { det, unit in
                // Convertir metros → unidad seleccionada en el AR app
                measureUnit = unit
                cVal = unit.format(det.size.x)
                lVal = unit.format(det.size.z)
                aVal = unit.format(det.size.y)
                if camTrigger == .unitario && !volsVal.isEmpty && !pesoUnitVal.isEmpty {
                    agregarItem()
                }
                showCamera = false
            }
        }
    }

    // MARK: - Sub-vistas del formulario

    private var minutaRow: some View {
        HStack {
            ZStack {
                RoundedRectangle(cornerRadius: 8).stroke(Color.green, lineWidth: 1.5)
                    .background(Color.latamCard.cornerRadius(8))
                HStack {
                    Text(minuta).foregroundColor(.white)
                        .font(.system(size: 15, weight: .medium)).padding(.leading, 12)
                    Spacer()
                    Image(systemName: "xmark.circle").foregroundColor(.white.opacity(0.35)).padding(.trailing, 12)
                }
            }
            .frame(height: 46)
            Image(systemName: "pencil").foregroundColor(.white).font(.system(size: 18))
        }
        .padding(.horizontal, 20)
    }

    private var dimensionesRow: some View {
        HStack(spacing: 8) {
            MedicionField(label: "C (\(measureUnit.rawValue))", value: $cVal)
            MedicionField(label: "L (\(measureUnit.rawValue))", value: $lVal)
            MedicionField(label: "A (\(measureUnit.rawValue))", value: $aVal)
            Button(action: { camTrigger = .manual; showCamera = true }) {
                Image(systemName: "camera.fill")
                    .font(.system(size: 20)).foregroundColor(.white)
                    .frame(width: 46, height: 52)
                    .background(Color.latamCard).cornerRadius(8)
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.latamCardBorder, lineWidth: 1))
            }
        }
        .padding(.horizontal, 20)
    }

    private var pesosRow: some View {
        HStack(spacing: 8) {
            MedicionField(label: "VOLS.", value: $volsVal, keyboardType: .numberPad)
            MedicionField(label: "PESO UN.", value: $pesoUnitVal, keyboardType: .decimalPad)
            MedicionField(label: "PESO TOTAL", value: .constant(pesoTotalStr), enabled: false)
            Spacer().frame(width: 46)
        }
        .padding(.horizontal, 20)
    }

    private var camposListos: Bool {
        !cVal.isEmpty && !lVal.isEmpty && !aVal.isEmpty && !volsVal.isEmpty && !pesoUnitVal.isEmpty
    }

    private var agregarBtn: some View {
        Button(action: {
            // Si todos los campos están llenos → agregar directo, sin sheet
            if camposListos { agregarItem() } else { showSheet = true }
        }) {
            HStack(spacing: 8) {
                Image(systemName: "gearshape.fill")
                Text("AGREGAR").font(.system(size: 15, weight: .heavy))
            }
            .foregroundColor(.white)
            .frame(maxWidth: .infinity).frame(height: 50)
            .background(camposListos ? Color.green.opacity(0.8) : Color.latamCard)
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.latamCardBorder, lineWidth: 1))
            .cornerRadius(10)
        }
        .padding(.horizontal, 20)
    }

    // MARK: - Tabla de ítems
    private var tablaItems: some View {
        VStack(spacing: 0) {
            // Encabezado
            HStack {
                Text("#").frame(width: 22, alignment: .center)
                Text("VOLS").frame(maxWidth: .infinity)
                Text("PESO UN.").frame(maxWidth: .infinity)
                Text("PESO TOTAL").frame(maxWidth: .infinity)
                Text("C•L•A").frame(maxWidth: .infinity)
            }
            .font(.system(size: 10, weight: .bold))
            .foregroundColor(.white.opacity(0.55))
            .padding(.vertical, 8).padding(.horizontal, 10)

            Divider().background(Color.white.opacity(0.15))

            ForEach(Array(items.enumerated()), id: \.element.id) { idx, item in
                HStack {
                    Text("\(idx+1)").frame(width: 22, alignment: .center)
                    Text("\(item.vols)").frame(maxWidth: .infinity)
                    Text(String(format: "%.0f", item.pesoUnit)).frame(maxWidth: .infinity)
                    Text(String(format: "%.0f", item.pesoTotal)).frame(maxWidth: .infinity)
                    Text(item.cla).frame(maxWidth: .infinity)
                }
                .font(.system(size: 12)).foregroundColor(.white)
                .padding(.vertical, 6).padding(.horizontal, 10)
                Divider().background(Color.white.opacity(0.10))
            }

            // Totales
            HStack(spacing: 8) {
                totalCard(value: "\(totalVols)", label: "VOLÚMENES")
                totalCard(value: String(format: "%.0f kg", totalPesoReal), label: "PESO REAL")
                totalCard(value: String(format: "%.1f kg", totalPesoCubado), label: "PESO CUBADO")
            }
            .padding(.top, 10).padding(.bottom, 4)
        }
        .padding(12)
        .background(Color.latamCard)
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.latamCardBorder, lineWidth: 1))
        .cornerRadius(12)
        .padding(.horizontal, 20)
    }

    private func totalCard(value: String, label: String) -> some View {
        VStack(spacing: 2) {
            Text(value).font(.system(size: 18, weight: .heavy)).foregroundColor(.white)
            Text(label).font(.system(size: 9, weight: .medium)).foregroundColor(.white.opacity(0.55))
        }
        .frame(maxWidth: .infinity).padding(.vertical, 10)
        .background(Color.white.opacity(0.07)).cornerRadius(8)
    }

    // MARK: - Botones finales
    private var botonesFinales: some View {
        VStack(spacing: 10) {
            Button(action: finalizarMinuta) {
                HStack(spacing: 6) {
                    Image(systemName: "checkmark.square.fill")
                    Text("FINALIZAR").font(.system(size: 15, weight: .heavy))
                }
                .foregroundColor(.white).frame(maxWidth: .infinity).frame(height: 50)
                .background(Color.red).cornerRadius(10)
            }
            Button(action: { appState.path = [.minuta] }) {
                HStack(spacing: 6) {
                    Image(systemName: "arrow.counterclockwise")
                    Text("NUEVA MINUTA").font(.system(size: 14, weight: .medium))
                }
                .foregroundColor(.white).frame(maxWidth: .infinity).frame(height: 44)
                .background(Color.latamCard)
                .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.latamCardBorder, lineWidth: 1))
                .cornerRadius(10)
            }
        }
        .padding(.horizontal, 20).padding(.vertical, 14)
        .background(Color.latamBlue)
    }

    // MARK: - Sheet de tipo
    private var tipoSheet: some View {
        TipoCapturaSheet(
            onUnitario: {
                showSheet = false
                camTrigger = .unitario
                showCamera = true
            },
            onLote: {
                showSheet = false
                camTrigger = .lote
                showCamera = true
            },
            onAgregarManual: {
                agregarItem()
                showSheet = false
            }
        )
    }

    // MARK: - Helpers
    private var pesoTotalStr: String {
        let v = Double(volsVal) ?? 0
        let p = Double(pesoUnitVal) ?? 0
        return v > 0 && p > 0 ? String(format: "%.1f", v * p) : ""
    }

    private var totalVols: Int { items.reduce(0) { $0 + $1.vols } }
    private var totalPesoReal: Double { items.reduce(0) { $0 + $1.pesoTotal } }
    private var totalPesoCubado: Double { items.reduce(0) { $0 + $1.pesoCubado } }

    // Converts a field value (in measureUnit) to cm for internal storage.
    private func toCm(_ str: String) -> Double? {
        guard let val = Double(str) else { return nil }
        switch measureUnit {
        case .cm:     return val
        case .m:      return val * 100
        case .inches: return val * 2.54
        }
    }

    private func agregarItem() {
        guard let c = toCm(cVal), let l = toCm(lVal), let a = toCm(aVal),
              let v = Int(volsVal), let p = Double(pesoUnitVal),
              c > 0, l > 0, a > 0, v > 0 else { return }
        items.append(MedicionItem(c: c, l: l, a: a, vols: v, pesoUnit: p))
        cVal = ""; lVal = ""; aVal = ""; volsVal = ""; pesoUnitVal = ""
    }

    private func finalizarMinuta() {
        let record = MinutaRecord(numero: minuta, fecha: Date(), items: items)
        appState.minutas.append(record)
        let email = appState.userEmail
        Task {
            do {
                try await SheetsUploader.shared.appendRows(minuta: record, userEmail: email)
            } catch {
                // Los datos ya se guardaron localmente — el error de red no bloquea la app
                print("SheetsUploader: \(error.localizedDescription)")
            }
        }
        appState.path.removeAll()
    }
}

// MARK: - Campo de medición reutilizable
struct MedicionField: View {
    let label: String
    @Binding var value: String
    var keyboardType: UIKeyboardType = .decimalPad
    var enabled: Bool = true

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.system(size: 9, weight: .bold))
                .foregroundColor(.white.opacity(0.5))
            TextField("", text: $value)
                .keyboardType(keyboardType)
                .foregroundColor(.white)
                .disabled(!enabled)
                .padding(8)
                .background(enabled ? Color.latamCard : Color.white.opacity(0.05))
                .cornerRadius(8)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.latamCardBorder, lineWidth: 1))
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Sheet de tipo de captura
struct TipoCapturaSheet: View {
    let onUnitario: () -> Void
    let onLote: () -> Void
    let onAgregarManual: () -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ZStack {
            Color.latamBlue.ignoresSafeArea()
            VStack(spacing: 0) {
                Spacer()
                VStack(spacing: 10) {
                    Text("AGREGAR")
                        .font(.system(size: 13, weight: .heavy))
                        .foregroundColor(.white.opacity(0.6))
                        .padding(.bottom, 4)

                    sheetBtn("UNITARIO", action: onUnitario)
                    sheetBtn("LOTE", action: onLote)
                    Button(action: { dismiss() }) {
                        HStack(spacing: 6) {
                            Image(systemName: "camera")
                            Text("SOLO FOTO").font(.system(size: 15, weight: .semibold))
                        }
                        .foregroundColor(.white).frame(maxWidth: .infinity).frame(height: 52)
                        .background(Color.latamCard)
                        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.latamCardBorder, lineWidth: 1))
                        .cornerRadius(10)
                    }
                    .padding(.horizontal, 20)
                }
                .padding(.bottom, 30)
            }
        }
        .presentationDetents([.fraction(0.32)])
        .presentationDragIndicator(.visible)
    }

    private func sheetBtn(_ title: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title).font(.system(size: 15, weight: .semibold))
                .foregroundColor(.white).frame(maxWidth: .infinity).frame(height: 52)
                .background(Color.latamCard)
                .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.latamCardBorder, lineWidth: 1))
                .cornerRadius(10)
        }
        .padding(.horizontal, 20)
    }
}

// MARK: - Wrapper de la cámara AR
struct ARCameraWrapper: View {
    let onConfirm: (DetectionInfo, ARViewModel.MeasureUnit) -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ContentView(onConfirm: { det, unit in
            onConfirm(det, unit)
        })
        .ignoresSafeArea()
        .overlay(alignment: .topLeading) {
            Button(action: { dismiss() }) {
                HStack(spacing: 4) {
                    Image(systemName: "chevron.left")
                    Text("Volver")
                }
                .foregroundColor(.white)
                .padding(.horizontal, 16).padding(.vertical, 10)
                .background(.black.opacity(0.4)).cornerRadius(20)
                .padding(16)
            }
        }
    }
}
