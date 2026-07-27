import SwiftUI

struct MeditionFormView: View {
    @StateObject private var vm = MeditionFormViewModel()
    @State private var showAR = false

    var body: some View {
        NavigationStack {
            ZStack {
                AppTheme.background.ignoresSafeArea()

                ScrollView {
                    VStack(spacing: 16) {
                        minutaSection
                        dimensionsSection
                        adicionarButton
                        if !vm.volumes.isEmpty {
                            volumeListSection
                            totalsSection
                            actionButtons
                        }
                    }
                    .padding(16)
                }
            }
            .navigationTitle("REGISTRAR MEDIÇÃO")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarColorScheme(.dark, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("< Voltar") {}
                        .foregroundColor(AppTheme.textPrimary)
                }
            }
        }
        .sheet(isPresented: $showAR) {
            ARMeasurementView { measurement in
                vm.aplicarMedicao(measurement)
            }
        }
    }

    // MARK: - Número da minuta
    private var minutaSection: some View {
        HStack(spacing: 10) {
            LargeFormFieldView(
                label: "Número da minuta",
                text: $vm.minutaNumber,
                trailingIcon: "qrcode.viewfinder"
            )

            Button(action: {}) {
                Image(systemName: "pencil.and.scribble")
                    .font(.system(size: 20))
                    .foregroundColor(AppTheme.textPrimary)
                    .frame(width: 44, height: 56)
                    .background(AppTheme.card)
                    .cornerRadius(10)
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(AppTheme.fieldBorder, lineWidth: 1)
                    )
            }
            .padding(.top, 20)
        }
    }

    // MARK: - Campos de dimensão e peso
    private var dimensionsSection: some View {
        VStack(spacing: 12) {
            // C / L / A + câmera
            HStack(spacing: 8) {
                FormFieldView(label: "C (cm)", text: $vm.comprimento,
                              keyboardType: .decimalPad)
                FormFieldView(label: "L (cm)", text: $vm.largura,
                              keyboardType: .decimalPad)
                FormFieldView(label: "A (cm)", text: $vm.altura,
                              keyboardType: .decimalPad)

                Button(action: { showAR = true }) {
                    Image(systemName: "camera.fill")
                        .font(.system(size: 22))
                        .foregroundColor(AppTheme.textPrimary)
                        .frame(width: 44, height: 56)
                        .background(AppTheme.card)
                        .cornerRadius(10)
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .stroke(vm.comprimento.isEmpty ? AppTheme.fieldBorder : AppTheme.green,
                                        lineWidth: 1.5)
                        )
                }
                .padding(.top, 18)
            }

            // VOLS / PESO UNIT / PESO TOTAL
            HStack(spacing: 8) {
                FormFieldView(label: "VOLS.", text: $vm.quantidadeStr,
                              keyboardType: .numberPad)
                FormFieldView(label: "PESO UNIT.", text: $vm.pesoUnitStr,
                              keyboardType: .decimalPad)
                FormFieldView(label: "PESO TOTAL", text: .constant(vm.pesoTotal),
                              isReadOnly: true)
            }

            // Aviso de foto pré-registrada
            if vm.photoAttached {
                HStack(spacing: 6) {
                    Image(systemName: "photo")
                        .font(.caption)
                        .foregroundColor(AppTheme.textSecondary)
                    Text("1 foto pré-registrada — serão anexadas ao próximo ADICIONAR")
                        .font(.caption)
                        .foregroundColor(AppTheme.textSecondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    // MARK: - Botão ADICIONAR
    private var adicionarButton: some View {
        Button(action: vm.adicionar) {
            HStack(spacing: 10) {
                Image(systemName: "cube.fill")
                    .font(.system(size: 18))
                Text("ADICIONAR")
                    .font(.system(size: 16, weight: .bold))
                    .tracking(1)
            }
            .foregroundColor(AppTheme.textPrimary)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
            .background(
                LinearGradient(
                    colors: [AppTheme.accent, AppTheme.accent.opacity(0.7)],
                    startPoint: .leading, endPoint: .trailing
                )
            )
            .cornerRadius(12)
        }
        .opacity(canAdicionar ? 1.0 : 0.5)
        .disabled(!canAdicionar)
    }

    private var canAdicionar: Bool {
        !vm.comprimento.isEmpty && !vm.largura.isEmpty && !vm.altura.isEmpty &&
        !vm.quantidadeStr.isEmpty && !vm.pesoUnitStr.isEmpty
    }

    // MARK: - Lista de volumes adicionados
    private var volumeListSection: some View {
        VStack(spacing: 0) {
            VolumeTableHeaderView()
            ForEach(Array(vm.volumes.enumerated()), id: \.element.id) { index, vol in
                VolumeRowView(index: index + 1, volume: vol)
            }
        }
        .background(AppTheme.card)
        .cornerRadius(12)
    }

    // MARK: - Totais
    private var totalsSection: some View {
        VStack(spacing: 8) {
            TotalsCardView(
                totalVolumes: vm.totalVolumes,
                pesoReal: vm.totalPesoReal,
                pesoCubado: vm.totalPesoCubado
            )

            Text("Última medição: \(Date().formatted(date: .numeric, time: .shortened))")
                .font(.caption)
                .foregroundColor(AppTheme.textSecondary)
        }
    }

    // MARK: - Botões de ação finais
    private var actionButtons: some View {
        VStack(spacing: 12) {
            Button(action: vm.finalizar) {
                HStack {
                    if vm.isLoading {
                        ProgressView()
                            .tint(.white)
                            .scaleEffect(0.8)
                    } else {
                        Image(systemName: "checkmark.circle.fill")
                    }
                    Text(vm.isLoading ? "FINALIZANDO..." : "FINALIZAR")
                        .font(.system(size: 16, weight: .bold))
                        .tracking(1)
                }
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
                .background(AppTheme.pink)
                .cornerRadius(12)
            }
            .disabled(vm.isLoading)

            Button(action: vm.novaMinnuta) {
                HStack(spacing: 6) {
                    Image(systemName: "arrow.counterclockwise")
                    Text("NOVA MINUTA")
                        .font(.system(size: 14, weight: .semibold))
                }
                .foregroundColor(AppTheme.textSecondary)
            }
        }
    }
}

#Preview {
    MeditionFormView()
}
