import SwiftUI

struct ConsultarView: View {
    @EnvironmentObject var appState: AppState

    private static let dateFmt: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .short
        f.timeStyle = .short
        return f
    }()

    var body: some View {
        ZStack {
            Color.latamBlue.ignoresSafeArea()

            VStack(spacing: 0) {
                Text("CONSULTAR MINUTAS")
                    .font(.system(size: 14, weight: .heavy))
                    .foregroundColor(.white)
                    .padding(.top, 16)
                    .padding(.bottom, 14)

                if appState.minutas.isEmpty {
                    Spacer()
                    VStack(spacing: 12) {
                        Image(systemName: "tray")
                            .font(.system(size: 48))
                            .foregroundColor(.white.opacity(0.25))
                        Text("No hay minutas registradas")
                            .font(.system(size: 15))
                            .foregroundColor(.white.opacity(0.45))
                    }
                    Spacer()
                } else {
                    ScrollView {
                        VStack(spacing: 12) {
                            ForEach(appState.minutas.reversed()) { record in
                                MinutaCard(record: record)
                            }
                        }
                        .padding(.horizontal, 20)
                        .padding(.bottom, 24)
                    }
                }
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
    }
}

// MARK: - Tarjeta de minuta
private struct MinutaCard: View {
    let record: MinutaRecord

    private static let dateFmt: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "dd/MM/yyyy HH:mm"
        return f
    }()

    @State private var expanded = false

    var body: some View {
        VStack(spacing: 0) {
            // Encabezado colapsable
            Button(action: { withAnimation(.easeInOut(duration: 0.2)) { expanded.toggle() } }) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Minuta \(record.numero)")
                            .font(.system(size: 15, weight: .heavy))
                            .foregroundColor(.white)
                        Text(Self.dateFmt.string(from: record.fecha))
                            .font(.system(size: 11))
                            .foregroundColor(.white.opacity(0.5))
                    }
                    Spacer()
                    HStack(spacing: 16) {
                        VStack(spacing: 1) {
                            Text("\(record.totalVols)")
                                .font(.system(size: 16, weight: .heavy)).foregroundColor(.white)
                            Text("VOLS")
                                .font(.system(size: 9)).foregroundColor(.white.opacity(0.5))
                        }
                        VStack(spacing: 1) {
                            Text(String(format: "%.0f kg", record.totalPesoReal))
                                .font(.system(size: 14, weight: .bold)).foregroundColor(.white)
                            Text("REAL")
                                .font(.system(size: 9)).foregroundColor(.white.opacity(0.5))
                        }
                        Image(systemName: expanded ? "chevron.up" : "chevron.down")
                            .foregroundColor(.white.opacity(0.4))
                            .font(.system(size: 12))
                    }
                }
                .padding(.horizontal, 14).padding(.vertical, 12)
            }
            .buttonStyle(.plain)

            // Detalle de ítems (expandible)
            if expanded {
                Divider().background(Color.white.opacity(0.15))

                // Cabecera de tabla
                HStack {
                    Text("#").frame(width: 22, alignment: .center)
                    Text("VOLS").frame(maxWidth: .infinity)
                    Text("PESO UN.").frame(maxWidth: .infinity)
                    Text("PESO TOT.").frame(maxWidth: .infinity)
                    Text("C·L·A").frame(maxWidth: .infinity)
                }
                .font(.system(size: 9, weight: .bold))
                .foregroundColor(.white.opacity(0.5))
                .padding(.horizontal, 10).padding(.vertical, 6)

                ForEach(Array(record.items.enumerated()), id: \.element.id) { idx, item in
                    Divider().background(Color.white.opacity(0.08))
                    HStack {
                        Text("\(idx + 1)").frame(width: 22, alignment: .center)
                        Text("\(item.vols)").frame(maxWidth: .infinity)
                        Text(String(format: "%.0f", item.pesoUnit)).frame(maxWidth: .infinity)
                        Text(String(format: "%.0f", item.pesoTotal)).frame(maxWidth: .infinity)
                        Text(item.cla).frame(maxWidth: .infinity)
                    }
                    .font(.system(size: 12)).foregroundColor(.white)
                    .padding(.horizontal, 10).padding(.vertical, 5)
                }

                // Totales
                Divider().background(Color.white.opacity(0.15))
                HStack(spacing: 8) {
                    summaryChip(value: "\(record.totalVols)", label: "VOLS")
                    summaryChip(value: String(format: "%.0f kg", record.totalPesoReal), label: "REAL")
                    summaryChip(value: String(format: "%.1f kg", record.totalPesoCubado), label: "CUBADO")
                }
                .padding(.horizontal, 10).padding(.vertical, 8)
            }
        }
        .background(Color.latamCard)
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.latamCardBorder, lineWidth: 1))
        .cornerRadius(12)
    }

    private func summaryChip(value: String, label: String) -> some View {
        VStack(spacing: 2) {
            Text(value).font(.system(size: 14, weight: .heavy)).foregroundColor(.white)
            Text(label).font(.system(size: 9)).foregroundColor(.white.opacity(0.5))
        }
        .frame(maxWidth: .infinity).padding(.vertical, 6)
        .background(Color.white.opacity(0.07)).cornerRadius(6)
    }
}
