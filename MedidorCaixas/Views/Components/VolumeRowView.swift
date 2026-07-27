import SwiftUI

struct VolumeRowView: View {
    let index: Int
    let volume: Volume

    var body: some View {
        HStack(spacing: 0) {
            Text("#\(index)")
                .frame(width: 36, alignment: .leading)
                .foregroundColor(AppTheme.textSecondary)
                .font(.caption)

            Text("\(volume.quantidade)")
                .frame(maxWidth: .infinity)
                .foregroundColor(AppTheme.textPrimary)
                .font(.system(size: 14, weight: .bold))

            Text(String(format: "%.0f", volume.pesoUnit))
                .frame(maxWidth: .infinity)
                .foregroundColor(AppTheme.textPrimary)
                .font(.system(size: 14, weight: .bold))

            Text(String(format: "%.0f", volume.pesoTotal))
                .frame(maxWidth: .infinity)
                .foregroundColor(AppTheme.textPrimary)
                .font(.system(size: 14, weight: .bold))

            Text(volume.dimensoesText)
                .frame(maxWidth: .infinity, alignment: .trailing)
                .foregroundColor(AppTheme.textSecondary)
                .font(.caption)
        }
        .padding(.vertical, 10)
        .padding(.horizontal, 4)

        Divider().background(AppTheme.fieldBorder)
    }
}

struct VolumeTableHeaderView: View {
    var body: some View {
        HStack(spacing: 0) {
            Text("#")
                .frame(width: 36, alignment: .leading)
            Text("VOLS")
                .frame(maxWidth: .infinity)
            Text("PESO UN.")
                .frame(maxWidth: .infinity)
            Text("PESO TOTAL")
                .frame(maxWidth: .infinity)
            Text("C×L×A")
                .frame(maxWidth: .infinity, alignment: .trailing)
        }
        .font(.caption2)
        .foregroundColor(AppTheme.textSecondary)
        .padding(.vertical, 6)
        .padding(.horizontal, 4)

        Divider().background(AppTheme.fieldBorder)
    }
}

struct TotalsCardView: View {
    let totalVolumes: Int
    let pesoReal: Double
    let pesoCubado: Double

    var body: some View {
        HStack {
            VStack(spacing: 2) {
                Text("\(totalVolumes)")
                    .font(.system(size: 28, weight: .bold))
                    .foregroundColor(AppTheme.textPrimary)
                Text("VOLUMES")
                    .font(.caption2)
                    .foregroundColor(AppTheme.yellow)
            }
            .frame(maxWidth: .infinity)

            Divider().background(AppTheme.fieldBorder).frame(height: 50)

            VStack(spacing: 2) {
                Text(String(format: "%.0f kg", pesoReal))
                    .font(.system(size: 22, weight: .bold))
                    .foregroundColor(AppTheme.textPrimary)
                Text("PESO REAL")
                    .font(.caption2)
                    .foregroundColor(AppTheme.yellow)
            }
            .frame(maxWidth: .infinity)

            Divider().background(AppTheme.fieldBorder).frame(height: 50)

            VStack(spacing: 2) {
                Text(String(format: "%.1f kg", pesoCubado))
                    .font(.system(size: 22, weight: .bold))
                    .foregroundColor(AppTheme.textPrimary)
                Text("PESO CUBADO")
                    .font(.caption2)
                    .foregroundColor(AppTheme.yellow)
            }
            .frame(maxWidth: .infinity)
        }
        .padding()
        .background(AppTheme.card)
        .cornerRadius(12)
    }
}
