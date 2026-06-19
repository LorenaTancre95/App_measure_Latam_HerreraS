import SwiftUI

struct HomeView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        ZStack {
            Color.latamBlue.ignoresSafeArea()

            VStack(spacing: 0) {
                // Barra superior
                HStack(alignment: .center, spacing: 12) {
                    ZStack {
                        Circle().fill(Color.red).frame(width: 42, height: 42)
                        Text(String(appState.userName.prefix(1)).uppercased())
                            .foregroundColor(.white).font(.headline.bold())
                    }
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Hola, \(appState.userName)")
                            .foregroundColor(.white).font(.subheadline.bold())
                        Text(appState.userEmail)
                            .foregroundColor(.white.opacity(0.55)).font(.caption)
                    }
                    Spacer()
                    Image(systemName: "moon.fill")
                        .foregroundColor(.yellow).font(.system(size: 18))
                    Image(systemName: "power")
                        .foregroundColor(.white).font(.system(size: 18))
                }
                .padding(.horizontal, 20)
                .padding(.top, 16)

                Spacer()

                // Logo LATAM
                Image("LatamLogo")
                    .resizable()
                    .scaledToFit()
                    .frame(height: 54)
                    .padding(.bottom, 52)

                // Tarjetas principales
                VStack(spacing: 16) {
                    NavigationLink(value: AppRoute.minuta) {
                        HomeCard(icon: "shippingbox.fill", title: "REGISTRAR CUBAJE")
                    }
                    .buttonStyle(.plain)

                    NavigationLink(value: AppRoute.consultar) {
                        HomeCard(icon: "magnifyingglass", title: "CONSULTAR MINUTA")
                    }
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 20)

                Spacer()
            }
        }
        .navigationBarHidden(true)
    }
}

struct HomeCard: View {
    let icon: String
    let title: String

    var body: some View {
        HStack(spacing: 20) {
            Image(systemName: icon)
                .font(.system(size: 38))
                .foregroundColor(.white)
                .frame(width: 56, height: 56)
            Text(title)
                .font(.system(size: 17, weight: .heavy))
                .foregroundColor(.white)
                .multilineTextAlignment(.leading)
            Spacer()
            Image(systemName: "chevron.right")
                .foregroundColor(.white.opacity(0.4))
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 28)
        .background(Color.latamCard)
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .stroke(Color.latamCardBorder, lineWidth: 1)
        )
        .cornerRadius(16)
    }
}
