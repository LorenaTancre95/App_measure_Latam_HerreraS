import SwiftUI
import GoogleSignIn

struct HomeView: View {
    @EnvironmentObject var appState: AppState
    @ObservedObject private var signInMgr = GoogleSignInManager.shared

    var body: some View {
        ZStack {
            Color.latamBlue.ignoresSafeArea()

            if signInMgr.isSignedIn {
                homeContent
                    .transition(.opacity)
            } else {
                loginContent
                    .transition(.opacity)
            }
        }
        .navigationBarHidden(true)
        .onChange(of: signInMgr.isSignedIn) { _, signedIn in
            if signedIn {
                syncUserFromGoogle()
            }
        }
        .onAppear {
            if signInMgr.isSignedIn { syncUserFromGoogle() }
        }
    }

    // MARK: - Login screen

    private var loginContent: some View {
        VStack(spacing: 0) {
            Spacer()

            // Logo LATAM
            Image("LatamLogo")
                .resizable()
                .scaledToFit()
                .frame(height: 54)
                .padding(.bottom, 48)

            Text("CUBAJE AR")
                .font(.system(size: 22, weight: .heavy))
                .foregroundColor(.white)
                .padding(.bottom, 8)

            Text("Iniciá sesión con tu cuenta LATAM\npara guardar las mediciones")
                .font(.system(size: 14))
                .foregroundColor(.white.opacity(0.6))
                .multilineTextAlignment(.center)
                .padding(.bottom, 40)

            Button(action: googleSignIn) {
                HStack(spacing: 12) {
                    Image(systemName: "person.crop.circle.badge.checkmark")
                        .font(.system(size: 20))
                    Text("Iniciar sesión con Google")
                        .font(.system(size: 16, weight: .bold))
                }
                .foregroundColor(.black)
                .frame(maxWidth: .infinity)
                .frame(height: 54)
                .background(Color.white)
                .cornerRadius(14)
            }
            .padding(.horizontal, 40)

            Spacer()
        }
    }

    // MARK: - Home screen (logueado)

    private var homeContent: some View {
        VStack(spacing: 0) {
            // Barra superior con usuario
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
                // Cerrar sesión
                Button(action: signOut) {
                    Image(systemName: "power")
                        .foregroundColor(.white.opacity(0.7)).font(.system(size: 18))
                }
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

    // MARK: - Helpers

    private func googleSignIn() {
        guard let vc = UIApplication.shared.connectedScenes
            .compactMap({ $0 as? UIWindowScene })
            .first?.windows.first?.rootViewController else { return }
        signInMgr.signIn(presenting: vc)
    }

    private func signOut() {
        signInMgr.signOut()
        appState.userName = ""
        appState.userEmail = ""
    }

    private func syncUserFromGoogle() {
        let user = GIDSignIn.sharedInstance.currentUser
        appState.userName  = user?.profile?.name  ?? "Usuario"
        appState.userEmail = user?.profile?.email ?? ""
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
