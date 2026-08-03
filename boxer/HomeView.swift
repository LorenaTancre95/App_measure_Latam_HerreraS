import SwiftUI

struct HomeView: View {
    @EnvironmentObject var appState: AppState

    // Email persistido entre sesiones
    @AppStorage("userEmail") private var storedEmail = ""
    @State private var emailInput = ""
    @FocusState private var emailFocused: Bool

    private var isLoggedIn: Bool { !storedEmail.isEmpty }

    var body: some View {
        ZStack {
            Color.latamBlue.ignoresSafeArea()
            if isLoggedIn {
                homeContent
            } else {
                loginContent
            }
        }
        .navigationBarHidden(true)
        .onAppear {
            if isLoggedIn {
                appState.userEmail = storedEmail
                appState.userName  = nameFrom(storedEmail)
            }
        }
    }

    // MARK: - Pantalla de login

    private var loginContent: some View {
        VStack(spacing: 0) {
            Spacer()

            Image("LatamLogo")
                .resizable().scaledToFit().frame(height: 54)
                .padding(.bottom, 40)

            Text("CUBAJE AR")
                .font(.system(size: 22, weight: .heavy))
                .foregroundColor(.white)
                .padding(.bottom, 6)

            Text("Ingresá tu email LATAM para continuar")
                .font(.system(size: 14))
                .foregroundColor(.white.opacity(0.6))
                .padding(.bottom, 32)

            // Campo de email
            HStack {
                Image(systemName: "envelope")
                    .foregroundColor(.white.opacity(0.5))
                TextField("", text: $emailInput,
                          prompt: Text("nombre@latam.com").foregroundColor(.white.opacity(0.35)))
                    .foregroundColor(.white)
                    .keyboardType(.emailAddress)
                    .autocapitalization(.none)
                    .focused($emailFocused)
            }
            .padding(.horizontal, 16).padding(.vertical, 14)
            .background(Color.latamCard)
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.latamCardBorder, lineWidth: 1))
            .cornerRadius(12)
            .padding(.horizontal, 40)
            .padding(.bottom, 16)

            Button(action: login) {
                Text("CONTINUAR")
                    .font(.system(size: 16, weight: .heavy))
                    .foregroundColor(.black)
                    .frame(maxWidth: .infinity).frame(height: 52)
                    .background(emailInput.contains("@") ? Color.white : Color.white.opacity(0.3))
                    .cornerRadius(14)
            }
            .disabled(!emailInput.contains("@"))
            .padding(.horizontal, 40)

            Spacer()
        }
        .onAppear { emailFocused = true }
    }

    // MARK: - Pantalla principal

    private var homeContent: some View {
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
                Button(action: logout) {
                    Image(systemName: "power")
                        .foregroundColor(.white.opacity(0.7)).font(.system(size: 18))
                }
            }
            .padding(.horizontal, 20).padding(.top, 16)

            Spacer()

            Image("LatamLogo")
                .resizable().scaledToFit().frame(height: 54)
                .padding(.bottom, 52)

            VStack(spacing: 16) {
                NavigationLink(value: AppRoute.minuta) {
                    HomeCard(icon: "shippingbox.fill", title: "REGISTRAR CUBAJE")
                }.buttonStyle(.plain)

                NavigationLink(value: AppRoute.consultar) {
                    HomeCard(icon: "magnifyingglass", title: "CONSULTAR MINUTA")
                }.buttonStyle(.plain)
            }
            .padding(.horizontal, 20)

            Spacer()
        }
    }

    // MARK: - Helpers

    private func login() {
        let email = emailInput.trimmingCharacters(in: .whitespaces).lowercased()
        guard email.contains("@") else { return }
        storedEmail        = email
        appState.userEmail = email
        appState.userName  = nameFrom(email)
    }

    private func logout() {
        storedEmail        = ""
        appState.userEmail = ""
        appState.userName  = ""
        emailInput         = ""
    }

    // "silvinaherrera.acidlabs@latam.com" → "Silvinaherrera"
    private func nameFrom(_ email: String) -> String {
        let prefix = email.components(separatedBy: "@").first ?? email
        let name   = prefix.components(separatedBy: ".").first ?? prefix
        return name.prefix(1).uppercased() + name.dropFirst()
    }
}

struct HomeCard: View {
    let icon: String
    let title: String

    var body: some View {
        HStack(spacing: 20) {
            Image(systemName: icon)
                .font(.system(size: 38)).foregroundColor(.white).frame(width: 56, height: 56)
            Text(title)
                .font(.system(size: 17, weight: .heavy)).foregroundColor(.white).multilineTextAlignment(.leading)
            Spacer()
            Image(systemName: "chevron.right").foregroundColor(.white.opacity(0.4))
        }
        .padding(.horizontal, 24).padding(.vertical, 28)
        .background(Color.latamCard)
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(Color.latamCardBorder, lineWidth: 1))
        .cornerRadius(16)
    }
}
