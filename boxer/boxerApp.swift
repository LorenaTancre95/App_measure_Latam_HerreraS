import SwiftUI

@main
struct boxerApp: App {
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            NavigationStack(path: $appState.path) {
                HomeView()
                    .navigationDestination(for: AppRoute.self) { route in
                        switch route {
                        case .minuta:
                            MinutaView()
                        case .medicion(let numero):
                            MedicionView(minuta: numero)
                        case .consultar:
                            ConsultarView()
                        }
                    }
            }
            .environmentObject(appState)
            .preferredColorScheme(.dark)
        }
    }
}
