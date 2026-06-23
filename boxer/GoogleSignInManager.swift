import Foundation
import Combine
import GoogleSignIn
import UIKit

@MainActor
final class GoogleSignInManager: ObservableObject {
    static let shared = GoogleSignInManager()

    @Published var isSignedIn: Bool = false
    @Published var userEmail: String? = nil

    private init() {
        refresh()
    }

    func refresh() {
        let user = GIDSignIn.sharedInstance.currentUser
        isSignedIn = user != nil
        userEmail = user?.profile?.email
    }

    private let driveScope = "https://www.googleapis.com/auth/drive.file"

    func signIn(presenting viewController: UIViewController) {
        GIDSignIn.sharedInstance.signIn(
            withPresenting: viewController,
            hint: nil,
            additionalScopes: [driveScope]
        ) { result, error in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if let error {
                    print("GoogleSignIn error: \(error.localizedDescription)")
                    return
                }
                self.isSignedIn = result?.user != nil
                self.userEmail = result?.user.profile?.email
            }
        }
    }

    func signOut() {
        GIDSignIn.sharedInstance.signOut()
        isSignedIn = false
        userEmail = nil
    }

    func restorePreviousSignIn() {
        GIDSignIn.sharedInstance.restorePreviousSignIn { [weak self] user, _ in
            guard let self else { return }
            Task { @MainActor in
                self.isSignedIn = user != nil
                self.userEmail = user?.profile?.email
            }
        }
    }
}
