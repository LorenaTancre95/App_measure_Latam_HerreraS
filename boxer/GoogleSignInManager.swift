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

    let driveScope  = "https://www.googleapis.com/auth/drive.file"
    let sheetsScope = "https://www.googleapis.com/auth/spreadsheets"

    private var allScopes: [String] { [driveScope, sheetsScope] }

    var hasDriveScope: Bool {
        GIDSignIn.sharedInstance.currentUser?.grantedScopes?.contains(driveScope) == true
    }

    func signIn(presenting viewController: UIViewController) {
        GIDSignIn.sharedInstance.signIn(
            withPresenting: viewController,
            hint: nil,
            additionalScopes: allScopes
        ) { result, error in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if let error { print("GoogleSignIn error: \(error.localizedDescription)"); return }
                self.isSignedIn = result?.user != nil
                self.userEmail = result?.user.profile?.email
            }
        }
    }

    func grantDriveScope(presenting viewController: UIViewController) {
        guard let user = GIDSignIn.sharedInstance.currentUser else { signIn(presenting: viewController); return }
        user.addScopes(allScopes, presenting: viewController) { result, error in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if let error { print("addScopes error: \(error.localizedDescription)"); return }
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
