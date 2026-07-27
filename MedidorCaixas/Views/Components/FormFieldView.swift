import SwiftUI

struct FormFieldView: View {
    let label: String
    @Binding var text: String
    var keyboardType: UIKeyboardType = .default
    var isReadOnly: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.caption2)
                .foregroundColor(AppTheme.textSecondary)
                .padding(.horizontal, 8)

            TextField("", text: $text)
                .keyboardType(keyboardType)
                .disabled(isReadOnly)
                .foregroundColor(AppTheme.textPrimary)
                .font(.system(size: 16, weight: .medium))
                .padding(.horizontal, 10)
                .padding(.vertical, 10)
                .background(AppTheme.fieldBG)
                .cornerRadius(8)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(AppTheme.fieldBorder, lineWidth: 1)
                )
        }
    }
}

struct LargeFormFieldView: View {
    let label: String
    @Binding var text: String
    var trailingIcon: String? = nil
    var onIconTap: (() -> Void)? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.caption)
                .foregroundColor(AppTheme.textSecondary)
                .padding(.horizontal, 4)

            HStack {
                TextField("", text: $text)
                    .foregroundColor(AppTheme.textPrimary)
                    .font(.system(size: 17, weight: .medium))

                if let icon = trailingIcon {
                    Button(action: { onIconTap?() }) {
                        Image(systemName: icon)
                            .foregroundColor(AppTheme.textSecondary)
                    }
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 14)
            .background(AppTheme.fieldBG)
            .cornerRadius(10)
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .stroke(AppTheme.fieldBorder, lineWidth: 1)
            )
        }
    }
}
