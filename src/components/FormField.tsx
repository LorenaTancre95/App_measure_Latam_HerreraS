import React from 'react';
import { View, Text, TextInput, StyleSheet, KeyboardTypeOptions } from 'react-native';
import { Colors } from '../theme';

interface Props {
  label: string;
  value: string;
  onChangeText?: (t: string) => void;
  keyboardType?: KeyboardTypeOptions;
  editable?: boolean;
}

export function FormField({ label, value, onChangeText, keyboardType = 'default', editable = true }: Props) {
  return (
    <View style={styles.wrapper}>
      <Text style={styles.label}>{label}</Text>
      <TextInput
        style={[styles.input, !editable && styles.readonly]}
        value={value}
        onChangeText={onChangeText}
        keyboardType={keyboardType}
        editable={editable}
        placeholderTextColor={Colors.textSecondary}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: { flex: 1 },
  label: {
    fontSize: 10,
    color: Colors.textSecondary,
    marginBottom: 4,
    paddingHorizontal: 2,
  },
  input: {
    backgroundColor: Colors.fieldBG,
    borderWidth: 1,
    borderColor: Colors.fieldBorder,
    borderRadius: 8,
    color: Colors.textPrimary,
    fontSize: 16,
    fontWeight: '500',
    paddingHorizontal: 10,
    paddingVertical: 10,
  },
  readonly: {
    opacity: 0.7,
  },
});
