import { requireOptionalNativeModule } from 'expo-modules-core';

export interface MeasurementResult {
  comprimento: number;
  largura: number;
  altura: number;
}

// Returns null in Expo Go or on Android (no native module bundled).
const Mod = requireOptionalNativeModule('LidarBoxMeasure');

/** True when the native LiDAR module is available (real EAS build on iOS). */
export function isLidarSupported(): boolean {
  return Mod != null;
}

/**
 * Opens the full-screen native ARKit camera and returns the confirmed
 * box dimensions. Rejects with code "CANCELLED" if the user closes
 * without confirming.
 */
export async function openARCamera(): Promise<MeasurementResult> {
  if (!Mod) throw new Error('LiDAR module not available');
  return Mod.openARCamera() as Promise<MeasurementResult>;
}
