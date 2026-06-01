import {
  requireNativeViewManager,
  NativeModule,
  requireOptionalNativeModule,
} from 'expo-modules-core';
import { ViewProps } from 'react-native';

export type ARMode = 'auto' | 'manual';

export interface MeasurementEvent {
  comprimento: number;
  largura: number;
  altura: number;
}

export interface ARBoxViewProps extends ViewProps {
  mode: ARMode;
  onMeasurementUpdate?: (event: { nativeEvent: MeasurementEvent }) => void;
  onMeasurementConfirmed?: (event: { nativeEvent: MeasurementEvent }) => void;
  onPlaneFound?: () => void;
}

const NativeView = requireNativeViewManager('LidarBoxMeasure');

export { NativeView as ARBoxView };

const NativeLidarModule: NativeModule | null =
  requireOptionalNativeModule('LidarBoxMeasure');

export function isLidarSupported(): boolean {
  return NativeLidarModule != null;
}
