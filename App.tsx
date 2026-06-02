import React from 'react';
import { FormScreen } from './src/screens/FormScreen';
import { StatusBar } from 'expo-status-bar';

export type RootStackParamList = {
  Form: undefined;
  ARCamera: { onMeasurement: (m: any) => void };
};

export default function App() {
  return (
    <>
      <StatusBar style="light" />
      <FormScreen />
    </>
  );
}
