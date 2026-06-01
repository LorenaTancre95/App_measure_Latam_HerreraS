import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { FormScreen } from './src/screens/FormScreen';
import { ARCameraScreen } from './src/screens/ARCameraScreen';
import { BoxMeasurement } from './src/models/BoxMeasurement';
import { StatusBar } from 'expo-status-bar';

export type RootStackParamList = {
  Form: undefined;
  ARCamera: { onMeasurement: (m: BoxMeasurement) => void };
};

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function App() {
  return (
    <NavigationContainer>
      <StatusBar style="light" />
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        <Stack.Screen name="Form"     component={FormScreen} />
        <Stack.Screen name="ARCamera" component={ARCameraScreen}
          options={{ presentation: 'fullScreenModal' }} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
