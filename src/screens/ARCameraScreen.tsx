import React, { useState, useCallback } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet,
  SafeAreaView, Animated, TextInput, ScrollView,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../App';
import { BoxMeasurement, volumeM3, pesoCubado } from '../models/BoxMeasurement';
import { Colors } from '../theme';

type Props = NativeStackScreenProps<RootStackParamList, 'ARCamera'>;
type ARMode = 'auto' | 'manual';

// Em Expo Go o módulo nativo não está disponível — usamos entrada manual.
// Quando compilado com EAS (build real), o LiDAR é ativado automaticamente.
let ARBoxView: React.ComponentType<any> | null = null;
try {
  ARBoxView = require('../../modules/lidar-box-measure').ARBoxView;
} catch {
  ARBoxView = null;
}

const lidarAvailable = ARBoxView !== null;

export function ARCameraScreen({ navigation, route }: Props) {
  const [mode, setMode] = useState<ARMode>('auto');
  const [measurement, setMeasurement] = useState<BoxMeasurement | null>(null);
  const [confirmed, setConfirmed] = useState(false);

  // Campos manuais
  const [manC, setManC] = useState('');
  const [manL, setManL] = useState('');
  const [manA, setManA] = useState('');

  const pulseAnim = React.useRef(new Animated.Value(1)).current;
  React.useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1.2, duration: 700, useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 1.0, duration: 700, useNativeDriver: true }),
      ])
    ).start();
  }, []);

  const handleMeasurementUpdate = useCallback((e: any) => {
    const { comprimento, largura, altura } = e.nativeEvent;
    setMeasurement({ comprimento, largura, altura });
  }, []);

  const handleConfirmManual = () => {
    const c = parseFloat(manC), l = parseFloat(manL), a = parseFloat(manA);
    if (!c || !l || !a) return;
    setMeasurement({ comprimento: c, largura: l, altura: a });
    setConfirmed(true);
  };

  const handleConfirmAR = () => {
    if (measurement) setConfirmed(true);
  };

  const handleUsar = () => {
    if (!measurement) return;
    route.params.onMeasurement(measurement);
    navigation.goBack();
  };

  const handleRemedir = () => {
    setConfirmed(false);
    setMeasurement(null);
    setManC(''); setManL(''); setManA('');
  };

  return (
    <View style={styles.container}>

      {/* ── LIDAR disponível (build EAS real) ── */}
      {lidarAvailable && ARBoxView && (
        <>
          <ARBoxView
            style={StyleSheet.absoluteFill}
            mode={mode}
            onMeasurementUpdate={handleMeasurementUpdate}
          />
          <ScanFrame color={confirmed ? Colors.green : Colors.yellow} />
          <View style={styles.crosshairWrapper} pointerEvents="none">
            <Animated.View style={[
              styles.crosshair,
              { borderColor: confirmed ? Colors.green : Colors.yellow },
              { transform: [{ scale: pulseAnim }] },
            ]} />
          </View>
        </>
      )}

      {/* ── Expo Go: fundo simulado ── */}
      {!lidarAvailable && (
        <View style={styles.simulatedCamera}>
          <Text style={styles.expoGoLabel}>
            📱 Expo Go — Modo Prévia
          </Text>
          <Text style={styles.expoGoSub}>
            LiDAR ativo no build final (EAS){'\n'}
            Insira as medidas manualmente para testar o fluxo
          </Text>
        </View>
      )}

      {/* Tabs AUTO / MANUAL */}
      <SafeAreaView style={styles.topBar}>
        <View style={styles.modePicker}>
          {(['auto', 'manual'] as ARMode[]).map(m => (
            <TouchableOpacity
              key={m}
              style={[styles.modeBtn, mode === m && styles.modeBtnSelected]}
              onPress={() => setMode(m)}
            >
              <Text style={[styles.modeBtnText, mode === m && styles.modeBtnTextSelected]}>
                {m.toUpperCase()}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </SafeAreaView>

      {/* Painel inferior */}
      <View style={styles.bottomPanel}>
        {confirmed && measurement ? (
          <ConfirmedPanel
            measurement={measurement}
            onRemedir={handleRemedir}
            onUsar={handleUsar}
          />
        ) : lidarAvailable ? (
          /* Painel AR real */
          <View>
            <View style={styles.statusBar}>
              <Text style={styles.statusText}>
                {measurement
                  ? `C: ${Math.round(measurement.comprimento)} × L: ${Math.round(measurement.largura)} × A: ${Math.round(measurement.altura)} cm`
                  : 'Aponte para a caixa'}
              </Text>
            </View>
            <View style={styles.calibBar}>
              <View style={styles.calibDot} />
              <Text style={styles.calibText}>Calibrando...</Text>
              {measurement && (
                <TouchableOpacity style={styles.confirmBtn} onPress={handleConfirmAR}>
                  <Text style={styles.confirmBtnText}>Confirmar</Text>
                </TouchableOpacity>
              )}
            </View>
          </View>
        ) : (
          /* Painel manual (Expo Go) */
          <ManualInputPanel
            c={manC} l={manL} a={manA}
            onC={setManC} onL={setManL} onA={setManA}
            onConfirm={handleConfirmManual}
          />
        )}
      </View>
    </View>
  );
}

// ─── Sub-components ────────────────────────────────────────────────────────────

function ManualInputPanel({ c, l, a, onC, onL, onA, onConfirm }: {
  c: string; l: string; a: string;
  onC: (v: string) => void;
  onL: (v: string) => void;
  onA: (v: string) => void;
  onConfirm: () => void;
}) {
  const valid = !!parseFloat(c) && !!parseFloat(l) && !!parseFloat(a);
  return (
    <View style={styles.manualPanel}>
      <Text style={styles.manualTitle}>Inserir medidas manualmente</Text>
      <View style={styles.manualRow}>
        <ManualField label="C (cm)" value={c} onChange={onC} />
        <ManualField label="L (cm)" value={l} onChange={onL} />
        <ManualField label="A (cm)" value={a} onChange={onA} />
      </View>
      <TouchableOpacity
        style={[styles.manualConfirmBtn, !valid && { opacity: 0.4 }]}
        onPress={onConfirm}
        disabled={!valid}
      >
        <Text style={styles.manualConfirmText}>✓  Confirmar medição</Text>
      </TouchableOpacity>
    </View>
  );
}

function ManualField({ label, value, onChange }: {
  label: string; value: string; onChange: (v: string) => void;
}) {
  return (
    <View style={{ flex: 1, marginHorizontal: 4 }}>
      <Text style={styles.manualFieldLabel}>{label}</Text>
      <TextInput
        style={styles.manualFieldInput}
        value={value}
        onChangeText={onChange}
        keyboardType="decimal-pad"
        placeholderTextColor={Colors.textSecondary}
        placeholder="0"
      />
    </View>
  );
}

function ConfirmedPanel({ measurement, onRemedir, onUsar }: {
  measurement: BoxMeasurement;
  onRemedir: () => void;
  onUsar: () => void;
}) {
  return (
    <View style={styles.confirmedPanel}>
      <Text style={styles.confirmedTitle}>✅  Medição confirmada</Text>
      <Text style={styles.confirmedSub}>
        {`Vol: ${volumeM3(measurement).toFixed(4)} m³  •  Peso Cubado: ${pesoCubado(measurement).toFixed(1)} kg`}
      </Text>
      <View style={styles.actionRow}>
        <TouchableOpacity style={styles.btnRemedir} onPress={onRemedir}>
          <Text style={styles.btnRmedirText}>↺  Remedir</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.btnUsar} onPress={onUsar}>
          <Text style={styles.btnUsarText}>✓  Usar</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

function ScanFrame({ color }: { color: string }) {
  const size = 28;
  return (
    <View style={[StyleSheet.absoluteFill, { margin: 40 }]} pointerEvents="none">
      {[
        { top: 0,    left: 0,    borderTopWidth: 3,    borderLeftWidth: 3  },
        { top: 0,    right: 0,   borderTopWidth: 3,    borderRightWidth: 3 },
        { bottom: 0, left: 0,    borderBottomWidth: 3, borderLeftWidth: 3  },
        { bottom: 0, right: 0,   borderBottomWidth: 3, borderRightWidth: 3 },
      ].map((corner, i) => (
        <View key={i} style={[{ position: 'absolute', width: size, height: size, borderColor: color }, corner]} />
      ))}
    </View>
  );
}

// ─── Styles ────────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  container:        { flex: 1, backgroundColor: '#000' },
  topBar:           { position: 'absolute', top: 0, left: 0, right: 0 },
  modePicker:       {
    flexDirection: 'row', backgroundColor: 'rgba(0,0,0,0.6)',
    borderRadius: 8, margin: 16, overflow: 'hidden',
  },
  modeBtn:          { flex: 1, paddingVertical: 10, alignItems: 'center', borderRadius: 8 },
  modeBtnSelected:  { backgroundColor: Colors.white },
  modeBtnText:      { color: Colors.white, fontSize: 13 },
  modeBtnTextSelected: { color: '#000', fontWeight: '700' },

  crosshairWrapper: { ...StyleSheet.absoluteFillObject, alignItems: 'center', justifyContent: 'center' },
  crosshair:        { width: 36, height: 36, borderRadius: 18, borderWidth: 2.5 },

  simulatedCamera: {
    flex: 1, backgroundColor: '#0a0a2a',
    alignItems: 'center', justifyContent: 'center', padding: 32,
  },
  expoGoLabel:    { color: Colors.yellow, fontSize: 18, fontWeight: '700', marginBottom: 12 },
  expoGoSub:      { color: Colors.textSecondary, fontSize: 13, textAlign: 'center', lineHeight: 20 },

  bottomPanel:    { position: 'absolute', bottom: 0, left: 0, right: 0 },

  statusBar:      { backgroundColor: 'rgba(0,0,0,0.55)', alignItems: 'center', paddingVertical: 12 },
  statusText:     { color: Colors.white, fontSize: 16, fontWeight: '600' },
  calibBar:       {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: 'rgba(255,204,0,0.15)',
    paddingHorizontal: 16, paddingVertical: 10,
  },
  calibDot:       { width: 10, height: 10, borderRadius: 5, backgroundColor: Colors.yellow, marginRight: 8 },
  calibText:      { color: Colors.white, fontSize: 12, flex: 1 },
  confirmBtn:     { backgroundColor: Colors.yellow, borderRadius: 20, paddingHorizontal: 16, paddingVertical: 6 },
  confirmBtnText: { color: '#000', fontWeight: '700', fontSize: 13 },

  manualPanel:    { backgroundColor: 'rgba(10,10,40,0.97)', padding: 16 },
  manualTitle:    { color: Colors.textSecondary, fontSize: 12, marginBottom: 10, textAlign: 'center' },
  manualRow:      { flexDirection: 'row', marginBottom: 12 },
  manualFieldLabel: { color: Colors.textSecondary, fontSize: 10, marginBottom: 4 },
  manualFieldInput: {
    backgroundColor: Colors.fieldBG, borderWidth: 1, borderColor: Colors.fieldBorder,
    borderRadius: 8, color: Colors.white, fontSize: 18, fontWeight: '600',
    paddingHorizontal: 10, paddingVertical: 10, textAlign: 'center',
  },
  manualConfirmBtn: {
    backgroundColor: Colors.green, borderRadius: 12,
    paddingVertical: 14, alignItems: 'center',
  },
  manualConfirmText: { color: '#000', fontWeight: '700', fontSize: 15 },

  confirmedPanel: { backgroundColor: 'rgba(0,0,0,0.85)', alignItems: 'center', paddingVertical: 12 },
  confirmedTitle: { color: Colors.green, fontSize: 16, fontWeight: '700' },
  confirmedSub:   { color: Colors.textSecondary, fontSize: 13, marginTop: 4 },
  actionRow:      { flexDirection: 'row', gap: 12, marginTop: 12, marginHorizontal: 16 },
  btnRemedir:     {
    flex: 1, padding: 14, borderRadius: 12,
    borderWidth: 1, borderColor: Colors.fieldBorder,
    alignItems: 'center', backgroundColor: Colors.card,
  },
  btnRmedirText:  { color: Colors.white, fontWeight: '600', fontSize: 15 },
  btnUsar:        { flex: 1, padding: 14, borderRadius: 12, backgroundColor: Colors.green, alignItems: 'center' },
  btnUsarText:    { color: '#000', fontWeight: '700', fontSize: 15 },
});
