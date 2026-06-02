import React, { useState, useCallback } from 'react';
import {
  View, Text, ScrollView, TouchableOpacity,
  StyleSheet, SafeAreaView, ActivityIndicator, Image, Modal,
} from 'react-native';
import { FormField } from '../components/FormField';
import { VolumeTableHeader, VolumeRow, TotalsCard } from '../components/VolumeRow';
import { BoxMeasurement } from '../models/BoxMeasurement';
import { Volume, pesoTotal, pesoCubadoTotal } from '../models/Volume';
import { Colors, Spacing, Radius } from '../theme';
import { ARCameraScreen } from './ARCameraScreen';
import { isLidarSupported, openARCamera } from '../../modules/lidar-box-measure';

export function FormScreen() {
  const [minuta, setMinuta]       = useState('');
  const [C, setC]                 = useState('');
  const [L, setL]                 = useState('');
  const [A, setA]                 = useState('');
  const [vols, setVols]           = useState('');
  const [pesoUnit, setPesoUnit]   = useState('');
  const [volumes, setVolumes]     = useState<Volume[]>([]);
  const [photoAttached, setPhoto] = useState(false);
  const [finalizing, setFinal]    = useState(false);
  const [cameraOpen, setCameraOpen] = useState(false);

  const pesoTotalCalc = React.useMemo(() => {
    const q = parseFloat(vols), p = parseFloat(pesoUnit);
    return isNaN(q) || isNaN(p) ? '' : (q * p).toFixed(3);
  }, [vols, pesoUnit]);

  const totalVolumes   = volumes.reduce((s, v) => s + v.quantidade, 0);
  const totalPesoReal  = volumes.reduce((s, v) => s + pesoTotal(v), 0);
  const totalPesoCub   = volumes.reduce((s, v) => s + pesoCubadoTotal(v), 0);

  const canAdicionar = C && L && A && vols && pesoUnit;

  const onMeasurement = useCallback((m: BoxMeasurement) => {
    setC(String(Math.round(m.comprimento)));
    setL(String(Math.round(m.largura)));
    setA(String(Math.round(m.altura)));
    setPhoto(true);
  }, []);

  const adicionar = () => {
    const vol: Volume = {
      id: Date.now().toString(),
      comprimento: parseFloat(C),
      largura:     parseFloat(L),
      altura:      parseFloat(A),
      quantidade:  parseInt(vols),
      pesoUnit:    parseFloat(pesoUnit),
    };
    setVolumes(prev => [...prev, vol]);
    setC(''); setL(''); setA(''); setVols(''); setPesoUnit('');
    setPhoto(false);
  };

  const finalizar = () => {
    setFinal(true);
    setTimeout(() => setFinal(false), 2000); // substituir por chamada API
  };

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.header}>
        <Image
          source={require('../../assets/logo_latam.png')}
          style={styles.logo}
          resizeMode="contain"
        />
        <Text style={styles.headerTitle}>REGISTRAR MEDIÇÃO</Text>
      </View>

      <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">

        {/* Número da minuta */}
        <View style={styles.row}>
          <View style={{ flex: 1 }}>
            <FormField label="Número da minuta" value={minuta} onChangeText={setMinuta} />
          </View>
          <TouchableOpacity style={styles.iconBtn}>
            <Text style={styles.iconBtnText}>✏️</Text>
          </TouchableOpacity>
        </View>

        {/* C / L / A + câmera */}
        <View style={[styles.row, { marginTop: Spacing.md }]}>
          <FormField label="C (cm)" value={C} onChangeText={setC} keyboardType="decimal-pad" />
          <View style={{ width: Spacing.sm }} />
          <FormField label="L (cm)" value={L} onChangeText={setL} keyboardType="decimal-pad" />
          <View style={{ width: Spacing.sm }} />
          <FormField label="A (cm)" value={A} onChangeText={setA} keyboardType="decimal-pad" />
          <View style={{ width: Spacing.sm }} />
          <TouchableOpacity
            style={[styles.iconBtn, { borderColor: photoAttached ? Colors.green : Colors.fieldBorder }]}
            onPress={async () => {
              if (isLidarSupported()) {
                try {
                  const m = await openARCamera();
                  onMeasurement(m);
                } catch {
                  // user cancelled — do nothing
                }
              } else {
                setCameraOpen(true);
              }
            }}
          >
            <Text style={styles.iconBtnText}>📷</Text>
          </TouchableOpacity>
        </View>

        {/* VOLS / PESO UNIT / PESO TOTAL */}
        <View style={[styles.row, { marginTop: Spacing.sm }]}>
          <FormField label="VOLS."      value={vols}     onChangeText={setVols}     keyboardType="number-pad" />
          <View style={{ width: Spacing.sm }} />
          <FormField label="PESO UNIT." value={pesoUnit} onChangeText={setPesoUnit} keyboardType="decimal-pad" />
          <View style={{ width: Spacing.sm }} />
          <FormField label="PESO TOTAL" value={pesoTotalCalc} editable={false} />
        </View>

        {photoAttached && (
          <Text style={styles.photoNote}>
            📷 1 foto pré-registrada — serão anexadas ao próximo ADICIONAR
          </Text>
        )}

        {/* ADICIONAR */}
        <TouchableOpacity
          style={[styles.adicionarBtn, !canAdicionar && styles.disabled]}
          onPress={adicionar}
          disabled={!canAdicionar}
        >
          <Text style={styles.adicionarText}>⬡  ADICIONAR</Text>
        </TouchableOpacity>

        {/* Lista de volumes */}
        {volumes.length > 0 && (
          <View style={styles.card}>
            <VolumeTableHeader />
            {volumes.map((v, i) => <VolumeRow key={v.id} index={i + 1} volume={v} />)}
          </View>
        )}

        {/* Totais */}
        {volumes.length > 0 && (
          <>
            <TotalsCard
              totalVolumes={totalVolumes}
              pesoReal={totalPesoReal}
              pesoCubado={totalPesoCub}
            />
            <Text style={styles.lastMeasure}>
              Última medição: {new Date().toLocaleString('pt-BR')}
            </Text>

            {/* FINALIZAR */}
            <TouchableOpacity style={styles.finalizarBtn} onPress={finalizar} disabled={finalizing}>
              {finalizing
                ? <ActivityIndicator color="#fff" />
                : <Text style={styles.finalizarText}>FINALIZAR</Text>
              }
            </TouchableOpacity>

            <TouchableOpacity onPress={() => setVolumes([])}>
              <Text style={styles.novaMinuta}>↺  NOVA MINUTA</Text>
            </TouchableOpacity>
          </>
        )}

      </ScrollView>

      <Modal
        visible={cameraOpen}
        animationType="slide"
        onRequestClose={() => setCameraOpen(false)}
      >
        <ARCameraScreen
          onMeasurement={(m) => { onMeasurement(m); setCameraOpen(false); }}
          onClose={() => setCameraOpen(false)}
        />
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe:        { flex: 1, backgroundColor: Colors.background },
  header:      {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: Spacing.lg, paddingVertical: 10,
    backgroundColor: Colors.background,
    borderBottomWidth: 2,
    borderBottomColor: Colors.accent,
  },
  logo:        { width: 120, height: 40 },
  headerTitle: { color: Colors.white, fontSize: 11, fontWeight: '700', letterSpacing: 1 },
  scroll:      { padding: Spacing.lg, paddingBottom: 40 },

  row:         { flexDirection: 'row', alignItems: 'flex-end' },
  iconBtn:     {
    width: 48, height: 48, borderRadius: Radius.sm,
    borderWidth: 1.5, borderColor: Colors.fieldBorder,
    backgroundColor: Colors.card,
    alignItems: 'center', justifyContent: 'center',
    marginBottom: 1, marginLeft: Spacing.sm,
  },
  iconBtnText: { fontSize: 20 },

  photoNote:   { color: Colors.textSecondary, fontSize: 11, marginTop: 6 },

  adicionarBtn: {
    marginTop: Spacing.lg,
    backgroundColor: Colors.accent,
    borderRadius: Radius.md,
    paddingVertical: 16,
    alignItems: 'center',
    shadowColor: Colors.accent,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 8,
    elevation: 6,
  },
  disabled:    { opacity: 0.4 },
  adicionarText: { color: Colors.white, fontSize: 16, fontWeight: '700', letterSpacing: 1 },

  card:        { backgroundColor: Colors.card, borderRadius: Radius.md, marginTop: Spacing.lg },

  lastMeasure: { color: Colors.textSecondary, fontSize: 11, textAlign: 'center', marginVertical: 8 },

  finalizarBtn: {
    marginTop: Spacing.sm,
    backgroundColor: Colors.pink,
    borderRadius: Radius.md,
    paddingVertical: 16,
    alignItems: 'center',
  },
  finalizarText: { color: Colors.white, fontSize: 16, fontWeight: '700', letterSpacing: 1 },

  novaMinuta:  { color: Colors.textSecondary, textAlign: 'center', fontSize: 13, marginTop: 12 },
});
