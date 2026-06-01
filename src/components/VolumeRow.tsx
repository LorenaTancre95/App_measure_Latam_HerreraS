import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Volume, pesoTotal, dimensoesText } from '../models/Volume';
import { Colors } from '../theme';

export function VolumeTableHeader() {
  return (
    <View style={styles.row}>
      <Text style={[styles.cell, styles.header, { width: 32 }]}>#</Text>
      <Text style={[styles.cell, styles.header]}>VOLS</Text>
      <Text style={[styles.cell, styles.header]}>PESO UN.</Text>
      <Text style={[styles.cell, styles.header]}>PESO TOT.</Text>
      <Text style={[styles.cell, styles.header, { textAlign: 'right' }]}>C×L×A</Text>
    </View>
  );
}

export function VolumeRow({ index, volume }: { index: number; volume: Volume }) {
  return (
    <>
      <View style={styles.row}>
        <Text style={[styles.cell, styles.indexCell]}>#{index}</Text>
        <Text style={[styles.cell, styles.valueCell]}>{volume.quantidade}</Text>
        <Text style={[styles.cell, styles.valueCell]}>{volume.pesoUnit.toFixed(0)}</Text>
        <Text style={[styles.cell, styles.valueCell]}>{pesoTotal(volume).toFixed(0)}</Text>
        <Text style={[styles.cell, styles.dimCell]}>{dimensoesText(volume)}</Text>
      </View>
      <View style={styles.divider} />
    </>
  );
}

export function TotalsCard({
  totalVolumes,
  pesoReal,
  pesoCubado,
}: {
  totalVolumes: number;
  pesoReal: number;
  pesoCubado: number;
}) {
  return (
    <View style={styles.totalsCard}>
      <TotalItem value={String(totalVolumes)} label="VOLUMES" />
      <View style={styles.vDivider} />
      <TotalItem value={`${pesoReal.toFixed(0)} kg`} label="PESO REAL" />
      <View style={styles.vDivider} />
      <TotalItem value={`${pesoCubado.toFixed(1)} kg`} label="PESO CUBADO" />
    </View>
  );
}

function TotalItem({ value, label }: { value: string; label: string }) {
  return (
    <View style={styles.totalItem}>
      <Text style={styles.totalValue}>{value}</Text>
      <Text style={styles.totalLabel}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    paddingHorizontal: 4,
  },
  cell: { flex: 1, color: Colors.textPrimary, fontSize: 13 },
  header: { fontSize: 10, color: Colors.textSecondary },
  indexCell: { width: 32, flex: 0, color: Colors.textSecondary, fontSize: 12 },
  valueCell: { fontWeight: '700' },
  dimCell: { textAlign: 'right', color: Colors.textSecondary, fontSize: 12 },
  divider: { height: 1, backgroundColor: Colors.fieldBorder, marginHorizontal: 4 },
  totalsCard: {
    flexDirection: 'row',
    backgroundColor: Colors.card,
    borderRadius: 12,
    padding: 16,
  },
  totalItem: { flex: 1, alignItems: 'center' },
  totalValue: { fontSize: 22, fontWeight: '700', color: Colors.textPrimary },
  totalLabel: { fontSize: 10, color: Colors.yellow, marginTop: 2 },
  vDivider: { width: 1, backgroundColor: Colors.fieldBorder, marginHorizontal: 4 },
});
