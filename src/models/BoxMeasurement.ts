export interface BoxMeasurement {
  comprimento: number;  // C em cm
  largura: number;      // L em cm
  altura: number;       // A em cm
}

export function volumeM3(m: BoxMeasurement): number {
  return (m.comprimento * m.largura * m.altura) / 1_000_000;
}

export function pesoCubado(m: BoxMeasurement): number {
  return (m.comprimento * m.largura * m.altura) / 6_000;
}

export function displayText(m: BoxMeasurement): string {
  return `C: ${Math.round(m.comprimento)} × L: ${Math.round(m.largura)} × A: ${Math.round(m.altura)} cm`;
}
