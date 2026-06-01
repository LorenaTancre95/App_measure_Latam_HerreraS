export interface Volume {
  id: string;
  comprimento: number;
  largura: number;
  altura: number;
  quantidade: number;
  pesoUnit: number;
}

export function pesoTotal(v: Volume): number {
  return v.quantidade * v.pesoUnit;
}

export function pesoCubadoTotal(v: Volume): number {
  return (v.comprimento * v.largura * v.altura / 6_000) * v.quantidade;
}

export function dimensoesText(v: Volume): string {
  return `${Math.round(v.comprimento)}×${Math.round(v.largura)}×${Math.round(v.altura)}`;
}
