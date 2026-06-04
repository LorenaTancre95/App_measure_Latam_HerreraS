#!/usr/bin/env python3
"""
Test script para model_core.mlmodel (YOLO26n-seg, 1 clase: 'package')

Instalar:  pip install coremltools pillow numpy
Ejecutar:
  python test_model.py                      # imagen sintetica
  python test_model.py /ruta/foto.jpg       # foto real (solo macOS)

- En Windows/Linux: muestra spec del modelo (shapes, metadata, nombres de outputs)
- En macOS:         ademas hace inferencia real y muestra detecciones
"""

import sys
import platform

# ── Dependencias ─────────────────────────────────────────────────────────────
try:
    import coremltools as ct
except ImportError:
    sys.exit("ERROR: pip install coremltools pillow numpy")

try:
    import numpy as np
except ImportError:
    sys.exit("ERROR: pip install numpy")

IS_MACOS = platform.system() == "Darwin"

MODEL_PATH = "modules/lidar-box-measure/ios/model_core.mlmodel"
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else None


# ══════════════════════════════════════════════════════════════════════════════
# 1. Cargar modelo y mostrar spec (funciona en Windows + macOS)
# ══════════════════════════════════════════════════════════════════════════════

print(f"Cargando modelo: {MODEL_PATH}")
model = ct.models.MLModel(MODEL_PATH)
spec  = model.get_spec()
desc  = spec.description

print("\n" + "="*55)
print("INPUTS")
print("="*55)
for inp in desc.input:
    t = inp.type
    if t.HasField("imageType"):
        it = t.imageType
        cs = {0: "GRAYSCALE", 10: "BGR", 65552: "RGB", 3: "GRAYSCALE_FLOAT16"}.get(it.colorSpace, str(it.colorSpace))
        print(f"  '{inp.name}'  image {it.width}x{it.height}  colorspace={cs}")
    elif t.HasField("multiArrayType"):
        mt = t.multiArrayType
        print(f"  '{inp.name}'  MultiArray {list(mt.shape)}")
    else:
        print(f"  '{inp.name}'  {t}")

print("\n" + "="*55)
print("OUTPUTS")
print("="*55)
for out in desc.output:
    t = out.type
    if t.HasField("multiArrayType"):
        mt = t.multiArrayType
        dtype = {1:"FLOAT32", 2:"DOUBLE", 3:"INT32", 5:"FLOAT16"}.get(mt.dataType, str(mt.dataType))
        print(f"  '{out.name}'  MultiArray shape={list(mt.shape)}  dtype={dtype}")
    else:
        print(f"  '{out.name}'  {t}")

print("\n" + "="*55)
print("METADATA")
print("="*55)
ud = dict(desc.metadata.userDefined)
for k in ["names","imgsz","task","end2end","nms","batch","stride","channels"]:
    if k in ud:
        print(f"  {k}: {ud[k]}")

print("\n" + "="*55)
print("NOTA PARA SWIFT (BoxDetectionCoordinator)")
print("="*55)
for out in desc.output:
    t = out.type
    if t.HasField("multiArrayType"):
        mt = t.multiArrayType
        shape = list(mt.shape)
        C = shape[-1] if shape else 0
        if C >= 5:
            print(f"  '{out.name}' → array de detecciones")
            print(f"    shape: {shape}")
            print(f"    columnas: x1 y1 x2 y2 conf class [mask_coefs…]")
            print(f"    en Swift: arr[base+0..4] = x1,y1,x2,y2,conf")
        else:
            print(f"  '{out.name}' → prototipos de mascara (ignorar en Swift)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Inferencia (solo macOS)
# ══════════════════════════════════════════════════════════════════════════════

if not IS_MACOS:
    print(f"\n{'='*55}")
    print(f"INFERENCIA: omitida (solo macOS — sistema actual: {platform.system()})")
    print(f"Para probar inferencia, ejecuta este script en un Mac.")
    print("="*55)
    sys.exit(0)

try:
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("ERROR: pip install pillow")


def make_synthetic_box(w=640, h=640):
    """Imagen sintetica: fondo claro + rectangulo marron (simula caja de carton)."""
    img = Image.new("RGB", (w, h), (245, 240, 230))
    draw = ImageDraw.Draw(img)
    bx1, by1 = int(w * 0.12), int(h * 0.18)
    bx2, by2 = int(w * 0.88), int(h * 0.82)
    draw.rectangle([bx1, by1, bx2, by2], fill=(170, 118, 55), outline=(90, 58, 20), width=5)
    # grietas / pliegues de caja
    mid_y = (by1 + by2) // 2
    draw.line([(bx1, mid_y), (bx2, mid_y)], fill=(120, 78, 30), width=2)
    # area de etiqueta blanca
    draw.rectangle([bx1+30, by1+30, bx2-30, mid_y-20],
                   fill=(255, 255, 255), outline=(180, 180, 180), width=1)
    return img


if IMAGE_PATH:
    print(f"\nCargando imagen real: {IMAGE_PATH}")
    img = Image.open(IMAGE_PATH).convert("RGB").resize((640, 640))
else:
    print("\nUsando imagen sintetica 640x640 (caja de carton simulada)...")
    img = make_synthetic_box()
    img.save("/tmp/test_input.png")
    print("Guardada en /tmp/test_input.png")

print(f"Tamano imagen: {img.size}  modo: {img.mode}")

print("\nEjecutando inferencia…")
out = model.predict({"image": img})
print("Inferencia completada.\n")

print("="*55)
print("SHAPES DE SALIDA")
print("="*55)
for k, v in out.items():
    arr = np.array(v)
    print(f"  '{k}': shape={arr.shape}  dtype={arr.dtype}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}")


def parse_detections(name, arr):
    C = arr.shape[-1]
    if C < 5:
        return
    data = arr[0] if len(arr.shape) == 3 else arr
    N    = data.shape[0]

    coord_max = data[:, 2].max()
    coord_fmt = "PIXEL (÷640 para normalizar)" if coord_max > 2 else "NORMALIZADO 0-1"

    print(f"\n{'='*55}")
    print(f"DETECCIONES EN '{name}'  ({N} filas x {C} cols)")
    print(f"Formato coords: {coord_fmt}")
    print(f"{'='*55}")
    print(f"{'idx':>5}  {'x1':>7} {'y1':>7} {'x2':>7} {'y2':>7}  {'conf':>6}  cls")
    print("-"*55)

    found = 0
    for i in range(N):
        row  = data[i]
        conf = float(row[4])
        if conf < 0.01:
            continue
        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        cls  = int(row[5]) if C > 5 else -1
        print(f"{i:>5}  {x1:>7.2f} {y1:>7.2f} {x2:>7.2f} {y2:>7.2f}  {conf:>6.4f}  {cls}")
        found += 1

    if found == 0:
        print("  (sin detecciones con conf > 0.01)")
        top5_idx = np.argsort(data[:, 4])[-5:][::-1]
        print(f"  Top-5 conf: {[f'{data[i,4]:.4f}' for i in top5_idx]}")
    else:
        print(f"\n  TOTAL: {found} deteccion(es)")

    # Consejo para Swift
    print(f"\n  En Swift extractBestBox:")
    print(f"    conf threshold 0.25 → {'OK detectaria' if found>0 and data[:,4].max()>0.25 else 'FALLA (max conf=' + str(round(float(data[:,4].max()),3)) + ')' }")
    if coord_max > 2:
        print(f"    scale = 1/640 (coords en pixeles) ← el codigo Swift ya hace esto ✓")
    else:
        print(f"    scale = 1.0 (coords ya normalizadas) ← el codigo Swift ya hace esto ✓")


print()
for k, v in out.items():
    arr = np.array(v)
    parse_detections(k, arr)
