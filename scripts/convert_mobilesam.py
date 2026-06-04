#!/usr/bin/env python3
"""
Convierte MobileSAM (mobile_sam.pt) a dos modelos CoreML:
  - sam_encoder.mlmodel  : TinyViT image encoder
  - sam_decoder.mlmodel  : Mask decoder con 1 punto de prompt fijo

Requisitos (macOS):
  pip install torch torchvision coremltools git+https://github.com/ChaoningZhang/MobileSAM.git

Uso:
  python3 scripts/convert_mobilesam.py
  # Salida: modules/lidar-box-measure/ios/sam_encoder.mlmodel
  #          modules/lidar-box-measure/ios/sam_decoder.mlmodel
"""

import sys
import os
import torch
import torch.nn as nn
import coremltools as ct
import numpy as np

OUT_DIR    = os.path.join(os.path.dirname(__file__),
                          "..", "modules", "lidar-box-measure", "ios")
CKPT_PATH  = "/tmp/mobile_sam.pt"
IMAGE_SIZE = 1024  # MobileSAM canonical input


# ── Descargar checkpoint ──────────────────────────────────────────────────────

def download_checkpoint():
    if os.path.exists(CKPT_PATH) and os.path.getsize(CKPT_PATH) > 1_000_000:
        print(f"Checkpoint ya existe: {CKPT_PATH}  ({os.path.getsize(CKPT_PATH)//1_000_000} MB)")
        return
    try:
        from huggingface_hub import hf_hub_download
        print("Descargando mobile_sam.pt desde HuggingFace…")
        path = hf_hub_download(
            repo_id="dhkim2810/MobileSAM",
            filename="mobile_sam.pt",
            local_dir="/tmp",
        )
        print(f"Descargado: {path}  ({os.path.getsize(path)//1_000_000} MB)")
    except Exception as e:
        print(f"huggingface_hub falló ({e}), intentando wget…")
        ret = os.system(
            "wget -q --show-progress -O /tmp/mobile_sam.pt "
            "https://huggingface.co/dhkim2810/MobileSAM/resolve/main/mobile_sam.pt"
        )
        if ret != 0:
            sys.exit("ERROR: no se pudo descargar mobile_sam.pt")


# ── Importar MobileSAM ────────────────────────────────────────────────────────

def load_sam():
    # Buscar MobileSAM en PYTHONPATH o en /tmp/MobileSAM (clonado en CI)
    mobile_sam_path = "/tmp/MobileSAM"
    if mobile_sam_path not in sys.path and os.path.isdir(mobile_sam_path):
        sys.path.insert(0, mobile_sam_path)
    try:
        from mobile_sam import sam_model_registry
    except ImportError:
        sys.exit(
            "ERROR: MobileSAM no encontrado. "
            "Clonar: git clone https://github.com/ChaoningZhang/MobileSAM.git /tmp/MobileSAM"
        )
    print("Cargando mobile_sam.pt…")
    sam = sam_model_registry["vit_t"](checkpoint=CKPT_PATH)
    sam.eval()
    return sam


# ── Wrappers trazables ────────────────────────────────────────────────────────

class EncoderWrapper(nn.Module):
    """Sólo el image encoder de SAM — entrada normalizada RGB."""
    def __init__(self, sam):
        super().__init__()
        self.encoder = sam.image_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)   # [1, 256, 64, 64]


class DecoderWrapper(nn.Module):
    """
    Decoder SAM con 1 punto de prompt (centro de pantalla).
    Entradas:
        embedding  : [1, 256, 64, 64]
        point_x    : escalar float en espacio 1024px
        point_y    : escalar float en espacio 1024px
    Salida:
        mask_logits: [1, 1, 256, 256]  (aplicar sigmoid > 0.5 para binario)
    """
    def __init__(self, sam):
        super().__init__()
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder   = sam.mask_decoder

    def forward(self, embedding: torch.Tensor,
                point_x: torch.Tensor,
                point_y: torch.Tensor) -> torch.Tensor:
        coords = torch.stack([point_x, point_y], dim=-1).float()   # [1, 1, 2]
        coords = coords.unsqueeze(0).unsqueeze(0)                   # [1, 1, 1, 2]
        labels = torch.ones(1, 1, dtype=torch.int)                  # foreground

        sparse_emb, dense_emb = self.prompt_encoder(
            points=(coords, labels), boxes=None, masks=None
        )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings    = embedding,
            image_pe            = self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings = sparse_emb,
            dense_prompt_embeddings  = dense_emb,
            multimask_output    = False,
        )
        return low_res_masks   # [1, 1, 256, 256]


# ── Conversión ────────────────────────────────────────────────────────────────

def to_coreml(wrapper, dummy_inputs, input_specs, output_names, out_path, step):
    """
    Export wrapper → CoreML usando torch.export.export(strict=False).
    strict=False permite operaciones dinámicas de TinyViT (assert, len, etc.)
    y es la API recomendada para coremltools 8+.
    """
    args = dummy_inputs if isinstance(dummy_inputs, tuple) else (dummy_inputs,)

    print(f"  [{step}] torch.export.export(strict=False)…")
    with torch.no_grad():
        exported = torch.export.export(wrapper, args, strict=False)

    print(f"  [{step}] Convirtiendo ExportedProgram → CoreML…")
    mlmodel = ct.convert(
        exported,
        inputs=input_specs,
        outputs=[ct.TensorType(name=n) for n in output_names],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS15,
        convert_to="mlprogram",
    )
    mlmodel.save(out_path)
    print(f"  [{step}] Guardado: {out_path}")
    return out_path


def convert_encoder(sam):
    print("\n[1/2] Convirtiendo encoder…")
    return to_coreml(
        EncoderWrapper(sam),
        torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        [ct.TensorType(name="image", shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)],
        ["embedding"],
        os.path.join(OUT_DIR, "sam_encoder.mlmodel"),
        "1/2",
    )


def convert_decoder(sam):
    print("\n[2/2] Convirtiendo decoder…")
    return to_coreml(
        DecoderWrapper(sam),
        (torch.zeros(1, 256, 64, 64),
         torch.tensor([512.0]),
         torch.tensor([512.0])),
        [
            ct.TensorType(name="embedding", shape=(1, 256, 64, 64), dtype=np.float32),
            ct.TensorType(name="point_x",   shape=(1,),              dtype=np.float32),
            ct.TensorType(name="point_y",   shape=(1,),              dtype=np.float32),
        ],
        ["mask_logits"],
        os.path.join(OUT_DIR, "sam_decoder.mlmodel"),
        "2/2",
    )


# ── Test rápido ───────────────────────────────────────────────────────────────

def quick_test(enc_path, dec_path):
    import sys
    if sys.platform != "darwin":
        print("\nTest de inferencia: omitido (solo macOS)")
        return

    print("\nTest de inferencia…")
    from PIL import Image
    enc = ct.models.MLModel(enc_path)
    dec = ct.models.MLModel(dec_path)

    # Imagen gris 1024×1024
    img_arr = np.random.randint(100, 200, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    img     = Image.fromarray(img_arr)

    # Normalizar a [-1, 1] aprox (SAM usa pixel_mean / pixel_std)
    tensor = (np.array(img).astype(np.float32) / 255.0).transpose(2, 0, 1)
    tensor = tensor[np.newaxis]  # [1, 3, H, W]

    emb_out = enc.predict({"image": tensor})
    emb_key = list(emb_out.keys())[0]
    emb     = emb_out[emb_key]
    print(f"  embedding shape: {np.array(emb).shape}")

    mask_out = dec.predict({
        "embedding": np.array(emb).astype(np.float32),
        "point_x":   np.array([512.0], dtype=np.float32),
        "point_y":   np.array([512.0], dtype=np.float32),
    })
    mask_key = list(mask_out.keys())[0]
    mask     = np.array(mask_out[mask_key])
    print(f"  mask shape: {mask.shape}  min={mask.min():.3f}  max={mask.max():.3f}")

    # Bounding box de la mascara
    binary = (mask[0, 0] > 0).astype(np.uint8)
    ys, xs = np.where(binary)
    if len(xs) > 0:
        print(f"  Deteccion: x=[{xs.min()},{xs.max()}] y=[{ys.min()},{ys.max()}]  "
              f"pixels={len(xs)}")
    else:
        print("  Sin deteccion (imagen sintetica sin objeto)")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    download_checkpoint()
    sam = load_sam()

    with torch.no_grad():
        enc_path = convert_encoder(sam)
        dec_path = convert_decoder(sam)

    quick_test(enc_path, dec_path)

    print("\n✓ Conversion completa")
    print(f"  {enc_path}")
    print(f"  {dec_path}")
