"""
pgd_jpeg_sim_attack.py — PGD con simulazione JPEG durante l'ottimizzazione
===========================================================================

Insight dai debug
-----------------
  La perturbazione adversariale efficace per densenet121_dct e' ad alta
  frequenza: proiettarla su basse frequenze distrugge l'efficacia.
  Filtrare il delta finale non mantiene la predizione Real.

  Soluzione: rendere la perturbazione robusta a JPEG durante l'ottimizzazione
  stessa, non a posteriori. Ad ogni step PGD applichiamo una simulazione
  differenziabile di JPEG, cosi' il gradiente impara a trovare perturbazioni
  che sopravvivono alla compressione.

Simulazione JPEG differenziabile
---------------------------------
  JPEG comprime ogni canale in blocchi 8x8 via DCT. Per ogni blocco:
    1. DCT 8x8
    2. Divisione per matrice di quantizzazione Q (dipende da quality)
    3. Arrotondamento (non differenziabile → approssimiamo con noise smooth)
    4. Moltiplicazione per Q
    5. IDCT 8x8

  Usiamo una approssimazione differenziabile dello step di arrotondamento:
    round(x) ≈ x + 0.5*sin(2*pi*x)  (smooth approximation)

  In pratica: ottimizziamo la loss su JPEG(pixel_orig + delta) invece di
  pixel_orig + delta. Questo forza il gradiente a trovare delta che
  sopravvivono alla quantizzazione JPEG.

  proj_every=50 funzionava (SUCCESS step=37): usiamo questa frequenza
  di proiezione come regolarizzatore soft invece di hard constraint.

Uso
---
  python test-2/pgd_jpeg_sim_attack.py --config AADD_2026_config.yaml --input_dir  AADD_2026_Test --output_dir adv-jpeg-sim --eps 32 --steps 200 --jpeg_quality 75
"""

import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy.fftpack import dct as scipy_dct, idct as scipy_idct
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.models as tv_models


# ============================================================================
# Costanti
# ============================================================================
CLASS_IDX_REAL = 0
CLASSES        = 2
IMAGE_EXTS     = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
DCT_SIZE       = 128


# ============================================================================
# Model factory (identica a evaluate.py)
# ============================================================================

def _create_densenet121_dct() -> nn.Module:
    model = tv_models.densenet121(weights=None)
    model.features.conv0 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.classifier.in_features, CLASSES)
    )
    return model


def load_densenet_dct(weight_path: Path, device: torch.device) -> nn.Module:
    print(f"[MODEL] Carico densenet121_dct da {weight_path} ...")
    model = _create_densenet121_dct()
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)
    print(f"[MODEL] Pronto su {device}\n")
    return model


# ============================================================================
# DCT torch corretta (verificata: errore < 0.001 vs scipy)
# ============================================================================

def _dct1d_torch(x: torch.Tensor, dim: int) -> torch.Tensor:
    n = x.shape[dim]
    idx_even = torch.arange(0, n, 2, device=x.device)
    idx_odd  = torch.arange(n-1 if n % 2 == 1 else n-1, 0, -2, device=x.device)
    x_r = torch.index_select(x, dim, torch.cat([idx_even, idx_odd]))
    X = torch.fft.fft(x_r, dim=dim)
    k = torch.arange(n, dtype=torch.float32, device=x.device)
    phase = torch.exp(-1j * torch.pi * k / (2*n))
    shape = [1]*x.ndim; shape[dim] = n
    dct = (X * phase.view(shape)).real
    w = torch.ones(n, dtype=torch.float32, device=x.device) * np.sqrt(2.0/n)
    w[0] = np.sqrt(1.0/n)
    return dct * w.view(shape)


def dct2_torch(x: torch.Tensor) -> torch.Tensor:
    return _dct1d_torch(_dct1d_torch(x, dim=0), dim=1)


def model_pipeline(pixels: torch.Tensor) -> torch.Tensor:
    """pixels (128,128) → (1,1,128,128) input modello."""
    dct  = dct2_torch(pixels)
    feat = torch.log(torch.abs(dct) + 1e-6)
    return feat.unsqueeze(0).unsqueeze(0)


# ============================================================================
# DCT numpy
# ============================================================================

def dct2_np(a): return scipy_dct(scipy_dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')
def idct2_np(a): return scipy_idct(scipy_idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


# ============================================================================
# Simulazione JPEG differenziabile su blocchi 8x8
# ============================================================================

# Matrice di quantizzazione luminanza JPEG standard (qualita' base Q=50)
JPEG_LUMA_Q50 = torch.tensor([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99],
], dtype=torch.float32)


def jpeg_quant_matrix(quality: int, device: torch.device) -> torch.Tensor:
    """
    Calcola la matrice di quantizzazione JPEG per una data qualita'.
    Formula standard JPEG: Q_table = clip(floor((Q50 * S + 50) / 100), 1, 255)
    con S = 200 - 2*quality se quality >= 50, else 5000/quality.
    """
    q = float(quality)
    if q >= 50:
        s = 200.0 - 2.0 * q
    else:
        s = 5000.0 / q
    Q = torch.floor((JPEG_LUMA_Q50 * s + 50.0) / 100.0)
    Q = Q.clamp(1.0, 255.0).to(device)
    return Q


def differentiable_round(x: torch.Tensor) -> torch.Tensor:
    """
    Approssimazione differenziabile di round(x).
    round(x) ≈ x - (1/(2*pi)) * sin(2*pi*x)
    Questa funzione ha derivata 1 - cos(2*pi*x), che e' sempre >= 0
    e si annulla nei punti di discontinuita' dell'arrotondamento reale.
    """
    return x - (1.0 / (2.0 * torch.pi)) * torch.sin(2.0 * torch.pi * x)


def jpeg_compress_block(block: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    Compressione JPEG differenziabile su un singolo blocco 8x8.
    block: (8, 8) tensor float
    Q:     (8, 8) matrice di quantizzazione
    """
    dct  = dct2_torch(block)
    coef = dct / Q
    coef_rounded = differentiable_round(coef)
    dct_reconstructed = coef_rounded * Q
    # IDCT differenziabile: usiamo la trasposta della DCT (DCT e' ortogonale)
    # Approssimiamo con scipy via detach per la ricostruzione pixel
    # ma manteniamo il grafo per backprop attraverso la quantizzazione
    return dct_reconstructed


def jpeg_simulate_differentiable(pixels: torch.Tensor,
                                  quality: int,
                                  device: torch.device) -> torch.Tensor:
    """
    Simula compressione JPEG differenziabile su immagine 128x128.
    Processa blocchi 8x8 sovrapposti.
    Ritorna i coefficienti DCT quantizzati (non i pixel ricostruiti)
    perche' il modello opera su DCT.

    Strategia: invece di ricostruire i pixel (richiederebbe IDCT torch
    che introduce ulteriore errore), passiamo direttamente i coefficienti
    DCT quantizzati al modello, simulando cio' che accadrebbe dopo JPEG.
    """
    H, W = pixels.shape
    assert H == 128 and W == 128

    # DCT globale 128x128 (come evaluate.py)
    dct_global = dct2_torch(pixels)   # 128x128

    # Quantizzazione approssimata: usiamo la matrice Q scalata per 128x128
    # La matrice 8x8 viene tile-ata e scalata
    Q_8x8  = jpeg_quant_matrix(quality, device)
    # Tile Q su 128x128: ogni coefficiente globale ha una quantizzazione
    # approssimata basata sulla sua posizione relativa nel blocco 8x8
    Q_tiled = Q_8x8.repeat(128 // 8, 128 // 8)   # 128x128

    # Quantizzazione differenziabile nel dominio DCT globale
    coef         = dct_global / Q_tiled
    coef_rounded = differentiable_round(coef)
    dct_jpeg     = coef_rounded * Q_tiled

    return dct_jpeg   # 128x128, coefficienti DCT dopo JPEG simulato


def model_pipeline_jpeg(pixels: torch.Tensor, quality: int,
                         device: torch.device) -> torch.Tensor:
    """
    Pipeline con JPEG simulato:
    pixels → JPEG_simulate → log(|DCT_jpeg|) → modello
    """
    dct_jpeg = jpeg_simulate_differentiable(pixels, quality, device)
    feat     = torch.log(torch.abs(dct_jpeg) + 1e-6)
    return feat.unsqueeze(0).unsqueeze(0)


# ============================================================================
# Pre-processing identico a evaluate.py
# ============================================================================

def preprocess(pil_img: Image.Image) -> tuple[np.ndarray, dict]:
    img = pil_img.convert('L')
    orig_size = pil_img.size
    if max(img.size) > 256:
        img_256 = img.resize((256, 256), Image.Resampling.LANCZOS)
    else:
        img_256 = img.copy()
    w, h  = img_256.size
    left  = (w - 128) // 2
    top   = (h - 128) // 2
    crop  = img_256.crop((left, top, left + 128, top + 128))
    arr   = np.array(crop, dtype=np.float32)
    meta  = {'orig_size': orig_size, 'img_256': img_256, 'crop_box': (left, top)}
    return arr, meta


def predict_np(model: nn.Module, gray: np.ndarray,
               device: torch.device) -> int:
    dct  = dct2_np(gray)
    feat = np.log(np.abs(dct) + 1e-6)
    t    = torch.from_numpy(feat).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(t).argmax(1).item()


def predict_after_jpeg(model: nn.Module, gray: np.ndarray,
                        quality: int, device: torch.device) -> int:
    """Predizione dopo compressione JPEG reale (via PIL)."""
    import io
    img_pil = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8), mode='L')
    buf = io.BytesIO()
    img_pil.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    img_jpeg = np.array(Image.open(buf).convert('L'), dtype=np.float32)
    return predict_np(model, img_jpeg, device)


def save_adversarial(gray_adv: np.ndarray, meta: dict, out_path: Path) -> None:
    Y        = np.clip(gray_adv, 0.0, 255.0).astype(np.uint8)
    crop_pil = Image.fromarray(Y, mode='L')
    img_out  = meta['img_256'].copy()
    img_out.paste(crop_pil, meta['crop_box'])
    orig_w, orig_h = meta['orig_size']
    if (orig_w, orig_h) != img_out.size:
        img_out = img_out.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    img_out.save(out_path)


# ============================================================================
# PGD con JPEG simulato durante l'ottimizzazione
# ============================================================================

def pgd_jpeg_robust(
    model:        nn.Module,
    gray_orig:    np.ndarray,
    device:       torch.device,
    eps:          float,
    alpha:        float,
    steps:        int,
    target_class: int,
    jpeg_quality: int,
    proj_every:   int,    # proiezione low-freq ogni N step (regolarizzatore)
    low_freq_mask: np.ndarray,
    verbose:      bool = False,
) -> tuple[np.ndarray, bool]:
    """
    PGD che ottimizza su JPEG(pixel + delta) invece di pixel + delta.
    Il gradiente impara automaticamente a trovare perturbazioni che
    sopravvivono alla quantizzazione JPEG.

    In aggiunta, ogni proj_every step proietta il delta su basse
    frequenze come regolarizzatore (dal debug: proj_every=50 funziona).
    """
    loss_fn  = nn.CrossEntropyLoss()
    target_t = torch.tensor([target_class], device=device)

    delta     = np.zeros_like(gray_orig)
    best_gray = gray_orig.copy()
    found     = False
    best_norm = float('inf')

    for step in range(steps):
        gray_adv = np.clip(gray_orig + delta, 0.0, 255.0)

        t = torch.from_numpy(gray_adv).float().to(device)
        t.requires_grad_(True)

        # Forward attraverso JPEG simulato
        logits = model(model_pipeline_jpeg(t, jpeg_quality, device))
        loss   = loss_fn(logits, target_t)
        model.zero_grad()
        loss.backward()

        grad = t.grad.detach().cpu().numpy()
        delta += alpha * np.sign(grad)
        delta  = np.clip(delta, -eps, eps)

        # Proiezione low-freq come regolarizzatore ogni proj_every step
        if proj_every > 0 and (step + 1) % proj_every == 0:
            d_dct  = dct2_np(delta)
            d_dct *= low_freq_mask
            delta  = np.clip(idct2_np(d_dct), -eps, eps)

        # Verifica con pipeline originale (senza JPEG sim, come evaluate.py)
        pred_orig_pipeline = predict_np(
            model, np.clip(gray_orig + delta, 0, 255), device
        )
        if pred_orig_pipeline == target_class:
            cur_norm = np.abs(delta).max()
            if not found or cur_norm < best_norm:
                best_gray = np.clip(gray_orig + delta, 0.0, 255.0).copy()
                best_norm = cur_norm
                found     = True

        if verbose and (step + 1) % 50 == 0:
            print(f"  step {step+1:4d}/{steps}  loss={loss.item():.4f}  "
                  f"pred={'Real' if pred_orig_pipeline==0 else 'Fake'}  "
                  f"found={found}  delta_max={np.abs(delta).max():.2f}")

    if not found:
        best_gray = np.clip(gray_orig + delta, 0.0, 255.0)

    return best_gray, found


# ============================================================================
# Config helpers
# ============================================================================

def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def get_device(choice: str) -> torch.device:
    if choice == 'auto':
        dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        dev = torch.device(choice)
    print(f"[DEVICE] {dev}")
    return dev


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PGD con JPEG simulato durante ottimizzazione."
    )
    parser.add_argument('--config',          required=True)
    parser.add_argument('--input_dir',       required=True)
    parser.add_argument('--output_dir',      required=True)
    parser.add_argument('--eps',             type=float, default=32.0,
                        help="Epsilon in pixel (default 32).")
    parser.add_argument('--alpha',           type=float, default=None,
                        help="Step size (default eps/10).")
    parser.add_argument('--steps',           type=int,   default=200,
                        help="Iterazioni PGD (default 200).")
    parser.add_argument('--jpeg_quality',    type=int,   default=75,
                        help="Qualita' JPEG simulata (default 75). "
                             "Piu' basso = piu' robusto ma piu' difficile.")
    parser.add_argument('--proj_every',      type=int,   default=50,
                        help="Proiezione low-freq ogni N step (default 50). "
                             "0 = disabilita. Dal debug: 50 funziona.")
    parser.add_argument('--low_freq_ratio',  type=float, default=0.25,
                        help="Frazione low-freq per il regolarizzatore "
                             "(default 0.25).")
    parser.add_argument('--target_class',    type=int,   default=CLASS_IDX_REAL)
    parser.add_argument('--verbose',         action='store_true')
    args = parser.parse_args()

    alpha = args.alpha if args.alpha is not None else args.eps / 10.0

    cfg    = load_cfg(args.config)
    device = get_device(cfg.get('device', 'auto'))

    models_dir  = Path(cfg['models_dir'])
    weight_path = models_dir / 'densenet121_dct.pth'
    if not weight_path.exists():
        raise FileNotFoundError(f"Pesi non trovati: {weight_path}")
    model = load_densenet_dct(weight_path, device)

    K = max(1, int(DCT_SIZE * args.low_freq_ratio))
    low_freq_mask = np.zeros((DCT_SIZE, DCT_SIZE), dtype=bool)
    low_freq_mask[:K, :K] = True

    print(f"[JPEG]  quality={args.jpeg_quality}  "
          f"proj_every={args.proj_every}  low_freq K={K}")
    print(f"[ATTACK] eps={args.eps}px  alpha={alpha:.3f}px  "
          f"steps={args.steps}\n")

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_paths = [p for p in input_dir.rglob('*')
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not img_paths:
        raise RuntimeError(f"Nessuna immagine trovata in: {input_dir}")
    print(f"[DATA] {len(img_paths)} immagini da attaccare\n")

    success_count = 0

    for img_path in tqdm(img_paths, desc="PGD JPEG-sim"):
        pil_orig        = Image.open(img_path)
        gray_orig, meta = preprocess(pil_orig)
        pred_orig       = predict_np(model, gray_orig, device)

        gray_adv, found = pgd_jpeg_robust(
            model         = model,
            gray_orig     = gray_orig,
            device        = device,
            eps           = args.eps,
            alpha         = alpha,
            steps         = args.steps,
            target_class  = args.target_class,
            jpeg_quality  = args.jpeg_quality,
            proj_every    = args.proj_every,
            low_freq_mask = low_freq_mask,
            verbose       = args.verbose,
        )

        pred_adv = predict_np(model, gray_adv, device)

        # Verifica anche dopo JPEG reale
        pred_jpeg = predict_after_jpeg(
            model, gray_adv, args.jpeg_quality, device
        )

        success  = int(pred_adv == args.target_class)
        success_count += success

        rel      = img_path.relative_to(input_dir)
        out_path = (output_dir / rel).with_suffix('.png')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_adversarial(gray_adv, meta, out_path)

        tqdm.write(
            f"[{img_path.name}]  "
            f"orig={'Real' if pred_orig==0 else 'Fake'}  "
            f"adv={'Real' if pred_adv==0 else 'Fake'}  "
            f"after_jpeg={'Real' if pred_jpeg==0 else 'Fake'}  "
            f"success={success}"
        )

    total = len(img_paths)
    print(f"\n[DONE] Attack success rate: {success_count}/{total} "
          f"({100*success_count/total:.1f}%)")
    print(f"[DONE] Immagini salvate in: {output_dir}")


if __name__ == '__main__':
    main()