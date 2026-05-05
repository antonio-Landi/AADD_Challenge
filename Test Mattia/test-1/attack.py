"""
test-1/attack.py — Per-model adversarial attack for AADD-2026
==============================================================
Per ogni modello in {vit_b_16, densenet121_dct}:
  1. Applica PGD con loss score-aware:
       L = −log P(Real | x_adv) − λ_ssim · SSIM(x_adv, x_orig)
     Il vincolo L∞ ε controlla implicitamente LPIPS.
  2. Salva le immagini avversariali in  dataset_adv/<model>/<nome>.png
  3. Dopo ogni immagine: valuta con entrambi i classificatori e aggiorna
     dataset_adv/<model>/results.json  con metriche per-immagine e score totale.

Formula dello score (identica a AADD_2026_evaluation.py):
  sim_weight      = α·SSIM + (1−α)·(1−LPIPS)          α = 0.5
  pair_contrib    = Σ_c  w_c · sim_weight · I(pred_c == Real)
  total_score     = Σ_images  pair_contrib
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import lpips as lpips_module
from skimage.metrics import structural_similarity as sk_ssim

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
TEST_DIR   = ROOT / "AADD_2026_Test"
MODELS_DIR = ROOT / "models_weights"
ADV_BASE   = ROOT / "dataset_adv"

# ── Score constants — devono corrispondere a AADD_2026_evaluation.py ──────────
CLASS_IDX_REAL = 0
ALPHA_SCORE    = 0.5                                # peso SSIM in sim_weight
CLF_WEIGHTS    = {"vit_b_16": 1.0, "densenet121_dct": 1.0}

# ── Iperparametri dell'attacco ─────────────────────────────────────────────────
EPSILON      = 8 / 255    # budget L∞ (in [0,1])
N_STEPS      = 40         # iterazioni PGD
STEP_SIZE    = EPSILON / 10
LAMBDA_SSIM  = 0.3        # peso del termine SSIM nella loss
ATTACK_RES   = 256        # risoluzione per il calcolo del gradiente

# ── Normalizzazione ImageNet ───────────────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# ════════════════════════════════════════════════════════════════════════════════
# Caricamento modelli  (identico a AADD_2026_evaluation.py)
# ════════════════════════════════════════════════════════════════════════════════

def _load_vit_b16(device: torch.device) -> nn.Module:
    from torchvision.models import vit_b_16
    m = vit_b_16(weights=None)
    m.heads.head = nn.Linear(m.heads.head.in_features, 2)
    m.load_state_dict(torch.load(MODELS_DIR / "vit_b_16.pth", map_location=device))
    return m.eval().to(device)


def _load_densenet121_dct(device: torch.device) -> nn.Module:
    from torchvision.models import densenet121
    m = densenet121(weights=None)
    m.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.classifier.in_features, 2))
    m.load_state_dict(torch.load(MODELS_DIR / "densenet121_dct.pth", map_location=device))
    return m.eval().to(device)


_LOADERS = {"vit_b_16": _load_vit_b16, "densenet121_dct": _load_densenet121_dct}


# ════════════════════════════════════════════════════════════════════════════════
# Pre-processing differenziabile  (pipeline identica allo script di valutazione)
# ════════════════════════════════════════════════════════════════════════════════

def _preprocess_vit(x: torch.Tensor) -> torch.Tensor:
    """[B,3,H,W] ∈ [0,1]  →  [B,3,224,224] normalizzato ImageNet"""
    x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
    x = x[:, :, 16:240, 16:240]                         # center-crop → 224
    m = torch.tensor(_MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(_STD,  device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


class _DCT2(nn.Module):
    """DCT-II 2D ortonormale via moltiplicazione matriciale (completamente differenziabile).
    Equivale a scipy.fftpack.dct(..., norm='ortho') applicata su entrambi gli assi.
    """
    def __init__(self, n: int = 128):
        super().__init__()
        k = torch.arange(n, dtype=torch.float64).unsqueeze(1)
        m = torch.arange(n, dtype=torch.float64).unsqueeze(0)
        W = torch.cos(torch.pi * k * (2 * m + 1) / (2 * n)).float()
        W[0]  *= (1.0 / n) ** 0.5
        W[1:] *= (2.0 / n) ** 0.5
        self.register_buffer("W", W)   # [n, n]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, n, n]   →   DCT2D: W @ x @ W^T
        return self.W @ x @ self.W.t()


_dct2_cache: dict = {}


def _get_dct2(device: torch.device) -> _DCT2:
    key = str(device)
    if key not in _dct2_cache:
        _dct2_cache[key] = _DCT2(128).to(device)
    return _dct2_cache[key]


def _preprocess_dct(x: torch.Tensor) -> torch.Tensor:
    """[B,3,H,W] ∈ [0,1]  →  [B,1,128,128] log-DCT (ITU-R BT.601 grayscale)"""
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    gray = F.interpolate(gray, size=(256, 256), mode="bilinear", align_corners=False)
    gray = gray[:, :, 64:192, 64:192]                   # center-crop → 128
    dct  = _get_dct2(x.device)(gray)
    return torch.log(torch.abs(dct) + 1e-6)


_PREPROCESSORS = {
    "vit_b_16":        _preprocess_vit,
    "densenet121_dct": _preprocess_dct,
}


# ════════════════════════════════════════════════════════════════════════════════
# SSIM differenziabile  (usato nella loss dell'attacco)
# ════════════════════════════════════════════════════════════════════════════════

def _gauss_kernel(sz: int, sigma: float, C: int, device: torch.device) -> torch.Tensor:
    half = sz // 2
    xs = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    g  = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    k  = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    return k.expand(C, 1, sz, sz).contiguous()


def diff_ssim(x: torch.Tensor, y: torch.Tensor, win: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """SSIM differenziabile su [B,C,H,W] ∈ [0,1]. Restituisce scalare medio."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    C = x.shape[1]
    k = _gauss_kernel(win, sigma, C, x.device)
    p = win // 2

    mx  = F.conv2d(x,   k, padding=p, groups=C)
    my  = F.conv2d(y,   k, padding=p, groups=C)
    sx2 = F.conv2d(x * x, k, padding=p, groups=C) - mx ** 2
    sy2 = F.conv2d(y * y, k, padding=p, groups=C) - my ** 2
    sxy = F.conv2d(x * y, k, padding=p, groups=C) - mx * my

    num = (2 * mx * my + C1) * (2 * sxy + C2)
    den = (mx ** 2 + my ** 2 + C1) * (sx2 + sy2 + C2)
    return (num / den).mean()


# ════════════════════════════════════════════════════════════════════════════════
# Loss score-aware
# ════════════════════════════════════════════════════════════════════════════════

def _loss(logits: torch.Tensor, x_adv: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:
    """
    L = −log P(Real | x_adv) − λ_ssim · SSIM(x_adv, x_orig)

    Minimizzare L:
      • massimizza P(Real)  → indicator = 1  → contribuisce allo score
      • massimizza SSIM     → sim_weight alto → moltiplica il contributo score
    Il vincolo L∞ (ε) limita implicitamente LPIPS ≈ basso.
    """
    ce = -F.log_softmax(logits, dim=-1)[:, CLASS_IDX_REAL].mean()
    s  = diff_ssim(x_adv, x_orig)
    return ce - LAMBDA_SSIM * s


# ════════════════════════════════════════════════════════════════════════════════
# Attacco PGD
# ════════════════════════════════════════════════════════════════════════════════

def pgd(model: nn.Module, preprocess, x_orig: torch.Tensor) -> torch.Tensor:
    """
    PGD con vincolo L∞.
    Il gradiente è calcolato a ATTACK_RES×ATTACK_RES per efficienza (segnale
    più forte rispetto alla risoluzione originale 1024×1024).
    Il delta finale viene ricampionato bilinearmente alla risoluzione originale
    (interpolazione bilineare = combinazione convessa → preserva L∞).
    """
    H, W = x_orig.shape[-2:]

    x256  = F.interpolate(x_orig.detach(), size=(ATTACK_RES, ATTACK_RES),
                          mode="bilinear", align_corners=False)
    delta = torch.zeros_like(x256)

    for _ in range(N_STEPS):
        delta.requires_grad_(True)
        x_adv  = (x256 + delta).clamp(0.0, 1.0)
        logits = model(preprocess(x_adv))
        loss   = _loss(logits, x_adv, x256)
        loss.backward()

        with torch.no_grad():
            delta = (delta - STEP_SIZE * delta.grad.sign()).clamp(-EPSILON, EPSILON)
            delta = (x256 + delta).clamp(0.0, 1.0) - x256

    delta_full = F.interpolate(delta.detach(), size=(H, W),
                               mode="bilinear", align_corners=False)
    return (x_orig + delta_full).clamp(0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════════════
# Valutazione inline  (logica identica a AADD_2026_evaluation.py)
# ════════════════════════════════════════════════════════════════════════════════

def _eval_pair(
    orig_np:    np.ndarray,   # [H,W,3] uint8
    adv_np:     np.ndarray,   # [H,W,3] uint8
    adv_tensor: torch.Tensor, # [1,3,H,W] float in [0,1]
    all_models: dict,
    lpips_fn:   nn.Module,
    device:     torch.device,
) -> dict:
    """
    Calcola SSIM, LPIPS, sim_weight e predizioni per-classificatore.
    Il formato del dizionario restituito segue quello di AADD_2026_evaluation.py.
    """
    # SSIM per canale → media  (data_range=255 come nello script ufficiale)
    ssim_val = float(np.mean([
        sk_ssim(orig_np[:, :, c], adv_np[:, :, c], data_range=255)
        for c in range(3)
    ]))

    # LPIPS  — normalizzazione [−1, 1] come in np_to_lpips_tensor() ufficiale
    orig_t = torch.from_numpy(orig_np).permute(2, 0, 1).float() / 127.5 - 1.0
    adv_t  = adv_tensor[0].cpu().float() * 2.0 - 1.0
    with torch.no_grad():
        lp_val = float(lpips_fn(
            orig_t.unsqueeze(0).to(device),
            adv_t.unsqueeze(0).to(device),
        ).item())

    sim_weight = ALPHA_SCORE * ssim_val + (1.0 - ALPHA_SCORE) * (1.0 - lp_val)

    per_clf      = {}
    pair_contrib = 0.0

    for name, model in all_models.items():
        with torch.no_grad():
            logits    = model(_PREPROCESSORS[name](adv_tensor.to(device)))
            pred_idx  = int(logits.argmax(dim=-1).item())
            prob_real = float(torch.softmax(logits, dim=-1)[0, CLASS_IDX_REAL].item())

        label     = "Real" if pred_idx == CLASS_IDX_REAL else "Fake"
        indicator = 1 if pred_idx == CLASS_IDX_REAL else 0
        contrib   = CLF_WEIGHTS[name] * sim_weight * indicator

        per_clf[name] = {
            "prediction": label,
            "prob_real":  prob_real,
            "indicator":  indicator,
            "contribution": contrib,
        }
        pair_contrib += contrib

    return {
        "ssim":             ssim_val,
        "lpips":            lp_val,
        "sim_weight":       sim_weight,
        "per_classifier":   per_clf,
        "pair_contribution": pair_contrib,
    }


# ════════════════════════════════════════════════════════════════════════════════
# Loop principale per un singolo modello di attacco
# ════════════════════════════════════════════════════════════════════════════════

def attack_model(
    target_name: str,
    all_models:  dict,
    lpips_fn:    nn.Module,
    device:      torch.device,
    img_paths:   list,
) -> float:
    """
    Esegue l'attacco PGD usando `target_name` come modello sorgente.
    Le immagini avversariali vengono salvate in dataset_adv/<target_name>/.
    Dopo ogni immagine aggiorna dataset_adv/<target_name>/results.json.
    Restituisce lo score totale.
    """
    target_model = all_models[target_name]
    preprocess   = _PREPROCESSORS[target_name]

    out_dir = ADV_BASE / target_name
    out_dir.mkdir(parents=True, exist_ok=True)

    records      = []
    total_score  = 0.0

    for img_path in tqdm(img_paths, desc=f"[{target_name}]"):
        # ── carica immagine originale ──────────────────────────────────────
        orig_pil = Image.open(img_path).convert("RGB")
        orig_np  = np.array(orig_pil, dtype=np.uint8)
        orig_t   = (
            torch.from_numpy(orig_np).permute(2, 0, 1).float() / 255.0
        ).unsqueeze(0).to(device)

        # ── genera immagine avversariale ───────────────────────────────────
        with torch.enable_grad():
            adv_t = pgd(target_model, preprocess, orig_t)

        adv_np = (adv_t[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        # ── salva immagine avversariale ────────────────────────────────────
        Image.fromarray(adv_np).save(out_dir / img_path.name)

        # ── valutazione inline (entrambi i classificatori) ─────────────────
        ev    = _eval_pair(orig_np, adv_np, adv_t.detach(), all_models, lpips_fn, device)
        entry = {"image": img_path.name, **ev}
        records.append(entry)
        total_score += ev["pair_contribution"]

        # ── aggiorna JSON con risultati cumulativi ─────────────────────────
        n_fooled = sum(
            1 for r in records
            if r["per_classifier"][target_name]["prediction"] == "Real"
        )
        summary = {
            "attack_model":  target_name,
            "epsilon":       EPSILON,
            "n_steps":       N_STEPS,
            "lambda_ssim":   LAMBDA_SSIM,
            "n_processed":   len(records),
            "n_fooled_by_target": n_fooled,
            "total_score":   total_score,
            "images":        records,
        }
        with open(out_dir / "results.json", "w") as f:
            json.dump(summary, f, indent=2)

    return total_score


# ════════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nCaricamento modelli...")
    all_models = {name: _LOADERS[name](device) for name in CLF_WEIGHTS}

    print("Caricamento LPIPS (AlexNet)...")
    lpips_fn = lpips_module.LPIPS(net="alex").eval().to(device)

    img_paths = sorted(TEST_DIR.glob("*.png"))
    print(f"Immagini di test: {len(img_paths)}\n")

    ADV_BASE.mkdir(parents=True, exist_ok=True)

    total_scores: dict = {}

    for target_name in CLF_WEIGHTS:
        print(f"\n{'='*60}")
        print(f"Attacco con modello sorgente: {target_name}")
        print(f"  ε={EPSILON:.4f}  steps={N_STEPS}  λ_ssim={LAMBDA_SSIM}")
        print(f"  Output: {ADV_BASE / target_name}")
        print('='*60)

        score = attack_model(target_name, all_models, lpips_fn, device, img_paths)
        total_scores[target_name] = score
        print(f"\nScore totale [{target_name}]: {score:.4f}")

    print(f"\n{'='*60}")
    print("RIEPILOGO FINALE")
    print('='*60)
    for name, score in total_scores.items():
        print(f"  {name:<25s}  score = {score:.4f}")
    print(f"\nJSON salvati in:")
    for name in CLF_WEIGHTS:
        print(f"  {ADV_BASE / name / 'results.json'}")


if __name__ == "__main__":
    main()
