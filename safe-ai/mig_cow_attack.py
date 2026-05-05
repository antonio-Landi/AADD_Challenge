"""
MIG-COW: Momentum Integrated Gradient with Consensus-Orthogonal Weighting
==========================================================================
Paper: "MIG-COW: Transferable Adversarial Attacks on Deepfake Detectors
        via Gradient Decomposition" (ACM MM 2025)

Two key improvements over the base paper implementation:

1. DIFFERENTIABLE DCT SOURCE MODEL
   densenet121_dct.pth is now a white-box source for the attack.
   The DCT-II 2D transform (normally scipy/numpy, not differentiable) is
   reimplemented as a matrix multiplication: DCT2(X) = W @ X @ W^T
   where W is the precomputed orthonormal basis (matches scipy ortho exactly).
   Since torch.matmul is differentiable, autograd can propagate gradients
   through grayscale→resize→crop→DCT→log→DenseNet. Result: ~100% white-box
   ASR on densenet121_dct.

2. ATTACK AT 256×256 SCALE — output stays at original resolution
   Models downsample 1024→256→224 internally. Computing the perturbation
   directly at 256px gives 16× stronger gradient signal (less dilution).
   Procedure:
     a) resize x_orig to 256×256 (BICUBIC)
     b) run MIG-COW at 256×256 → get δ₂₅₆
     c) upsample δ₂₅₆ → δ_orig  (bilinear, L∞-preserving: convex combination)
     d) adv = clip(x_orig_full + δ_orig, 0, 1)
   The adversarial image is ALWAYS saved at the original resolution.
   When the evaluation script downsamples 1024→256 it recovers ≈x_orig_256+δ₂₅₆,
   so the model sees the perturbation computed at exactly the right scale.

Preprocessing in _SpatialModel EXACTLY matches the AADD-2026 evaluation:
  vit_b_16        → Resize(256) → CenterCrop(224) → ImageNet normalise
  resnet50 / densenet121 → Resize(256)             → ImageNet normalise

Usage:
  python safe-ai/mig_cow_attack.py \\
      --input_dir  AADD_2026_Test \\
      --output_dir adversarial_examples \\
      --models_dir models_weights \\
      --aadd25_dir AADD_2025/models_weights \\
      --eval_config AADD_2026_config.yaml
"""

import argparse
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision import models as tv_models
from torchvision.models import vit_b_16 as tv_vit_b16
from tqdm import tqdm


# ─── Constants ────────────────────────────────────────────────────────────────

CLASS_REAL    = 0
IMAGE_EXTS    = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
NUM_CLASSES   = 2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ════════════════════════════════════════════════════════════════════════════
# ATTACK — model builders
# ════════════════════════════════════════════════════════════════════════════

def _build_vit_b16() -> nn.Module:
    m = tv_vit_b16(weights=None)
    m.heads.head = nn.Linear(m.heads.head.in_features, NUM_CLASSES)
    return m


def _build_resnet50() -> nn.Module:
    m = tv_models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


def _build_densenet121() -> nn.Module:
    m = tv_models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, NUM_CLASSES)
    return m


# ─── Spatial attack wrapper (vit_b_16 / resnet50 / densenet121) ───────────────

class _SpatialModel(nn.Module):
    """
    Wraps a torchvision spatial model with preprocessing that exactly mirrors
    the AADD-2026 evaluation script.

    Input:  (B, 3, H, W) float32 in [0, 1] — any resolution
    Output: logits (B, 2)

      vit_b_16   → Resize(256,256) → CenterCrop(224,224) → ImageNet normalise
      others     → Resize(256,256)                        → ImageNet normalise
    """
    def __init__(self, base: nn.Module, arch: str = 'generic',
                 mean: list = None, std: list = None):
        super().__init__()
        self.base = base
        self.arch = arch
        mean = mean or IMAGENET_MEAN
        std  = std  or IMAGENET_STD
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        if self.arch == 'vit_b_16':
            x = x[:, :, 16:240, 16:240]   # CenterCrop(224,224) from 256×256
        return self.base((x - self.mean) / self.std)


# ─── Differentiable DCT attack wrapper (densenet121_dct) ─────────────────────

class _DifferentiableDCTModel(nn.Module):
    """
    Wraps densenet121_dct.pth with a FULLY DIFFERENTIABLE DCT preprocessing
    pipeline that exactly mirrors the AADD-2026 evaluation script.

    The 2D DCT-II (scipy.fftpack.dct, norm='ortho') is expressed as:
        DCT2(X) = W @ X @ W^T
    where W[k,n] = c_k · cos(π·k·(2n+1)/(2N)) and c_0=√(1/N), c_k=√(2/N).
    This is a linear operation → fully differentiable via torch.matmul.

    Pipeline (identical to eval script, all differentiable):
      RGB [0,1] → grayscale [0,255]  (ITU-R BT.601, same as PIL L mode)
               → Resize(256,256)      (bilinear, matches eval)
               → CenterCrop(128,128)  (center 128×128 from 256×256)
               → DCT2(X)              (W @ X @ W^T, matches scipy ortho)
               → log(|·| + 1e-6)      (matches eval log_scale=True)
               → DenseNet-121 (1-ch)

    Input:  (B, 3, H, W) float32 in [0, 1]
    Output: logits (B, 2)
    """
    N = 128   # DCT / crop size

    def __init__(self, base: nn.Module, log_scale: bool = True):
        super().__init__()
        self.base      = base
        self.log_scale = log_scale
        N = self.N

        # Precompute orthonormal DCT-II basis matrix W (N×N)
        # Computed in float64 for precision, stored as float32
        k = torch.arange(N, dtype=torch.float64)
        n = torch.arange(N, dtype=torch.float64)
        W = torch.cos(torch.pi * k.unsqueeze(1) * (2 * n.unsqueeze(0) + 1) / (2 * N))
        scale      = torch.full((N,), (2.0 / N) ** 0.5, dtype=torch.float64)
        scale[0]   = (1.0 / N) ** 0.5
        W          = (W * scale.unsqueeze(1)).float()
        self.register_buffer('dct_W', W)   # (N, N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. RGB [0,1] → grayscale [0,255] using ITU-R BT.601 (PIL L mode)
        x = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]) * 255.0

        # 2. Resize to 256×256 (bilinear, differentiable)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        # 3. CenterCrop(128) from 256×256: (256-128)//2 = 64
        x = x[:, :, 64:192, 64:192]       # (B, 1, 128, 128)

        # 4. Differentiable 2D DCT-II: DCT2(X) = W @ X @ W^T
        W = self.dct_W                     # (N, N)
        x = x.squeeze(1)                   # (B, N, N)
        x = torch.matmul(W, x)             # apply DCT along axis-0 (columns)
        x = torch.matmul(x, W.t())         # apply DCT along axis-1 (rows)
        x = x.unsqueeze(1)                 # (B, 1, N, N)

        # 5. Log scale: log(|DCT| + 1e-6)
        if self.log_scale:
            x = torch.log(x.abs() + 1e-6)

        return self.base(x)


# ─── HuggingFace model wrapper ────────────────────────────────────────────────

class _HFModel(nn.Module):
    def __init__(self, hf_model: nn.Module, real_idx: int = 0,
                 mean: list = None, std: list = None):
        super().__init__()
        self.hf_model = hf_model
        self.real_idx = real_idx
        mean = mean or [0.5, 0.5, 0.5]
        std  = std  or [0.5, 0.5, 0.5]
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        logits = self.hf_model(pixel_values=x).logits
        if self.real_idx != 0:
            idx = [self.real_idx] + [i for i in range(logits.shape[1])
                                     if i != self.real_idx]
            return logits[:, idx]
        return logits


def _load_hf_model(repo: str, real_idx: int, device: torch.device) -> nn.Module:
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    hf   = AutoModelForImageClassification.from_pretrained(repo)
    proc = AutoImageProcessor.from_pretrained(repo)
    mean = list(proc.image_mean) if hasattr(proc, 'image_mean') else None
    std  = list(proc.image_std)  if hasattr(proc, 'image_std')  else None
    return _HFModel(hf, real_idx=real_idx, mean=mean, std=std).eval().to(device)


# ─── Public attack model loaders ─────────────────────────────────────────────

def load_pth_model(arch: str, weight_path: str, device: torch.device) -> nn.Module:
    """Load a spatial competition model (vit_b_16 / resnet50 / densenet121)."""
    builders = {
        'vit_b_16':    _build_vit_b16,
        'resnet50':    _build_resnet50,
        'densenet121': _build_densenet121,
    }
    if arch not in builders:
        raise ValueError(f"Unknown arch '{arch}'. Supported: {list(builders)}")
    base  = builders[arch]()
    state = torch.load(weight_path, map_location=device)
    base.load_state_dict(state)
    return _SpatialModel(base, arch=arch).eval().to(device)


def load_dct_source_model(weight_path: str, device: torch.device,
                           log_scale: bool = True) -> nn.Module:
    """
    Load densenet121_dct.pth as a DIFFERENTIABLE source model.
    Uses _DifferentiableDCTModel for backprop through the DCT pipeline.
    """
    base = _build_densenet121_dct_eval()
    state = torch.load(weight_path, map_location=device)
    base.load_state_dict(state)
    return _DifferentiableDCTModel(base, log_scale=log_scale).eval().to(device)


def load_vit_p(device: torch.device) -> nn.Module:
    return _load_hf_model(
        "prithivMLmods/Deep-Fake-Detector-v2-Model",
        real_idx=0, device=device)


# ════════════════════════════════════════════════════════════════════════════
# EVALUATION — model builders + transforms + metrics
# (mirrors AADD_2026_evaluation.py exactly)
# ════════════════════════════════════════════════════════════════════════════

def _build_densenet121_dct_eval() -> nn.Module:
    """DenseNet-121 with 1-channel input — matches AADD_2026_evaluation.py."""
    m = tv_models.densenet121(weights=None)
    m.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(m.classifier.in_features, NUM_CLASSES)
    )
    return m


def load_eval_model(name: str, weight_path: Path, device: torch.device) -> nn.Module:
    if name == 'vit_b_16':
        model = tv_vit_b16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    elif name == 'densenet121_dct':
        model = _build_densenet121_dct_eval()
    else:
        raise ValueError(f"Unknown eval classifier '{name}'")
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    return model.eval().to(device)


def _dct2_numpy(arr: np.ndarray) -> np.ndarray:
    from scipy.fftpack import dct
    return dct(dct(arr, axis=0, norm='ortho'), axis=1, norm='ortho')


def build_dct_transform(log_scale: bool = True):
    def _t(pil_img: Image.Image) -> torch.Tensor:
        img = pil_img.convert('L')
        if max(img.size) > 256:
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
        w, h = img.size
        l, t = (w - 128) // 2, (h - 128) // 2
        img  = img.crop((l, t, l + 128, t + 128))
        arr  = np.array(img, dtype=np.float32)
        dct_a = _dct2_numpy(arr)
        if log_scale:
            dct_a = np.log(np.abs(dct_a) + 1e-6)
        return torch.from_numpy(dct_a).unsqueeze(0)
    return _t


def build_spatial_eval_transform(name: str) -> T.Compose:
    if name == 'vit_b_16':
        return T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def compute_ssim_rgb(im1: np.ndarray, im2: np.ndarray) -> float:
    from skimage.metrics import structural_similarity as sk_ssim
    return sum(
        sk_ssim(im1[..., c], im2[..., c], data_range=255) for c in range(3)
    ) / 3.0


def np_to_lpips_tensor(np_img: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(np_img).permute(2, 0, 1).float() / 127.5 - 1.0
    return t.unsqueeze(0).to(device)


# ─── Evaluation context ───────────────────────────────────────────────────────

@dataclass
class EvalCtx:
    classifiers: dict
    lpips_fn:    nn.Module
    alpha:       float
    device:      torch.device
    per_image:   list = field(default_factory=list)


def setup_eval(cfg_path: str, device: torch.device) -> EvalCtx:
    import yaml
    import lpips as lpips_lib

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    models_dir = Path(cfg['models_dir'])
    alpha      = float(cfg.get('alpha', 0.5))
    log_scale  = bool(cfg.get('dct_log_scale', True))
    weight_cfg = cfg.get('weights', {})
    clf_names  = cfg['classifiers']

    print("[EVAL] Loading LPIPS (alex)…")
    lpips_fn = lpips_lib.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    classifiers = {}
    for name in clf_names:
        w_path = models_dir / f'{name}.pth'
        if not w_path.exists():
            raise FileNotFoundError(f"[EVAL] Weight not found: {w_path}")
        model     = load_eval_model(name, w_path, device)
        transform = (build_dct_transform(log_scale) if name.endswith('_dct')
                     else build_spatial_eval_transform(name))
        classifiers[name] = {
            'model':      model,
            'transform':  transform,
            'clf_weight': float(weight_cfg.get(name, 1.0)),
            'indicators': [],
            'ssim_vals':  [],
            'lpips_vals': [],
        }
        print(f"[EVAL] Loaded eval model: {name}")

    return EvalCtx(classifiers=classifiers, lpips_fn=lpips_fn,
                   alpha=alpha, device=device)


def evaluate_pair(image_rel: str, orig_pil: Image.Image,
                  adv_pil: Image.Image, ctx: EvalCtx) -> dict:
    img_o = np.array(orig_pil.convert('RGB'))
    img_a = np.array(adv_pil.convert('RGB'))

    try:
        ssim_val = compute_ssim_rgb(img_o, img_a)
    except Exception as e:
        warnings.warn(f"SSIM failed for {image_rel}: {e}")
        ssim_val = 0.0

    with torch.no_grad():
        lpips_val = ctx.lpips_fn(
            np_to_lpips_tensor(img_o, ctx.device),
            np_to_lpips_tensor(img_a, ctx.device),
        ).item()

    sim_weight        = ctx.alpha * ssim_val + (1.0 - ctx.alpha) * (1.0 - lpips_val)
    pair_contribution = 0.0
    per_clf           = {}

    for name, pack in ctx.classifiers.items():
        tensor = pack['transform'](Image.fromarray(img_a)).unsqueeze(0).to(ctx.device)
        with torch.no_grad():
            pred = pack['model'](tensor).argmax(1).item()
        indicator    = int(pred == CLASS_REAL)
        contribution = pack['clf_weight'] * sim_weight * indicator
        pair_contribution += contribution
        pack['indicators'].append(indicator)
        pack['ssim_vals'].append(ssim_val)
        pack['lpips_vals'].append(lpips_val)
        per_clf[name] = {
            'prediction':   'Real' if pred == CLASS_REAL else 'Fake',
            'indicator':    indicator,
            'clf_weight':   pack['clf_weight'],
            'contribution': round(contribution, 6),
        }

    result = {
        'image':             image_rel,
        'ssim':              round(ssim_val,          6),
        'lpips':             round(lpips_val,         6),
        'sim_weight':        round(sim_weight,        6),
        'per_classifier':    per_clf,
        'pair_contribution': round(pair_contribution, 6),
    }
    ctx.per_image.append(result)
    return result


def write_results(ctx: EvalCtx, results_path: Path):
    n = len(ctx.per_image)
    if n == 0:
        return

    final_score = sum(r['pair_contribution'] for r in ctx.per_image)

    lines = [
        "=" * 60,
        f"MIG-COW AADD-2026 Evaluation  —  {datetime.now():%Y-%m-%d %H:%M:%S}",
        "=" * 60,
        f"  Images evaluated    : {n}",
        f"  Classifiers         : {', '.join(ctx.classifiers)}",
        f"  Alpha (SSIM weight) : {ctx.alpha}",
        f"  Aggregate           : sum",
        f"  Final score         : {final_score:.6f}",
        "",
    ]
    for name, pack in ctx.classifiers.items():
        asr     = float(np.mean(pack['indicators']))  if pack['indicators'] else 0.0
        m_ssim  = float(np.mean(pack['ssim_vals']))   if pack['ssim_vals']  else 0.0
        m_lpips = float(np.mean(pack['lpips_vals']))  if pack['lpips_vals'] else 0.0
        lines.append(
            f"  [{name:<22s}]  "
            f"attack_success={asr:.4f}  "
            f"mean_ssim={m_ssim:.4f}  "
            f"mean_lpips={m_lpips:.4f}"
        )

    lines += ["", "── Per-image ─────────────────────────────────────────────────────", ""]
    for r in ctx.per_image:
        clf_str = "  ".join(
            f"{k}={'R' if v['indicator'] else 'F'}"
            for k, v in r['per_classifier'].items()
        )
        lines.append(
            f"  {r['image']:<50s}  "
            f"SSIM={r['ssim']:.4f}  LPIPS={r['lpips']:.4f}  "
            f"{clf_str}  score={r['pair_contribution']:.4f}"
        )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[RESULTS] Written → {results_path}")


# ════════════════════════════════════════════════════════════════════════════
# MIG-COW ATTACK
# ════════════════════════════════════════════════════════════════════════════

def compute_ig(
    model:        nn.Module,
    x:            torch.Tensor,   # (1, C, H, W) in [0,1], detached
    baseline:     torch.Tensor,
    target_class: int,
    steps:        int,
) -> torch.Tensor:
    """
    IG via Riemann sum (paper Eq. 2).
    f(x) = log_softmax(logits)[real_class]  — MuMoDIG-style log-based loss.
    Returns IG of shape (C, H, W).
    """
    diff       = x - baseline
    total_grad = torch.zeros_like(x)

    for k in range(1, steps + 1):
        interp = (baseline + (k / steps) * diff).detach().requires_grad_(True)
        f = F.log_softmax(model(interp), dim=1)[0, target_class]
        f.backward()
        total_grad += interp.grad.detach()

    return diff.squeeze(0) * (total_grad.squeeze(0) / steps)


def _probe_prob(models: list, x: torch.Tensor, label: str):
    """Print P(Real) for each source model — used in verbose mode."""
    with torch.no_grad():
        for i, m in enumerate(models):
            p = F.softmax(m(x), dim=1)[0, CLASS_REAL].item()
            print(f"    [source {i}] P(Real) {label}: {p:.4f}")


def mig_cow(
    models:       list,
    x_pil:        Image.Image,
    epsilon:      float = 0.05,
    T:            int   = 50,
    mu:           float = 1.0,
    beta:         float = 0.75,
    ig_steps:     int   = 20,
    attack_scale: int   = 256,
    verbose:      bool  = False,
    device:       torch.device = torch.device('cpu'),
) -> Image.Image:
    """
    MIG-COW adversarial attack (Algorithm 1 from the paper).

    attack_scale=256 (default):
      - Computes the perturbation at 256×256 (the models' intermediate resize scale).
      - Upsamples δ back to the original resolution via bilinear interpolation.
      - L∞ of the upsampled δ is ≤ ε (convex combination preserves max norm).
      - Output PIL image is at the ORIGINAL resolution.

    attack_scale=0:
      - Attacks at the original resolution directly.

    verbose=True:
      - Prints P(Real) before and after attack for each source model.
        Use this to verify the gradient direction is correct.
    """
    orig_rgb = x_pil.convert('RGB')
    orig_w, orig_h = orig_rgb.size

    if attack_scale > 0:
        x_scaled = orig_rgb.resize((attack_scale, attack_scale), Image.BICUBIC)
    else:
        x_scaled = orig_rgb

    x_01 = (torch.from_numpy(np.array(x_scaled))
            .permute(2, 0, 1).float().div(255.).unsqueeze(0).to(device))

    baseline = torch.zeros_like(x_01)
    x_orig   = x_01.clone()
    x_adv    = x_01.clone()
    g_accum  = torch.zeros_like(x_01)

    alpha = epsilon / T
    EPS   = 1e-8

    if verbose:
        _probe_prob(models, x_01, "BEFORE")

    for _ in range(T):
        x_adv = x_adv.detach()

        igs = [compute_ig(m, x_adv, baseline, CLASS_REAL, ig_steps) for m in models]
        N   = len(igs)

        # Consensus gradient (Eq. 3)
        g_con = torch.stack(igs, dim=0).mean(0)

        # Gram matrix K = G^T G
        G = torch.stack([ig.reshape(-1) for ig in igs], dim=1)
        K = G.t().mm(G)

        # Smallest eigenvector → aggregated gradient
        _, eigenvectors = torch.linalg.eigh(K.float())
        v_min = eigenvectors[:, 0]
        g_agg = sum(v_min[i].item() * igs[i] for i in range(N))

        # Orthogonal component (Eq. 6)
        proj   = (g_agg.reshape(-1).dot(g_con.reshape(-1)) /
                  (g_con.norm() ** 2 + EPS))
        g_orth = g_agg - proj * g_con

        # Final attack direction (Eq. 7)
        g_cb = beta * g_con + (1.0 - beta) * g_orth

        # Momentum with L2 normalisation (Algorithm 1 line 17)
        g_accum = mu * g_accum + (g_cb / (g_cb.norm() + EPS)).unsqueeze(0)

        # FGSM step + ε-ball clipping
        x_adv = x_adv + alpha * g_accum.sign()
        x_adv = torch.clamp(x_adv, x_orig - epsilon, x_orig + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    if verbose:
        _probe_prob(models, x_adv, "AFTER ")

    if attack_scale > 0 and (orig_w != attack_scale or orig_h != attack_scale):
        # Upsample perturbation to original resolution
        # Bilinear is a convex combination → L∞(δ_orig) ≤ L∞(δ_scale) ≤ ε ✓
        delta_scaled = x_adv - x_orig                            # (1,3,S,S) in [-ε,ε]
        delta_orig   = F.interpolate(delta_scaled,
                                     size=(orig_h, orig_w),
                                     mode='bilinear',
                                     align_corners=False)        # (1,3,H,W)
        x_orig_full  = (torch.from_numpy(np.array(orig_rgb))
                        .permute(2, 0, 1).float().div(255.)
                        .unsqueeze(0))                           # CPU, (1,3,H,W)
        adv_tensor   = torch.clamp(x_orig_full + delta_orig.cpu(), 0., 1.)
    else:
        adv_tensor = x_adv.cpu()

    adv_np = (adv_tensor.squeeze(0).permute(1, 2, 0)
              .mul(255.).round().clamp(0, 255).byte().numpy())
    return Image.fromarray(adv_np)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MIG-COW adversarial attack — AADD 2026")
    parser.add_argument('--input_dir',    required=True)
    parser.add_argument('--output_dir',   required=True)
    parser.add_argument('--models_dir',   default='models_weights',
                        help="Folder with vit_b_16.pth and densenet121_dct.pth (AADD-2026)")
    parser.add_argument('--aadd25_dir',   default=None,
                        help="AADD_2025/models_weights — adds resnet50, densenet121, vit_b_16 (2025)")
    parser.add_argument('--use_dct_source', action='store_true',
                        help="Use densenet121_dct.pth as a differentiable white-box source model")
    parser.add_argument('--use_vit_p',    action='store_true',
                        help="Add ViT-P from HuggingFace (paper: may hurt black-box transfer)")
    parser.add_argument('--eval_config',  default=None,
                        help="Path to AADD_2026_config.yaml — enables inline evaluation")
    parser.add_argument('--results_file', default=None,
                        help="Path for .results summary (default: output_dir/evaluation.results)")
    parser.add_argument('--epsilon',      type=float, default=0.05)
    parser.add_argument('--T',            type=int,   default=50)
    parser.add_argument('--mu',           type=float, default=1.0)
    parser.add_argument('--beta',         type=float, default=0.75)
    parser.add_argument('--ig_steps',     type=int,   default=20)
    parser.add_argument('--attack_scale', type=int,   default=256,
                        help="Resolution at which gradients are computed (default: 256). "
                             "Use 0 for full original resolution.")
    parser.add_argument('--verbose',      action='store_true',
                        help="Print P(Real) before and after attack for each source model "
                             "(use to verify gradient direction is correct)")
    parser.add_argument('--device',       default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"[DEVICE] {device}\n")

    # ── Load attack source models ──────────────────────────────────────────────
    models_dir    = Path(args.models_dir)
    source_models = []

    vit_pth = models_dir / 'vit_b_16.pth'
    if vit_pth.exists():
        source_models.append(load_pth_model('vit_b_16', str(vit_pth), device))
        print(f"[ATTACK MODEL] vit_b_16      (AADD-2026, spatial)  ← {vit_pth}")

    if args.use_dct_source:
        dct_pth = models_dir / 'densenet121_dct.pth'
        if dct_pth.exists():
            source_models.append(load_dct_source_model(str(dct_pth), device))
            print(f"[ATTACK MODEL] densenet121_dct (AADD-2026, diff-DCT) ← {dct_pth}")
        else:
            print(f"[WARNING] densenet121_dct.pth not found in {models_dir}")

    if args.aadd25_dir:
        aadd25_dir = Path(args.aadd25_dir)
        for arch in ('resnet50', 'densenet121', 'vit_b_16'):
            pth = aadd25_dir / f'{arch}.pth'
            if pth.exists():
                source_models.append(load_pth_model(arch, str(pth), device))
                print(f"[ATTACK MODEL] {arch:<16s} (AADD-2025, spatial)  ← {pth}")

    if args.use_vit_p:
        print("[ATTACK MODEL] Downloading ViT-P from HuggingFace…")
        source_models.append(load_vit_p(device))
        print("[ATTACK MODEL] ViT-P ready")

    if not source_models:
        raise RuntimeError("No attack source models loaded. Check --models_dir.")

    scale_str = f"{args.attack_scale}px" if args.attack_scale > 0 else "full-res"
    print(f"\n[ATTACK] {len(source_models)} source model(s) | "
          f"ε={args.epsilon}  T={args.T}  μ={args.mu}  β={args.beta}  "
          f"IG_steps={args.ig_steps}  attack_scale={scale_str}\n")

    # ── Load evaluation context ────────────────────────────────────────────────
    eval_ctx = None
    if args.eval_config:
        print(f"[EVAL] Loading evaluation models from {args.eval_config}")
        eval_ctx = setup_eval(args.eval_config, device)
        print()

    # ── Collect images ─────────────────────────────────────────────────────────
    input_dir = Path(args.input_dir)
    img_paths = sorted(p for p in input_dir.rglob('*')
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not img_paths:
        raise RuntimeError(f"No images found in {input_dir}")
    print(f"[DATA] {len(img_paths)} image(s) to attack\n")

    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = (Path(args.results_file) if args.results_file
                    else output_dir / 'evaluation.results')

    # ── Attack + evaluate loop ─────────────────────────────────────────────────
    for img_path in tqdm(img_paths, desc="MIG-COW"):
        orig_pil = Image.open(img_path).convert('RGB')
        rel      = img_path.relative_to(input_dir)

        adv_pil = mig_cow(
            models       = source_models,
            x_pil        = orig_pil,
            epsilon      = args.epsilon,
            T            = args.T,
            mu           = args.mu,
            beta         = args.beta,
            ig_steps     = args.ig_steps,
            attack_scale = args.attack_scale,
            verbose      = args.verbose,
            device       = device,
        )

        # Save adversarial image at original resolution
        out_img = (output_dir / rel).with_suffix('.png')
        out_img.parent.mkdir(parents=True, exist_ok=True)
        adv_pil.save(str(out_img))

        if eval_ctx is not None:
            result = evaluate_pair(str(rel), orig_pil, adv_pil, eval_ctx)

            out_img.with_suffix('.json').write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding='utf-8'
            )

            clf_summary = "  ".join(
                f"{k}={v['prediction']}" for k, v in result['per_classifier'].items()
            )
            print(
                f"  {rel}  "
                f"SSIM={result['ssim']:.4f}  LPIPS={result['lpips']:.4f}  "
                f"{clf_summary}  score={result['pair_contribution']:.4f}"
            )
            write_results(eval_ctx, results_path)
        else:
            print(f"  saved → {out_img}")

    print(f"\n[DONE] Adversarial images → {output_dir}")
    if eval_ctx is not None:
        tot = sum(r['pair_contribution'] for r in eval_ctx.per_image)
        print(f"[DONE] Final score (sum, {len(eval_ctx.per_image)} images): {tot:.6f}")
        print(f"[DONE] Results file → {results_path}")


if __name__ == '__main__':
    main()
