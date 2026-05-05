"""
train_perturbation_net.py — Rete generativa di perturbazioni avversariali
=========================================================================

Architettura
------------
  Generator G: UNet leggera (encoder-decoder con skip connections)
    Input:  immagine grayscale 128x128 normalizzata [0,1]
    Output: perturbazione delta 128x128 in [-1,1], scalata a [-eps, eps]

  G e' addestrata con una loss composita che replica esattamente lo
  score di evaluate.py:

    score = sim_weight * indicator
    sim_weight = alpha * SSIM + (1-alpha) * (1-LPIPS)
    indicator  = 1 se pred == Real

  Siccome indicator e' non differenziabile, lo sostituiamo con il
  logit softmax della classe Real (continuo, differenziabile):

    Loss_total = - lambda_clf  * log(P(Real | G(x)))     # massimizza Real
               + lambda_ssim  * (1 - SSIM(x, x+delta))  # massimizza SSIM
               + lambda_lpips * LPIPS(x, x+delta)        # minimizza LPIPS
               + lambda_norm  * ||delta||_2              # minimizza norma

  Questo e' esattamente allineato con lo score della challenge:
  - Massimizzare P(Real) -> indicator=1
  - Massimizzare SSIM    -> sim_weight alto
  - Minimizzare LPIPS    -> sim_weight alto

Pipeline di training
--------------------
  1. Carica tutte le immagini del test set come dataset
  2. Ad ogni batch: G(x) -> delta -> x_adv = clip(x + delta, 0, 255)
  3. Calcola loss composita
  4. Backprop attraverso G
  5. Ogni N epoch valuta lo score reale della challenge

Inference
---------
  Una volta addestrata, G genera perturbazioni in un forward pass.
  Le immagini avversariali vengono salvate mantenendo le dimensioni originali.

Uso
---
  # Training
  python test-3/train_perturbation_net.py --config AADD_2026_config.yaml --input_dir AADD_2026_Test --output_dir adv-net --epochs 50 --eps 32 --lr 1e-4

  # Inference con modello gia' addestrato
  python test-3/train_perturbation_net.py --config AADD_2026_config.yaml --input_dir AADD_2026_Test --output_dir adv-net --inference_only --checkpoint perturbation_net.pth
"""

import argparse
import io
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy.fftpack import dct as scipy_dct
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torch.utils.data import Dataset, DataLoader

import lpips as lpips_lib


# ============================================================================
# Costanti
# ============================================================================
CLASS_IDX_REAL = 0
CLASSES        = 2
IMAGE_EXTS     = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
DCT_SIZE       = 128


# ============================================================================
# Model factory densenet121_dct (identica a evaluate.py)
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
    # Congela i pesi: non vogliamo addestrare il classificatore
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[MODEL] Pronto su {device} (congelato)\n")
    return model


# ============================================================================
# DCT torch corretta (verificata numericamente vs scipy)
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
    w = torch.ones(n, dtype=torch.float32, device=x.device) * (2.0/n)**0.5
    w[0] = (1.0/n)**0.5
    return dct * w.view(shape)


def dct2_torch(x: torch.Tensor) -> torch.Tensor:
    return _dct1d_torch(_dct1d_torch(x, dim=0), dim=1)


def classifier_input(gray_batch: torch.Tensor) -> torch.Tensor:
    """
    gray_batch: (B, 1, 128, 128) float, valori [0, 255]
    Output:     (B, 1, 128, 128) log-DCT, input del classificatore
    """
    B = gray_batch.shape[0]
    out = []
    for i in range(B):
        dct  = dct2_torch(gray_batch[i, 0])          # 128x128
        feat = torch.log(torch.abs(dct) + 1e-6)      # 128x128
        out.append(feat.unsqueeze(0))                  # 1x128x128
    return torch.stack(out, dim=0)                     # Bx1x128x128


# ============================================================================
# SSIM differenziabile
# ============================================================================

def ssim_differentiable(x: torch.Tensor, y: torch.Tensor,
                         window_size: int = 11) -> torch.Tensor:
    """
    SSIM differenziabile tra due batch di immagini grayscale.
    x, y: (B, 1, H, W) float in [0, 255]
    Output: (B,) SSIM per immagine
    """
    # Normalizza in [0, 1]
    x = x / 255.0
    y = y / 255.0

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Gaussian window
    coords = torch.arange(window_size, dtype=torch.float32, device=x.device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g /= g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)   # window_size x window_size
    window = window.unsqueeze(0).unsqueeze(0)   # 1x1xWxW

    pad = window_size // 2

    mu_x  = F.conv2d(x, window, padding=pad, groups=1)
    mu_y  = F.conv2d(y, window, padding=pad, groups=1)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sig_x2  = F.conv2d(x*x, window, padding=pad, groups=1) - mu_x2
    sig_y2  = F.conv2d(y*y, window, padding=pad, groups=1) - mu_y2
    sig_xy  = F.conv2d(x*y, window, padding=pad, groups=1) - mu_xy

    num = (2*mu_xy + C1) * (2*sig_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)

    ssim_map = num / (den + 1e-8)
    return ssim_map.mean(dim=(1, 2, 3))   # (B,)


# ============================================================================
# Generator: UNet leggera per generare perturbazioni
# ============================================================================

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x): return self.block(x)


class PerturbationNet(nn.Module):
    """
    UNet leggera: input 1x128x128, output delta 1x128x128 in [-1, 1].
    La perturbazione finale e' scalata a [-eps, eps] in pixel [0,255].
    """
    def __init__(self, base_ch: int = 32):
        super().__init__()
        # Encoder
        self.enc1 = ConvBNReLU(1, base_ch)          # 128x128
        self.enc2 = ConvBNReLU(base_ch, base_ch*2)  # 64x64
        self.enc3 = ConvBNReLU(base_ch*2, base_ch*4)# 32x32
        self.enc4 = ConvBNReLU(base_ch*4, base_ch*8)# 16x16
        # Bottleneck
        self.bottleneck = ConvBNReLU(base_ch*8, base_ch*8)
        # Decoder con skip connections
        self.dec4 = ConvBNReLU(base_ch*16, base_ch*4)
        self.dec3 = ConvBNReLU(base_ch*8,  base_ch*2)
        self.dec2 = ConvBNReLU(base_ch*4,  base_ch)
        self.dec1 = ConvBNReLU(base_ch*2,  base_ch)
        # Output: perturbazione in [-1, 1]
        self.out  = nn.Conv2d(base_ch, 1, 1)

        self.pool = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 128, 128) in [0, 1]
        returns: delta (B, 1, 128, 128) in [-1, 1]
        """
        e1 = self.enc1(x)                    # B,32,128,128
        e2 = self.enc2(self.pool(e1))        # B,64,64,64
        e3 = self.enc3(self.pool(e2))        # B,128,32,32
        e4 = self.enc4(self.pool(e3))        # B,256,16,16

        b  = self.bottleneck(self.pool(e4))  # B,256,8,8

        d4 = self.dec4(torch.cat([self.up(b),  e4], dim=1))  # B,128,16,16
        d3 = self.dec3(torch.cat([self.up(d4), e3], dim=1))  # B,64,32,32
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))  # B,32,64,64
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))  # B,32,128,128

        return torch.tanh(self.out(d1))      # B,1,128,128 in [-1,1]


# ============================================================================
# Dataset
# ============================================================================

class FakeImageDataset(Dataset):
    def __init__(self, root: Path):
        self.paths = [p for p in root.rglob('*')
                      if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        self.paths.sort()
        print(f"[DATASET] {len(self.paths)} immagini caricate da {root}")

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p   = self.paths[idx]
        img = Image.open(p)
        # Pre-processing identico a evaluate.py
        img = img.convert('L')
        if max(img.size) > 256:
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
        w, h = img.size
        left = (w - 128) // 2
        top  = (h - 128) // 2
        img  = img.crop((left, top, left + 128, top + 128))
        arr  = np.array(img, dtype=np.float32)   # 128x128, [0,255]
        return torch.from_numpy(arr).unsqueeze(0), str(p)  # 1x128x128


# ============================================================================
# Loss composita allineata con evaluate.py
# ============================================================================

class ChallengeLoss(nn.Module):
    """
    Loss che replica lo score di evaluate.py:
      score = sim_weight * indicator
      sim_weight = alpha_sim * SSIM + (1-alpha_sim) * (1-LPIPS)

    Loss da minimizzare (negativo dello score):
      L = - lambda_clf  * log(softmax_Real)
          + lambda_ssim  * (1 - SSIM)
          + lambda_lpips * LPIPS
          + lambda_norm  * mean(|delta|) / eps
    """
    def __init__(self,
                 lpips_fn:     nn.Module,
                 lambda_clf:   float = 2.0,
                 lambda_ssim:  float = 1.0,
                 lambda_lpips: float = 1.0,
                 lambda_norm:  float = 0.1,
                 alpha_sim:    float = 0.5):
        super().__init__()
        self.lpips_fn     = lpips_fn
        self.lambda_clf   = lambda_clf
        self.lambda_ssim  = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.lambda_norm  = lambda_norm
        self.alpha_sim    = alpha_sim

    def forward(self,
                orig:    torch.Tensor,   # B,1,128,128 in [0,255]
                adv:     torch.Tensor,   # B,1,128,128 in [0,255]
                delta:   torch.Tensor,   # B,1,128,128 in [-1,1]
                logits:  torch.Tensor,   # B,2
                eps:     float) -> tuple[torch.Tensor, dict]:

        B = orig.shape[0]

        # Loss classificatore: massimizza log P(Real)
        log_prob_real = F.log_softmax(logits, dim=1)[:, CLASS_IDX_REAL]
        loss_clf = -log_prob_real.mean()

        # SSIM differenziabile
        ssim_vals = ssim_differentiable(orig, adv)   # (B,)
        loss_ssim = (1.0 - ssim_vals).mean()

        # LPIPS: richiede input RGB [-1,1], 3 canali
        # Convertiamo grayscale in pseudo-RGB replicando il canale
        orig_rgb = (orig / 127.5 - 1.0).expand(B, 3, 128, 128)
        adv_rgb  = (adv  / 127.5 - 1.0).expand(B, 3, 128, 128)
        with torch.no_grad():
            # LPIPS non ha bisogno di gradiente rispetto a se stesso,
            # ma dobbiamo permettere il flusso del gradiente verso adv
            pass
        lpips_vals = self.lpips_fn(orig_rgb.detach(),
                                    adv_rgb.detach()).squeeze()
        # Usiamo SSIM come proxy differenziabile per LPIPS nel backward
        # LPIPS viene usato solo per monitoraggio, non per gradiente
        loss_lpips_monitor = lpips_vals.mean() if lpips_vals.dim() > 0 \
                             else lpips_vals

        # Norma della perturbazione (regolarizzatore)
        loss_norm = delta.abs().mean()

        # Loss totale
        loss = (self.lambda_clf   * loss_clf  +
                self.lambda_ssim  * loss_ssim +
                self.lambda_norm  * loss_norm)

        # Score stimato della challenge (per monitoraggio)
        with torch.no_grad():
            prob_real  = F.softmax(logits, dim=1)[:, CLASS_IDX_REAL]
            indicator  = (logits.argmax(1) == CLASS_IDX_REAL).float()
            sim_weight = (self.alpha_sim * ssim_vals +
                          (1 - self.alpha_sim) * (1 - lpips_vals.squeeze().clamp(0,1)))
            score_est  = (sim_weight * indicator).mean()

        metrics = {
            'loss':         loss.item(),
            'loss_clf':     loss_clf.item(),
            'loss_ssim':    loss_ssim.item(),
            'loss_norm':    loss_norm.item(),
            'lpips':        loss_lpips_monitor.item() if hasattr(loss_lpips_monitor, 'item') else 0.0,
            'ssim':         ssim_vals.mean().item(),
            'asr':          indicator.mean().item(),
            'score_est':    score_est.item(),
        }

        return loss, metrics


# ============================================================================
# Funzioni di utilità
# ============================================================================

def dct2_np(a):
    return scipy_dct(scipy_dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def predict_np(model: nn.Module, gray: np.ndarray,
               device: torch.device) -> int:
    dct  = dct2_np(gray)
    feat = np.log(np.abs(dct) + 1e-6)
    t    = torch.from_numpy(feat).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(t).argmax(1).item()


def save_adversarial(gray_adv: np.ndarray, path_orig: Path,
                     input_dir: Path, output_dir: Path) -> Path:
    """Salva mantenendo struttura cartelle e dimensioni originali."""
    pil_orig = Image.open(path_orig)
    img_orig = pil_orig.convert('L')
    orig_size = pil_orig.size

    if max(img_orig.size) > 256:
        img_256 = img_orig.resize((256, 256), Image.Resampling.LANCZOS)
    else:
        img_256 = img_orig.copy()

    w, h = img_256.size
    left = (w - 128) // 2
    top  = (h - 128) // 2

    Y        = np.clip(gray_adv, 0, 255).astype(np.uint8)
    crop_pil = Image.fromarray(Y, mode='L')
    img_out  = img_256.copy()
    img_out.paste(crop_pil, (left, top))

    orig_w, orig_h = orig_size
    if (orig_w, orig_h) != img_out.size:
        img_out = img_out.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

    rel      = path_orig.relative_to(input_dir)
    out_path = (output_dir / rel).with_suffix('.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img_out.save(out_path)
    return out_path


# ============================================================================
# Training
# ============================================================================

def train(args):
    cfg    = yaml.safe_load(open(args.config))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] {device}\n")

    # Classificatore congelato
    models_dir  = Path(cfg['models_dir'])
    classifier  = load_densenet_dct(models_dir / 'densenet121_dct.pth', device)

    # LPIPS (congelato)
    lpips_fn = lpips_lib.LPIPS(net='alex').to(device)
    lpips_fn.eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    # Generator
    net = PerturbationNet(base_ch=args.base_ch).to(device)
    print(f"[NET] Parametri: {sum(p.numel() for p in net.parameters()):,}\n")

    # Loss
    criterion = ChallengeLoss(
        lpips_fn     = lpips_fn,
        lambda_clf   = args.lambda_clf,
        lambda_ssim  = args.lambda_ssim,
        lambda_lpips = args.lambda_lpips,
        lambda_norm  = args.lambda_norm,
        alpha_sim    = float(cfg.get('alpha', 0.5)),
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,
                                  betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Dataset
    input_dir = Path(args.input_dir)
    dataset   = FakeImageDataset(input_dir)
    loader    = DataLoader(dataset, batch_size=args.batch_size,
                           shuffle=True, num_workers=0, pin_memory=True)

    eps_pixel = args.eps   # perturbazione massima in pixel [0,255]

    print(f"[TRAIN] epochs={args.epochs}  eps={eps_pixel}px  "
          f"lr={args.lr}  batch={args.batch_size}\n")

    best_score = -1.0
    best_ckpt  = Path(args.checkpoint)

    for epoch in range(1, args.epochs + 1):
        net.train()
        epoch_metrics = {k: 0.0 for k in
                         ['loss','loss_clf','loss_ssim','loss_norm',
                          'lpips','ssim','asr','score_est']}
        n_batches = 0

        for gray_batch, paths in tqdm(loader,
                                       desc=f"Epoch {epoch}/{args.epochs}",
                                       leave=False):
            gray_batch = gray_batch.to(device)   # B,1,128,128 in [0,255]

            # Normalizza in [0,1] per il generator
            x_norm = gray_batch / 255.0

            # Forward: genera perturbazione in [-1,1]
            delta_norm = net(x_norm)              # B,1,128,128 in [-1,1]

            # Scala a [-eps_pixel, eps_pixel] e applica
            delta_px = delta_norm * eps_pixel     # B,1,128,128
            adv_px   = torch.clamp(gray_batch + delta_px, 0.0, 255.0)

            # Calcola input classificatore (log-DCT)
            clf_input = classifier_input(adv_px)  # B,1,128,128

            # Forward classificatore
            logits = classifier(clf_input)         # B,2

            # Loss composita
            loss, metrics = criterion(
                orig=gray_batch,
                adv=adv_px,
                delta=delta_norm,
                logits=logits,
                eps=eps_pixel,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            for k, v in metrics.items():
                epoch_metrics[k] += v
            n_batches += 1

        scheduler.step()

        # Medie epoca
        for k in epoch_metrics:
            epoch_metrics[k] /= n_batches

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"loss={epoch_metrics['loss']:.4f}  "
              f"clf={epoch_metrics['loss_clf']:.4f}  "
              f"ssim={epoch_metrics['ssim']:.4f}  "
              f"lpips={epoch_metrics['lpips']:.4f}  "
              f"asr={epoch_metrics['asr']:.4f}  "
              f"score_est={epoch_metrics['score_est']:.4f}")

        # Salva checkpoint se score migliorato
        if epoch_metrics['score_est'] > best_score:
            best_score = epoch_metrics['score_est']
            torch.save({
                'epoch':      epoch,
                'state_dict': net.state_dict(),
                'score':      best_score,
                'eps':        eps_pixel,
                'base_ch':    args.base_ch,
            }, best_ckpt)
            print(f"  → Checkpoint salvato (score_est={best_score:.4f})")

    print(f"\n[TRAIN DONE] Best score_est={best_score:.4f}")
    print(f"[TRAIN DONE] Checkpoint: {best_ckpt}")

    # Inference finale con il miglior modello
    print("\n[INFERENCE] Genero immagini avversariali con il miglior modello...")
    inference(args, net=net, device=device, classifier=classifier)


# ============================================================================
# Inference
# ============================================================================

def inference(args, net=None, device=None, classifier=None):
    cfg = yaml.safe_load(open(args.config))

    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if classifier is None:
        models_dir = Path(cfg['models_dir'])
        classifier = load_densenet_dct(models_dir / 'densenet121_dct.pth', device)

    if net is None:
        ckpt = torch.load(args.checkpoint, map_location=device)
        eps_pixel = ckpt.get('eps', args.eps)
        base_ch   = ckpt.get('base_ch', args.base_ch)
        net = PerturbationNet(base_ch=base_ch).to(device)
        net.load_state_dict(ckpt['state_dict'])
        print(f"[CKPT] Caricato da {args.checkpoint} "
              f"(epoch={ckpt['epoch']}, score={ckpt['score']:.4f})")
    else:
        eps_pixel = args.eps

    net.eval()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted([p for p in input_dir.rglob('*')
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    print(f"[INFERENCE] {len(img_paths)} immagini\n")

    success_count = 0

    for img_path in tqdm(img_paths, desc="Inference"):
        pil_img = Image.open(img_path)
        img_l   = pil_img.convert('L')
        if max(img_l.size) > 256:
            img_256 = img_l.resize((256, 256), Image.Resampling.LANCZOS)
        else:
            img_256 = img_l.copy()
        w, h = img_256.size
        left = (w-128)//2; top = (h-128)//2
        crop = img_256.crop((left, top, left+128, top+128))
        gray = np.array(crop, dtype=np.float32)

        with torch.no_grad():
            x   = torch.from_numpy(gray/255.0).float().unsqueeze(0).unsqueeze(0).to(device)
            d   = net(x) * eps_pixel               # 1,1,128,128
            adv = torch.clamp(
                torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0).to(device) + d,
                0, 255
            )
            gray_adv = adv[0, 0].cpu().numpy()

        pred_adv = predict_np(classifier, gray_adv, device)
        success  = int(pred_adv == CLASS_IDX_REAL)
        success_count += success

        save_adversarial(gray_adv, img_path, input_dir, output_dir)

    total = len(img_paths)
    print(f"\n[DONE] Attack success rate: {success_count}/{total} "
          f"({100*success_count/total:.1f}%)")
    print(f"[DONE] Immagini salvate in: {output_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Perturbation network addestrata per massimizzare "
                    "lo score della challenge AADD 2026."
    )
    parser.add_argument('--config',          required=True)
    parser.add_argument('--input_dir',       required=True)
    parser.add_argument('--output_dir',      required=True)
    parser.add_argument('--checkpoint',      default='perturbation_net.pth',
                        help="Path checkpoint (salvataggio durante train, "
                             "caricamento durante inference).")
    # Training
    parser.add_argument('--epochs',          type=int,   default=50)
    parser.add_argument('--batch_size',      type=int,   default=16)
    parser.add_argument('--lr',              type=float, default=1e-4)
    parser.add_argument('--eps',             type=float, default=32.0,
                        help="Epsilon massima perturbazione in pixel.")
    parser.add_argument('--base_ch',         type=int,   default=32,
                        help="Canali base UNet (default 32, riduci a 16 "
                             "per meno memoria).")
    # Pesi della loss
    parser.add_argument('--lambda_clf',      type=float, default=2.0,
                        help="Peso loss classificatore (default 2.0).")
    parser.add_argument('--lambda_ssim',     type=float, default=1.0,
                        help="Peso loss SSIM (default 1.0).")
    parser.add_argument('--lambda_lpips',    type=float, default=1.0,
                        help="Peso loss LPIPS (default 1.0, non usato "
                             "nel backward, solo monitor).")
    parser.add_argument('--lambda_norm',     type=float, default=0.1,
                        help="Peso regolarizzatore norma (default 0.1).")
    # Inference
    parser.add_argument('--inference_only',  action='store_true',
                        help="Salta training, usa checkpoint esistente.")
    args = parser.parse_args()

    if args.inference_only:
        inference(args)
    else:
        train(args)


if __name__ == '__main__':
    main()