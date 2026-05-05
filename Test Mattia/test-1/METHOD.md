# Attacco Avversariale per AADD-2026 — Spiegazione del Metodo

## Obiettivo

Le immagini in `AADD_2026_Test` sono **fake** (generate da modelli di deepfake), ma i due classificatori del challenge le classificano correttamente come "Fake". L'obiettivo è **ingannare i classificatori** facendogli predire "Real", aggiungendo una perturbazione impercettibile ai pixel — senza alterare visibilmente le immagini.

Lo **score** della challenge premia sia il successo dell'attacco sia l'impercettibilità della perturbazione:

```
sim_weight   = 0.5 · SSIM(x_adv, x_orig) + 0.5 · (1 − LPIPS(x_adv, x_orig))
pair_contrib = Σ_c  w_c · sim_weight · I(pred_c == Real)
total_score  = Σ_immagini  pair_contrib
```

Per massimizzare lo score bisogna quindi:
1. Far classificare l'immagine come "Real" → `I = 1`
2. Mantenere l'immagine visivamente vicina all'originale → `sim_weight` alto

---

## Modelli bersaglio

L'attacco è eseguito **separatamente** per ciascuno dei due classificatori:

| Modello | Input | Pipeline |
|---|---|---|
| `vit_b_16` | RGB 224×224 | Resize 256 → CenterCrop 224 → Norm ImageNet |
| `densenet121_dct` | DCT 1-canale 128×128 | Grayscale → Resize 256 → CenterCrop 128 → DCT-II → log |

Per ogni modello viene creata una cartella distinta in `dataset_adv/<model>/`.

---

## Algoritmo: PGD (Projected Gradient Descent)

L'attacco è una variante di **I-FGSM / PGD** (Madry et al., 2018), che esegue `T = 40` passi di discesa del gradiente sulla loss, proiettando ad ogni passo nel vincolo L∞.

### Loop PGD

Dato un'immagine originale `x_orig`, si inizializza `δ = 0` e si itera:

```
per t = 1 … T:
    x_adv = clip(x_orig + δ, 0, 1)
    logits = model( preprocess(x_adv) )
    L      = loss(logits, x_adv, x_orig)

    g = ∂L / ∂δ                         # backpropagation

    δ ← δ − α · sign(g)                 # passo FGSM
    δ ← clip(δ, −ε, +ε)                 # proiezione L∞
    δ ← clip(x_orig + δ, 0, 1) − x_orig # proiezione [0,1]
```

**Parametri:**

| Simbolo | Valore | Significato |
|---|---|---|
| ε | 8/255 ≈ 0.031 | Budget L∞ massimo per pixel |
| α | ε/10 | Dimensione del passo |
| T | 40 | Numero di iterazioni |

---

## Loss Score-Aware

La loss è progettata per massimizzare direttamente i fattori che compongono lo score:

```
L(x_adv, x_orig) = −log P(Real | x_adv)  −  λ_ssim · SSIM(x_adv, x_orig)
                        ↑                          ↑
               massimizza P(Real)        massimizza sim_weight
               → indicator = 1          → moltiplica il contributo score
```

Con `λ_ssim = 0.3`.

**Perché questa forma?**

- `−log P(Real)` è la cross-entropy verso la classe "Real" (indice 0). Minimizzarla spinge il modello a predire "Real" con probabilità crescente.
- `−SSIM` spinge la perturbazione a conservare la struttura locale dell'immagine (medie e varianze locali), il che corrisponde direttamente al fattore `SSIM` dentro `sim_weight`.
- Il vincolo L∞ (ε = 8/255) limita la magnitudine di ogni pixel, tenendo basso LPIPS in modo implicito — perturbazioni piccole sono quasi sempre impercettibili anche a livello percettuale.

La combinazione dei due termini quindi allinea la loss alla formula dello score: `sim_weight · I(pred == Real)`.

---

## SSIM Differenziabile

Lo standard SSIM è calcolato con `skimage` (non differenziabile). Per poterlo usare nella loss, è reimplementato in PyTorch con convoluzione gaussiana:

```
SSIM(x, y) = mean_pixel [  (2·μ_x·μ_y + C1)(2·σ_xy + C2)  /
                            (μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2)  ]
```

dove `μ`, `σ²`, `σ_xy` sono stimati tramite convoluzione con un kernel gaussiano 11×11 (σ = 1.5). La derivata ∂SSIM/∂x è calcolata automaticamente da autograd di PyTorch.

---

## DCT Differenziabile

Il classificatore `densenet121_dct` riceve in input i **coefficienti DCT** dell'immagine in scala di grigi. La pipeline ufficiale usa `scipy.fftpack.dct(..., norm='ortho')`, che **non è differenziabile** — il gradiente non può fluire attraverso di essa.

Per risolvere il problema si implementa la DCT-II 2D come **moltiplicazione matriciale**, che PyTorch sa derivare:

```
Y = W · X · Wᵀ
```

La matrice ortogonale W è costruita una volta sola e memorizzata come buffer:

```
W[0, m]  = sqrt(1/N)
W[k, m]  = sqrt(2/N) · cos( π · k · (2m+1) / (2N) )    per k ≥ 1
```

Questa è esattamente la forma normalizzata (`norm='ortho'`) di scipy, quindi i gradienti fluiscono attraverso la DCT verso i pixel dell'immagine.

---

## Attacco a Risoluzione Ridotta (256×256)

Le immagini originali sono 1024×1024. Calcolare il gradiente a piena risoluzione è svantaggioso:

- Ogni pixel a 1024×1024 contribuisce solo a una piccola porzione dell'input del modello (224×224 o 128×128).
- Il gradiente risultante è **diluito** di un fattore `(1024/256)² = 16×`.

**Soluzione:** Il gradiente viene calcolato sull'immagine ridimensionata a **256×256**:

```
x_256  = resize(x_orig, 256×256)          # downscale bilineare
δ_256  = PGD(model, x_256)                # 40 iterazioni a 256×256
δ_full = resize(δ_256, H×W)               # upsample bilineare
x_adv  = clip(x_orig + δ_full, 0, 1)      # applica all'originale
```

L'upsampling bilineare è una combinazione convessa dei valori adiacenti, quindi il vincolo L∞ è **preservato**: `‖δ_full‖∞ ≤ ‖δ_256‖∞ ≤ ε`.

---

## Pre-processing Differenziabile

Per permettere la backpropagation attraverso il pre-processing, tutte le operazioni sono implementate in PyTorch:

### ViT B/16
```
x  → interpolate(256×256, bilinear)
   → x[:, :, 16:240, 16:240]          # center-crop 224×224
   → (x − mean_ImageNet) / std_ImageNet
```

### DenseNet-121 DCT
```
x  → 0.299·R + 0.587·G + 0.114·B      # ITU-R BT.601 grayscale
   → interpolate(256×256, bilinear)
   → x[:, :, 64:192, 64:192]           # center-crop 128×128
   → W · x · Wᵀ                        # DCT-II ortonormale
   → log(|·| + 1e-6)                   # log-scale
```

Queste pipeline corrispondono esattamente a quelle di `AADD_2026_evaluation.py`.

---

## Valutazione Inline

Dopo aver generato ogni immagine avversariale, lo script esegue una valutazione completa **senza ricaricare i modelli dal disco** — la logica è identica a quella di `AADD_2026_evaluation.py`.

Per ogni coppia (originale, avversariale):

1. **SSIM** — calcolato per canale R/G/B con `skimage` a piena risoluzione, poi mediato.
2. **LPIPS** — calcolato con AlexNet (`lpips.LPIPS(net='alex')`), immagini normalizzate in `[−1, 1]` con `/127.5 − 1`.
3. **sim_weight** = 0.5 · SSIM + 0.5 · (1 − LPIPS)
4. **Predizione** di entrambi i classificatori sull'immagine avversariale (indipendentemente da quale è stato il modello di attacco).
5. **pair_contribution** = Σ_c  w_c · sim_weight · I(pred_c == Real)

I risultati vengono salvati in `dataset_adv/<model>/results.json` e aggiornati **dopo ogni singola immagine**.

---

## Struttura dell'output

```
dataset_adv/
├── vit_b_16/
│   ├── 000.png          immagine avversariale
│   ├── 001.png
│   ├── ...
│   └── results.json     aggiornato dopo ogni immagine
└── densenet121_dct/
    ├── 000.png
    ├── ...
    └── results.json
```

### Formato `results.json`

```json
{
  "attack_model": "vit_b_16",
  "epsilon": 0.0314,
  "n_steps": 40,
  "lambda_ssim": 0.3,
  "n_processed": 1600,
  "n_fooled_by_target": 1423,
  "total_score": 712.5,
  "images": [
    {
      "image": "000.png",
      "ssim": 0.9821,
      "lpips": 0.0312,
      "sim_weight": 0.4755,
      "per_classifier": {
        "vit_b_16":        {"prediction": "Real", "prob_real": 0.93, "indicator": 1, "contribution": 0.4755},
        "densenet121_dct": {"prediction": "Fake", "prob_real": 0.12, "indicator": 0, "contribution": 0.0}
      },
      "pair_contribution": 0.4755
    },
    ...
  ]
}
```

---

## Riepilogo dei componenti

```
Immagine originale (1024×1024)
        │
        ▼
  Downscale 256×256
        │
        ▼
┌───────────────────────────────┐
│  PGD loop (T=40)              │
│                               │
│  x_adv = clip(x + δ, 0, 1)   │
│  logits = model(preprocess(x_adv)) │
│                               │
│  L = −log P(Real)             │
│      − 0.3 · SSIM(x_adv,x)   │
│                               │
│  δ ← δ − α · sign(∂L/∂δ)     │
│  δ ← clip(δ, −ε, +ε)         │
└───────────────────────────────┘
        │
        ▼
  Upsample δ → 1024×1024
        │
        ▼
  x_adv = clip(x_orig + δ, 0, 1)
        │
        ├──► Salva PNG
        │
        └──► Valutazione (SSIM, LPIPS, predizioni) → aggiorna JSON
```
