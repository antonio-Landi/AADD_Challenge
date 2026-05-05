"""
fft_batch.py — Genera gli spettri FFT 2D per una cartella di immagini.

Uso:
    python fft_batch.py <cartella_input> [cartella_output]

Se cartella_output non viene specificata, viene creata <cartella_input>_fft accanto all'input.
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless, nessuna finestra
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

SUPPORTED = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

plt.rcParams.update({
    'figure.facecolor': '#0f0f0f',
    'axes.facecolor':   '#0f0f0f',
    'text.color':       'white',
    'axes.labelcolor':  'white',
    'xtick.color':      'white',
    'ytick.color':      'white',
})


def load_gray(path: Path) -> np.ndarray:
    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.float64)


def compute_fft(img: np.ndarray):
    F_shift = np.fft.fftshift(np.fft.fft2(img))
    log_mag = np.log1p(np.abs(F_shift))
    phase   = np.angle(F_shift)
    return F_shift, log_mag, phase


def draw_spectrum(img: np.ndarray, log_mag: np.ndarray, phase: np.ndarray,
                  name: str) -> plt.Figure:
    H, W = img.shape

    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor('#0f0f0f')
    fig.suptitle(name, color='white', fontsize=13, y=1.01)
    gs = GridSpec(1, 3, figure=fig, wspace=0.08)

    # --- Originale ---
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax0.set_title('Originale', color='white')
    ax0.axis('off')

    # --- Spettro ampiezza (centrato) ---
    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(log_mag, cmap='inferno',
                     extent=[-W // 2, W // 2, -H // 2, H // 2],
                     origin='lower', interpolation='nearest')
    ax1.set_title('Spettro FFT (ampiezza log)', color='white')
    ax1.set_xlabel('Frequenza U', color='#aaaaaa', fontsize=9)
    ax1.set_ylabel('Frequenza V', color='#aaaaaa', fontsize=9)
    for spine in ax1.spines.values():
        spine.set_edgecolor('#444')
    # Cerchi concentrici bassa / media / alta frequenza
    r_max = min(H, W) // 2
    for frac, lbl in [(0.10, 'bassa'), (0.35, 'media'), (0.70, 'alta')]:
        r = r_max * frac
        ax1.add_patch(plt.Circle((0, 0), r, color='cyan', fill=False,
                                  linestyle='--', linewidth=0.8, alpha=0.5))
        ax1.text(r * 0.72, r * 0.72, lbl, color='cyan', fontsize=7, alpha=0.7)
    ax1.axhline(0, color='white', lw=0.4, alpha=0.3)
    ax1.axvline(0, color='white', lw=0.4, alpha=0.3)
    plt.colorbar(im1, ax=ax1, label='log(1+|F|)', fraction=0.046, pad=0.04)

    # --- Fase ---
    ax2 = fig.add_subplot(gs[2])
    im2 = ax2.imshow(phase, cmap='hsv', interpolation='nearest')
    ax2.set_title('Spettro di fase', color='white')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, label='fase (rad)', fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def process_folder(input_dir: Path, output_dir: Path, dpi: int = 150) -> None:
    images = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in SUPPORTED]
    if not images:
        print(f"Nessuna immagine trovata in '{input_dir}'.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(images)
    print(f"Trovate {total} immagini → output in '{output_dir}'")

    for i, path in enumerate(images, 1):
        try:
            img = load_gray(path)
            F_shift, log_mag, phase = compute_fft(img)
            fig = draw_spectrum(img, log_mag, phase, path.name)
            out_path = output_dir / (path.stem + '_fft.png')
            fig.savefig(out_path, dpi=dpi, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"  [{i}/{total}] {path.name} → {out_path.name}")
        except Exception as exc:
            print(f"  [{i}/{total}] ERRORE su '{path.name}': {exc}")

    print("Completato.")


def main() -> None:
    parser = argparse.ArgumentParser(description='FFT batch — spettri di Fourier per una cartella di immagini')
    parser.add_argument('input',  type=Path, help='Cartella con le immagini sorgente')
    parser.add_argument('output', type=Path, nargs='?', default=None,
                        help='Cartella di destinazione (default: <input>_fft)')
    parser.add_argument('--dpi', type=int, default=150, help='DPI delle immagini output (default: 150)')
    args = parser.parse_args()

    input_dir: Path = args.input.resolve()
    if not input_dir.is_dir():
        sys.exit(f"Errore: '{input_dir}' non è una cartella valida.")

    output_dir: Path = args.output.resolve() if args.output else input_dir.parent / (input_dir.name + '_fft')

    process_folder(input_dir, output_dir, dpi=args.dpi)


if __name__ == '__main__':
    main()
