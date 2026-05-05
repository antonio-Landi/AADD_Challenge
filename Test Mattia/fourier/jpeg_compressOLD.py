"""
jpeg_compress.py — Comprime una cartella di immagini in JPEG con quality factor configurabile.

Uso:
    python jpeg_compress.py <cartella_input> [cartella_output] [--qf 70]
"""

import sys
import argparse
from pathlib import Path

from PIL import Image

SUPPORTED = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def compress_folder(input_dir: Path, output_dir: Path, qf: int) -> None:
    images = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in SUPPORTED]
    if not images:
        print(f"Nessuna immagine trovata in '{input_dir}'.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(images)
    print(f"Trovate {total} immagini — QF={qf} → output in '{output_dir}'")

    for i, path in enumerate(images, 1):
        try:
            img = Image.open(path).convert('RGB')
            out_path = output_dir / (path.stem + '.jpg')
            img.save(out_path, format='JPEG', quality=qf, subsampling=2)
            print(f"  [{i}/{total}] {path.name} → {out_path.name}")
        except Exception as exc:
            print(f"  [{i}/{total}] ERRORE su '{path.name}': {exc}")

    print("Completato.")


def main() -> None:
    parser = argparse.ArgumentParser(description='Compressione JPEG batch')
    parser.add_argument('input',  type=Path, help='Cartella con le immagini sorgente')
    parser.add_argument('output', type=Path, nargs='?', default=None,
                        help='Cartella di destinazione (default: <input>_jpeg<qf>)')
    parser.add_argument('--qf', type=int, default=70, metavar='1-95',
                        help='Quality factor JPEG, 1=minima qualità 95=massima (default: 70)')
    args = parser.parse_args()

    if not 1 <= args.qf <= 95:
        sys.exit("Errore: --qf deve essere tra 1 e 95.")

    input_dir: Path = args.input.resolve()
    if not input_dir.is_dir():
        sys.exit(f"Errore: '{input_dir}' non è una cartella valida.")

    output_dir: Path = (args.output.resolve() if args.output
                        else input_dir.parent / f"{input_dir.name}_jpeg{args.qf}")

    compress_folder(input_dir, output_dir, args.qf)


if __name__ == '__main__':
    main()
