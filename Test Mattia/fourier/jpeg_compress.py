"""
jpeg_compress.py - Simulatore JPEG/social per testare robustezza AADD.

Default consigliato:
    python jpeg_compress.py AADD_2026_Test AADD_2026_Test_fb92

Griglia utile per la challenge:
    python jpeg_compress.py adversarial_examples robust_check --preset challenge

Nota AADD:
    di default preserva i nomi originali (es. 000.png), ma scrive byte JPEG.
    PIL li apre correttamente e questo evita rinomine non volute nello ZIP.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps


SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


@dataclass(frozen=True)
class Preset:
    qfs: tuple[int, ...]
    max_side: int | None
    subsampling: int
    optimize: bool = True
    progressive: bool = False


PRESETS: dict[str, Preset] = {
    # Paper 2310.12708v1: Facebook usa QF=92 per la maggior parte delle immagini
    # e ridimensiona solo quando la risoluzione supera 2048 px.
    "facebook": Preset(qfs=(92,), max_side=2048, subsampling=2),
    # Stesso paper: q = 92/58/85 per Facebook/WeChat/QQ.
    "wechat": Preset(qfs=(58,), max_side=2048, subsampling=2),
    "qq": Preset(qfs=(85,), max_side=2048, subsampling=2),
    # La challenge non dichiara un QF singolo: questa griglia serve per stress test.
    "challenge": Preset(qfs=(92, 90, 85, 80, 75), max_side=2048, subsampling=2),
    "custom": Preset(qfs=(70,), max_side=None, subsampling=2),
}


def parse_qfs(raw: str) -> tuple[int, ...]:
    qfs: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        qf = int(item)
        if not 1 <= qf <= 95:
            raise argparse.ArgumentTypeError("ogni QF deve essere tra 1 e 95")
        qfs.append(qf)
    if not qfs:
        raise argparse.ArgumentTypeError("specifica almeno un QF")
    return tuple(dict.fromkeys(qfs))


def collect_images(input_dir: Path, recursive: bool) -> list[Path]:
    iterator: Iterable[Path]
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(
        p for p in iterator
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )


def to_rgb(img: Image.Image, alpha_background: str) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        bg_color = (255, 255, 255) if alpha_background == "white" else (0, 0, 0)
        background = Image.new("RGB", img.size, bg_color)
        background.paste(img.convert("RGBA"), mask=img.convert("RGBA").getchannel("A"))
        return background
    return img.convert("RGB")


def resize_max_side(img: Image.Image, max_side: int | None) -> Image.Image:
    if not max_side:
        return img
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def output_path_for(
    image_path: Path,
    input_dir: Path,
    output_dir: Path,
    output_mode: str,
    recursive: bool,
) -> Path:
    rel = image_path.relative_to(input_dir) if recursive else Path(image_path.name)
    if output_mode == "jpg":
        rel = rel.with_suffix(".jpg")
    return output_dir / rel


def save_as_jpeg(
    src: Path,
    dst: Path,
    qf: int,
    preset: Preset,
    output_mode: str,
    alpha_background: str,
    keep_metadata: bool,
    dry_run: bool,
) -> tuple[tuple[int, int], tuple[int, int]]:
    with Image.open(src) as opened:
        before_size = opened.size
        img = to_rgb(opened, alpha_background)
        img = resize_max_side(img, preset.max_side)
        after_size = img.size

        if dry_run:
            return before_size, after_size

        dst.parent.mkdir(parents=True, exist_ok=True)

        save_kwargs = {
            "format": "JPEG",
            "quality": qf,
            "subsampling": preset.subsampling,
            "optimize": preset.optimize,
            "progressive": preset.progressive,
        }
        if keep_metadata:
            if "exif" in opened.info:
                save_kwargs["exif"] = opened.info["exif"]
            if "icc_profile" in opened.info:
                save_kwargs["icc_profile"] = opened.info["icc_profile"]

        # output_mode="preserve" puo produrre 000.png con contenuto JPEG.
        # E' intenzionale per rispettare i nomi AADD e simulare re-encoding.
        img.save(dst, **save_kwargs)
        return before_size, after_size


def infer_output_dir(input_dir: Path, preset_name: str, qfs: tuple[int, ...]) -> Path:
    if len(qfs) == 1:
        return input_dir.parent / f"{input_dir.name}_{preset_name}_qf{qfs[0]}"
    return input_dir.parent / f"{input_dir.name}_{preset_name}_grid"


def compress_folder(
    input_dir: Path,
    output_dir: Path,
    preset_name: str,
    preset: Preset,
    qfs: tuple[int, ...],
    output_mode: str,
    recursive: bool,
    alpha_background: str,
    keep_metadata: bool,
    expected_count: int | None,
    dry_run: bool,
    quiet: bool,
) -> None:
    images = collect_images(input_dir, recursive)
    if not images:
        print(f"Nessuna immagine trovata in '{input_dir}'.")
        return
    if expected_count is not None and len(images) != expected_count:
        raise SystemExit(
            f"Errore: trovate {len(images)} immagini, ma --expected-count={expected_count}."
        )

    multi_qf = len(qfs) > 1
    print(
        f"Trovate {len(images)} immagini | preset={preset_name} | "
        f"QF={','.join(map(str, qfs))} | output_mode={output_mode}"
    )
    if preset.max_side:
        print(f"Resize max-side: {preset.max_side}px")
    if dry_run:
        print("Dry-run: nessun file verra scritto.")

    total_jobs = len(images) * len(qfs)
    done = 0

    for qf in qfs:
        qf_output_dir = output_dir / f"qf{qf}" if multi_qf else output_dir
        for src in images:
            done += 1
            dst = output_path_for(src, input_dir, qf_output_dir, output_mode, recursive)
            before_size, after_size = save_as_jpeg(
                src=src,
                dst=dst,
                qf=qf,
                preset=preset,
                output_mode=output_mode,
                alpha_background=alpha_background,
                keep_metadata=keep_metadata,
                dry_run=dry_run,
            )
            resize_note = "" if before_size == after_size else f" {before_size}->{after_size}"
            if not quiet:
                print(f"  [{done}/{total_jobs}] QF={qf} {src.name} -> {dst.name}{resize_note}")

    print("Completato.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compressione JPEG batch con preset social/AADD."
    )
    parser.add_argument("input", type=Path, help="Cartella con le immagini sorgente")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="Cartella di destinazione (default: <input>_<preset>_qfXX/grid)",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="facebook",
        help="Pipeline predefinita (default: facebook, QF=92 dal paper OSN).",
    )
    parser.add_argument(
        "--qf",
        type=int,
        default=None,
        metavar="1-95",
        help="Quality factor singolo; sovrascrive il preset.",
    )
    parser.add_argument(
        "--qf-grid",
        type=parse_qfs,
        default=None,
        metavar="92,90,85,80,75",
        help="Lista di QF; crea sottocartelle qfXX e sovrascrive il preset.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=None,
        help="Lato massimo in pixel; sovrascrive il preset. Usa 0 per disattivare.",
    )
    parser.add_argument(
        "--subsampling",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="0=4:4:4, 1=4:2:2, 2=4:2:0; sovrascrive il preset.",
    )
    parser.add_argument(
        "--output-mode",
        choices=("preserve", "jpg"),
        default="preserve",
        help="preserve mantiene i nomi originali; jpg cambia estensione in .jpg.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Processa sottocartelle preservando la struttura relativa.",
    )
    parser.add_argument(
        "--alpha-background",
        choices=("white", "black"),
        default="white",
        help="Sfondo per immagini con alpha channel (default: white).",
    )
    parser.add_argument(
        "--keep-metadata",
        action="store_true",
        help="Mantiene EXIF/ICC quando presenti. Default: strip metadata stile social.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="Fallisce se il numero immagini non coincide; per AADD usare 1600.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra cosa verrebbe scritto senza creare file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Mostra solo il riepilogo, utile per griglie grandi.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_dir = args.input.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Errore: '{input_dir}' non e' una cartella valida.")

    if args.qf is not None and not 1 <= args.qf <= 95:
        raise SystemExit("Errore: --qf deve essere tra 1 e 95.")
    if args.qf is not None and args.qf_grid is not None:
        raise SystemExit("Errore: usa solo uno tra --qf e --qf-grid.")
    if args.max_side is not None and args.max_side < 0:
        raise SystemExit("Errore: --max-side deve essere >= 0.")

    base = PRESETS[args.preset]
    qfs = args.qf_grid or ((args.qf,) if args.qf is not None else base.qfs)
    max_side = base.max_side if args.max_side is None else (None if args.max_side == 0 else args.max_side)
    subsampling = base.subsampling if args.subsampling is None else args.subsampling
    preset = Preset(
        qfs=qfs,
        max_side=max_side,
        subsampling=subsampling,
        optimize=base.optimize,
        progressive=base.progressive,
    )

    output_dir = args.output.resolve() if args.output else infer_output_dir(input_dir, args.preset, qfs)
    if output_dir == input_dir and not args.dry_run:
        raise SystemExit(
            "Errore: la cartella output coincide con input. "
            "Scegli una cartella diversa per non sovrascrivere le immagini sorgente."
        )

    compress_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        preset_name=args.preset,
        preset=preset,
        qfs=qfs,
        output_mode=args.output_mode,
        recursive=args.recursive,
        alpha_background=args.alpha_background,
        keep_metadata=args.keep_metadata,
        expected_count=args.expected_count,
        dry_run=args.dry_run,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
