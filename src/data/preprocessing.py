"""
Load and preprocess raw BBBC021 TIFF images into normalised 224×224 RGB PNGs.

Run as a script:
    python src/data/preprocessing.py --raw_dir data/raw --output_dir data/processed
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from tqdm import tqdm


MOA_CLASSES = [
    "Actin disruptors",
    "Aurora kinase inhibitors",
    "Cholesterol-lowering",
    "DNA damage",
    "DNA replication",
    "Eg5 kinesin inhibitors",
    "Epithelial",
    "Kinase inhibitors",
    "Microtubule destabilizers",
    "Microtubule stabilizers",
    "Protein degradation",
    "Protein synthesis",
]

MOA_TO_IDX = {moa: i for i, moa in enumerate(MOA_CLASSES)}


def percentile_normalise(channel: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Stretch channel to [0, 255] using percentile clipping."""
    p_low, p_high = np.percentile(channel, [low, high])
    clipped = np.clip(channel, p_low, p_high)
    if p_high == p_low:
        return np.zeros_like(clipped, dtype=np.uint8)
    normalised = (clipped - p_low) / (p_high - p_low) * 255.0
    return normalised.astype(np.uint8)


def load_tiff_as_rgb(tiff_path: Path) -> np.ndarray:
    """
    Load a 3-channel BBBC021 TIFF (DAPI, Tubulin, Actin) and convert to an
    8-bit RGB array of shape (H, W, 3).
    """
    img = tifffile.imread(str(tiff_path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=0)

    channels = []
    for c in range(3):
        channels.append(percentile_normalise(img[c] if img.ndim == 3 else img))
    return np.stack(channels, axis=-1)


def preprocess_dataset(
    raw_dir: Path,
    output_dir: Path,
    metadata_csv: Path,
    moa_csv: Path,
    target_size: int = 224,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    images_df = pd.read_csv(metadata_csv)
    moa_df = pd.read_csv(moa_csv)
    merged = images_df.merge(moa_df, on="compound", how="inner")
    merged = merged[merged["moa"].isin(MOA_CLASSES)]

    skipped = 0
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Preprocessing"):
        tiff_path = raw_dir / row["Image_FileName_DAPI"].replace(".tif", "")
        # Try to find the composite TIFF or individual channel files
        composite = raw_dir / f"{Path(row['Image_FileName_DAPI']).stem}_composite.tif"
        src = composite if composite.exists() else raw_dir / row["Image_FileName_DAPI"]

        if not src.exists():
            skipped += 1
            continue

        try:
            rgb = load_tiff_as_rgb(src)
            pil_img = Image.fromarray(rgb).resize((target_size, target_size), Image.BILINEAR)

            moa_label = row["moa"].replace(" ", "_").replace("/", "-")
            class_dir = output_dir / moa_label
            class_dir.mkdir(exist_ok=True)

            out_path = class_dir / f"{src.stem}.png"
            pil_img.save(out_path)
        except Exception as exc:
            print(f"  Warning: failed to process {src.name}: {exc}")
            skipped += 1

    total = len(merged)
    print(f"\nDone. Processed {total - skipped}/{total} images. Skipped {skipped}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess BBBC021 TIFF images")
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--metadata_csv", type=Path, default=Path("data/metadata/BBBC021_v1_image.csv"))
    parser.add_argument("--moa_csv", type=Path, default=Path("data/metadata/BBBC021_v1_moa.csv"))
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()

    preprocess_dataset(args.raw_dir, args.output_dir, args.metadata_csv, args.moa_csv, args.size)


if __name__ == "__main__":
    main()
