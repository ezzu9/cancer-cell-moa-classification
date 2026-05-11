"""PyTorch Dataset for the preprocessed BBBC021 images (compound-held-out splits)."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class BBBC021Dataset(Dataset):
    """
    Loads preprocessed 224×224 PNG images organised as:
        processed_dir/<moa_class>/<image>.png

    Supports NSC (Not-Same-Compound) splits by filtering on compound names.
    """

    def __init__(
        self,
        processed_dir: Path,
        metadata_df: pd.DataFrame,
        transform: Optional[Callable] = None,
        compounds: Optional[List[str]] = None,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.transform = transform

        df = metadata_df.copy()
        if compounds is not None:
            df = df[df["compound"].isin(compounds)]

        self.samples: List[Tuple[Path, int]] = []
        self.class_to_idx: Dict[str, int] = {}
        self._build_index(df)

    def _build_index(self, df: pd.DataFrame) -> None:
        classes = sorted(df["moa"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        for _, row in df.iterrows():
            moa_dir = self.processed_dir / row["moa"].replace(" ", "_").replace("/", "-")
            img_path = moa_dir / f"{Path(row['Image_FileName_DAPI']).stem}.png"
            if img_path.exists():
                self.samples.append((img_path, self.class_to_idx[row["moa"]]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label


def make_nsc_splits(
    metadata_df: pd.DataFrame,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create compound-level train/val/test splits so that images of the same
    compound are never shared across splits (NSC protocol).
    Returns three lists of compound names.
    """
    rng = np.random.default_rng(seed)
    compounds = metadata_df["compound"].unique().tolist()
    rng.shuffle(compounds)

    n = len(compounds)
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))

    test_compounds = compounds[:n_test]
    val_compounds = compounds[n_test: n_test + n_val]
    train_compounds = compounds[n_test + n_val:]
    return train_compounds, val_compounds, test_compounds
