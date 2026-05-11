"""Unit tests for dataset loading and NSC split utilities."""

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import make_nsc_splits


@pytest.fixture
def dummy_metadata():
    compounds = [f"compound_{i:03d}" for i in range(20)]
    moa_cycle = [
        "Actin disruptors", "Aurora kinase inhibitors", "Cholesterol-lowering",
        "DNA damage", "DNA replication",
    ]
    rows = []
    for i, c in enumerate(compounds):
        for j in range(5):
            rows.append({
                "compound": c,
                "moa": moa_cycle[i % len(moa_cycle)],
                "Image_FileName_DAPI": f"{c}_img{j:02d}_DAPI.tif",
            })
    return pd.DataFrame(rows)


def test_nsc_splits_no_overlap(dummy_metadata):
    train_c, val_c, test_c = make_nsc_splits(dummy_metadata)
    assert len(set(train_c) & set(val_c)) == 0
    assert len(set(train_c) & set(test_c)) == 0
    assert len(set(val_c) & set(test_c)) == 0


def test_nsc_splits_cover_all_compounds(dummy_metadata):
    train_c, val_c, test_c = make_nsc_splits(dummy_metadata)
    all_compounds = set(dummy_metadata["compound"].unique())
    union = set(train_c) | set(val_c) | set(test_c)
    assert union == all_compounds


def test_nsc_splits_reproducible(dummy_metadata):
    splits_a = make_nsc_splits(dummy_metadata, seed=0)
    splits_b = make_nsc_splits(dummy_metadata, seed=0)
    for a, b in zip(splits_a, splits_b):
        assert sorted(a) == sorted(b)
