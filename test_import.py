"""Smoke test for PULSE's batched longitudinal imputation contract."""
from pathlib import Path
import sys
import warnings

import torch

REPO_ROOT = Path(__file__).resolve().parent
PULSE_ROOT = REPO_ROOT / "LongitudinalGeneration"
if str(PULSE_ROOT) not in sys.path:
    sys.path.insert(0, str(PULSE_ROOT))

from pulse.model import PulseModel


def test_model_impute_missing_preserves_single_patient_batch_dimensions():
    model = PulseModel(
        modality_dims={"modality_a": 61, "modality_b": 251},
        latent_dim=16,
        hidden_dim=32,
        temporal_model="recurrent",
    ).eval()
    visits = [
        {"modality_a": torch.zeros((1, 61)), "modality_b": torch.zeros((1, 251))},
        {"modality_a": torch.ones((1, 61)), "modality_b": None},
    ]
    masks = [
        {"modality_a": 0, "modality_b": 0},
        {"modality_a": 0, "modality_b": 2},
    ]

    with warnings.catch_warnings(record=True) as observed_warnings:
        warnings.simplefilter("always")
        reconstructed = model.impute_missing(visits, masks)

    assert len(reconstructed) == 2
    assert tuple(reconstructed[0]["modality_a"].shape) == (1, 61)
    assert tuple(reconstructed[0]["modality_b"].shape) == (1, 251)
    assert tuple(reconstructed[1]["modality_a"].shape) == (1, 61)
    assert tuple(reconstructed[1]["modality_b"].shape) == (1, 251)
    assert not any("target size" in str(w.message) for w in observed_warnings)
