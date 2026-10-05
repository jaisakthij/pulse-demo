"""Synthetic model-space vectors for safe software demonstrations."""
from __future__ import annotations

import numpy as np


def generate_synthetic_profile(
    lab_dim: int,
    metabolomics_dim: int,
    followup_input_dim: int,
    *,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Generate Gaussian feature vectors, not a clinically realistic patient."""
    dimensions = (lab_dim, metabolomics_dim, followup_input_dim)
    if any(not isinstance(size, int) or size <= 0 for size in dimensions):
        raise ValueError("All feature dimensions must be positive integers.")
    generator = rng if rng is not None else np.random.default_rng()
    return {
        "baseline_labs": generator.standard_normal(lab_dim).astype(np.float32),
        "baseline_metabolomics": generator.standard_normal(metabolomics_dim).astype(np.float32),
        "followup_input": generator.standard_normal(followup_input_dim).astype(np.float32),
    }
