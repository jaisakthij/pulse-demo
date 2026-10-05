import numpy as np
import pytest

from pulse_build.synthetic import generate_synthetic_profile


def test_profile_vectors_match_requested_dimensions_and_are_finite():
    profile = generate_synthetic_profile(
        lab_dim=61,
        metabolomics_dim=251,
        followup_input_dim=61,
        rng=np.random.default_rng(17),
    )

    assert profile["baseline_labs"].shape == (61,)
    assert profile["baseline_metabolomics"].shape == (251,)
    assert profile["followup_input"].shape == (61,)
    assert all(values.dtype == np.float32 for values in profile.values())
    assert all(np.isfinite(values).all() for values in profile.values())


def test_generator_is_reproducible_with_seeded_rng_and_varies_across_seeds():
    first = generate_synthetic_profile(61, 251, 61, rng=np.random.default_rng(1))
    repeated = generate_synthetic_profile(61, 251, 61, rng=np.random.default_rng(1))
    other = generate_synthetic_profile(61, 251, 61, rng=np.random.default_rng(2))

    assert np.array_equal(first["baseline_labs"], repeated["baseline_labs"])
    assert not np.array_equal(first["baseline_labs"], other["baseline_labs"])


def test_generator_rejects_nonpositive_dimensions():
    with pytest.raises(ValueError, match="dimensions must be positive"):
        generate_synthetic_profile(0, 251, 61)
