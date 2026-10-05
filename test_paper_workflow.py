import numpy as np
import pytest
import warnings
import torch

from pulse_build.engine import PulseEngine
from pulse_build.io import Patient, PatientVisit


CHECKPOINT = "ckpt_small/checkpoint_epoch_60.pt"


def test_paper_workflow_reconstructs_followup_metabolomics_vector():
    engine = PulseEngine(ckpt_path=CHECKPOINT)
    assert engine.checkpoint_loaded is True
    assert engine.checkpoint_epoch == 60
    assert engine.model.training is False

    predicted = engine.impute_followup_metabolomics(
        baseline_labs=np.zeros(61, dtype=np.float32),
        baseline_metabolomics=np.zeros(251, dtype=np.float32),
        followup_labs=np.zeros(61, dtype=np.float32),
    )

    assert predicted.shape == (251,)
    assert np.isfinite(predicted).all()


def test_checkpoint_loading_uses_safe_weights_only_mode(monkeypatch):
    real_load = torch.load
    observed = {}

    def recording_load(*args, **kwargs):
        observed["weights_only"] = kwargs.get("weights_only")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    engine = PulseEngine(ckpt_path=CHECKPOINT)

    assert engine.checkpoint_loaded
    assert observed["weights_only"] is True


def test_paper_workflow_rejects_missing_checkpoint():
    engine = PulseEngine(ckpt_path=None)

    with pytest.raises(RuntimeError, match="trained checkpoint"):
        engine.impute_followup_metabolomics(
            baseline_labs=np.zeros(61, dtype=np.float32),
            baseline_metabolomics=np.zeros(251, dtype=np.float32),
            followup_labs=np.zeros(61, dtype=np.float32),
        )


def test_reverse_direction_reconstructs_followup_labs():
    engine = PulseEngine(ckpt_path=CHECKPOINT)

    predicted = engine.impute_followup_labs(
        baseline_labs=np.zeros(61, dtype=np.float32),
        baseline_metabolomics=np.zeros(251, dtype=np.float32),
        followup_metabolomics=np.zeros(251, dtype=np.float32),
    )

    assert predicted.shape == (61,)
    assert np.isfinite(predicted).all()


def test_reverse_direction_rejects_missing_checkpoint():
    engine = PulseEngine(ckpt_path=None)

    with pytest.raises(RuntimeError, match="trained checkpoint"):
        engine.impute_followup_labs(
            baseline_labs=np.zeros(61, dtype=np.float32),
            baseline_metabolomics=np.zeros(251, dtype=np.float32),
            followup_metabolomics=np.zeros(251, dtype=np.float32),
        )


def test_reverse_direction_rejects_wrong_followup_dimension():
    engine = PulseEngine(ckpt_path=CHECKPOINT)

    with pytest.raises(ValueError, match="followup_metabolomics.*251"):
        engine.impute_followup_labs(
            baseline_labs=np.zeros(61, dtype=np.float32),
            baseline_metabolomics=np.zeros(251, dtype=np.float32),
            followup_metabolomics=np.zeros(61, dtype=np.float32),
        )


def test_uncertainty_is_reported_as_unavailable_not_as_absolute_z_score():
    engine = PulseEngine(ckpt_path=CHECKPOINT)
    patient = Patient(
        patient_id="test", age=50, sex="F", ancestry="Unspecified", site="test",
        visits=[PatientVisit(date="2025-01-01", labs={"glucose": 95.0})],
    )

    uncertainty = engine.uncertainty(patient)

    assert uncertainty["available"] is False
    assert "reason" in uncertainty
    assert "mean_abs_z" not in uncertainty


def test_legacy_patient_imputation_does_not_broadcast_features_as_batch():
    engine = PulseEngine(ckpt_path=CHECKPOINT)
    patient = Patient(
        patient_id="legacy-test", age=50, sex="F", ancestry="Unspecified", site="test",
        visits=[PatientVisit(date="2025-01-01", labs={"glucose": 95.0, "hdl": 52.0})],
    )

    with warnings.catch_warnings(record=True) as observed_warnings:
        warnings.simplefilter("always")
        result = engine.impute(patient)

    assert len(result.visits) == 1
    assert len(result.visits[0].imputations) == 12
    assert not any("target size" in str(w.message) for w in observed_warnings)


def test_latent_states_returns_one_state_per_patient_visit():
    engine = PulseEngine(ckpt_path=CHECKPOINT)
    patient = Patient(
        patient_id="latent-test", age=50, sex="F", ancestry="Unspecified", site="test",
        visits=[PatientVisit(date="2025-01-01", labs={"glucose": 95.0})],
    )

    states = engine.latent_states(patient)

    assert len(states) == 1
    assert len(states[0]) == engine.model.latent_dim


def test_paper_workflow_rejects_wrong_feature_dimensions():
    engine = PulseEngine(ckpt_path=CHECKPOINT)

    with pytest.raises(ValueError, match="baseline_labs.*61"):
        engine.impute_followup_metabolomics(
            baseline_labs=np.zeros(12, dtype=np.float32),
            baseline_metabolomics=np.zeros(251, dtype=np.float32),
            followup_labs=np.zeros(61, dtype=np.float32),
        )
