"""Inference engine: wrap the PULSE model, expose imputation + uncertainty."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(os.path.dirname(_HERE), "LongitudinalGeneration")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from pulse.model import PulseModel
from pulse.data.dataset import collate_visits
from pulse_build.io import (
    LAB_DIM, LAB_SLOTS, LAB_RANGES, Patient, PatientVisit, build_visit_vector,
    tensor_or_none, zscore,
)

MODALITY_DIMS = {"modality_a": LAB_DIM, "modality_b": 251}
LATENT_DIM = 16
HIDDEN_DIM = 32


@dataclass
class ImputationResult:
    biomarker: str
    measured: Optional[float]
    imputed_z: float
    imputed_raw: float
    unit: str
    mean: float
    sd: float


@dataclass
class VisitResult:
    date: str
    imputations: List[ImputationResult] = field(default_factory=list)


@dataclass
class PatientResult:
    patient_id: str
    visits: List[VisitResult] = field(default_factory=list)
    latent: List[List[float]] = field(default_factory=list)


@dataclass
class CohortStats:
    cohort: str
    means: Dict[str, float] = field(default_factory=dict)
    sds: Dict[str, float] = field(default_factory=dict)
    n: int = 0
    labels: Dict[str, int] = field(default_factory=dict)


class PulseEngine:
    """Inference engine over a PULSE checkpoint (inference only)."""

    def __init__(self, ckpt_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self._ckpt_path = ckpt_path
        self.checkpoint_epoch: Optional[int] = None
        self.model = PulseModel(
            modality_dims=MODALITY_DIMS, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM,
            temporal_model="recurrent").to(device)
        self.model.eval()
        if ckpt_path and os.path.isfile(ckpt_path):
            self.load(ckpt_path)

    def load(self, path: str):
        """Load a PULSE checkpoint into the engine."""
        import torch as _torch
        cp = _torch.load(path, map_location=self.device, weights_only=True)
        cfg = cp.get("model_config")
        if cfg:
            self.model = PulseModel(**cfg).to(self.device)
        sd = cp.get("model_state_dict")
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()
        self._ckpt_path = path
        self.checkpoint_epoch = cp.get("epoch")
        print(f"Engine loaded checkpoint: {path} (epoch={self.checkpoint_epoch})")

    @property
    def checkpoint_loaded(self) -> bool:
        return self._ckpt_path is not None and os.path.isfile(self._ckpt_path)

    @torch.no_grad()
    def impute_followup_metabolomics(
        self,
        baseline_labs: np.ndarray,
        baseline_metabolomics: np.ndarray,
        followup_labs: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct follow-up metabolomics from follow-up labs and baseline context.

        The three vectors must already be preprocessed exactly as the model's
        training inputs. This matches the public UK Biobank task shape; it does
        not make a synthetic-trained checkpoint equivalent to the paper model.
        """
        if not self.checkpoint_loaded:
            raise RuntimeError("A trained checkpoint is required before inference.")
        if "modality_a" not in self.model.modality_dims or "modality_b" not in self.model.modality_dims:
            raise ValueError("Checkpoint must define modality_a (labs) and modality_b (metabolomics).")

        expected_a = int(self.model.modality_dims["modality_a"])
        expected_b = int(self.model.modality_dims["modality_b"])
        vectors = {
            "baseline_labs": (baseline_labs, expected_a),
            "baseline_metabolomics": (baseline_metabolomics, expected_b),
            "followup_labs": (followup_labs, expected_a),
        }
        tensors = {}
        for name, (values, expected) in vectors.items():
            arr = np.asarray(values, dtype=np.float32).reshape(-1)
            if arr.size != expected:
                raise ValueError(f"{name} must contain exactly {expected} features; received {arr.size}.")
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} must contain only finite, preprocessed values.")
            tensors[name] = torch.as_tensor(arr, dtype=torch.float32, device=self.device).unsqueeze(0)

        visits = [
            {"modality_a": tensors["baseline_labs"], "modality_b": tensors["baseline_metabolomics"]},
            {"modality_a": tensors["followup_labs"], "modality_b": None},
        ]
        masks = [
            {"modality_a": 0, "modality_b": 0},
            {"modality_a": 0, "modality_b": 2},
        ]
        generated = self.model.impute_missing(visits, masks)
        output = generated[-1].get("modality_b")
        if output is None:
            raise RuntimeError("The model did not return a follow-up metabolomics reconstruction.")
        return output.detach().cpu().numpy().reshape(-1)

    @torch.no_grad()
    def impute_followup_labs(
        self,
        baseline_labs: np.ndarray,
        baseline_metabolomics: np.ndarray,
        followup_metabolomics: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct follow-up routine labs from paired baseline data and follow-up metabolomics.

        This is the reverse modality direction of the metabolomics workflow.
        Inputs must already use the checkpoint's training-time feature scale.
        """
        if not self.checkpoint_loaded:
            raise RuntimeError("A trained checkpoint is required before inference.")
        if "modality_a" not in self.model.modality_dims or "modality_b" not in self.model.modality_dims:
            raise ValueError("Checkpoint must define modality_a (labs) and modality_b (metabolomics).")

        expected_a = int(self.model.modality_dims["modality_a"])
        expected_b = int(self.model.modality_dims["modality_b"])
        vectors = {
            "baseline_labs": (baseline_labs, expected_a),
            "baseline_metabolomics": (baseline_metabolomics, expected_b),
            "followup_metabolomics": (followup_metabolomics, expected_b),
        }
        tensors = {}
        for name, (values, expected) in vectors.items():
            arr = np.asarray(values, dtype=np.float32).reshape(-1)
            if arr.size != expected:
                raise ValueError(f"{name} must contain exactly {expected} features; received {arr.size}.")
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} must contain only finite, preprocessed values.")
            tensors[name] = torch.as_tensor(arr, dtype=torch.float32, device=self.device).unsqueeze(0)

        visits = [
            {"modality_a": tensors["baseline_labs"], "modality_b": tensors["baseline_metabolomics"]},
            {"modality_a": None, "modality_b": tensors["followup_metabolomics"]},
        ]
        masks = [
            {"modality_a": 0, "modality_b": 0},
            {"modality_a": 2, "modality_b": 0},
        ]
        generated = self.model.impute_missing(visits, masks)
        output = generated[-1].get("modality_a")
        if output is None:
            raise RuntimeError("The model did not return a follow-up routine-lab reconstruction.")
        return output.detach().cpu().numpy().reshape(-1)

    def _to_pulse(self, patient: Patient) -> Tuple[List[Dict[str, Any]], List[Dict[str, int]]]:
        visits_data, missing_masks = [], []
        for visit in patient.visits:
            vec = build_visit_vector(visit)
            visits_data.append({"modality_a": vec, "modality_b": None})
            missing_masks.append({"modality_a": 0, "modality_b": 2})
        return visits_data, missing_masks

    def _batch_patient_visits(self, visits_data: List[Dict[str, Any]]) -> List[Dict[str, Optional[torch.Tensor]]]:
        """Convert one-patient feature vectors to the model's (batch, features) contract."""
        batched_visits: List[Dict[str, Optional[torch.Tensor]]] = []
        for visit_data in visits_data:
            batched_visit: Dict[str, Optional[torch.Tensor]] = {}
            for modality, values in visit_data.items():
                tensor = tensor_or_none(values)
                if tensor is not None:
                    tensor = tensor.to(device=self.device, dtype=torch.float32)
                    if tensor.ndim == 1:
                        tensor = tensor.unsqueeze(0)
                    elif tensor.ndim != 2 or tensor.shape[0] != 1:
                        raise ValueError(f"{modality} for a single Patient must have shape (features,) or (1, features).")
                batched_visit[modality] = tensor
            batched_visits.append(batched_visit)
        return batched_visits

    @torch.no_grad()
    def impute(self, patient: Patient) -> PatientResult:
        visits_data, missing_masks = self._to_pulse(patient)
        visits_t = self._batch_patient_visits(visits_data)
        recon = self.model.impute_missing(visits_t, missing_masks)

        result = PatientResult(patient_id=patient.patient_id)
        for visit, recon_visit in zip(patient.visits, recon):
            vr = VisitResult(date=visit.date)
            arr = recon_visit.get("modality_a")
            if arr is None:
                result.visits.append(vr)
                continue
            a = arr.detach().cpu().numpy().reshape(-1)
            for name, slot in LAB_SLOTS.items():
                z = float(a[slot]) if slot < len(a) else 0.0
                r = LAB_RANGES.get(name, {"mean": 0.0, "sd": 1.0, "unit": ""})
                raw = z * r["sd"] + r["mean"]
                measured = visit.labs.get(name)
                vr.imputations.append(ImputationResult(
                    biomarker=name, measured=measured, imputed_z=z, imputed_raw=raw,
                    unit=r["unit"], mean=r["mean"], sd=r["sd"]))
            result.visits.append(vr)
        return result

    @torch.no_grad()
    def latent_states(self, patient: Patient) -> List[List[float]]:
        """Return one latent visit-state vector per visit for a single patient."""
        visits_data, missing_masks = self._to_pulse(patient)
        visits_t = self._batch_patient_visits(visits_data)
        output = self.model(
            visits_t,
            missing_masks,
            return_all_visit_states=True,
        )
        states = output.get("visit_states")
        if states is None:
            return []
        return [state[0].detach().cpu().tolist() for state in states]

    def uncertainty(self, patient: Patient, n_samples: int = 8) -> Dict[str, Any]:
        """Report uncertainty status; this checkpoint does not estimate it."""
        return {
            "available": False,
            "reason": (
                "No calibrated predictive interval or uncertainty estimate is implemented "
                "for the current model/checkpoint."
            ),
        }
