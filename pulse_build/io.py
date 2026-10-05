"""I/O helpers: load a patient's sparse routine labs, normalise, run PULSE imputation."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Reference ranges — these are illustrative population stats, NOT medical
# thresholds.  Values are z-scored against them before the model sees them.
# --------------------------------------------------------------------------- #
LAB_RANGES: Dict[str, Dict[str, float]] = {
    "glucose":       {"mean": 95.0, "sd": 15.0,  "unit": "mg/dL"},
    "ldl":           {"mean": 115.0, "sd": 35.0,  "unit": "mg/dL"},
    "hdl":           {"mean": 55.0,  "sd": 15.0,  "unit": "mg/dL"},
    "triglycerides": {"mean": 130.0, "sd": 70.0,  "unit": "mg/dL"},
    "hba1c":         {"mean": 5.4,  "sd": 0.6,   "unit": "%"},
    "crp":           {"mean": 2.1,  "sd": 4.5,   "unit": "mg/L"},
    "creatinine":    {"mean": 0.95, "sd": 0.25,  "unit": "mg/dL"},
    "alt":           {"mean": 22.0, "sd": 12.0,  "unit": "U/L"},
    "ast":           {"mean": 25.0, "sd": 11.0,  "unit": "U/L"},
    "bmi":           {"mean": 27.0, "sd": 5.0,   "unit": "kg/m2"},
    "systolic_bp":   {"mean": 122.0,"sd": 18.0,  "unit": "mmHg"},
    "heart_rate":    {"mean": 72.0, "sd": 10.0,  "unit": "bpm"},
}

# Map a routine lab name -> position in the 61-dim "modality_a" vector.
# The remaining slots are other routine labs we don't expose in the UI.
LAB_DIM = 61
LAB_SLOTS: Dict[str, int] = {
    "glucose": 0, "ldl": 1, "hdl": 2, "triglycerides": 3,
    "hba1c": 4, "crp": 5, "creatinine": 6, "alt": 7,
    "ast": 8, "bmi": 9, "systolic_bp": 10, "heart_rate": 11,
}


@dataclass
class PatientVisit:
    """One visit's worth of routine labs (sparse — many may be missing)."""
    date: str
    labs: Dict[str, float] = field(default_factory=dict)


@dataclass
class Patient:
    patient_id: str
    age: int
    sex: str
    ancestry: str
    site: str
    visits: List[PatientVisit] = field(default_factory=list)


def zscore(value: float, lab: str) -> float:
    """Normalise a lab value against the population reference range."""
    r = LAB_RANGES.get(lab)
    if not r or r["sd"] <= 0:
        return 0.0
    return (float(value) - r["mean"]) / r["sd"]


def build_visit_vector(visit: PatientVisit) -> np.ndarray:
    """Pack a visit's sparse labs into the 61-dim modality_a vector (NaN = missing)."""
    vec = np.full(LAB_DIM, np.nan, dtype=np.float32)
    for name, value in visit.labs.items():
        slot = LAB_SLOTS.get(name)
        if slot is not None and 0 <= slot < LAB_DIM:
            vec[slot] = zscore(value, name)
    return vec


def visits_to_pulse_format(patient: Patient):
    """Convert a Patient into PULSE's (visits_data, missing_masks) contract."""
    visits_data: List[Dict[str, Any]] = []
    missing_masks: List[Dict[str, int]] = []
    for visit in patient.visits:
        vec = build_visit_vector(visit)
        # modality_a = routine labs; modality_b = deep profile (always unobserved)
        visits_data.append({"modality_a": vec, "modality_b": None})
        missing_masks.append({"modality_a": 0, "modality_b": 2})
    return visits_data, missing_masks


def load_patient_from_json(path: str) -> Patient:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return Patient(
        patient_id=str(raw.get("patient_id", "unknown")),
        age=int(raw.get("age", 0)),
        sex=str(raw.get("sex", "unknown")),
        ancestry=str(raw.get("ancestry", "unknown")),
        site=str(raw.get("site", "unknown")),
        visits=[PatientVisit(date=v["date"], labs=dict(v.get("labs", {}))) for v in raw.get("visits", [])],
    )


def save_patient(patient: Patient, path: str) -> None:
    raw = {
        "patient_id": patient.patient_id,
        "age": patient.age,
        "sex": patient.sex,
        "ancestry": patient.ancestry,
        "site": patient.site,
        "visits": [{"date": v.date, "labs": v.labs} for v in patient.visits],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(raw, fh, indent=2)


def tensor_or_none(x) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    arr = np.asarray(x, dtype=np.float32)
    arr = np.where(np.isnan(arr), 0.0, arr)
    return torch.as_tensor(arr)