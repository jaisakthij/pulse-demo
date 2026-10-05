"""Validation for preprocessed longitudinal feature CSVs."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def validate_upload_size(size_bytes: int, max_bytes: int = MAX_UPLOAD_BYTES) -> None:
    if size_bytes < 0:
        raise ValueError("Uploaded file size cannot be negative.")
    if size_bytes > max_bytes:
        if max_bytes % (1024 * 1024) == 0:
            limit = f"{max_bytes // (1024 * 1024)} MiB"
        else:
            limit = f"{max_bytes} bytes"
        raise ValueError(f"CSV exceeds the {limit} upload limit.")


def validate_feature_table(
    data: pd.DataFrame,
    input_columns: Sequence[str],
    truth_columns: Sequence[str],
    max_rows: int = 100,
) -> tuple[pd.DataFrame, bool]:
    """Validate required inputs and return a numeric copy plus truth availability."""
    if data.empty:
        raise ValueError("CSV contains no sample rows.")
    if len(data) > max_rows:
        raise ValueError(f"CSV is limited to {max_rows} samples; received {len(data)}.")
    if data.columns.duplicated().any():
        repeated = str(data.columns[data.columns.duplicated()][0])
        raise ValueError(f"CSV contains a duplicate column name: `{repeated}`.")

    required_columns = ["patient_id", "population", *input_columns]
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(
            f"CSV is missing {len(missing)} required columns. First missing: `{missing[0]}`."
        )

    present_truth = [column for column in truth_columns if column in data.columns]
    if present_truth and len(present_truth) != len(truth_columns):
        raise ValueError(
            f"For evaluation, include all {len(truth_columns)} measured follow-up columns or none."
        )

    for column in ("patient_id", "population"):
        values = data[column]
        for position, value in enumerate(values):
            if pd.isna(value) or not str(value).strip():
                raise ValueError(
                    f"CSV row {position + 2}, column '{column}' must be non-empty."
                )

    patient_ids = data["patient_id"].astype(str).str.strip()
    duplicate_mask = patient_ids.duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_id = patient_ids.loc[duplicate_mask].iloc[0]
        raise ValueError(
            f"patient_id `{duplicate_id}` must appear only once; this benchmark requires one row per patient."
        )

    numeric_columns = [*input_columns, *present_truth]
    numeric = data.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    numeric_values = numeric.to_numpy(dtype=np.float64)
    invalid = ~np.isfinite(numeric_values)
    if invalid.any():
        row_position, column_position = np.argwhere(invalid)[0]
        column = numeric_columns[column_position]
        raise ValueError(
            f"CSV row {row_position + 2}, column '{column}' must contain a finite numeric value."
        )

    validated = data.copy()
    for column in numeric_columns:
        validated[column] = numeric[column].astype(np.float32)
    return validated, bool(present_truth)
