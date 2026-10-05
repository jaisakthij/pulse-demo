import numpy as np
import pandas as pd
import pytest

from pulse_build.validation import validate_feature_table, validate_upload_size


INPUT_COLS = ["baseline_lab_001"]
TRUTH_COLS = ["measured_followup_lab_001", "measured_followup_lab_002"]


def _table(**extra_columns):
    data = {"patient_id": ["P1", "P2"], "population": ["A", "A"], INPUT_COLS[0]: [1.0, 2.0]}
    data.update(extra_columns)
    return pd.DataFrame(data)


def test_upload_size_limit_accepts_boundary_and_rejects_oversize():
    assert validate_upload_size(10, max_bytes=10) is None

    with pytest.raises(ValueError, match="10 bytes"):
        validate_upload_size(11, max_bytes=10)


def test_validates_and_normalizes_numeric_feature_columns():
    table = _table(measured_followup_lab_001=[3, 4], measured_followup_lab_002=[5, 6])

    validated, has_truth = validate_feature_table(table, INPUT_COLS, TRUTH_COLS)

    assert has_truth is True
    assert validated[INPUT_COLS[0]].dtype.kind == "f"
    assert validated[TRUTH_COLS].to_numpy().tolist() == [[3.0, 5.0], [4.0, 6.0]]


def test_rejects_nonfinite_input_with_row_and_column():
    table = _table()
    table.loc[1, INPUT_COLS[0]] = np.inf

    with pytest.raises(ValueError, match=r"row 3, column 'baseline_lab_001'.*finite"):
        validate_feature_table(table, INPUT_COLS, TRUTH_COLS)


def test_rejects_partial_measured_outcome_columns():
    table = _table(measured_followup_lab_001=[3, 4])

    with pytest.raises(ValueError, match="include all 2 measured follow-up columns or none"):
        validate_feature_table(table, INPUT_COLS, TRUTH_COLS)


def test_rejects_blank_sample_metadata():
    table = _table()
    table.loc[0, "patient_id"] = " "

    with pytest.raises(ValueError, match="row 2, column 'patient_id'.*non-empty"):
        validate_feature_table(table, INPUT_COLS, TRUTH_COLS)


def test_rejects_too_many_rows():
    table = _table()

    with pytest.raises(ValueError, match="limited to 1 samples"):
        validate_feature_table(table, INPUT_COLS, TRUTH_COLS, max_rows=1)


def test_rejects_duplicate_patient_ids_for_one_row_per_person_benchmark():
    table = _table()
    table.loc[1, "patient_id"] = " P1 "

    with pytest.raises(ValueError, match="patient_id.*appear only once"):
        validate_feature_table(table, INPUT_COLS, TRUTH_COLS)
