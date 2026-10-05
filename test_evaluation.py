import numpy as np
import pandas as pd
import pytest

from pulse_build.evaluation import (
    compute_confirmed_heldout_benchmark_metrics,
    compute_feature_benchmark_metrics,
    compute_longitudinal_change_vectors,
    build_longitudinal_change_scatter_data,
    compute_population_benchmark_summary,
    compute_paired_feature_wilcoxon,
    compute_carry_forward_predictions,
    create_evaluation_template,
    feature_metrics_to_long_frame,
    feature_metric_distributions_by_method,
)


def test_metrics_summarize_featurewise_pearson_and_errors():
    predictions = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    measured = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 2.0]])

    metrics = compute_feature_benchmark_metrics(predictions, measured)

    assert metrics["pearson_r"]["mean"] == pytest.approx(0.5)
    assert metrics["pearson_r"]["median"] == pytest.approx(0.5)
    assert metrics["mae"]["mean"] == pytest.approx(2 / 3)
    assert metrics["mse"]["mean"] == pytest.approx(8 / 3)
    assert metrics["pearson_r"]["n_valid_features"] == 2
    assert metrics["n_samples"] == 3
    assert metrics["feature_metrics"][0]["feature_id"] == "metab_001"
    assert metrics["feature_metrics"][0]["pearson_r"] == pytest.approx(1.0)
    assert metrics["feature_metrics"][1]["pearson_r"] == pytest.approx(0.0)


def test_metrics_mark_pearson_unavailable_for_single_sample():
    metrics = compute_feature_benchmark_metrics([[1.0, 2.0]], [[2.0, 4.0]])

    assert metrics["pearson_r"] is None
    assert metrics["mse"]["n_valid_features"] == 2
    assert metrics["mae"]["mean"] == pytest.approx(1.5)
    assert all(row["pearson_r"] is None for row in metrics["feature_metrics"])


def test_metrics_support_lab_feature_ids():
    metrics = compute_feature_benchmark_metrics(
        [[1.0, 2.0], [2.0, 3.0]],
        [[1.0, 2.0], [2.0, 3.0]],
        feature_prefix="lab",
    )

    assert metrics["feature_metrics"][0]["feature_id"] == "lab_001"


def test_evaluation_template_leaves_inputs_and_measured_outcomes_blank():
    template = create_evaluation_template(
        input_columns=["baseline_lab_001", "followup_lab_001"],
        truth_columns=["measured_followup_metab_001"],
    )

    assert list(template.columns) == [
        "patient_id",
        "population",
        "baseline_lab_001",
        "followup_lab_001",
        "measured_followup_metab_001",
    ]
    assert template.loc[0, "patient_id"] == "SAMPLE-001"
    assert template.loc[0, "population"] == "Unspecified"
    assert template.loc[0, "baseline_lab_001"] == ""
    assert template.loc[0, "followup_lab_001"] == ""
    assert template.loc[0, "measured_followup_metab_001"] == ""


def test_evaluation_template_rejects_duplicate_columns():
    with pytest.raises(ValueError, match="unique"):
        create_evaluation_template(
            input_columns=["baseline_lab_001"],
            truth_columns=["baseline_lab_001"],
        )


def test_metrics_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        compute_feature_benchmark_metrics([[1.0, 2.0]], [[1.0]])


def test_heldout_benchmark_metrics_are_withheld_without_patient_level_confirmation():
    metrics = compute_confirmed_heldout_benchmark_metrics(
        [[1.0, 2.0], [2.0, 3.0]],
        [[1.1, 1.9], [1.9, 3.1]],
        heldout_confirmed=False,
    )

    assert metrics is None


def test_heldout_benchmark_metrics_run_after_patient_level_confirmation():
    metrics = compute_confirmed_heldout_benchmark_metrics(
        [[1.0, 2.0], [2.0, 3.0]],
        [[1.1, 1.9], [1.9, 3.1]],
        heldout_confirmed=True,
    )

    assert metrics is not None
    assert metrics["n_samples"] == 2


def test_feature_metrics_long_frame_omits_unavailable_correlations():
    frame = feature_metrics_to_long_frame([
        {"feature_id": "metab_001", "pearson_r": 0.8, "mae": 0.2, "mse": 0.04},
        {"feature_id": "metab_002", "pearson_r": None, "mae": 0.3, "mse": 0.09},
    ])

    assert list(frame.columns) == ["feature_id", "metric", "score"]
    assert len(frame) == 5
    assert not frame.isna().any().any()
    assert set(frame["metric"]) == {"pearson_r", "mae", "mse"}
    assert frame.loc[frame["feature_id"] == "metab_002", "metric"].tolist() == ["mae", "mse"]


def test_carry_forward_predictions_select_baseline_features_for_target_modality():
    baseline_labs = [[10.0, 20.0], [11.0, 21.0]]
    baseline_metabolomics = [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]

    assert compute_carry_forward_predictions(
        baseline_labs, baseline_metabolomics, target_modality="metabolomics"
    ).tolist() == baseline_metabolomics
    assert compute_carry_forward_predictions(
        baseline_labs, baseline_metabolomics, target_modality="routine labs"
    ).tolist() == baseline_labs


def test_feature_metric_distribution_frame_preserves_method_and_feature_rows():
    frame = feature_metric_distributions_by_method({
        "Loaded checkpoint": [
            {"feature_id": "metab_001", "pearson_r": 0.8, "mae": 0.2, "mse": 0.04},
            {"feature_id": "metab_002", "pearson_r": None, "mae": 0.3, "mse": 0.09},
        ],
        "Carry-forward baseline": [
            {"feature_id": "metab_001", "pearson_r": 0.5, "mae": 0.4, "mse": 0.16},
            {"feature_id": "metab_002", "pearson_r": 0.6, "mae": 0.5, "mse": 0.25},
        ],
    })

    assert list(frame.columns) == ["series", "feature_id", "metric", "score"]
    assert len(frame) == 11
    assert set(frame["series"]) == {"Loaded checkpoint", "Carry-forward baseline"}
    assert frame.loc[
        (frame["series"] == "Carry-forward baseline")
        & (frame["feature_id"] == "metab_002")
        & (frame["metric"] == "mae"),
        "score",
    ].item() == pytest.approx(0.5)


def test_longitudinal_change_vectors_subtract_each_persons_baseline():
    predicted_change, measured_change = compute_longitudinal_change_vectors(
        predictions=[[3.0, 4.0], [7.0, 8.0]],
        measured=[[4.0, 7.0], [8.0, 9.0]],
        baseline=[[1.0, 2.0], [5.0, 6.0]],
    )

    assert predicted_change.tolist() == [[2.0, 2.0], [2.0, 2.0]]
    assert measured_change.tolist() == [[3.0, 5.0], [3.0, 3.0]]


def test_longitudinal_change_vectors_reject_shape_mismatches():
    with pytest.raises(ValueError, match="same shape"):
        compute_longitudinal_change_vectors(
            predictions=[[1.0, 2.0]], measured=[[1.0]], baseline=[[0.0, 0.0]]
        )


def test_population_summary_withholds_small_groups_and_scores_eligible_groups():
    predictions = np.arange(18, dtype=float).reshape(9, 2)
    measured = predictions + 1.0
    populations = ["Alpha"] * 5 + ["Beta"] * 4

    summary = compute_population_benchmark_summary(
        predictions, measured, populations, heldout_confirmed=True, minimum_group_size=5
    )

    assert summary is not None
    alpha = summary.loc[summary["population"] == "Alpha"].iloc[0]
    beta = summary.loc[summary["population"] == "Beta"].iloc[0]
    assert alpha["status"] == "Summarized"
    assert alpha["n_samples"] == 5
    assert alpha["median_mae"] is not None
    assert beta["status"] == "Withheld (fewer than 5)"
    assert beta["n_samples"] == "<5"
    assert pd.isna(beta["median_mae"])


def test_population_summary_is_withheld_without_holdout_attestation():
    summary = compute_population_benchmark_summary(
        [[1.0], [2.0]], [[1.1], [1.9]], ["A", "A"], heldout_confirmed=False
    )

    assert summary is None


def test_change_scatter_data_includes_checkpoint_persistence_and_identity_line():
    frame = build_longitudinal_change_scatter_data(
        predictions=[[3.0, 10.0], [5.0, 14.0]],
        measured=[[4.0, 11.0], [6.0, 15.0]],
        baseline=[[1.0, 10.0], [2.0, 12.0]],
        feature_index=0,
    )

    assert list(frame.columns) == ["method", "kind", "measured_change", "predicted_change"]
    assert "patient_id" not in frame.columns
    model = frame.loc[(frame["method"] == "Loaded checkpoint") & (frame["kind"] == "Observation")]
    baseline = frame.loc[(frame["method"] == "Carry-forward baseline") & (frame["kind"] == "Observation")]
    identity = frame.loc[frame["kind"] == "Identity line"]
    assert model["measured_change"].tolist() == [3.0, 4.0]
    assert model["predicted_change"].tolist() == [2.0, 3.0]
    assert baseline["predicted_change"].tolist() == [0.0, 0.0]
    assert identity["measured_change"].tolist() == identity["predicted_change"].tolist()
    assert len(identity) == 2


def test_change_scatter_data_rejects_out_of_range_feature():
    with pytest.raises(ValueError, match="feature_index"):
        build_longitudinal_change_scatter_data(
            predictions=[[1.0]], measured=[[2.0]], baseline=[[0.0]], feature_index=1
        )


def test_paired_wilcoxon_aligns_by_feature_id_and_reports_differences():
    primary = [
        {"feature_id": "metab_001", "pearson_r": 0.8, "mae": 0.4, "mse": 0.3},
        {"feature_id": "metab_002", "pearson_r": 0.6, "mae": 0.4, "mse": 0.5},
        {"feature_id": "metab_003", "pearson_r": None, "mae": 0.4, "mse": 0.7},
    ]
    comparator = [
        {"feature_id": "metab_002", "pearson_r": 0.4, "mae": 0.1, "mse": 0.2},
        {"feature_id": "metab_001", "pearson_r": 0.7, "mae": 0.1, "mse": 0.1},
        {"feature_id": "metab_003", "pearson_r": None, "mae": 0.1, "mse": 0.3},
    ]

    result = compute_paired_feature_wilcoxon(primary, comparator)

    mae = result.loc[result["metric"] == "mae"].iloc[0]
    pearson = result.loc[result["metric"] == "pearson_r"].iloc[0]
    assert mae["n_pairs"] == 3
    assert mae["median_primary_minus_comparator"] == pytest.approx(0.3)
    assert pearson["n_pairs"] == 2
    assert np.isfinite(mae["p_value"])


def test_paired_wilcoxon_handles_all_zero_differences_and_rejects_id_mismatch():
    same = [
        {"feature_id": "x", "pearson_r": 0.5, "mae": 1.0, "mse": 1.0},
        {"feature_id": "y", "pearson_r": 0.7, "mae": 2.0, "mse": 4.0},
    ]
    result = compute_paired_feature_wilcoxon(same, same)
    assert set(result["p_value"]) == {1.0}

    with pytest.raises(ValueError, match="same feature IDs"):
        compute_paired_feature_wilcoxon(
            same, [{"feature_id": "y", "pearson_r": 0.5, "mae": 1.0, "mse": 1.0}]
        )
