"""Benchmark metrics for predicted versus measured feature vectors."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def create_evaluation_template(
    input_columns: Sequence[str],
    truth_columns: Sequence[str],
) -> pd.DataFrame:
    """Return a blank upload template with explicit measured outcome columns."""
    if not input_columns:
        raise ValueError("At least one input column is required.")
    if not truth_columns:
        raise ValueError("At least one measured outcome column is required.")
    columns = ["patient_id", "population", *input_columns, *truth_columns]
    if len(columns) != len(set(columns)):
        raise ValueError("Template column names must be unique.")
    row = {"patient_id": "SAMPLE-001", "population": "Unspecified"}
    row.update({column: "" for column in [*input_columns, *truth_columns]})
    return pd.DataFrame([row], columns=columns)


def compute_feature_benchmark_metrics(
    predictions: Any,
    measured: Any,
    feature_prefix: str = "metab",
) -> dict[str, Any]:
    """Summarize feature-wise Pearson r, MAE, and MSE over samples x features.

    Pearson r is calculated across samples separately for each feature, then
    summarized across non-constant features. MAE and MSE are calculated per
    feature across samples and summarized across features. At least two samples
    are needed for Pearson r.
    """
    pred = np.asarray(predictions, dtype=np.float64)
    truth = np.asarray(measured, dtype=np.float64)
    if pred.ndim != 2 or truth.ndim != 2:
        raise ValueError("predictions and measured must both be 2D arrays (samples, features).")
    if pred.shape != truth.shape:
        raise ValueError("predictions and measured must have the same shape.")
    if pred.shape[0] == 0 or pred.shape[1] == 0:
        raise ValueError("At least one sample and one feature are required.")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("predictions and measured must contain only finite values.")

    def summarize(values: list[float]) -> dict[str, float | int] | None:
        if not values:
            return None
        array = np.asarray(values, dtype=np.float64)
        q1, median, q3 = np.percentile(array, [25, 50, 75])
        return {
            "mean": float(np.mean(array)),
            "median": float(median),
            "q1": float(q1),
            "q3": float(q3),
            "n_valid_features": int(array.size),
        }

    difference = pred - truth
    feature_mae = np.mean(np.abs(difference), axis=0)
    feature_mse = np.mean(np.square(difference), axis=0)
    feature_correlations: list[float] = []
    feature_metrics: list[dict[str, float | str | None]] = []
    for feature_index in range(pred.shape[1]):
        predicted_feature = pred[:, feature_index]
        measured_feature = truth[:, feature_index]
        correlation: float | None = None
        if pred.shape[0] >= 2 and np.ptp(predicted_feature) > 0 and np.ptp(measured_feature) > 0:
            candidate = float(np.corrcoef(predicted_feature, measured_feature)[0, 1])
            if np.isfinite(candidate):
                correlation = candidate
                feature_correlations.append(candidate)
        feature_metrics.append({
            "feature_id": f"{feature_prefix}_{feature_index + 1:03d}",
            "pearson_r": correlation,
            "mae": float(feature_mae[feature_index]),
            "mse": float(feature_mse[feature_index]),
        })

    return {
        "pearson_r": summarize(feature_correlations),
        "mae": summarize(feature_mae.tolist()),
        "mse": summarize(feature_mse.tolist()),
        "n_samples": int(pred.shape[0]),
        "n_features": int(pred.shape[1]),
        "feature_metrics": feature_metrics,
    }


def compute_confirmed_heldout_benchmark_metrics(
    predictions: Any,
    measured: Any,
    feature_prefix: str = "metab",
    *,
    heldout_confirmed: bool,
) -> dict[str, Any] | None:
    """Score only after the operator attests to patient-level holdout separation.

    The app cannot independently establish training-set separation; this flag is
    an explicit operator confirmation, not an automated leakage audit.
    """
    if not heldout_confirmed:
        return None
    return compute_feature_benchmark_metrics(
        predictions,
        measured,
        feature_prefix=feature_prefix,
    )


def feature_metrics_to_long_frame(feature_metrics: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Convert per-feature scores to chart-friendly rows, skipping unavailable values."""
    rows: list[dict[str, Any]] = []
    for feature in feature_metrics:
        feature_id = feature.get("feature_id")
        if not feature_id:
            raise ValueError("Each feature metric row must include a feature_id.")
        for metric in ("pearson_r", "mae", "mse"):
            value = feature.get(metric)
            if value is None:
                continue
            score = float(value)
            if not np.isfinite(score):
                raise ValueError(f"Feature metric `{metric}` must be finite when present.")
            rows.append({"feature_id": str(feature_id), "metric": metric, "score": score})
    return pd.DataFrame(rows, columns=["feature_id", "metric", "score"])


def compute_carry_forward_predictions(
    baseline_labs: Any,
    baseline_metabolomics: Any,
    *,
    target_modality: str,
) -> np.ndarray:
    """Copy the baseline target modality as a persistence/carry-forward prediction."""
    if target_modality == "metabolomics":
        selected = baseline_metabolomics
    elif target_modality == "routine labs":
        selected = baseline_labs
    else:
        raise ValueError("target_modality must be 'metabolomics' or 'routine labs'.")
    predictions = np.asarray(selected, dtype=np.float64)
    if predictions.ndim != 2 or predictions.shape[0] == 0 or predictions.shape[1] == 0:
        raise ValueError("The selected baseline modality must be a non-empty 2D samples-by-features array.")
    if not np.isfinite(predictions).all():
        raise ValueError("The selected baseline modality must contain only finite values.")
    return predictions.copy()


def feature_metric_distributions_by_method(
    method_metrics: dict[str, Sequence[dict[str, Any]]],
) -> pd.DataFrame:
    """Build long-form, method-labeled rows for feature-score distribution charts."""
    frames: list[pd.DataFrame] = []
    for method, feature_metrics in method_metrics.items():
        if not method.strip():
            raise ValueError("Method labels must be non-empty.")
        frame = feature_metrics_to_long_frame(feature_metrics)
        frame.insert(0, "series", method)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["series", "feature_id", "metric", "score"])
    return pd.concat(frames, ignore_index=True)


def compute_paired_feature_wilcoxon(
    primary_feature_metrics: Sequence[dict[str, Any]],
    comparator_feature_metrics: Sequence[dict[str, Any]],
) -> pd.DataFrame:
    """Compare matched feature scores with two-sided paired Wilcoxon tests.

    Differences are primary minus comparator; Pearson r favors positive values,
    whereas MAE/MSE favor negative values. This is an exploratory feature-level
    test and does not model dependence between correlated features.
    """
    def index_rows(rows: Sequence[dict[str, Any]], method: str) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        for row in rows:
            feature_id = str(row.get("feature_id", "")).strip()
            if not feature_id:
                raise ValueError(f"{method} metrics require non-empty feature IDs.")
            if feature_id in indexed:
                raise ValueError(f"{method} metrics contain duplicate feature IDs.")
            indexed[feature_id] = row
        return indexed

    primary = index_rows(primary_feature_metrics, "Primary")
    comparator = index_rows(comparator_feature_metrics, "Comparator")
    if set(primary) != set(comparator):
        raise ValueError("Primary and comparator metrics must contain the same feature IDs.")

    results: list[dict[str, Any]] = []
    for metric in ("pearson_r", "mae", "mse"):
        paired: list[tuple[float, float]] = []
        for feature_id in sorted(primary):
            left = primary[feature_id].get(metric)
            right = comparator[feature_id].get(metric)
            if left is None or right is None or pd.isna(left) or pd.isna(right):
                continue
            left_value = float(left)
            right_value = float(right)
            if not np.isfinite(left_value) or not np.isfinite(right_value):
                raise ValueError(f"Paired `{metric}` values must be finite when present.")
            paired.append((left_value, right_value))
        differences = np.asarray([left - right for left, right in paired], dtype=np.float64)
        if differences.size < 2:
            p_value = None
            status = "Insufficient paired features"
        elif np.all(differences == 0):
            p_value = 1.0
            status = "All paired differences are zero"
        else:
            p_value = float(wilcoxon(
                [left for left, _ in paired],
                [right for _, right in paired],
                alternative="two-sided",
                zero_method="pratt",
                method="auto",
            ).pvalue)
            status = "Exploratory; correlated features are not independent"
        results.append({
            "metric": metric,
            "n_pairs": int(differences.size),
            "median_primary_minus_comparator": float(np.median(differences)) if differences.size else None,
            "p_value": p_value,
            "status": status,
        })
    return pd.DataFrame(results, columns=[
        "metric", "n_pairs", "median_primary_minus_comparator", "p_value", "status"
    ])


def compute_population_benchmark_summary(
    predictions: Any,
    measured: Any,
    populations: Sequence[Any],
    feature_prefix: str = "metab",
    *,
    heldout_confirmed: bool,
    minimum_group_size: int = 5,
) -> pd.DataFrame | None:
    """Summarize held-out metrics by population, withholding very small groups.

    The population label is post-hoc metadata; it is not supplied to the model.
    Small-group withholding is a display precaution, not a privacy guarantee or
    a statistical sufficiency threshold.
    """
    if not heldout_confirmed:
        return None
    pred = np.asarray(predictions, dtype=np.float64)
    truth = np.asarray(measured, dtype=np.float64)
    if pred.ndim != 2 or truth.ndim != 2 or pred.shape != truth.shape:
        raise ValueError("predictions and measured must be same-shape 2D arrays.")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("predictions and measured must contain only finite values.")
    if len(populations) != pred.shape[0]:
        raise ValueError("Each sample must have exactly one population label.")
    if minimum_group_size < 2:
        raise ValueError("minimum_group_size must be at least 2.")
    labels: list[str] = []
    for index, value in enumerate(populations):
        if pd.isna(value) or not str(value).strip():
            raise ValueError(f"Population label at sample {index} must be non-empty.")
        labels.append(str(value).strip())

    rows: list[dict[str, Any]] = []
    for population in sorted(set(labels), key=str.casefold):
        mask = np.asarray([label == population for label in labels], dtype=bool)
        sample_count = int(mask.sum())
        row: dict[str, Any] = {
            "population": population,
            "n_samples": sample_count if sample_count >= minimum_group_size else f"<{minimum_group_size}",
            "status": "Summarized" if sample_count >= minimum_group_size else f"Withheld (fewer than {minimum_group_size})",
            "median_pearson_r": None,
            "median_mae": None,
            "median_mse": None,
        }
        if sample_count >= minimum_group_size:
            metrics = compute_feature_benchmark_metrics(pred[mask], truth[mask], feature_prefix=feature_prefix)
            for metric_key in ("pearson_r", "mae", "mse"):
                summary = metrics[metric_key]
                row[f"median_{metric_key}"] = None if summary is None else summary["median"]
        rows.append(row)
    return pd.DataFrame(rows, columns=[
        "population", "n_samples", "status", "median_pearson_r", "median_mae", "median_mse"
    ])


def build_longitudinal_change_scatter_data(
    predictions: Any,
    measured: Any,
    baseline: Any,
    feature_index: int,
) -> pd.DataFrame:
    """Build an ID-free feature-level change scatter for two methods and y=x."""
    predicted = np.asarray(predictions, dtype=np.float64)
    observed = np.asarray(measured, dtype=np.float64)
    baseline_values = np.asarray(baseline, dtype=np.float64)
    if predicted.ndim != 2 or observed.ndim != 2 or baseline_values.ndim != 2:
        raise ValueError("predictions, measured, and baseline must be 2D arrays.")
    if predicted.shape != observed.shape or predicted.shape != baseline_values.shape:
        raise ValueError("predictions, measured, and baseline must have the same shape.")
    if not np.isfinite(predicted).all() or not np.isfinite(observed).all() or not np.isfinite(baseline_values).all():
        raise ValueError("predictions, measured, and baseline must contain only finite values.")
    if isinstance(feature_index, bool) or not isinstance(feature_index, (int, np.integer)):
        raise ValueError("feature_index must be an integer.")
    if feature_index < 0 or feature_index >= predicted.shape[1]:
        raise ValueError("feature_index is outside the feature range.")

    predicted_change, measured_change = compute_longitudinal_change_vectors(
        predicted, observed, baseline_values
    )
    x_values = measured_change[:, feature_index]
    methods = {
        "Loaded checkpoint": predicted_change[:, feature_index],
        "Carry-forward baseline": np.zeros(predicted.shape[0], dtype=np.float64),
    }
    rows: list[dict[str, Any]] = []
    for method, y_values in methods.items():
        rows.extend({
            "method": method,
            "kind": "Observation",
            "measured_change": float(x_value),
            "predicted_change": float(y_value),
        } for x_value, y_value in zip(x_values, y_values))
    lower = float(min(np.min(x_values), *(np.min(values) for values in methods.values())))
    upper = float(max(np.max(x_values), *(np.max(values) for values in methods.values())))
    if lower == upper:
        lower -= 0.5
        upper += 0.5
    rows.extend({
        "method": "Ideal agreement",
        "kind": "Identity line",
        "measured_change": value,
        "predicted_change": value,
    } for value in (lower, upper))
    return pd.DataFrame(rows, columns=["method", "kind", "measured_change", "predicted_change"])


def compute_longitudinal_change_vectors(
    predictions: Any,
    measured: Any,
    baseline: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted and measured changes from each participant's baseline."""
    predicted = np.asarray(predictions, dtype=np.float64)
    observed = np.asarray(measured, dtype=np.float64)
    baseline_values = np.asarray(baseline, dtype=np.float64)
    if predicted.ndim != 2 or observed.ndim != 2 or baseline_values.ndim != 2:
        raise ValueError("predictions, measured, and baseline must be 2D samples-by-features arrays.")
    if predicted.shape != observed.shape or predicted.shape != baseline_values.shape:
        raise ValueError("predictions, measured, and baseline must have the same shape.")
    if not np.isfinite(predicted).all() or not np.isfinite(observed).all() or not np.isfinite(baseline_values).all():
        raise ValueError("predictions, measured, and baseline must contain only finite values.")
    return predicted - baseline_values, observed - baseline_values
