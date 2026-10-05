from pathlib import Path
import shutil

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


REVERSE_TASK = "Predict follow-up routine labs (reverse direction)"


def test_app_displays_data_privacy_caution():
    app = AppTest.from_file("C:/Users/jaisa/OneDrive/Desktop/Soft skills music/pulse-build/app.py").run(timeout=60)

    assert any("Do not upload identifiable patient data" in item.value for item in app.warning)


def test_random_synthetic_profile_generator_creates_fresh_sample():
    app = AppTest.from_file("app.py").run(timeout=60)

    assert app.button[0].label == "Generate random synthetic profile"
    assert any("not a clinically realistic patient" in item.value for item in app.caption)

    app.button[0].click().run(timeout=60)
    first_input = next(frame.value for frame in app.dataframe if "baseline_lab_001" in frame.value.columns)
    first_id = first_input.loc[0, "patient_id"]
    first_values = first_input.loc[0, [f"baseline_lab_{i:03d}" for i in range(1, 11)]].to_numpy()
    assert str(first_id).startswith("SYNTH-")
    assert any("Generated synthetic input profile" in item.value for item in app.subheader)

    app.button[0].click().run(timeout=60)
    second_input = next(frame.value for frame in app.dataframe if "baseline_lab_001" in frame.value.columns)
    second_id = second_input.loc[0, "patient_id"]
    second_values = second_input.loc[0, [f"baseline_lab_{i:03d}" for i in range(1, 11)]].to_numpy()

    assert second_id != first_id
    assert not (first_values == second_values).all()
    assert not app.exception


def test_plain_language_infographic_explains_population_vs_individual():
    app = AppTest.from_file("app.py").run(timeout=60)
    markdown_text = "\n".join(item.value for item in app.markdown).lower()

    assert "learns shared temporal and cross-modal patterns from a cohort" in markdown_text
    assert "uses this participant's baseline paired profile as historical context" in markdown_text
    assert "does not train a separate model for each participant" in markdown_text
    assert "not a measurement or diagnosis" in markdown_text
    assert "pulse-flow-infographic" in markdown_text
    assert "prefers-reduced-motion: reduce" in markdown_text
    assert "follow-up routine labs" in markdown_text
    assert "follow-up metabolomics" in markdown_text


def test_reverse_mode_infographic_updates_input_and_estimated_output():
    app = AppTest.from_file("app.py").run(timeout=60)
    app.radio[0].set_value(REVERSE_TASK).run(timeout=60)
    markdown_text = "\n".join(item.value for item in app.markdown).lower()

    assert "measured follow-up metabolomics" in markdown_text
    assert "estimated follow-up routine labs" in markdown_text


def test_baseline_feature_pie_chart_explains_what_its_slices_mean():
    app = AppTest.from_file("app.py").run(timeout=60)
    markdown_text = "\n".join(item.value for item in app.markdown).lower()

    assert "pulse-feature-pie" in markdown_text
    assert "61 routine-lab feature slots" in markdown_text
    assert "251 metabolomics feature slots" in markdown_text
    assert "19.6%" in markdown_text
    assert "80.4%" in markdown_text
    assert "not feature importance" in markdown_text
    assert "not accuracy" in markdown_text


def test_synthetic_and_prediction_values_are_explained_in_app():
    app = AppTest.from_file("app.py").run(timeout=60)
    app.button[0].click().run(timeout=60)
    all_text = "\n".join(
        item.value
        for items in (app.markdown, app.caption, app.info)
        for item in items
    ).lower()

    input_frame = next(frame.value for frame in app.dataframe if "baseline_lab_001" in frame.value.columns)
    output_frame = next(frame.value for frame in app.dataframe if "predicted_metab_001" in frame.value.columns)
    input_example = f"baseline_lab_001 = {float(input_frame.loc[0, 'baseline_lab_001']):.4f}".lower()
    output_example = f"predicted_metab_001 = {float(output_frame.loc[0, 'predicted_metab_001']):.4f}".lower()

    assert "how to read these values" in all_text
    assert input_example in all_text
    assert "randomly generated" in all_text
    assert "not a measured lab result" in all_text
    assert output_example in all_text
    assert "not a clinical high/low" in all_text
    assert "no lab units or reference range" in all_text


def test_reverse_direction_explains_its_predicted_lab_values():
    app = AppTest.from_file("app.py").run(timeout=60)
    app.radio[0].set_value(REVERSE_TASK).run(timeout=60)
    app.button[0].click().run(timeout=60)
    all_text = "\n".join(item.value for items in (app.markdown, app.caption, app.info) for item in items).lower()

    output_frame = next(frame.value for frame in app.dataframe if "predicted_lab_001" in frame.value.columns)
    output_example = f"predicted_lab_001 = {float(output_frame.loc[0, 'predicted_lab_001']):.4f}".lower()
    assert output_example in all_text
    assert "follow-up routine labs" in all_text


def test_missing_metrics_message_offers_an_evaluation_workflow():
    app = AppTest.from_file("app.py").run(timeout=60)

    download_labels = [button.label for button in app.get("download_button")]
    assert any("Download evaluation CSV template" in label for label in download_labels)

    app.button[0].click().run(timeout=60)
    visible_text = "\n".join(
        item.value
        for items in (app.info, app.caption, app.markdown)
        for item in items
    ).lower()
    assert "no measured follow-up metabolomics supplied" in visible_text
    assert "predictions alone cannot be scored" in visible_text
    assert "fill every measured outcome column" in visible_text
    assert "do not use predictions as measured truth" in visible_text


def test_app_explains_paper_benchmark_and_requires_holdout_attestation_for_metrics():
    app = AppTest.from_file("app.py").run(timeout=60)
    text = "\n".join(
        item.value
        for items in (app.markdown, app.caption, app.info, app.warning)
        for item in items
    ).lower()

    assert "70/30" in text
    assert "7,000" in text and "3,000" in text
    assert "training" in text and "z-score normalization for the static comparators" in text
    assert "does not establish that same preprocessing for main pulse" in text
    assert "static comparators receive only follow-up routine labs" in text
    assert "fig. 2c" in text and "r = 0.850" in text
    assert "all visits" in text and "patient-level" in text
    assert "midas" in text and "scvaeit" in text and "stabmap" in text
    assert "cited paper's uk biobank benchmark reports 61 routine-lab and 251 metabolomics features" in text
    assert "public framework example" not in text
    assert len(app.checkbox) == 1
    assert "held-out" in app.checkbox[0].label.lower()
    assert app.checkbox[0].value is False
    assert not app.exception


def test_app_withholds_measured_outcome_scores_until_holdout_is_confirmed():
    rows = [{"patient_id": patient_id, "population": "Test cohort"} for patient_id in ("TEST-1", "TEST-2")]
    for index in range(1, 62):
        rows[0][f"baseline_lab_{index:03d}"] = index / 100.0
        rows[1][f"baseline_lab_{index:03d}"] = index / 100.0 + 0.1
        rows[0][f"followup_lab_{index:03d}"] = index / 200.0
        rows[1][f"followup_lab_{index:03d}"] = index / 200.0 + 0.1
    for index in range(1, 252):
        rows[0][f"baseline_metab_{index:03d}"] = index / 1000.0
        rows[1][f"baseline_metab_{index:03d}"] = index / 1000.0 + 0.1
        rows[0][f"measured_followup_metab_{index:03d}"] = index / 1000.0 + 0.25
        rows[1][f"measured_followup_metab_{index:03d}"] = index / 1000.0 + 0.35
    csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")

    app = AppTest.from_file("app.py").run(timeout=60)
    app.file_uploader[0].set_value(("heldout.csv", csv_bytes, "text/csv")).run(timeout=60)
    app.button[1].click().run(timeout=60)

    assert any("scores are withheld" in item.value.lower() for item in app.warning)
    assert not any("Operator-attested descriptive scores" in item.value for item in app.subheader)

    app.checkbox[0].set_value(True).run(timeout=60)
    app.button[1].click().run(timeout=60)

    assert not app.exception, [str(item.message) for item in app.exception]
    assert any("Operator-attested descriptive scores" in item.value for item in app.subheader)
    assert any("Checkpoint vs carry-forward baseline" in item.value for item in app.subheader)
    assert any("Paired feature-level comparison" in item.value for item in app.subheader)
    wilcoxon_frame = next(
        frame.value for frame in app.dataframe
        if {"Metric", "Wilcoxon p (two-sided)"}.issubset(frame.value.columns)
    )
    assert set(wilcoxon_frame["Metric"]) == {"pearson_r", "mae", "mse"}
    comparison = next(
        frame.value for frame in app.dataframe
        if {"Method", "Median MAE"}.issubset(frame.value.columns)
    )
    carry_forward_mae = comparison.loc[
        comparison["Method"] == "Carry-forward baseline", "Median MAE"
    ].iloc[0]
    assert carry_forward_mae == pytest.approx(0.25)
    assert any("Across-feature metric distributions" in item.value for item in app.subheader)
    assert any("Longitudinal change agreement" in item.value for item in app.subheader)
    change_comparison = next(
        frame.value for frame in app.dataframe
        if "Median change Pearson r" in frame.value.columns
    )
    baseline_change_r = change_comparison.loc[
        change_comparison["Method"] == "Carry-forward baseline", "Median change Pearson r"
    ].iloc[0]
    assert pd.isna(baseline_change_r)
    assert any("Figure 2c-style change plot" in item.value for item in app.subheader)
    assert any("Select placeholder feature for change plot" == item.label for item in app.selectbox)
    assert any("0.850" in item.value for item in app.markdown)
    assert any("Top 10 placeholder features by Pearson r" in item.label for item in app.expander)
    assert any("Population-stratified held-out results" in item.value for item in app.subheader)
    population_frame = next(
        frame.value for frame in app.dataframe
        if {"Population", "Status", "Method"}.issubset(frame.value.columns)
    )
    assert set(population_frame["Status"]) == {"Withheld (fewer than 5)"}


def test_reverse_task_carry_forward_uses_baseline_labs():
    rows = [{"patient_id": patient_id, "population": "Test cohort"} for patient_id in ("REV-1", "REV-2")]
    for index in range(1, 62):
        rows[0][f"baseline_lab_{index:03d}"] = index / 100.0
        rows[1][f"baseline_lab_{index:03d}"] = index / 100.0 + 0.1
        rows[0][f"measured_followup_lab_{index:03d}"] = index / 100.0 + 0.4
        rows[1][f"measured_followup_lab_{index:03d}"] = index / 100.0 + 0.5
    for index in range(1, 252):
        rows[0][f"baseline_metab_{index:03d}"] = index / 1000.0
        rows[1][f"baseline_metab_{index:03d}"] = index / 1000.0 + 0.1
        rows[0][f"followup_metab_{index:03d}"] = index / 900.0
        rows[1][f"followup_metab_{index:03d}"] = index / 900.0 + 0.1
    csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")

    app = AppTest.from_file("app.py").run(timeout=60)
    app.radio[0].set_value(REVERSE_TASK).run(timeout=60)
    app.file_uploader[0].set_value(("reverse-heldout.csv", csv_bytes, "text/csv")).run(timeout=60)
    app.checkbox[0].set_value(True).run(timeout=60)
    app.button[1].click().run(timeout=60)

    assert not app.exception, [str(item.message) for item in app.exception]
    comparison = next(
        frame.value for frame in app.dataframe
        if {"Method", "Median MAE"}.issubset(frame.value.columns)
    )
    carry_forward_mae = comparison.loc[
        comparison["Method"] == "Carry-forward baseline", "Median MAE"
    ].iloc[0]
    assert carry_forward_mae == pytest.approx(0.4)


def test_corrupt_checkpoint_is_reported_without_crashing(tmp_path, monkeypatch):
    checkpoint = tmp_path / "corrupt.pt"
    checkpoint.write_bytes(b"not a valid checkpoint")
    monkeypatch.setenv("PULSE_CKPT", str(checkpoint))

    app = AppTest.from_file("app.py").run(timeout=60)

    assert not app.exception
    assert any("Could not load checkpoint" in item.value for item in app.error)
    assert all(button.disabled for button in app.button)

    valid_checkpoint = Path(__file__).parent / "ckpt_small" / "checkpoint_epoch_60.pt"
    shutil.copyfile(valid_checkpoint, checkpoint)
    app.run(timeout=60)

    assert any("Checkpoint loaded" in item.value for item in app.success)
    assert not app.exception


def test_demo_results_survive_unrelated_widget_rerun():
    app = AppTest.from_file("app.py").run(timeout=60)
    app.button[0].click().run(timeout=60)
    assert any("Generated follow-up metabolomics" in header.value for header in app.subheader)

    app.text_input[0].set_value("EDITED-AFTER-RUN").run(timeout=60)

    assert any("Generated follow-up metabolomics" in header.value for header in app.subheader)
    assert any("most recent run" in item.value.lower() for item in app.caption)


def test_reverse_lab_task_can_be_selected_and_run():
    app = AppTest.from_file("app.py").run(timeout=60)

    assert len(app.radio) == 1
    assert REVERSE_TASK in app.radio[0].options

    app.radio[0].set_value(REVERSE_TASK).run(timeout=60)
    assert any("Reverse modality direction" in message.value for message in app.caption)

    app.button[0].click().run(timeout=60)

    assert not app.exception, [str(exception.message) for exception in app.exception]
    assert any("Generated follow-up routine labs" in header.value for header in app.subheader)
    assert any("No measured follow-up" in message.value for message in app.info)
