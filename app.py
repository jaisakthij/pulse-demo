"""Paper-task-shaped PULSE research demo; not the paper's trained model."""
from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import streamlit as st

from pulse_build import PulseEngine
from pulse_build.evaluation import (
    compute_confirmed_heldout_benchmark_metrics,
    compute_carry_forward_predictions,
    compute_longitudinal_change_vectors,
    compute_population_benchmark_summary,
    compute_paired_feature_wilcoxon,
    build_longitudinal_change_scatter_data,
    create_evaluation_template,
    feature_metric_distributions_by_method,
)
from pulse_build.validation import validate_feature_table, validate_upload_size
from pulse_build.synthetic import generate_synthetic_profile

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "ckpt_small" / "checkpoint_epoch_60.pt"
CHECKPOINT = Path(os.environ.get("PULSE_CKPT", str(DEFAULT_CHECKPOINT)))
LAB_DIM = 61
METAB_DIM = 251
LAB_COLS = [f"baseline_lab_{i:03d}" for i in range(1, LAB_DIM + 1)]
METAB_COLS = [f"baseline_metab_{i:03d}" for i in range(1, METAB_DIM + 1)]
FOLLOWUP_LAB_COLS = [f"followup_lab_{i:03d}" for i in range(1, LAB_DIM + 1)]
FOLLOWUP_METAB_COLS = [f"followup_metab_{i:03d}" for i in range(1, METAB_DIM + 1)]
MEASURED_FOLLOWUP_LAB_COLS = [f"measured_followup_lab_{i:03d}" for i in range(1, LAB_DIM + 1)]
MEASURED_FOLLOWUP_METAB_COLS = [f"measured_followup_metab_{i:03d}" for i in range(1, METAB_DIM + 1)]
BASELINE_COLS = LAB_COLS + METAB_COLS

st.set_page_config(page_title="PULSE · Longitudinal Multimodal Demo", page_icon="🧬", layout="wide")
st.title("🧬 PULSE · Longitudinal Multimodal Imputation")
st.markdown("**Research use only · Not a medical device · Longitudinal multimodal signal imputation from sparse routine labs**")

with st.expander("🚀 Quick Start Guide", expanded=False):
    st.markdown("""
    **Three ways to explore PULSE:**
    
    1. **Generate a demo profile** → Click "Generate random synthetic profile" below to see model-space examples
    2. **Upload your CSV** → Prepare a CSV with required columns (see template download) and upload it  
    3. **Evaluate with held-out data** → Include measured outcomes and confirm held-out status to see metrics
    
    **Key points:**
    - Values must already be in model scale (not raw clinical units)
    - This checkpoint is synthetic-trained—not the paper's UK Biobank model
    - Predictions are research artifacts, never clinical measurements
    """)

st.warning("⚠️ Disclaimer: Generated values are model predictions, NOT measurements. Never use for clinical decisions. Always defer to measured results and consult a physician.")
st.warning("Privacy: Do not upload identifiable patient data. If this app is hosted, uploaded files are sent to that server. This demo has no authentication or privacy/compliance safeguards; use only de-identified, approved research data.")
st.markdown(
    "### One PULSE model · two reconstruction directions\n"
    "Choose which follow-up modality to reconstruct. Both options use the same checkpoint and longitudinal PULSE architecture; "
    "they are task modes, not an ensemble. The available checkpoint and cohort limitations are shown below."
)


def checkpoint_signature(checkpoint_path: str) -> tuple[int, int]:
    try:
        stat = Path(checkpoint_path).stat()
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return -1, -1


@st.cache_resource(show_spinner="Loading PULSE architecture and checkpoint…")
def load_engine(checkpoint_path: str, checkpoint_stamp: tuple[int, int]):
    engine = PulseEngine(device="cpu")
    path = Path(checkpoint_path)
    if not path.is_file():
        return engine, f"Checkpoint not found: {checkpoint_path}"
    try:
        engine.load(str(path))
    except Exception as exc:
        return engine, f"Could not load checkpoint `{path.name}` ({type(exc).__name__}): {exc}"
    return engine, None


engine, load_error = load_engine(str(CHECKPOINT), checkpoint_signature(str(CHECKPOINT)))
with st.expander("Model and provenance", expanded=True):
    if engine.checkpoint_loaded:
        st.success(f"Checkpoint loaded: `{CHECKPOINT.name}`")
        config_path = CHECKPOINT.parent / "train_config.json"
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                st.write(f"Checkpoint metadata: {config.get('n_patients', 'unknown')} training examples; {config.get('n_visits', 'unknown')} visits; epoch {engine.checkpoint_epoch if engine.checkpoint_epoch is not None else 'unknown'}.")
                st.error("Important: this checkpoint was trained on generated synthetic Gaussian data, not UK Biobank data. Outputs are only a software demonstration—not the paper's trained model or results.")
            except (OSError, ValueError):
                st.warning("Checkpoint is present, but its training-provenance metadata could not be read.")
    else:
        st.error(load_error or "No trained checkpoint is loaded. Inference is disabled.")
        st.caption("Set PULSE_CKPT to a compatible checkpoint file to enable the demonstration.")
    st.caption(
    "The cited paper's UK Biobank benchmark reports 61 routine-lab and 251 metabolomics features. "
    "This prototype does not bundle the cohort, validated clinical feature map, or matched preprocessing parameters."
    )

with st.expander("Paper benchmark: what matches and what is missing", expanded=False):
    st.info("**Quick summary:** This app matches the paper's task shape and feature counts (61 labs, 251 metabolites) but uses a synthetic-trained checkpoint. It cannot reproduce the paper's UK Biobank results.")
    
    paper_col, demo_col = st.columns(2)
    with paper_col:
        st.markdown("**📄 Reference study protocol (Nature paper)**")
        st.markdown(
            "- Two-visit UK Biobank dataset: 61 routine-lab tests and 251 metabolomics biomarkers.\n"
            "- Main PULSE task: full baseline multimodal profile for historical context + follow-up routine labs → follow-up metabolomics.\n"
            "- Static comparators receive only follow-up routine labs at test time; their training visits are treated as independent paired samples without historical state.\n"
            "- Approximately 10,000 paper participants, split by participant: 7,000 train and 3,000 held-out test (70/30); all visits remain within one split. These are reference-study counts, not this app's cohort.\n"
            "- Methods specify log-transform then Z-score normalization for the static comparators; this does not establish that same preprocessing for main PULSE.\n"
            "- Fig. 2a: per-metabolite Pearson r, MAE, MSE and paired two-sided Wilcoxon comparisons across 251 metabolites; Fig. 2b: top ten by r; Fig. 2c: change scatter for a named large-LDL lipid ratio (r = 0.850).\n"
            "- Static comparator methods are MIDAS, scVAEIT, and StabMap."
        )
    with demo_col:
        st.markdown("**💻 This interactive prototype**")
        st.markdown(
            "- ✅ Matches the paper-shaped input/output direction and 61/251 feature dimensions, but accepts at most 100 uploaded rows—not the paper's 3,000-person test set.\n"
            "- ⚠️ Bundled checkpoint: 400 generated synthetic profiles and 3 visits—not UK Biobank or paper-trained weights.\n"
            "- ❌ No verified clinical analyte map, reference cohort, or compatible fitted preprocessing.\n"
            "- ❌ Does not run MIDAS/scVAEIT/StabMap. Carry-forward scores, Wilcoxon results, top-ten rankings, and change plots are prototype-only diagnostics; the reverse-lab task is exploratory."
        )
    st.warning(
        "This is a task-shape demonstration, not a replication. Evaluation scores are gated on a user attestation "
        "that all rows are patient-level held-out data; the app cannot verify overlap with the checkpoint's training set."
    )

TASK_METAB = "Predict follow-up metabolomics (paper direction)"
TASK_LAB = "Predict follow-up routine labs (reverse direction)"
task = st.radio("Prediction direction", [TASK_METAB, TASK_LAB], index=0, horizontal=True)
if task == TASK_METAB:
    FOLLOWUP_INPUT_COLS = FOLLOWUP_LAB_COLS
    GROUND_TRUTH_COLS = MEASURED_FOLLOWUP_METAB_COLS
    target_modality = "metabolomics"
    target_prefix = "metab"
    target_dim = METAB_DIM
    task_caption = "Paper-shaped direction: paired baseline labs + metabolites and follow-up labs predict follow-up metabolites."
else:
    FOLLOWUP_INPUT_COLS = FOLLOWUP_METAB_COLS
    GROUND_TRUTH_COLS = MEASURED_FOLLOWUP_LAB_COLS
    target_modality = "routine labs"
    target_prefix = "lab"
    target_dim = LAB_DIM
    task_caption = "Reverse modality direction: paired baseline labs + metabolites and follow-up metabolites predict follow-up labs. This is not the paper's reported benchmark direction."
st.caption(f"{task_caption} Both directions use the same checkpoint; no ensemble is being run.")

followup_input_modality = "routine labs" if task == TASK_METAB else "metabolomics"
st.markdown("### PULSE in plain language")
st.markdown(
    "At a high level, the paper's PULSE model learns shared temporal and cross-modal patterns from a cohort. "
    "At inference, it uses this participant's baseline paired profile as historical context plus the available follow-up modality to estimate the missing follow-up modality. "
    "It does not train a separate model for each participant, measure the missing modality, or diagnose. "
    "An estimate is not a measurement or diagnosis. The checkpoint in this demo is synthetic-trained and cannot provide valid patient predictions."
)
st.caption("This is a research concept, not a diagnosis or a new lab result. The checkpoint in this demo was trained on synthetic data, so its outputs are not valid patient predictions.")

flow_input = f"measured follow-up {followup_input_modality}"
flow_output = f"estimated follow-up {target_modality}"
flow_note = "Paper-shaped direction" if task == TASK_METAB else "Exploratory reverse direction · not the paper benchmark"
st.markdown(
    f"""<style>
.pulse-flow-infographic {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr) auto minmax(0, 1fr) auto minmax(0, 1fr);
  gap: 0.65rem;
  align-items: stretch;
  margin: 0.7rem 0 0.35rem;
}}
.pulse-flow-card {{
  min-width: 0;
  padding: 1rem;
  border: 1px solid rgba(120, 140, 160, 0.35);
  border-radius: 0.9rem;
  background: rgba(120, 140, 160, 0.08);
  color: inherit;
}}
.pulse-flow-card h4 {{ margin: 0.45rem 0; font-size: 1rem; line-height: 1.3; }}
.pulse-flow-card p {{ margin: 0; font-size: 0.9rem; line-height: 1.45; }}
.pulse-flow-kicker {{ font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; opacity: 0.72; }}
.pulse-flow-arrow {{ display: flex; align-items: center; justify-content: center; gap: 0.2rem; font-size: 1.2rem; opacity: 0.72; }}
.pulse-flow-signal {{ width: 0.48rem; height: 0.48rem; border-radius: 50%; background: var(--primary-color, #4c8bf5); animation: pulse-signal 1.8s ease-in-out infinite; }}
.pulse-flow-note {{ margin: 0.15rem 0 0.6rem; font-size: 0.78rem; opacity: 0.72; }}
@keyframes pulse-signal {{ 0%, 100% {{ transform: translateX(-3px); opacity: 0.4; }} 50% {{ transform: translateX(3px); opacity: 1; }} }}
@media (max-width: 900px) {{ .pulse-flow-infographic {{ grid-template-columns: 1fr; }} .pulse-flow-arrow {{ min-height: 1.3rem; transform: rotate(90deg); }} }}
@media (prefers-reduced-motion: reduce) {{ .pulse-flow-signal {{ animation: none; }} }}
</style>
<div class="pulse-flow-infographic" role="group" aria-label="PULSE longitudinal reconstruction flow">
  <div class="pulse-flow-card"><span class="pulse-flow-kicker">1 · Earlier visit</span><h4>Paired measurements</h4><p>Routine labs and metabolites collected together help describe this person's earlier state.</p></div>
  <div class="pulse-flow-arrow" aria-hidden="true"><span class="pulse-flow-signal"></span>→</div>
  <div class="pulse-flow-card"><span class="pulse-flow-kicker">2 · Later visit</span><h4>One measured modality</h4><p>{flow_input.capitalize()} are supplied; the other modality is missing.</p></div>
  <div class="pulse-flow-arrow" aria-hidden="true"><span class="pulse-flow-signal"></span>→</div>
  <div class="pulse-flow-card"><span class="pulse-flow-kicker">3 · Shared model</span><h4>Patterns learned across people</h4><p>The model combines shared cohort patterns with this person's own timeline.</p></div>
  <div class="pulse-flow-arrow" aria-hidden="true"><span class="pulse-flow-signal"></span>→</div>
  <div class="pulse-flow-card"><span class="pulse-flow-kicker">4 · Individual output</span><h4>{flow_output.capitalize()}</h4><p>An estimate for this record—not a measurement, diagnosis, or guaranteed result.</p></div>
</div>
<div class="pulse-flow-note">{flow_note}. The original paper's framework is population-trained but can produce person-level research estimates.</div>""",
    unsafe_allow_html=True,
)

baseline_feature_total = LAB_DIM + METAB_DIM
lab_share_pct = LAB_DIM / baseline_feature_total * 100
metab_share_pct = METAB_DIM / baseline_feature_total * 100
st.markdown(
    f"""<style>
.pulse-feature-chart {{ display: flex; flex-wrap: wrap; align-items: center; gap: 1.3rem; padding: 1rem; border: 1px solid rgba(120, 140, 160, 0.3); border-radius: 0.9rem; background: rgba(120, 140, 160, 0.06); color: inherit; }}
.pulse-feature-pie {{ width: 11rem; height: 11rem; flex: 0 0 11rem; border-radius: 50%; background: conic-gradient(#4c8bf5 0 {lab_share_pct:.2f}%, #39a887 {lab_share_pct:.2f}% 100%); }}
.pulse-feature-legend {{ flex: 1 1 18rem; min-width: 0; }}
.pulse-feature-legend h4 {{ margin: 0 0 0.7rem; }}
.pulse-feature-legend ul {{ list-style: none; padding: 0; margin: 0; }}
.pulse-feature-legend li {{ display: flex; align-items: center; gap: 0.55rem; margin: 0.45rem 0; }}
.pulse-feature-swatch {{ display: inline-block; width: 0.8rem; height: 0.8rem; flex: 0 0 0.8rem; border-radius: 0.2rem; }}
.pulse-feature-labs {{ background: #4c8bf5; }}
.pulse-feature-metab {{ background: #39a887; }}
.pulse-feature-legend strong {{ margin-left: auto; white-space: nowrap; }}
.pulse-feature-explanation {{ margin: 0.85rem 0 0; line-height: 1.5; }}
</style>
<figure class="pulse-feature-chart" aria-label="Baseline feature slot composition">
  <div class="pulse-feature-pie" role="img" aria-label="Pie chart: {LAB_DIM} routine-lab feature slots, {lab_share_pct:.1f} percent; {METAB_DIM} metabolomics feature slots, {metab_share_pct:.1f} percent."></div>
  <figcaption class="pulse-feature-legend">
    <h4>Baseline input feature slots · {baseline_feature_total} total</h4>
    <ul>
      <li><span class="pulse-feature-swatch pulse-feature-labs" aria-hidden="true"></span><span>{LAB_DIM} routine-lab feature slots</span><strong>{lab_share_pct:.1f}%</strong></li>
      <li><span class="pulse-feature-swatch pulse-feature-metab" aria-hidden="true"></span><span>{METAB_DIM} metabolomics feature slots</span><strong>{metab_share_pct:.1f}%</strong></li>
    </ul>
    <p class="pulse-feature-explanation"><b>How to read it:</b> The larger metabolomics slice means this demo has more placeholder metabolomics columns in its baseline input. It shows feature count only—not feature importance, not accuracy, and not a patient's biological composition.</p>
  </figcaption>
</figure>""",
    unsafe_allow_html=True,
)

st.subheader("Synthetic profile and population metadata")
pop_col, label_col = st.columns(2)
with label_col:
    synthetic_label = st.text_input(
        "Synthetic sample label (optional)",
        value="",
        placeholder="Auto-generated SYNTH ID",
        help="Used only by the random synthetic-profile generator; uploaded CSVs use their own patient_id values.",
    )
with pop_col:
    population = st.text_input("Population / cohort (metadata)", value="Unspecified", help="Recorded for stratification only; this checkpoint does not condition predictions on population.")
st.info("Inputs must already be in the model's training-time feature scale. Do not enter raw lab units: the matching clinical feature map and normalization parameters are not bundled.")
st.caption("The random generator creates Gaussian model-space vectors only—not a clinically realistic patient, demographics, or measurements.")

template_cols = BASELINE_COLS + FOLLOWUP_INPUT_COLS
template = pd.DataFrame([{col: 0.0 for col in template_cols}])
template.insert(0, "population", "Unspecified")
template.insert(0, "patient_id", "DEMO-001")
st.download_button(f"Download {target_modality} task CSV template", data=template.to_csv(index=False).encode("utf-8"), file_name=f"pulse_{target_prefix}_task_template.csv", mime="text/csv")
evaluation_template = create_evaluation_template(BASELINE_COLS + FOLLOWUP_INPUT_COLS, GROUND_TRUTH_COLS)
st.download_button(
    "Download evaluation CSV template",
    data=evaluation_template.to_csv(index=False).encode("utf-8"),
    file_name=f"pulse_{target_prefix}_evaluation_template.csv",
    mime="text/csv",
)
st.caption(
    f"To evaluate, fill the template's blank input cells and every measured outcome column "
    f"(`{GROUND_TRUTH_COLS[0]}` … `{GROUND_TRUTH_COLS[-1]}`) with observed follow-up data in the same preprocessing scale. "
    "Use held-out data; do not copy model predictions into measured outcome columns."
)

uploaded = st.file_uploader("Upload preprocessed longitudinal feature CSV", type=["csv"])
heldout_confirmed = st.checkbox(
    "Confirm this is a held-out participant-level test set (one row per participant; no overlap with training)",
    value=False,
    help="The app checks for duplicate IDs within this file, but cannot inspect checkpoint training IDs or prove cross-split independence. This is an operator attestation, not an audit.",
)
run_demo = st.button("Generate random synthetic profile", type="secondary", disabled=not engine.checkpoint_loaded)
run_upload = st.button("Run uploaded sample(s)", type="primary", disabled=not engine.checkpoint_loaded or uploaded is None)

if run_demo or run_upload:
    st.session_state.pop("pulse_last_run", None)
    source = "random synthetic profile" if run_demo else "uploaded CSV"
    synthetic_input = None
    if run_demo:
        profile = generate_synthetic_profile(LAB_DIM, METAB_DIM, len(FOLLOWUP_INPUT_COLS))
        patient_id = synthetic_label.strip() or f"SYNTH-{uuid.uuid4().hex[:8].upper()}"
        sample = {"patient_id": patient_id, "population": population, **profile}
        samples = [sample]
        synthetic_row = {"patient_id": patient_id, "population": population}
        synthetic_row.update(dict(zip(LAB_COLS, profile["baseline_labs"])))
        synthetic_row.update(dict(zip(METAB_COLS, profile["baseline_metabolomics"])))
        synthetic_row.update(dict(zip(FOLLOWUP_INPUT_COLS, profile["followup_input"])))
        synthetic_input = pd.DataFrame([synthetic_row])
        has_truth = False
    else:
        try:
            validate_upload_size(uploaded.size)
            data = pd.read_csv(uploaded)
            data, has_truth = validate_feature_table(
                data,
                input_columns=BASELINE_COLS + FOLLOWUP_INPUT_COLS,
                truth_columns=GROUND_TRUTH_COLS,
                max_rows=100,
            )
            samples = []
            for _, row in data.iterrows():
                features = row[BASELINE_COLS + FOLLOWUP_INPUT_COLS]
                sample = {
                    "patient_id": str(row["patient_id"]),
                    "population": str(row["population"]),
                    "baseline_labs": features[LAB_COLS].to_numpy(dtype=np.float32),
                    "baseline_metabolomics": features[METAB_COLS].to_numpy(dtype=np.float32),
                    "followup_input": features[FOLLOWUP_INPUT_COLS].to_numpy(dtype=np.float32),
                }
                if has_truth:
                    sample["measured_followup"] = row[GROUND_TRUTH_COLS].to_numpy(dtype=np.float32)
                samples.append(sample)
        except Exception as exc:
            st.error(f"Upload validation failed: {exc}")
            st.stop()

    try:
        output_rows = []
        prediction_vectors = []
        progress = st.progress(0, text="Reconstructing follow-up modality…")
        for index, sample in enumerate(samples):
            if task == TASK_METAB:
                prediction = engine.impute_followup_metabolomics(
                    sample["baseline_labs"], sample["baseline_metabolomics"], sample["followup_input"]
                )
            else:
                prediction = engine.impute_followup_labs(
                    sample["baseline_labs"], sample["baseline_metabolomics"], sample["followup_input"]
                )
            prediction_vectors.append(prediction)
            out = {"patient_id": sample["patient_id"], "population": sample["population"]}
            out.update({f"predicted_{target_prefix}_{i:03d}": float(value) for i, value in enumerate(prediction, start=1)})
            output_rows.append(out)
            progress.progress((index + 1) / len(samples), text=f"Processed {index + 1}/{len(samples)} sample(s)")
        result = pd.DataFrame(output_rows)
        metrics = None
        baseline_metrics = None
        change_metrics = None
        baseline_change_metrics = None
        population_metrics = None
        baseline_population_metrics = None
        evaluation_arrays = None
        if has_truth:
            measured_vectors = np.stack([sample["measured_followup"] for sample in samples])
            prediction_matrix = np.stack(prediction_vectors)
            baseline_prediction_matrix = compute_carry_forward_predictions(
                np.stack([sample["baseline_labs"] for sample in samples]),
                np.stack([sample["baseline_metabolomics"] for sample in samples]),
                target_modality=target_modality,
            )
            metrics = compute_confirmed_heldout_benchmark_metrics(
                prediction_matrix,
                measured_vectors,
                feature_prefix=target_prefix,
                heldout_confirmed=heldout_confirmed,
            )
            baseline_metrics = compute_confirmed_heldout_benchmark_metrics(
                baseline_prediction_matrix,
                measured_vectors,
                feature_prefix=target_prefix,
                heldout_confirmed=heldout_confirmed,
            )
            population_labels = [sample["population"] for sample in samples]
            population_metrics = compute_population_benchmark_summary(
                prediction_matrix,
                measured_vectors,
                population_labels,
                feature_prefix=target_prefix,
                heldout_confirmed=heldout_confirmed,
            )
            baseline_population_metrics = compute_population_benchmark_summary(
                baseline_prediction_matrix,
                measured_vectors,
                population_labels,
                feature_prefix=target_prefix,
                heldout_confirmed=heldout_confirmed,
            )
            if heldout_confirmed:
                evaluation_arrays = {
                    "predictions": prediction_matrix.copy(),
                    "measured": measured_vectors.copy(),
                    "baseline": baseline_prediction_matrix.copy(),
                }
            checkpoint_change, measured_change = compute_longitudinal_change_vectors(
                prediction_matrix, measured_vectors, baseline_prediction_matrix
            )
            baseline_change, _ = compute_longitudinal_change_vectors(
                baseline_prediction_matrix, measured_vectors, baseline_prediction_matrix
            )
            change_metrics = compute_confirmed_heldout_benchmark_metrics(
                checkpoint_change,
                measured_change,
                feature_prefix=target_prefix,
                heldout_confirmed=heldout_confirmed,
            )
            baseline_change_metrics = compute_confirmed_heldout_benchmark_metrics(
                baseline_change,
                measured_change,
                feature_prefix=target_prefix,
                heldout_confirmed=heldout_confirmed,
            )
        st.session_state["pulse_last_run"] = {
            "result": result,
            "metrics": metrics,
            "baseline_metrics": baseline_metrics,
            "change_metrics": change_metrics,
            "baseline_change_metrics": baseline_change_metrics,
            "population_metrics": population_metrics,
            "baseline_population_metrics": baseline_population_metrics,
            "evaluation_arrays": evaluation_arrays,
            "has_truth": has_truth,
            "heldout_confirmed": heldout_confirmed,
            "task": task,
            "target_modality": target_modality,
            "target_prefix": target_prefix,
            "source": source,
            "synthetic_input": synthetic_input,
        }
    except Exception as exc:
        st.error(f"Inference failed: {exc}")

last_run = st.session_state.get("pulse_last_run")
if last_run:
    result = last_run["result"]
    metrics = last_run["metrics"]
    baseline_metrics = last_run.get("baseline_metrics")
    change_metrics = last_run.get("change_metrics")
    baseline_change_metrics = last_run.get("baseline_change_metrics")
    population_metrics = last_run.get("population_metrics")
    baseline_population_metrics = last_run.get("baseline_population_metrics")
    evaluation_arrays = last_run.get("evaluation_arrays")
    output_modality = last_run["target_modality"]
    output_prefix = last_run["target_prefix"]
    synthetic_input = last_run.get("synthetic_input")
    if synthetic_input is not None:
        st.subheader("Generated synthetic input profile")
        st.caption("Random Gaussian model-space vectors with placeholder features—not a clinically realistic patient or measured values.")
        st.dataframe(synthetic_input.iloc[:, :min(12, synthetic_input.shape[1])], use_container_width=True, hide_index=True)
        example_input_column = next(
            column for column in LAB_COLS + METAB_COLS + FOLLOWUP_INPUT_COLS if column in synthetic_input.columns
        )
        example_input_value = float(synthetic_input.iloc[0][example_input_column])
        st.caption(
            f"Example input: {example_input_column} = {example_input_value:.4f}. This is a randomly generated model-space value, "
            "not a measured lab result or clinical value. The synthetic inputs are drawn around zero; their signs do not mean clinically high or low."
        )
        st.caption("The SYNTH ID is only a demo label. Population is metadata and does not change this checkpoint's prediction.")
        st.download_button(
            "Download synthetic input profile (CSV)",
            data=synthetic_input.to_csv(index=False).encode("utf-8"),
            file_name=f"pulse_synthetic_profile_{output_prefix}.csv",
            mime="text/csv",
        )
    st.subheader(f"Generated follow-up {output_modality} · standardized feature space")
    st.caption(
        f"Showing results from the most recent run ({last_run['source']}) for task “{last_run['task']}”. "
        "Rerun after changing inputs or task. Feature IDs are placeholders, not clinical names or units; "
        "prediction intervals and calibrated uncertainty are unavailable."
    )
    st.dataframe(result.iloc[:, :min(12, result.shape[1])], use_container_width=True, hide_index=True)
    first_prediction_column = next(column for column in result.columns if column.startswith("predicted_"))
    first_prediction_value = float(result.iloc[0][first_prediction_column])
    st.info(
        f"How to read these values: Example prediction: {first_prediction_column} = {first_prediction_value:.4f}. "
        "This is the model's estimate for an unnamed follow-up feature, not a measured result. Positive/negative only describes its position relative to zero in the model representation—not a clinical high/low. "
        "There are no lab units or reference range, so do not compare it with clinical ranges."
    )
    st.download_button(
        "Download generated features (CSV)",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name=f"pulse_followup_{output_prefix}_predictions.csv",
        mime="text/csv",
    )
    if metrics is not None:
        st.subheader(f"Operator-attested descriptive scores · {output_modality}")
        metric_cols = st.columns(3)
        for col, key, label in zip(metric_cols, ("pearson_r", "mae", "mse"), ("Pearson r", "MAE", "MSE")):
            summary = metrics[key]
            with col:
                if summary is None:
                    st.metric(label, "Unavailable")
                    st.caption("Pearson r needs at least two samples with non-constant values.")
                else:
                    st.metric(f"{label} · median", f"{summary['median']:.3f}")
                    st.caption(f"Across {summary['n_valid_features']} features · IQR {summary['q1']:.3f}–{summary['q3']:.3f}")
        st.caption(
            f"Per-feature scores summarize {metrics['n_features']} placeholder features from {metrics['n_samples']} uploaded participant(s). "
            "Holdout status is operator-attested, not independently verified. The app accepts at most 100 rows, so this is not the paper's 3,000-participant test set; "
            "scores are descriptive and do not validate this synthetic-trained checkpoint."
        )
        if baseline_metrics is not None:
            st.subheader("Checkpoint vs carry-forward baseline")
            st.caption(
                "Carry-forward predicts each follow-up feature by copying that participant's baseline value of the same modality. "
                "This simple sanity-check is not one of the paper's MIDAS/scVAEIT/StabMap comparators."
            )
            comparison_rows = []
            for method_name, method_metrics in (
                ("Loaded checkpoint", metrics),
                ("Carry-forward baseline", baseline_metrics),
            ):
                row = {"Method": method_name}
                for metric_key, metric_label in (
                    ("pearson_r", "Median Pearson r"),
                    ("mae", "Median MAE"),
                    ("mse", "Median MSE"),
                ):
                    summary = method_metrics[metric_key]
                    row[metric_label] = None if summary is None else summary["median"]
                comparison_rows.append(row)
            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
            paired_test = compute_paired_feature_wilcoxon(
                metrics["feature_metrics"], baseline_metrics["feature_metrics"]
            ).rename(columns={
                "metric": "Metric",
                "n_pairs": "Paired features",
                "median_primary_minus_comparator": "Median checkpoint − baseline",
                "p_value": "Wilcoxon p (two-sided)",
                "status": "Note",
            })
            st.subheader("Paired feature-level comparison · exploratory Wilcoxon")
            direction_note = (
                "The reference study uses this analysis pattern for its 251-metabolite task. "
                if last_run["task"] == TASK_METAB
                else "This reverse routine-lab direction is outside the reference benchmark. "
            )
            st.caption(
                direction_note
                + "Here the synthetic-trained checkpoint is compared with carry-forward, not MIDAS/scVAEIT/StabMap. "
                "Features are correlated, so feature-level pairs are not independent; p-values are unadjusted across three metrics and exploratory only. "
                "Positive Pearson-r differences favor the checkpoint; negative MAE/MSE differences favor it. These p-values do not establish clinical value or paper reproduction."
            )
            st.dataframe(paired_test, use_container_width=True, hide_index=True)
        if change_metrics is not None and baseline_change_metrics is not None:
            st.subheader("Longitudinal change agreement · exploratory")
            st.caption(
                "Change is Δ = follow-up minus baseline per participant and feature. This table aggregates placeholder-feature scores; carry-forward predicts zero change, so its Pearson r is unavailable. "
                "The paper's Fig. 2c instead analyzes one mapped biomarker across full PULSE, two input ablations, and three static methods; this is not that analysis."
            )
            change_rows = []
            for method_name, method_metrics in (
                ("Loaded checkpoint", change_metrics),
                ("Carry-forward baseline", baseline_change_metrics),
            ):
                row = {"Method": method_name}
                for metric_key, metric_label in (
                    ("pearson_r", "Median change Pearson r"),
                    ("mae", "Median change MAE"),
                    ("mse", "Median change MSE"),
                ):
                    summary = method_metrics[metric_key]
                    row[metric_label] = None if summary is None else summary["median"]
                change_rows.append(row)
            st.dataframe(pd.DataFrame(change_rows), use_container_width=True, hide_index=True)
            if evaluation_arrays is not None:
                feature_options = [
                    f"{output_prefix}_{index + 1:03d}"
                    for index in range(evaluation_arrays["predictions"].shape[1])
                ]
                selected_feature = st.selectbox(
                    "Select placeholder feature for change plot",
                    options=feature_options,
                    key=f"change_feature_{output_prefix}",
                    help="Feature IDs are placeholders; the app has no validated analyte-name mapping.",
                )
                scatter_data = build_longitudinal_change_scatter_data(
                    evaluation_arrays["predictions"],
                    evaluation_arrays["measured"],
                    evaluation_arrays["baseline"],
                    feature_index=feature_options.index(selected_feature),
                )
                st.subheader("Figure 2c-style change plot · placeholder feature only")
                st.caption(
                    "Each dot is one held-out row; identifiers are intentionally omitted. Horizontal axis: observed follow-up minus baseline. "
                    "Vertical axis: estimated follow-up minus baseline. Values are model-space differences, not clinical units. Closer to the dashed y=x line means closer agreement for this feature."
                )
                st.vega_lite_chart(
                    scatter_data,
                    {
                        "layer": [
                            {
                                "transform": [{"filter": "datum.kind === 'Observation'"}],
                                "mark": {"type": "point", "filled": True, "opacity": 0.72, "size": 70},
                                "encoding": {
                                    "x": {"field": "measured_change", "type": "quantitative", "title": "Observed change · model scale"},
                                    "y": {"field": "predicted_change", "type": "quantitative", "title": "Estimated change · model scale"},
                                    "color": {"field": "method", "type": "nominal", "title": "Method"},
                                    "tooltip": [
                                        {"field": "method", "type": "nominal", "title": "Method"},
                                        {"field": "measured_change", "type": "quantitative", "title": "Observed change"},
                                        {"field": "predicted_change", "type": "quantitative", "title": "Estimated change"},
                                    ],
                                },
                            },
                            {
                                "transform": [{"filter": "datum.kind === 'Identity line'"}],
                                "mark": {"type": "line", "color": "#777777", "strokeDash": [5, 5]},
                                "encoding": {
                                    "x": {"field": "measured_change", "type": "quantitative"},
                                    "y": {"field": "predicted_change", "type": "quantitative"},
                                },
                            },
                        ],
                        "width": 440,
                        "height": 360,
                    },
                )
                st.markdown(
                    "The cited paper's Fig. 2c reports **r = 0.850** for the free-cholesterol-to-total-lipid ratio in large LDL particles. "
                    "This chart cannot be matched to that analyte: the prototype's feature IDs are unmapped, and its checkpoint is synthetic-trained. "
                    "[Open the Nature paper](https://www.nature.com/articles/s43588-026-01026-5)."
                )
        if population_metrics is not None and baseline_population_metrics is not None:
            population_rows = []
            for method_name, method_frame in (
                ("Loaded checkpoint", population_metrics),
                ("Carry-forward baseline", baseline_population_metrics),
            ):
                for row in method_frame.to_dict(orient="records"):
                    population_rows.append({
                        "Population": row["population"],
                        "Participants": row["n_samples"],
                        "Status": row["status"],
                        "Method": method_name,
                        "Median Pearson r": row["median_pearson_r"],
                        "Median MAE": row["median_mae"],
                        "Median MSE": row["median_mse"],
                    })
            st.subheader("Population-stratified held-out results · descriptive")
            st.caption(
                "Population is metadata, not a model input. These scores are computed separately within each attested held-out group; "
                "groups with fewer than five records are withheld. The five-record display threshold is not a statistical sufficiency rule or privacy guarantee. "
                "Differences are descriptive only—not a fairness, causal, or population-equivalence test."
            )
            st.dataframe(pd.DataFrame(population_rows), use_container_width=True, hide_index=True)
        model_feature_metrics = pd.DataFrame(metrics["feature_metrics"]).assign(method="Loaded checkpoint")
        if baseline_metrics is not None:
            baseline_feature_metrics = pd.DataFrame(baseline_metrics["feature_metrics"]).assign(method="Carry-forward baseline")
            feature_metrics = pd.concat([model_feature_metrics, baseline_feature_metrics], ignore_index=True)
            long_metrics = feature_metric_distributions_by_method(
                {
                    "Loaded checkpoint": metrics["feature_metrics"],
                    "Carry-forward baseline": baseline_metrics["feature_metrics"],
                }
            )
        else:
            feature_metrics = model_feature_metrics
            long_metrics = feature_metric_distributions_by_method(
                {"Loaded checkpoint": metrics["feature_metrics"]}
            )
        st.subheader("Across-feature metric distributions")
        st.caption(
            "Each box summarizes one score per feature for the loaded checkpoint and carry-forward sanity-check. "
            "These are not the paper's multi-model results. Higher Pearson r is better; lower MAE/MSE are better."
        )
        chart_columns = st.columns(3)
        for chart_col, metric_key, chart_title in zip(
            chart_columns,
            ("pearson_r", "mae", "mse"),
            ("Pearson r by feature", "MAE by feature", "MSE by feature"),
        ):
            scores = long_metrics.loc[long_metrics["metric"] == metric_key, "score"]
            with chart_col:
                st.caption(chart_title)
                if scores.empty:
                    st.caption("Unavailable: needs at least two non-constant samples.")
                else:
                    chart_data = long_metrics.loc[
                        long_metrics["metric"] == metric_key, ["series", "score"]
                    ].dropna(subset=["score"])
                    st.vega_lite_chart(
                        chart_data,
                        {
                            "mark": {"type": "boxplot", "extent": 1.5},
                            "encoding": {
                                "x": {"field": "series", "type": "nominal", "title": None},
                                "y": {"field": "score", "type": "quantitative", "title": chart_title},
                                "color": {"field": "series", "type": "nominal", "title": None},
                            },
                        },
                        use_container_width=True,
                    )
        top_ten_by_r = (
            model_feature_metrics.dropna(subset=["pearson_r"])
            .sort_values("pearson_r", ascending=False)
            .head(10)[["feature_id", "pearson_r", "mae", "mse"]]
        )
        with st.expander("Top 10 placeholder features by Pearson r"):
            st.caption(
                "Ranking view inspired by the paper's top-metabolite panel. These are this upload's unnamed feature slots, "
                "not mapped metabolites or a reproduction of the paper's ranking."
            )
            if top_ten_by_r.empty:
                st.info("Unavailable: Pearson r needs at least two non-constant held-out values per feature.")
            else:
                st.dataframe(top_ten_by_r, use_container_width=True, hide_index=True)
        with st.expander("Per-feature comparison (placeholder IDs)"):
            st.dataframe(feature_metrics, use_container_width=True, hide_index=True)
            st.download_button(
                "Download per-feature metrics (CSV)",
                data=feature_metrics.to_csv(index=False).encode("utf-8"),
                file_name=f"pulse_{output_prefix}_feature_metrics.csv",
                mime="text/csv",
            )
    else:
        if last_run.get("has_truth"):
            st.warning("Measured outcomes were supplied, but benchmark scores are withheld until patient-level held-out status is confirmed.")
            st.caption(
                "This app cannot verify training-set independence. Confirm that one row represents each participant and that no participant appears in training or another split, then rerun."
            )
        else:
            st.info(f"No measured follow-up {output_modality} supplied; benchmark metrics were not computed.")
            st.caption(
                "Predictions alone cannot be scored. Download the evaluation CSV template above, fill every measured outcome column "
                "with actual observed follow-up values from held-out data, then upload it. Do not use predictions as measured truth."
            )
    if result["population"].nunique() > 1:
        predicted_cols = [column for column in result.columns if column.startswith("predicted_")]
        summary = result.groupby("population")[predicted_cols].agg(["mean", "std"])
        st.subheader("Exploratory population summary")
        st.caption("Descriptive only. Population is metadata, not a model input; groups are not adjusted or clinically comparable.")
        st.dataframe(summary, use_container_width=True)

st.divider()
st.caption("Framework reference: [Wu et al., ‘Longitudinal alignments and syntheses of multimodal clinical data for personalized medicine with the PULSE framework,’ Nature Computational Science (2026)](https://www.nature.com/articles/s43588-026-01026-5). This software demonstration is not the paper's benchmark pipeline.")
