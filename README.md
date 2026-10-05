# PULSE Longitudinal Multimodal Demo

**Research use only · Not a medical device · Longitudinal multimodal signal imputation from sparse routine labs**

This Streamlit app is a local, CPU inference demo with two selectable reconstruction directions. It uses one shared PULSE checkpoint; it does not blend two checkpoints. The app includes a plain-language guide, a task-aware workflow infographic with reduced-motion support, and a pie chart of baseline feature-slot counts. The pie chart shows placeholder column counts, not feature importance or accuracy. Attested held-out evaluation includes per-feature metrics, a top-10 placeholder ranking, an exploratory change-scatter patterned after Fig. 2c, a paired feature-level Wilcoxon comparison, and population-stratified summaries. None reproduces the paper's trained model, mapped biomarkers, cohort, comparators, or results.

## Paper alignment and limits

The cited Nature benchmark used about 10,000 UK Biobank participants with baseline and follow-up data: 61 routine-lab tests and 251 metabolomics biomarkers. It split people, not visits, into 7,000 training and 3,000 held-out test participants. Main PULSE reconstructed follow-up metabolites from the participant's full baseline multimodal profile plus follow-up routine labs. Static methods MIDAS, scVAEIT, and StabMap saw only follow-up labs at test time; their training visits were treated as independent paired samples without historical state.

The Methods specify log-transformation followed by Z-score normalization when preparing data for the adapted static single-cell frameworks; that passage does **not** establish the same preprocessing for main PULSE. Fig. 2a reports per-metabolite Pearson r, MAE, and MSE across the 3,000-person test set, with paired two-sided Wilcoxon comparisons across 251 metabolites and no multiple-comparison adjustment. Fig. 2b ranks the ten highest-r metabolites. Fig. 2c compares predicted and measured change for a named free-cholesterol-to-total-lipid ratio in large LDL particles (reported r = 0.850).

This app mirrors the forward task shape and dimensions only. It accepts at most 100 uploaded rows and contains no UK Biobank cohort, paper-trained weights, verified analyte map, fitted preprocessing, or MIDAS/scVAEIT/StabMap implementation. Its carry-forward baseline, Wilcoxon table, top-10 list, change plot, and population summaries are prototype-only diagnostics—not the paper's models or results. The bundled checkpoint was trained on generated synthetic data (400 profiles, 3 visits). [Reference paper](https://www.nature.com/articles/s43588-026-01026-5).

> **Important:** The bundled checkpoint was trained on generated synthetic data, not UK Biobank data. This app is not the paper's trained model or benchmark pipeline, and its predictions must not be used for clinical decisions.

## Run locally

From this folder, install the declared dependencies and launch the app:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8504
```

Open `http://127.0.0.1:8504` in a browser. To use a compatible checkpoint elsewhere, set `PULSE_CKPT` to its path before starting the app. Checkpoints are loaded in weights-only mode.

Run the test suite with:

```bash
python -m pytest -q -W error::UserWarning
```

## Prediction directions

1. **Follow-up metabolomics (paper-shaped direction):** baseline routine labs + baseline metabolomics + follow-up routine labs → predicted follow-up metabolomics.
2. **Follow-up routine labs (reverse direction):** baseline routine labs + baseline metabolomics + follow-up metabolomics → predicted follow-up routine labs. This is an exploratory reverse direction, not the paper's reported benchmark.

Both paths require the vectors to already be in the model's training-time feature scale. Do not enter raw clinical units. The feature labels are placeholders, not validated analyte names or units; a clinically mapped feature dictionary and matching preprocessing parameters are not bundled. The **Generate random synthetic profile** button creates fresh Gaussian model-space vectors with an auto-generated `SYNTH-…` label; it is a software test fixture, not a realistic patient simulator or clinical data source.

## CSV inputs

Every upload needs a unique `patient_id` (one row per participant), `population`, all 61 baseline lab columns, all 251 baseline metabolomics columns, and the follow-up input columns for the selected direction. Uploads are limited to 100 rows and 25 MiB. The downloadable template includes the required inputs. Use these exact column patterns:

- `baseline_lab_001` … `baseline_lab_061`
- `baseline_metab_001` … `baseline_metab_251`
- Paper-shaped direction: `followup_lab_001` … `followup_lab_061`
- Reverse direction: `followup_metab_001` … `followup_metab_251`

For descriptive comparisons only, an upload may also include every measured outcome column for the selected direction:

- `measured_followup_metab_001` … `measured_followup_metab_251`, or
- `measured_followup_lab_001` … `measured_followup_lab_061`.

The app provides a separate **Download evaluation CSV template** with blank input and measured-outcome columns. Fill every blank with correctly preprocessed input or observed follow-up values before uploading; leaving outcomes blank intentionally means metrics cannot be computed. Never copy model predictions into measured outcome columns.

Provide all measured outcome columns or none. Scores appear only after the operator attests that the data are participant-level held-out observations. The app rejects duplicate participant IDs within a file, but cannot check overlap with the checkpoint's training data or another split. The attestation is not an audit.

Pearson r is correlation between predicted and measured values across participants for each feature; it can be high despite systematic bias. MAE is mean absolute error and MSE is mean squared error per feature (so large misses count more). Displayed summaries are medians across features, not pooled participant-level effect estimates.

The paper also compares per-feature scores using paired two-sided Wilcoxon signed-rank tests. This prototype shows the same analysis pattern only for its synthetic-trained checkpoint versus carry-forward; it does not run MIDAS/scVAEIT/StabMap. Feature slots can be correlated, p-values are unadjusted across the three metrics, and results are exploratory—not evidence of clinical value or paper reproduction.

The longitudinal-change diagnostic calculates `follow-up − baseline` per participant and feature. Its selectable scatter plots one **placeholder** feature against measured change with an identity line. The paper's Fig. 2c reports r = 0.850 for the free-cholesterol-to-total-lipids ratio in large LDL; this app cannot match a placeholder slot to that analyte or reproduce that result. [Reference paper](https://www.nature.com/articles/s43588-026-01026-5).

## Data and safety

- Do not upload identifiable patient data. When hosted, uploaded files are sent to the host running Streamlit. This demo has no authentication or privacy/compliance safeguards.
- Generated values are model predictions, not measurements. Prediction intervals and calibrated uncertainty are unavailable.
- Population is metadata, not a model input. Attested held-out results can be summarized per population against carry-forward; groups with fewer than five rows are withheld as a display precaution. These comparisons are descriptive, not tests of fairness or population equivalence.
- The random synthetic-profile generator creates Gaussian model-space vectors with placeholder features; it is not a clinically realistic patient simulator or a source of measured values.

Framework reference: Wu et al., “Longitudinal alignments and syntheses of multimodal clinical data for personalized medicine with the PULSE framework,” *Nature Computational Science* (2026). This software demo is not the paper's benchmark pipeline.
