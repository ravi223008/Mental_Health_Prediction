# Mental Health Prediction — Data Pipeline

Clean, privacy-safe preprocessing for the mental-health dataset, with deterministic splits, fairness guardrails, and SMOTENC-ready artifacts.

> If you only have 2 minutes: run  
> `python "Data Pre.py"`  
> (the script will auto-search `data/raw/mental_health_data final data.csv` or `data/`)

---

## Quickstart

```bash
# 1) Key outputs (created under data/processed):
#    - clean_full.csv
#    - clean_numeric.csv
#    - clean_numeric_model.csv
#    - label_mappings.json
#    - splits_70_15_15_k5.csv
#    - sample_weights.csv
#    - smote_config.json
#    - imbalance_report.txt
#    - cleaning_summary.txt
#    Public RR3 table:
#    - reports/tables_public/mh_by_gender_rr3.csv
```

Optional flags:

```bash
python "Data Pre.py"   --input "data/raw/mental_health_data final data.csv"   --out_dir "data/processed"   --pub_dir "reports/tables_public"   --kfolds 5   --no_splits   # if you want to skip 70/15/15 + k-fold labels
```

---

## Data assumptions (flexible column matching)

The cleaner is tolerant to common header variants (case/space differences). It expects, when available:

- **Continuous:** `Age`, `Sleep_Hours`, `Work_Hours`, `Physical_Activity_Hours`, `Social_Media_Usage`
- **Ordinal/Categorical:** `Smoking_Habit`, `Alcohol_Consumption`, `Diet_Quality`, `Severity`, `Stress_Level`
- **Binary (mapped):** `Consultation_History`, `Medication_Usage`
- **Demographics:** `Gender`/`Sex` → normalized to `Gender_norm`; `Relationship_Status` → `_norm`
- **Target:** `Mental_Health_Condition` (text “yes/no”) or its label-encoded twin

The script clips out-of-range numeric values to NA using sensible bounds, deduplicates by `ID` if present, imputes (median for continuous; mode for integers/booleans), and produces both a **full** view and a **model** view.

---

## Artifacts (what you’ll find after running)

- `data/processed/clean_full.csv` — analysis-friendly table (keeps sensitive columns for audits)
- `data/processed/clean_numeric.csv` — numeric/encoded features
- `data/processed/clean_numeric_model.csv` — **model view** without direct sensitive columns (+ optional `*_isna` flags)
- `data/processed/label_mappings.json` — categorical encodings
- `data/processed/splits_70_15_15_k5.csv` — deterministic **70/15/15** split + **train-only k-fold** labels
- `data/processed/sample_weights.csv` — optional fairness weights (`w_label`, `w_group`, `w_combo`)
- `data/processed/smote_config.json` — SMOTE/SMOTENC feature indices & recommended knobs
- `data/processed/imbalance_report.txt` — class balance overview (overall + per fold)
- `data/processed/cleaning_summary.txt` — one-page summary (rows, columns, k-anon, split counts, fairness notes)
- `reports/tables_public/mh_by_gender_rr3.csv` — privacy-safe RR3 table (Gender × Mental_Health_Condition)

> **Backward-compat:** the cleaner also writes `Has_MH_condition_bin` into `clean_full.csv` so older training code keeps working.

---

## Train/dev usage pattern (recommended)

- **Train features:** `clean_numeric_model.csv` (no direct sensitive columns)  
- **Splits:** read `splits_70_15_15_k5.csv` and only oversample/fit within `split=="train"`; keep `val/test` untouched  
- **SMOTENC:** use `smote_config.json` to pick categorical indices; apply inside each training fold only  
- **Fairness:** consult `sample_weights.csv`, `clean_full.csv` (for audits), and `cleaning_summary.txt` warnings

---

# Versioning

Below is what we shipped in **v1**, and what changed in **v2**.

## v1 — Baseline (original repository cleaner)

**What it did**
- Robust CSV load; NA normalization; light header cleanup
- Type coercion for key continuous fields
- Text normalization and mapping for common lifestyle variables
- Range clipping to NA; deduplication by `ID` (if present)
- Label encoding for nominal fields; basic numeric view
- Exported:  
  - `clean_full.csv`, `clean_numeric.csv`, `label_mappings.json`  
  - (sometimes) a public table for counts

**What it did not include**
- No deterministic 70/15/15 split or CV labels
- No dedicated model view to remove sensitive columns
- No fairness diagnostics or sample weights
- No SMOTE/SMOTENC setup artifacts
- No RR3 small-cell protection guarantee on public tables

**How teams trained**
- Ad-hoc train/val/test splits done downstream
- Risk of training on direct sensitive features unless manually filtered
- Class imbalance handled case-by-case without shared configuration

---

## v2 — What changed relative to v1 (the lean, productionized pipeline)

**New capabilities**
1. **Deterministic splits:** Stratified **70/15/15** + **train-only k-fold** labels → `splits_70_15_15_k5.csv`.
2. **Privacy-safe model view:** `clean_numeric_model.csv` drops direct sensitive cols (e.g., `Gender_norm`/`_lbl`) and can add `*_isna` flags.
3. **Fairness guardrails:**  
   - Split-wise **representation & prevalence** summaries by sensitive group  
   - **Small-group warnings** in val/test (`< GROUP_MIN_PER_SPLIT`, default 50)  
   - **Missingness gaps** leaderboard across groups
4. **Optional sample weights:** `sample_weights.csv` with `w_label`, `w_group`, `w_combo`.
5. **SMOTENC-ready artifacts:**  
   - `smote_config.json` — categorical indices, feature columns, recommended knobs  
   - `imbalance_report.txt` — pos/neg counts overall and per training fold
6. **Public privacy tables (hardened):** `mh_by_gender_rr3.csv` uses RR3 rounding and k-anonymity checks; low-freq merges for Country/Occupation; `AgeBand` bucketing.
7. **Target inference is robust:** If text “yes/no” is missing, the pipeline resolves the target via label mappings.
8. **Reproducibility & ergonomics:**  
   - Repository-relative paths and CLI flags  
   - Sensible defaults, single-file drop-in replacement  
   - Backward-compatibility: writes `Has_MH_condition_bin` into `clean_full.csv`

**Behavioral changes (be aware)**
- **Default seed** is `760` (was commonly `42`)
- **Continuous range clipping** now standardized (out-of-bound → NA)
- **Imputation policy** centralized: median for continuous, mode for integer/bool
- **Sensitive features** are excluded from the model view by default (training should point to `clean_numeric_model.csv`)
- **Public counts** are RR3-protected; raw small cells are suppressed/rounded

**What you need to change in training**
- Point feature loading to `clean_numeric_model.csv` (or keep `clean_numeric.csv` but be explicit about columns)
- Consume `splits_70_15_15_k5.csv`; never oversample or tune on val/test
- Use `smote_config.json` if you apply SMOTENC; prefer `sample_weights.csv` for label/group balancing

---

## Troubleshooting

- **“CSV not found”**: put the raw file at `data/raw/mental_health_data final data.csv` (preferred) or `data/`. Or pass `--input`.
- **Unmapped tokens warning**: set `STRICT=True` in the script to turn warnings into errors for early catching.
- **Small-group warnings**: if your dataset is small, consider lowering `GROUP_MIN_PER_SPLIT`.
- **No target detected**: ensure `Mental_Health_Condition` (yes/no) exists, or its label-encoded counterpart is present along with `label_mappings.json`.

---

## Maintainers

- Data pipeline owner: _your-team-name_  
- Issues & PRs: please include the `cleaning_summary.txt` and your command line

---
