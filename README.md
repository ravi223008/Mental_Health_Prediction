# Mental_Health_Prediction

## Data Processing

**Scope.** This section documents how raw survey-style data are converted into clean, typed, and leak-free datasets ready for modeling. It covers inputs, transformations, outputs, and how to run the script.

### Purpose
- Produce **clean, typed, reproducible** features/tables.
- Enforce **leak-free preprocessing** (fit on train only; apply to val/test).
- Prepare for **class imbalance** handling during modeling (no oversampling at cleaning time).
- Maintain an **audit trail** (column lists, label map, missingness report).

### Where the code lives
- **Script:** `Data Process.py`
- **Data folders:**  
  - `data/raw/` — place original CSV(s) here  
  - `data/processed/` — script writes outputs here

### Input expectations
- A single wide CSV in `data/raw/` (e.g., `mental_health_data.csv`), UTF-8 encoded.
- Columns may include:
  - Continuous metrics (e.g., sleep duration, activity index)
  - Ordinal Likert-style responses
  - Binary indicators
  - One or more label-like fields (see **Label standardization**)

> If the file name or location differs, pass it explicitly via CLI (see **Run it**).

### Pipeline stages (high level)
1. **Column normalization**  
   Harmonize column names and standardize common boolean/category variants (`Yes/No`, `Y/N`, `1/0`).

2. **Type casting & feature grouping**  
   Partition features into four lists for consistent downstream use:  
   - `numeric` — continuous variables  
   - `ord` — ordered categoricals (Likert)  
   - `bin` — binary indicators (0/1)  
   - `lbl` — label columns (see next step)

3. **Label standardization (single source of truth)**  
   - Consolidate any historical/derived encodings into a single label `y`.  
   - Maintain a **versioned mapping** `M(Risk_lbl) → {0,1,2}` (e.g., {low, medium, high}).  
   - Treat legacy fields (e.g., `*_bin`, `*_code`) as **deterministic derivations** of `y`; exclude them from training features by default to prevent “multiple truths”.  
   - If two columns look identical (e.g., `Mental_Health_Condition_lbl` vs `Has_MH_condition_bin`), mark one as the **primary label** and keep the other only for audit checks.

4. **Missing-value handling**  
   - `numeric`: impute (mean/median) **fit on training folds only**; store parameters.  
   - `ord/bin`: merge rare levels; use explicit `Unknown/None` fallback.  
   - Log per-column missingness and chosen strategies.

5. **Stratified split (leak-free)**  
   - Default `train/val/test = 70/15/15` with fixed seed; stratified by `y`.  
   - **Fit** encoders/imputers/scalers on **train only**; **transform** val/test.  
   - Emit a **k-fold plan** for CV inside the training set.

6. **Imbalance preparation**  
   - **No SMOTE/oversampling** here (avoid contaminating val/test).  
   - Export class distribution and suggested configs for use **inside** training folds later.

7. **(Optional) Fairness guardrails**  
   - Summaries by protected attributes (if available) and simple slices to inform later threshold alignment or re-weighting during modeling.

### Outputs (`data/processed/`)
- `train.csv`, `val.csv`, `test.csv` — leak-free splits
- `columns_numeric.txt`, `columns_ord.txt`, `columns_bin.txt`, `columns_lbl.txt` — feature lists
- `label_map.json` — versioned `M(Risk_lbl) → {0,1,2}` used to define `y`
- `cleaning_report.md` — missingness, imputations, merges, distributions, class balance
- `folds.json` — k-fold configuration (training set only)
- *(Optional)* `transformers.pkl` — serialized transformers (encoders/imputers/scalers) fit on train

> These artifacts are consumed directly by the modeling stage (e.g., an XGBoost notebook/script).

### Run it
```bash
# A) Default: expects a CSV in data/raw/
python "Data Process.py"

# B) Explicit paths (flags may vary; see script header for full list)
python "Data Process.py" \
  --input "data/raw/mental_health_data.csv" \
  --outdir "data/processed" \
  --seed 769
