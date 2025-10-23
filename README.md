# Mental_Health_Prediction

## Data Processing

This section documents how raw survey-style data are converted into clean, typed, **leak-free** datasets ready for modeling.

### Purpose
- Produce **clean, typed, reproducible** features/tables.
- Enforce **leak-free preprocessing**: fit encoders/imputers/scalers on **train only**; **apply** to val/test.
- Keep an **audit trail** (column lists, label map, missingness report).
- Defer **class-imbalance handling** to the modeling stage (no oversampling here).

### Code & Folders
- **Script:** `Data Process.py`
- **Data directories:**
  - `data/raw/` — place original CSV(s) here
  - `data/processed/` — script writes outputs here

### Inputs
- A single, wide CSV in `data/raw/` (UTF-8).  
  Columns may include:
  - continuous metrics (e.g., sleep duration, activity index)
  - ordinal Likert-style responses
  - binary indicators
  - one or more label-like fields (see **Label standardization**)

### Pipeline (high level)
1. **Column normalization**  
   Harmonize column names and standardize common boolean/category variants (`Yes/No`, `Y/N`, `1/0`).

2. **Type casting & feature grouping**  
   Partition features into four lists for consistent downstream use:
   - `numeric` — continuous variables  
   - `ord` — ordered categoricals (Likert)  
   - `bin` — binary indicators (0/1)  
   - `lbl` — label columns

3. **Label standardization (single source of truth)**  
   - Consolidate any historical/derived encodings into a single label `y` via a versioned mapping `M(Risk_lbl) → {0,1,2}` (e.g., {low, medium, high}).  
   - Treat legacy fields (e.g., `*_bin`, `*_code`) as **deterministic derivations** of `y`; exclude them from training features by default to prevent “multiple truths”.  
   - If two columns look identical (e.g., `Mental_Health_Condition_lbl` vs `Has_MH_condition_bin`), mark one as the **primary label** and keep the other for audit only.

4. **Missing-value handling**  
   - `numeric`: impute (mean/median) **fit on training only**; store parameters.  
   - `ord/bin`: merge rare levels; use explicit `Unknown/None` fallback.  
   - Log per-column missingness and chosen strategies.

5. **Stratified split (leak-free)**  
   - Default `train/val/test = 70/15/15` with a fixed seed; stratified by `y`.  
   - **Fit** encoders/imputers/scalers on **train only**; **transform** val/test.  
   - Emit a **k-fold plan** (CV inside the training set).

6. **Imbalance preparation**  
   - **No SMOTE/oversampling** here (to avoid contaminating val/test).  
   - Export class-distribution diagnostics for use **inside** training folds later.

7. **Fairness guardrails**  
   - Summaries by protected attributes (if present) and simple slices to inform later threshold alignment or re-weighting during modeling.

### Outputs (under `data/processed/`)
- `train.csv`, `val.csv`, `test.csv` — leak-free splits  
- `columns_numeric.txt`, `columns_ord.txt`, `columns_bin.txt`, `columns_lbl.txt` — feature lists  
- `label_map.json` — versioned `M(Risk_lbl) → {0,1,2}` used to define `y`  
- `cleaning_report.md` — missingness, imputations, merges, distributions, class balance  
- `folds.json` — k-fold configuration (training set only)

> These artifacts are consumed directly by downstream modeling code/notebooks.


