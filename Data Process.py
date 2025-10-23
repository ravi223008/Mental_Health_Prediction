# Data Pre.py — lean cleaning + privacy-safe tables + stratified 70/15/15 + 5-fold (train only)
# + fairness guardrails (model view / warnings / optional weights)
# + SMOTE prep artifacts (diagnostics & config; NO oversampling here)
# -*- coding: utf-8 -*-

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

# ======================================================
# Project paths (relative to repository root)
# ======================================================
REPO_ROOT = Path(__file__).resolve().parent

# Default input search order: prefer data/raw/, then data/
RAW_FILENAME = "mental_health_data final data.csv"
DEFAULT_IN_CANDIDATES = [
    REPO_ROOT / "data" / "raw" / RAW_FILENAME,
    REPO_ROOT / "data" / RAW_FILENAME,
]
DEFAULT_OUT = REPO_ROOT / "data" / "processed"
DEFAULT_PUB = REPO_ROOT / "reports" / "tables_public"

# ======================================================
# Config knobs (can be overridden via CLI if needed)
# ======================================================
SEED = 760
STRICT = False               # True: raise on unmapped tokens; False: set to NA with a warning
LOW_FREQ = 20                # rare-category merge threshold for public view
K_MIN = 5                    # k-anonymity threshold
SENSITIVE_GROUP_COL = "Gender_norm"
MAKE_MODEL_VIEW = True       # export clean_numeric_model.csv with direct sensitive columns removed
ADD_MISSING_FLAGS = True     # append *_isna indicators for key numeric fields to the model view
GROUP_MIN_PER_SPLIT = 50     # warn if val/test have groups smaller than this
MAKE_WEIGHTS = True          # export sample_weights.csv (label & group balancing)
EMIT_LEGACY_TARGET = True    # also export Has_MH_condition_bin to remain backward-compatible

# Accepted ranges for continuous fields (outside → NA). Stress_Level is ordinal (excluded here).
RANGE: Dict[str, Tuple[float, float]] = {
    "Age": (10, 100),
    "Sleep_Hours": (0, 16),
    "Work_Hours": (0, 100),
    "Physical_Activity_Hours": (0, 20),
    "Social_Media_Usage": (0, 24),
}

# ======================================================
# Utilities
# ======================================================
def read_csv_robust(path: Path) -> pd.DataFrame:
    """
    Attempt to read CSV using a few common encodings before falling back to pandas default.
    """
    for enc in ("utf-8", "utf-8-sig", "gb18030", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[INFO] CSV loaded ({enc}) -> {df.shape}")
            return df
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(path)
    print(f"[INFO] CSV loaded (default) -> {df.shape}")
    return df


def norm_lower(s: pd.Series) -> pd.Series:
    """
    Normalize text to lowercase and strip spaces.
    """
    return s.astype("string").str.strip().str.lower()


def norm_token(s: pd.Series) -> pd.Series:
    """
    Normalize text to alphanumeric tokens separated by single spaces.
    """
    x = s.astype("string").str.lower()
    return (
        x.str.replace(r"[^a-z0-9]+", " ", regex=True)
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
    )


def find_col(df: pd.DataFrame, *names: str) -> Optional[str]:
    """
    Case/space-insensitive column matcher. Returns the actual column name or None.
    """
    norm_map = {c.lower().replace(" ", "_"): c for c in df.columns}
    for n in names:
        key = n.lower().replace(" ", "_")
        if key in norm_map:
            return norm_map[key]
    return None


def ensure_numeric(df: pd.DataFrame, cols: List[Optional[str]]) -> None:
    """
    Coerce columns to numeric if present; invalid values become NA.
    """
    for c in cols:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def to_binary(s: pd.Series) -> pd.Series:
    """
    Map common yes/no patterns to {1,0} (Int8). Unrecognized → NA.
    """
    yes = {"yes", "y", "true", "t", "1", "ever", "present", "positive"}
    no = {"no", "n", "false", "f", "0", "never", "absent", "negative"}
    x = norm_lower(s)
    y = pd.Series(pd.NA, index=s.index, dtype="Int8")
    return y.mask(x.isin(yes), 1).mask(x.isin(no), 0)


def label_encode_with_unknown(s: pd.Series, unknown: str = "Unknown") -> Tuple[pd.Series, Dict[str, int]]:
    """
    Label-encode a string series with a dedicated 'Unknown' bucket. Returns (codes, mapping).
    """
    s2 = s.replace(r"^\s*$", pd.NA, regex=True).fillna(unknown).astype("string")
    cats = pd.Categorical(s2)
    codes = pd.Series(cats.codes, index=s.index).astype("Int16")
    mapping = {str(cat): int(i) for i, cat in enumerate(cats.categories)}
    return codes, mapping


def drop_constant_lowvar(df: pd.DataFrame, min_unique: int = 2) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop columns with fewer than min_unique distinct values (including NA).
    """
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) < min_unique]
    return df.drop(columns=const_cols), const_cols


def clip_to_nan(df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]) -> None:
    """
    Set values outside [lo, hi] to NA for each column in bounds.
    """
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            bad = df[col].lt(lo) | df[col].gt(hi)
            if bad.any():
                print(f"[INFO] clip -> {col}: {bad.sum()} out-of-range set to NA")
            df.loc[bad, col] = pd.NA


def k_anonymity_violations(df: pd.DataFrame, quasi_cols: List[str], k: int = 5) -> pd.Series:
    """
    Return group sizes under k for k-anonymity check on the given quasi-identifiers.
    """
    g = df.groupby(quasi_cols, dropna=False).size()
    return g[g < k].sort_values()


def rr3_series(counts: pd.Series, seed: int = SEED) -> pd.Series:
    """
    Round-and-randomize-to-3 (RR3) for small-cell protection.
    """
    rng = np.random.default_rng(seed)
    base = 3
    out = []
    for n in counts.astype(int):
        if n <= 0:
            out.append(0)
            continue
        lo = (n // base) * base
        hi = lo + base
        if n == lo:
            out.append(n)
            continue
        p_hi = (n - lo) / base
        out.append(int(rng.choice([lo, hi], p=[1 - p_hi, p_hi])))
    return pd.Series(out, index=count.s.index if hasattr(counts, "s") else counts.index, dtype=int)


def merge_low_freq(s: pd.Series, min_count: int = LOW_FREQ, other_label: str = "Other") -> pd.Series:
    """
    Merge categories with frequency < min_count into 'Other'.
    """
    vc = s.value_counts(dropna=False)
    rare = set(vc[vc < min_count].index.tolist())
    return s.where(~s.isin(rare), other_label)


def map_tokens(s: pd.Series, vocab: Dict[str, int], name: str = "") -> pd.Series:
    """
    Map normalized tokens to integers using a vocabulary; unmapped tokens → NA (or raise if STRICT).
    """
    x = norm_token(s)
    # Minimal synonym expansion
    syn = {
        "non smoker": "non-smoker", "nonsmoker": "non-smoker", "ex smoker": "occasional smoker",
        "occasional": "occasional smoker", "moderate": "regular smoker", "regular": "regular smoker",
        "non drinker": "non-drinker", "no alcohol": "non-drinker", "teetotaler": "non-drinker",
        "social": "social drinker", "socially": "social drinker", "moderate drinker": "regular drinker",
        "poor": "unhealthy", "good": "healthy", "mild": "low", "severe": "high",
    }
    x = x.replace(syn)
    bad = sorted(x[~x.isin(vocab) & x.notna()].unique().tolist())
    if bad and STRICT:
        raise ValueError(f"[{name}] unmapped: {bad[:10]} ...")
    if bad:
        print(f"[WARN] [{name}] unmapped->NA: {bad[:10]} ...")
    return x.where(x.isin(vocab), pd.NA).map(vocab)

# ======================================================
# Splits & CV (no sklearn)
# ======================================================
def target_from_full(clean_full: pd.DataFrame, label_maps: Optional[Dict[str, Dict[str, int]]] = None) -> Optional[np.ndarray]:
    """
    Infer binary target y from clean_full:
      1) Text column Mental_Health_Condition / MHC / Condition (yes/no → 1/0)
      2) Otherwise, via label_mappings for Mental_Health_Condition_lbl
    """
    tcol = find_col(clean_full, "Mental_Health_Condition", "MHC", "Condition")
    if tcol:
        return (clean_full[tcol].astype("string").str.strip().str.lower() == "yes").astype(int).to_numpy()

    lbl = find_col(clean_full, "Mental_Health_Condition_lbl")
    if lbl and label_maps and "Mental_Health_Condition" in label_maps:
        inv = {int(v): str(k).lower() for k, v in label_maps["Mental_Health_Condition"].items()}
        arr = clean_full[lbl].to_numpy()
        return np.array([1 if inv.get(int(v), "") == "yes" else 0 for v in arr], dtype=int)
    return None


def stratified_70_15_15(y: np.ndarray, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create stratified masks for 70% train / 15% val / 15% test (by label).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    tr = np.zeros(n, bool)
    va = np.zeros(n, bool)
    te = np.zeros(n, bool)
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        m = len(idx)
        ntr = int(np.floor(0.70 * m))
        nva = int(np.floor(0.15 * m))
        tr[idx[:ntr]] = True
        va[idx[ntr:ntr + nva]] = True
        te[idx[ntr + nva:]] = True
    rest = ~tr & ~va & ~te
    te[rest] = True
    return tr, va, te


def stratified_kfold_on_train(y: np.ndarray, train_mask: np.ndarray, k: int = 5, seed: int = SEED) -> np.ndarray:
    """
    Assign k-fold IDs (0..k-1) within the training set, stratified by label.
    Non-training rows receive -1.
    """
    rng = np.random.default_rng(seed)
    cv = np.full(len(y), -1, int)
    for cls in [0, 1]:
        idx = np.where((y == cls) & train_mask)[0]
        rng.shuffle(idx)
        for fid, chunk in enumerate(np.array_split(idx, k)):
            if len(chunk):
                cv[chunk] = fid
    return cv

# ======================================================
# Privacy, fairness, and SMOTE preparation
# ======================================================
def make_privacy_tables(clean_full: pd.DataFrame, pub_dir: Path) -> str:
    """
    Build privacy-safe tables:
      - Age bands and rare-category merges for Country/Occupation in public view
      - k-anonymity check on quasi-identifiers
      - RR3-protected Gender x Mental_Health_Condition table
    Returns a short text summary for k-anonymity status.
    """
    pub = clean_full.copy()

    # Age banding (20-year bins starting at 10)
    agec = find_col(pub, "Age")
    if agec:
        pub["AgeBand"] = pd.cut(pd.to_numeric(pub[agec], errors="coerce"),
                                bins=list(range(10, 110, 20)), right=False)

    # Merge low-frequency categories
    if find_col(pub, "Occupation"):
        pub["Occupation_pub"] = merge_low_freq(pub[find_col(pub, "Occupation")])
    if find_col(pub, "Country"):
        pub["Country_pub"] = merge_low_freq(pub[find_col(pub, "Country")])

    # k-anonymity check
    qcols = [c for c in ["Gender_norm", "Country_pub", "Occupation_pub", "AgeBand"] if c in pub.columns]
    viol_txt = ""
    if qcols:
        viol = k_anonymity_violations(pub, qcols, k=K_MIN)
        viol_txt = f"k-anonymity <{K_MIN}: {len(viol)} groups"
        print(("[WARN] " if len(viol) > 0 else "[INFO] ") + viol_txt)

    # RR3: small-cell protection on Gender x Mental_Health_Condition
    if {"Gender_norm", "Mental_Health_Condition"}.issubset(pub.columns):
        tab = pub.pivot_table(index="Gender_norm", columns="Mental_Health_Condition",
                              aggfunc="size", fill_value=0)
        tab = tab.mask(tab < 3, 0)  # small cell suppression (deterministic floor)
        tab_rr3 = tab.apply(rr3_series, axis=0)
        pub_dir.mkdir(parents=True, exist_ok=True)
        out = pub_dir / "mh_by_gender_rr3.csv"
        out.write_text(tab_rr3.to_csv(index=True), encoding="utf-8")
        print(f"[INFO] RR3 table -> {out}")

    return viol_txt


def make_model_view_and_weights(
    clean_full: pd.DataFrame,
    clean_numeric: pd.DataFrame,
    out_dir: Path,
    label_maps: Dict[str, Dict[str, int]],
    splits_path: Path,
    lines_accum: List[str],
) -> None:
    """
    Export a model-friendly numeric table without direct sensitive columns,
    emit group representation diagnostics per split, compute missingness gaps,
    and (optionally) export sample weights.
    """
    fair_lines: List[str] = []

    # 1) Model view without direct sensitive columns
    if MAKE_MODEL_VIEW:
        drop = []
        if SENSITIVE_GROUP_COL in clean_numeric.columns:
            drop.append(SENSITIVE_GROUP_COL)
        lbl = f"{SENSITIVE_GROUP_COL}_lbl"
        if lbl in clean_numeric.columns:
            drop.append(lbl)
        keep_cols = [c for c in clean_numeric.columns if c not in drop]

        model_view = clean_numeric[keep_cols].copy()

        # Optional missingness indicators
        if ADD_MISSING_FLAGS:
            miss_cols = [
                c for c in ["Age", "Sleep_Hours", "Work_Hours", "Physical_Activity_Hours", "Social_Media_Usage"]
                if c in clean_full.columns
            ]
            for c in miss_cols:
                model_view[f"{c}_isna"] = clean_full[c].isna().astype("Int8")

        model_path = out_dir / "clean_numeric_model.csv"
        model_view.to_csv(model_path, index=False)
        fair_lines.append("[FAIR] Model view: clean_numeric_model.csv (no direct sensitive columns)")

    # 2) Group representation and prevalence per split
    y_arr = target_from_full(clean_full, label_maps)
    if splits_path.exists() and y_arr is not None and SENSITIVE_GROUP_COL in clean_full.columns:
        sp = pd.read_csv(splits_path)
        for split_name in ["train", "val", "test"]:
            idx = sp.index[sp["split"] == split_name]
            sub = clean_full.loc[idx]
            if len(sub) == 0:
                fair_lines.append(f"[FAIR] {split_name}: 0 rows")
                continue

            grp = sub.groupby(SENSITIVE_GROUP_COL)["Mental_Health_Condition"].agg(["count"])
            prev = pd.Series(y_arr[idx], index=sub.index)
            rep = sub.assign(y=prev.values).groupby(SENSITIVE_GROUP_COL)["y"].mean().rename("pos_rate")
            rep = pd.concat([grp, rep], axis=1).sort_values("count", ascending=False)

            # Warn on small groups in val/test
            if split_name in ("val", "test"):
                small = rep[rep["count"] < GROUP_MIN_PER_SPLIT]
                if len(small):
                    fair_lines.append(
                        f"[FAIR][WARN] {split_name} small groups <{GROUP_MIN_PER_SPLIT}: " +
                        ", ".join([f"{g}:{int(n)}" for g, n in small["count"].items()])
                    )

            fair_lines.append(
                f"[FAIR] {split_name} by {SENSITIVE_GROUP_COL}: " +
                ", ".join([f"{g}=n{int(n)}/pos{p:.2f}" for g, (n, p) in zip(rep.index, zip(rep['count'], rep['pos_rate']))])
            )

    # 3) Missingness gaps across groups (top features)
    if SENSITIVE_GROUP_COL in clean_full.columns:
        miss_cols = [
            c for c in ["Age", "Sleep_Hours", "Work_Hours", "Physical_Activity_Hours", "Social_Media_Usage"]
            if c in clean_full.columns
        ]
        gaps: List[Tuple[str, float]] = []
        for c in miss_cols:
            rates = clean_full.groupby(SENSITIVE_GROUP_COL)[c].apply(lambda s: s.isna().mean())
            if len(rates) >= 2:
                gaps.append((c, float(rates.max() - rates.min())))
        if gaps:
            gaps.sort(key=lambda x: x[1], reverse=True)
            fair_lines.append("[FAIR] Missingness gap (top): " + ", ".join([f"{c}:{g:.2f}" for c, g in gaps[:5]]))

    # 4) Optional sample weights (inverse label & group frequency; geometric combo)
    if MAKE_WEIGHTS and y_arr is not None:
        weights = pd.DataFrame({"row_id": np.arange(len(clean_full))})
        p1 = float(np.mean(y_arr))
        p0 = 1.0 - p1
        w_label = np.where(y_arr == 1, 1.0 / max(p1, 1e-6), 1.0 / max(p0, 1e-6))
        weights["w_label"] = w_label / np.mean(w_label)

        if SENSITIVE_GROUP_COL in clean_full.columns:
            g = clean_full[SENSITIVE_GROUP_COL].astype("string").fillna("Unknown")
            sizes = g.value_counts()
            w_group = g.map(lambda z: 1.0 / max(float(sizes.get(z, len(clean_full))), 1.0))
            weights["w_group"] = (w_group / w_group.mean()).values
            weights["w_combo"] = np.sqrt(weights["w_label"] * weights["w_group"])
        else:
            weights["w_group"] = 1.0
            weights["w_combo"] = weights["w_label"]

        weights.to_csv(out_dir / "sample_weights.csv", index=False)
        fair_lines.append("[FAIR] Optional weights -> sample_weights.csv (w_label, w_group, w_combo)")

    lines_accum += fair_lines


def make_smote_prep(
    clean_full: pd.DataFrame,
    clean_numeric: pd.DataFrame,
    out_dir: Path,
    label_maps: Dict[str, Dict[str, int]],
    splits_path: Path,
    lines_accum: List[str],
) -> None:
    """
    Prepare SMOTE/SMOTENC artifacts for the modeling stage (do NOT oversample here):
      - smote_config.json: feature columns, categorical indices for SMOTENC, recommended knobs
      - imbalance_report.txt: train set pos/neg and per-fold distribution
    """
    try:
        feat_path = out_dir / "clean_numeric_model.csv"
        if not feat_path.exists():
            feat_path = out_dir / "clean_numeric.csv"
        cols_probe = pd.read_csv(feat_path, nrows=1).columns.tolist()

        cat_idx = [i for i, c in enumerate(cols_probe) if c.endswith("_lbl") or c.endswith("_bin")]
        num_idx = [i for i in range(len(cols_probe)) if i not in cat_idx]

        if not splits_path.exists():
            lines_accum.append("SMOTE prep: skipped (no splits file)")
            return

        y_vec = target_from_full(clean_full, label_maps)
        if y_vec is None:
            lines_accum.append("SMOTE prep: cannot infer target")
            return

        sp = pd.read_csv(splits_path)
        train_rows = sp.index[sp["split"] == "train"].to_numpy()
        y_train = y_vec[train_rows]
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        minority = min(n_pos, n_neg)
        majority = max(n_pos, n_neg)
        ratio = float(minority / majority) if majority > 0 else 0.0

        rec_k = max(1, min(5, minority - 1))
        rec_strategy = 1.0 if ratio < 0.4 else "auto"
        rec_sampler = "SMOTENC" if len(cat_idx) > 0 else "SMOTE"

        fold_ids = sp.loc[sp["split"] == "train", "cv_fold"].values
        fold_counts = []
        kmax = int(np.nanmax(fold_ids)) if len(fold_ids) else -1
        for k in range(kmax + 1):
            mk = (fold_ids == k)
            yk = y_train[mk]
            fold_counts.append({
                "fold": int(k),
                "n_pos": int((yk == 1).sum()),
                "n_neg": int((yk == 0).sum()),
            })

        cfg = {
            "features_path": str(feat_path),
            "feature_columns": cols_probe,
            "categorical_indices": cat_idx,
            "numeric_indices": num_idx,
            "target_name": "Mental_Health_Condition",
            "train_stats": {"n_pos": n_pos, "n_neg": n_neg, "minority_ratio": round(ratio, 4)},
            "cv_folds": {"k": int(len(fold_counts)), "train_fold_counts": fold_counts},
            "recommended": {
                "sampler": rec_sampler,
                "sampling_strategy": rec_strategy,
                "k_neighbors": int(rec_k),
                "random_state": int(SEED),
            },
            "notes": "Apply SMOTENC inside each training fold only; never touch val/test.",
        }
        (out_dir / "smote_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = []
        lines.append(f"Feature table: {Path(cfg['features_path']).name}")
        lines.append(f"Categorical idx ({len(cat_idx)}): {cat_idx[:20]}{' ...' if len(cat_idx) > 20 else ''}")
        lines.append(f"Numeric idx     ({len(num_idx)}): {num_idx[:20]}{' ...' if len(num_idx) > 20 else ''}")
        lines.append(f"Train pos/neg: {n_pos}/{n_neg} (minority_ratio={ratio:.3f})")
        lines.append(
            f"Recommended: sampler={rec_sampler}, strategy={rec_strategy}, "
            f"k_neighbors={rec_k}, random_state={SEED}"
        )
        lines.append("Train folds: " + "; ".join([f"fold={d['fold']}, pos={d['n_pos']}, neg={d['n_neg']}" for d in fold_counts]))
        (out_dir / "imbalance_report.txt").write_text("\n".join(lines), encoding="utf-8")

        lines_accum.append(
            f"SMOTE prep: minority_ratio(train)={ratio:.3f}, cfg=smote_config.json, rpt=imbalance_report.txt"
        )
    except Exception as e:
        lines_accum.append(f"SMOTE prep: error -> {e}")

# ======================================================
# Main pipeline
# ======================================================
def run(args) -> None:
    """
    End-to-end cleaning:
      - robust CSV read, normalization, type coercion, range clipping, dedup
      - label-encoding (incl. target inference)
      - numeric/model views and imputations
      - privacy-safe public tables (k-anon summary + RR3)
      - stratified 70/15/15 + train-only k-fold labels
      - fairness guardrails & optional sample weights
      - SMOTE/SMOTENC configuration (no oversampling here)
    """
    # 1) Resolve input path: explicit argument or default candidates
    in_csv = Path(args.input).resolve() if args.input else None
    if not in_csv or not in_csv.exists():
        tried = []
        if in_csv:
            tried.append(str(in_csv))
        for cand in DEFAULT_IN_CANDIDATES:
            tried.append(str(cand))
            if cand.exists():
                in_csv = cand.resolve()
                break
        if not in_csv or not in_csv.exists():
            msg = " / ".join(tried)
            raise FileNotFoundError(
                f"CSV not found. Tried: {msg}\n"
                f"Hint: place raw file under {REPO_ROOT/'data'/'raw'} "
                f"or {REPO_ROOT/'data'} or pass --input"
            )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pub_dir = Path(args.pub_dir).resolve()
    pub_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df = read_csv_robust(in_csv)

    # Normalize NA tokens and clean column names
    df = (
        df.replace(r"^\s*$", pd.NA, regex=True)
          .replace(r"(?i)^(na|n/a|null|none|-|--)$", pd.NA, regex=True)
    )
    df.columns = [c.strip().replace("  ", " ").strip() for c in df.columns]
    n0 = len(df)

    # Force numeric coercion for key continuous fields
    num_candidates = ["Age", "Sleep_Hours", "Work_Hours", "Physical_Activity_Hours", "Social_Media_Usage"]
    ensure_numeric(df, [find_col(df, x, x.replace("_", " ")) for x in num_candidates
                        if find_col(df, x, x.replace("_", " "))])

    # Normalize key text fields
    gcol = find_col(df, "Gender", "Sex")
    if gcol:
        g = norm_lower(df[gcol])
        gender_map = {
            "male": "Male", "m": "Male", "man": "Male",
            "female": "Female", "f": "Female", "woman": "Female",
            "non-binary": "Other", "nonbinary": "Other", "nb": "Other", "other": "Other",
            "prefer not to say": "Unknown", "na": "Unknown", "n/a": "Unknown",
            "unknown": "Unknown", "": "Unknown",
        }
        df["Gender_norm"] = g.map(gender_map).fillna("Unknown")

    rcol = find_col(df, "Relationship_Status", "Relationship Status", "Marital_Status", "Marital Status", "MaritalStatus")
    if rcol:
        rs = norm_lower(df[rcol])
        rs_map = {
            "single": "single", "married": "married", "divorced": "divorced", "widowed": "widowed",
            "in a relationship": "in_relationship", "committed relationship": "in_relationship",
            "prefer not to say": "unknown", "unknown": "unknown", "": "unknown",
        }
        df["Relationship_Status_norm"] = rs.map(rs_map).fillna("unknown")

    for base in ["Consultation_History", "Medication_Usage"]:
        cname = find_col(df, base, base.replace("_", " "))
        if cname:
            df[f"{base}_bin"] = to_binary(df[cname])

    # Ordinal/boolean mappings (compact)
    conf = [
        (find_col(df, "Smoking_Habit", "Smoking Habit", "Smoking", "Smoking Status"),
         {"non-smoker": 0, "occasional smoker": 1, "regular smoker": 2, "heavy smoker": 3}, "Smoking_Habit"),
        (find_col(df, "Alcohol_Consumption", "Alcohol Consumption", "Alcohol", "Alcohol Use", "Alcohol Status"),
         {"non-drinker": 0, "social drinker": 1, "regular drinker": 2, "heavy drinker": 3}, "Alcohol_Consumption"),
        (find_col(df, "Diet_Quality", "Diet Quality", "Diet"),
         {"unhealthy": 0, "average": 1, "healthy": 2}, "Diet_Quality"),
        (find_col(df, "Severity", "Condition_Severity", "Severity Level"),
         {"low": 0, "medium": 1, "high": 2}, "Severity"),
        (find_col(df, "Stress_Level", "Stress Level"),
         {"low": 0, "medium": 1, "high": 2}, "Stress_Level"),
    ]
    for cname, vocab, name in conf:
        if cname:
            df[f"{name}_ord"] = map_tokens(df[cname], vocab, name=name).astype("Int8")
            if name in ("Smoking_Habit", "Alcohol_Consumption"):
                df[f"{name}_bin"] = (df[f"{name}_ord"] >= 2).astype("Int8")

    # Range clipping on continuous fields (out-of-range → NA)
    bounds: Dict[str, Tuple[float, float]] = {}
    for k, (lo, hi) in RANGE.items():
        cname = find_col(df, k, k.replace("_", " "))
        if cname:
            df[cname] = pd.to_numeric(df[cname], errors="coerce")
            bounds[cname] = (lo, hi)
    clip_to_nan(df, bounds)

    # Deduplicate by ID if present, else full-row dedup
    idc = find_col(df, "ID")
    df = df.drop_duplicates(subset=[idc]) if idc else df.drop_duplicates()

    # Full view (for analysis/fairness)
    clean_full = df.copy()

    # Label-encode nominal fields (also capture mapping)
    label_maps: Dict[str, Dict[str, int]] = {}
    for (name, cname) in {
        "Gender_norm": find_col(df, "Gender_norm"),
        "Occupation":  find_col(df, "Occupation", "Job", "Employment"),
        "Country":     find_col(df, "Country", "Location", "Region", "Nationality"),
        "Relationship_Status_norm": find_col(df, "Relationship_Status_norm"),
        "Mental_Health_Condition":  find_col(df, "Mental_Health_Condition", "Mental Health Condition", "Condition", "MHC", "MentalHealth"),
    }.items():
        if cname:
            codes, mapping = label_encode_with_unknown(df[cname])
            clean_full[f"{name}_lbl"] = codes
            label_maps[name] = mapping

    # Backward compatibility: export Has_MH_condition_bin if requested
    if EMIT_LEGACY_TARGET:
        y_tmp = target_from_full(clean_full, label_maps)
        if y_tmp is not None:
            clean_full["Has_MH_condition_bin"] = pd.Series(y_tmp, index=clean_full.index).astype("Int8")

    # Numeric view (modeling-ready)
    keep_suffix = ("_ord", "_bin", "_lbl")
    num_cols = [
        c for c in clean_full.columns
        if pd.api.types.is_numeric_dtype(clean_full[c]) or any(c.endswith(s) for s in keep_suffix)
    ]
    clean_numeric = clean_full[num_cols].copy()

    # Drop constant/low-var columns + impute
    clean_numeric, dropped = drop_constant_lowvar(clean_numeric, 2)
    for c in clean_numeric.columns:
        if pd.api.types.is_integer_dtype(clean_numeric[c]) or pd.api.types.is_bool_dtype(clean_numeric[c]):
            m = clean_numeric[c].mode(dropna=True)
            if len(m):
                clean_numeric[c] = clean_numeric[c].fillna(m.iloc[0])
        else:
            try:
                clean_numeric[c] = clean_numeric[c].fillna(clean_numeric[c].median())
            except Exception:
                clean_numeric[c] = clean_numeric[c].fillna(clean_numeric[c].mode(dropna=True).iloc[0])

    # Privacy-safe public tables
    viol_txt = make_privacy_tables(clean_full, pub_dir)

    # Write base artifacts
    out_full = out_dir / "clean_full.csv"
    out_num  = out_dir / "clean_numeric.csv"
    out_map  = out_dir / "label_mappings.json"
    clean_full.to_csv(out_full, index=False)
    clean_numeric.to_csv(out_num, index=False)
    out_map.write_text(json.dumps(label_maps, ensure_ascii=False, indent=2), encoding="utf-8")

    # Stratified 70/15/15 + train-only k-fold CV
    split_info = "Splits: skipped (--no_splits)"
    splits_path = out_dir / f"splits_70_15_15_k{args.kfolds}.csv"
    if not args.no_splits:
        y = target_from_full(clean_full, label_maps)
        if y is None:
            raise RuntimeError("Cannot locate Mental_Health_Condition (text or *_lbl via mapping).")
        tr, va, te = stratified_70_15_15(y, seed=SEED)
        cv_fold = stratified_kfold_on_train(y, tr, k=args.kfolds, seed=SEED)

        split = np.full(len(y), "unassigned", object)
        split[tr] = "train"
        split[va] = "val"
        split[te] = "test"
        assert (split != "unassigned").all() and (cv_fold[split != "train"] == -1).all()

        pd.DataFrame({
            "row_id": np.arange(len(y), dtype=int),
            "split": split,
            "cv_fold": cv_fold,
        }).to_csv(splits_path, index=False)

        counts = pd.Series(split).value_counts().to_dict()
        folds = pd.Series(cv_fold[cv_fold >= 0]).value_counts().sort_index().to_dict()
        split_info = f"Splits -> {splits_path.name} | counts={counts} | train folds={folds}"

    # Fairness guardrails (model view / warnings / optional weights)
    lines: List[str] = []
    make_model_view_and_weights(clean_full, clean_numeric, out_dir, label_maps, splits_path, lines)

    # SMOTE prep (config + diagnostics; no touching val/test)
    make_smote_prep(clean_full, clean_numeric, out_dir, label_maps, splits_path, lines)

    # Summary
    summ = out_dir / "cleaning_summary.txt"
    rows = [
        f"Input rows: {n0}, Output rows: {len(clean_full)}",
        f"Columns (full→numeric): {clean_full.shape[1]} → {clean_numeric.shape[1]}",
        f"Dropped constant/low-var: {dropped}",
        (f"Severity_ord NA rate: {clean_full['Severity_ord'].isna().mean():.3f}" if 'Severity_ord' in clean_full.columns else ""),
        (f"Stress_Level_ord NA rate: {clean_full['Stress_Level_ord'].isna().mean():.3f}" if 'Stress_Level_ord' in clean_full.columns else ""),
        (viol_txt or ""),
        split_info,
        *lines,
        f"Privacy-safe tables: {pub_dir}",
    ]
    summ.write_text("\n".join([r for r in rows if r]), encoding="utf-8")

    print("\n".join([r for r in rows if r]))
    print("Artifacts:")
    for p in [out_full, out_num, out_map, summ]:
        print(" -", p)
    if not args.no_splits:
        print(" -", splits_path)
    if (out_dir / "clean_numeric_model.csv").exists():
        print(" -", out_dir / "clean_numeric_model.csv")
    if (out_dir / "sample_weights.csv").exists():
        print(" -", out_dir / "sample_weights.csv")
    if (out_dir / "smote_config.json").exists():
        print(" -", out_dir / "smote_config.json")
    if (out_dir / "imbalance_report.txt").exists():
        print(" -", out_dir / "imbalance_report.txt")
    print(f"Done in {time.time() - t0:.2f}s.")

# ======================================================
# CLI
# ======================================================
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Lean cleaning + privacy + splits + fairness guardrails + SMOTE prep (legacy-compatible)"
    )
    ap.add_argument("--input",   default="", help="path to raw CSV (auto-searches data/raw and data if empty)")
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT), help="directory for processed artifacts")
    ap.add_argument("--pub_dir", default=str(DEFAULT_PUB), help="directory for privacy-safe public tables")
    ap.add_argument("--no_splits", action="store_true", help="skip 70/15/15 + train-only k-fold labels")
    ap.add_argument("--kfolds", type=int, default=5, help="number of CV folds on the training set")
    return ap


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        run(args)
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)
