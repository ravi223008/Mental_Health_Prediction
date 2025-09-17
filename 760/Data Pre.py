# Data Pre v4.py — team-style upgrade with RELATIVE PATHS (cross-platform, English comments)
# -*- coding: utf-8 -*-
import os, json, time, re, argparse
import numpy as np
import pandas as pd
from pathlib import Path

# =========================
# Project-relative paths
# =========================
# Assumes this script sits at the repo root. Raw CSV lives in data/raw/.
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_IN  = REPO_ROOT / "data" / "mental_health_data final data.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "processed"
DEFAULT_PUB = REPO_ROOT / "reports" / "tables_public"

# =========================
# Runtime knobs
# =========================
STRICT   = False   # True = fail fast on unmapped values; False = warn and set to NA (recommended for first run)
K_MIN    = 5       # k-anonymity threshold for public views
SEED     = 42      # RNG seed for RR3 randomized rounding (for reproducibility)
LOW_FREQ = 20      # Merge public categories with count < LOW_FREQ into "Other"

# =========================
# Helpers
# =========================
def read_csv_robust(path: Path):
    """Try several common encodings, print which one worked."""
    for enc in ("utf-8", "utf-8-sig", "gb18030", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[INFO] Loaded CSV with encoding={enc}, shape={df.shape}")
            return df
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(path)  # final fallback
    print(f"[INFO] Loaded CSV with default encoding, shape={df.shape}")
    return df

def norm_lower(s: pd.Series) -> pd.Series:
    """Lowercase + trim whitespace."""
    return s.astype("string").str.strip().str.lower()

def norm_token(s: pd.Series) -> pd.Series:
    """Lowercase, normalize non-alphanumerics to single spaces, then trim."""
    x = s.astype("string").str.lower()
    x = x.str.replace(r"[^a-z0-9]+", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    return x

def col(df, *candidates):
    """Return the first existing column among candidates (case/space-insensitive)."""
    norm_map = {c.lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "_")
        if key in norm_map: return norm_map[key]
    return None

def map_with_assert(s: pd.Series, mapping: dict, name=None) -> pd.Series:
    """Map string tokens using a standard vocabulary; optionally fail fast on unknowns."""
    x = norm_token(s)
    # lightweight synonym expansion; extend if needed
    mapping_ext = {
        # smoking
        "non smoker":"non-smoker","nonsmoker":"non-smoker","ex smoker":"occasional smoker",
        "occasional":"occasional smoker","moderate":"regular smoker","regular":"regular smoker",
        # alcohol
        "non drinker":"non-drinker","no alcohol":"non-drinker","teetotaler":"non-drinker",
        "social":"social drinker","socially":"social drinker","moderate drinker":"regular drinker",
        # diet
        "poor":"unhealthy","good":"healthy",
        # severity
        "mild":"low","moderate":"medium","severe":"high",
    }
    x = x.replace(mapping_ext)
    std_keys = set(mapping.keys())
    bad_vals = sorted(x[~x.isin(std_keys) & x.notna()].unique().tolist())
    if bad_vals and STRICT:
        raise ValueError(f"[{name or s.name}] unmapped values: {bad_vals[:10]} ...")
    elif bad_vals:
        print(f"[WARN] [{name or s.name}] unmapped (set NA): {bad_vals[:10]} ...")
    return x.where(x.isin(std_keys), other=pd.NA).map(mapping)

YES = {"yes","y","true","t","1","ever","present","positive"}
NO  = {"no","n","false","f","0","never","absent","negative"}
def to_binary(s: pd.Series) -> pd.Series:
    """Normalize yes/no-like tokens into {1,0}, keep missing as NA."""
    x = norm_lower(s)
    y = pd.Series(pd.NA, index=s.index, dtype="Int8")
    return y.mask(x.isin(YES), 1).mask(x.isin(NO), 0)

def label_encode_with_unknown(s: pd.Series, unknown_token="Unknown"):
    """Label-encode while preserving missing as 'Unknown' (human-readable)."""
    s2 = s.replace(r'^\s*$', pd.NA, regex=True).fillna(unknown_token).astype("string")
    cats = pd.Categorical(s2)
    codes = pd.Series(cats.codes, index=s.index)
    mapping = {str(cat): int(i) for i, cat in enumerate(cats.categories)}
    return codes.astype("Int16"), mapping

def drop_constant_lowvar(df: pd.DataFrame, min_unique=2):
    """Drop columns having fewer than min_unique distinct values (including NA)."""
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) < min_unique]
    return df.drop(columns=const_cols), const_cols

def ensure_numeric(df: pd.DataFrame, cols):
    """Coerce selected columns to numeric (invalid -> NaN)."""
    for c in cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

def k_anonymity_violations(df: pd.DataFrame, quasi_cols, k=5):
    """Return groups with frequency < k (k-anonymity check for public views)."""
    g = df.groupby(quasi_cols, dropna=False).size()
    return g[g < k].sort_values()

def rr3_series(counts: pd.Series, seed=SEED):
    """Random rounding to base 3 (RR3) applied to a vector of counts."""
    rng = np.random.default_rng(seed); base = 3; out = []
    for n in counts.astype(int).tolist():
        if n <= 0: out.append(0); continue
        lo = (n // base) * base; hi = lo + base
        if n == lo: out.append(n); continue
        p_hi = (n - lo) / base
        out.append(int(rng.choice([lo, hi], p=[1-p_hi, p_hi])))
    return pd.Series(out, index=counts.index, dtype=int)

def merge_low_freq(s: pd.Series, min_count=LOW_FREQ, other_label="Other"):
    """Merge categories with frequency < min_count into 'Other' (for public release)."""
    vc = s.value_counts(dropna=False)
    rare = set(vc[vc < min_count].index.tolist())
    return s.where(~s.isin(rare), other_label)

# Continuous ranges (exclude Stress_Level; it’s ordinal)
RANGE = {
    "Age": (10, 100),
    "Sleep_Hours": (0, 16),
    "Work_Hours": (0, 100),
    "Physical_Activity_Hours": (0, 20),
    "Social_Media_Usage": (0, 24),
}
def clip_to_nan(df: pd.DataFrame, bounds: dict):
    """Set out-of-range numeric values to NA."""
    for col,(lo,hi) in bounds.items():
        if col in df.columns:
            bad = df[col].lt(lo) | df[col].gt(hi)
            print(f"[INFO] {col}: out-of-range {bad.sum()} / {bad.size}")
            df.loc[bad, col] = pd.NA

# =========================
# Main
# =========================
def main(args):
    in_csv  = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve()
    pub_dir = Path(args.pub_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pub_dir.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise FileNotFoundError(
            f"CSV not found: {in_csv}\n"
            f"Hint: place the raw file at {REPO_ROOT / 'data' / 'raw'} or pass --input."
        )

    t0 = time.time()
    df = read_csv_robust(in_csv)

    # Treat visual blanks and common placeholders as missing
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    df = df.replace(r'(?i)^(na|n/a|null|none|-|--)$', pd.NA, regex=True)

    # Light column cleanup (preserve original names, just trim extra spaces)
    df.columns = [c.strip().replace("  "," ").strip() for c in df.columns]
    n0 = len(df)

    # Coerce obvious numeric fields
    numeric_candidates = ["Age","Sleep_Hours","Work_Hours","Physical_Activity_Hours","Social_Media_Usage"]
    ensure_numeric(df, [col(df, x, x.replace("_"," ")) for x in numeric_candidates if col(df, x, x.replace("_"," "))])

    # ---- Normalize key text fields
    gcol = col(df, "Gender", "Sex")
    if gcol:
        g = norm_lower(df[gcol])
        gender_map = {
            "male":"Male","m":"Male","man":"Male",
            "female":"Female","f":"Female","woman":"Female",
            "non-binary":"Other","nonbinary":"Other","nb":"Other","other":"Other",
            "prefer not to say":"Unknown","na":"Unknown","n/a":"Unknown","unknown":"Unknown","":"Unknown"
        }
        df["Gender_norm"] = g.map(gender_map).fillna("Unknown")

    rscol = col(df, "Relationship_Status","Relationship Status","Marital_Status","Marital Status","MaritalStatus")
    if rscol:
        rs = norm_lower(df[rscol])
        rs_map = {
            "single":"single","married":"married","divorced":"divorced","widowed":"widowed",
            "in a relationship":"in_relationship","committed relationship":"in_relationship",
            "prefer not to say":"unknown","unknown":"unknown","":"unknown"
        }
        df["Relationship_Status_norm"] = rs.map(rs_map).fillna("unknown")

    # ---- Binary fields (if present)
    for base in ["Consultation_History","Medication_Usage"]:
        cname = col(df, base, base.replace("_"," "))
        if cname: df[f"{base}_bin"] = to_binary(df[cname])

    # ---- Ordinal mappings (smoking/alcohol/diet/severity)
    smkcol = col(df, "Smoking_Habit","Smoking Habit","Smoking","Smoking Status")
    if smkcol:
        smk_map = {"non-smoker":0,"occasional smoker":1,"regular smoker":2,"heavy smoker":3}
        df["Smoking_Habit_ord"] = map_with_assert(df[smkcol], smk_map, name="Smoking_Habit").astype("Int8")
        df["Smoking_Habit_bin"] = (df["Smoking_Habit_ord"] >= 2).astype("Int8")

    alccol = col(df, "Alcohol_Consumption","Alcohol Consumption","Alcohol","Alcohol Use","Alcohol Status")
    if alccol:
        alc_map = {"non-drinker":0,"social drinker":1,"regular drinker":2,"heavy drinker":3}
        df["Alcohol_Consumption_ord"] = map_with_assert(df[alccol], alc_map, name="Alcohol_Consumption").astype("Int8")
        df["Alcohol_Consumption_bin"] = (df["Alcohol_Consumption_ord"] >= 2).astype("Int8")

    dietcol = col(df, "Diet_Quality","Diet Quality","Diet")
    if dietcol:
        diet_map = {"unhealthy":0,"average":1,"healthy":2}
        df["Diet_Quality_ord"] = map_with_assert(df[dietcol], diet_map, name="Diet_Quality").astype("Int8")

    sevcol = col(df, "Severity","Condition_Severity","Severity Level")
    if sevcol:
        sev = norm_token(df[sevcol]).map({"low":0,"medium":1,"high":2})
        if STRICT and sev.isna().any() and df[sevcol].notna().any():
            bad = sorted(norm_token(df[sevcol])[sev.isna() & df[sevcol].notna()].unique().tolist())
            raise ValueError(f"[Severity] unmapped: {bad[:10]} ...")
        df["Severity_ord"] = sev.astype("Int8")

    # ---- NEW: Stress_Level as ordinal (Low/Medium/High -> 0/1/2)
    scol = col(df, "Stress_Level","Stress Level")
    if scol:
        s = norm_token(df[scol]).map({"low":0,"medium":1,"high":2})
        df["Stress_Level_ord"] = s.astype("Int8")

    # ---- Clip continuous variables (exclude Stress_Level)
    bounds = {}
    for k,(lo,hi) in RANGE.items():
        cname = col(df, k, k.replace("_"," "))
        if cname:
            df[cname] = pd.to_numeric(df[cname], errors="coerce")
            bounds[cname] = (lo,hi)
    clip_to_nan(df, bounds)

    # ---- Drop duplicates
    if col(df, "ID"): df = df.drop_duplicates(subset=[col(df,"ID")])
    else: df = df.drop_duplicates()

    # ---- clean_full: analysis-friendly (keep readable text + derived cols)
    clean_full = df.copy()

    # ---- Label-encode nominal fields and store mappings
    label_maps = {}
    occ  = col(df, "Occupation","Job","Employment")
    ctry = col(df, "Country","Location","Region","Nationality")
    mhc  = col(df, "Mental_Health_Condition","Mental Health Condition","Condition","MHC","MentalHealth")
    for col_name, cname in {
        "Gender_norm": col(df, "Gender_norm"),
        "Occupation": occ,
        "Country": ctry,
        "Relationship_Status_norm": col(df, "Relationship_Status_norm"),
        "Mental_Health_Condition": mhc,
    }.items():
        if cname:
            codes, mapping = label_encode_with_unknown(df[cname], unknown_token="Unknown")
            clean_full[f"{col_name}_lbl"] = codes
            label_maps[col_name] = mapping

    # ---- clean_numeric: modeling-ready (numeric/ordinal/binary/label-encoded)
    keep_suffix = ("_ord","_bin","_lbl")
    num_cols = []
    for c in clean_full.columns:
        if pd.api.types.is_numeric_dtype(clean_full[c]) or any(c.endswith(suf) for suf in keep_suffix):
            num_cols.append(c)
    clean_numeric = clean_full[num_cols].copy()

    # Drop constants/low-variance columns
    clean_numeric, dropped_const = drop_constant_lowvar(clean_numeric, min_unique=2)
    print("[INFO] Dropped constant/low-var:", dropped_const)

    # Simple imputation: integers/bools -> mode; others -> median
    for c in clean_numeric.columns:
        if pd.api.types.is_integer_dtype(clean_numeric[c]) or pd.api.types.is_bool_dtype(clean_numeric[c]):
            mode = clean_numeric[c].mode(dropna=True)
            if len(mode): clean_numeric[c] = clean_numeric[c].fillna(mode.iloc[0])
        else:
            try:
                clean_numeric[c] = clean_numeric[c].fillna(clean_numeric[c].median())
            except Exception:
                mode = clean_numeric[c].mode(dropna=True)
                if len(mode): clean_numeric[c] = clean_numeric[c].fillna(mode.iloc[0])

    # ===== Privacy-safe public outputs: low-frequency merge + 20y AgeBand + RR3 =====
    cf_pub = clean_full.copy()
    agec = col(cf_pub, "Age")
    if agec:
        cf_pub["AgeBand"] = pd.cut(pd.to_numeric(cf_pub[agec], errors="coerce"),
                                   bins=list(range(10, 110, 20)), right=False)
    if col(cf_pub, "Occupation"):
        cf_pub["Occupation_pub"] = merge_low_freq(cf_pub[col(cf_pub, "Occupation")], min_count=LOW_FREQ)
    if col(cf_pub, "Country"):
        cf_pub["Country_pub"] = merge_low_freq(cf_pub[col(cf_pub, "Country")], min_count=LOW_FREQ)

    qcols = [c for c in ["Gender_norm","Country_pub","Occupation_pub","AgeBand"] if c in cf_pub.columns]
    viol_txt = ""
    if qcols:
        viol = k_anonymity_violations(cf_pub, qcols, k=K_MIN)
        viol_txt = f"k-anonymity <{K_MIN} groups (public view): {len(viol)}"
        print("[WARN]" if len(viol)>0 else "[INFO]", viol_txt)

    # Example public table with RR3: Gender × Mental_Health_Condition
    if {"Gender_norm","Mental_Health_Condition"}.issubset(cf_pub.columns):
        tab = cf_pub.pivot_table(index="Gender_norm", columns="Mental_Health_Condition",
                                 aggfunc="size", fill_value=0)
        tab_supp = tab.mask(tab < 3, 0)           # suppress small cells first
        tab_rr3  = tab_supp.apply(rr3_series, axis=0)
        (pub_dir / "mh_by_gender_rr3.csv").parent.mkdir(parents=True, exist_ok=True)
        tab_rr3.to_csv(pub_dir / "mh_by_gender_rr3.csv", index=True)
        print(f"[INFO] Published RR3 -> {pub_dir / 'mh_by_gender_rr3.csv'}")

    # ===== Write artifacts =====
    full_path = out_dir / "clean_full.csv"
    num_path  = out_dir / "clean_numeric.csv"
    map_path  = out_dir / "label_mappings.json"
    summ_path = out_dir / "cleaning_summary.txt"

    clean_full.to_csv(full_path, index=False)
    clean_numeric.to_csv(num_path, index=False)
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(label_maps, f, ensure_ascii=False, indent=2)

    # Summary (kept as a simple txt for auditability)
    lines = []
    lines.append(f"Input rows: {n0}, Output rows: {len(clean_full)}")
    lines.append(f"Columns (full→numeric): {clean_full.shape[1]} → {clean_numeric.shape[1]}")
    lines.append(f"Dropped constant/low-var cols: {dropped_const}")
    if "Severity_ord" in clean_full.columns:
        sev_na = clean_full["Severity_ord"].isna().mean()
        lines.append(f"Severity_ord NA in full: {sev_na:.3f} (kept NA in analysis; imputed in numeric)")
    if "Stress_Level_ord" in clean_full.columns:
        sl_na = clean_full["Stress_Level_ord"].isna().mean()
        lines.append(f"Added Stress_Level_ord (0/1/2); NA rate: {sl_na:.3f}")
    if qcols: lines.append(viol_txt + f" | LOW_FREQ merge threshold={LOW_FREQ}")
    lines.append("Privacy-safe outputs (RR3) written to: " + str(pub_dir))
    with open(summ_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("Artifacts:")
    print(" -", full_path)
    print(" -", num_path)
    print(" -", map_path)
    print(" -", summ_path)
    print(f"Done in {time.time()-t0:.2f}s.")

# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Team-style data cleaning (relative paths, privacy-safe outputs)")
    parser.add_argument("--input",  default=str(DEFAULT_IN),  help="path to raw CSV (relative or absolute)")
    parser.add_argument("--out_dir",default=str(DEFAULT_OUT), help="output dir for processed artifacts")
    parser.add_argument("--pub_dir",default=str(DEFAULT_PUB), help="output dir for privacy-safe public tables")
    args = parser.parse_args()
    main(args)
