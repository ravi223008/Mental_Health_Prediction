# Data Pre (lean).py — cleaning + privacy-safe tables + stratified 70/15/15 + 5-fold (train only)
# + gentle fairness guardrails (model view / warnings / optional weights)
# + SMOTE prep artifacts (diagnostics & config; NO oversampling here)
# -*- coding: utf-8 -*-
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------
# Project paths (relative)
# -----------------------
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_IN  = REPO_ROOT / "data" / "mental_health_data final data.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "processed"
DEFAULT_PUB = REPO_ROOT / "reports" / "tables_public"

# -----------------------
# Config knobs
# -----------------------
SEED     = 760
STRICT   = False          # if True, unknown tokens raise; else -> NA
LOW_FREQ = 20             # public view rare-category merge threshold
K_MIN    = 5              # k-anonymity threshold for public view
SENSITIVE_GROUP_COL = "Gender_norm"
MAKE_MODEL_VIEW   = True
ADD_MISSING_FLAGS = True
GROUP_MIN_PER_SPLIT = 50
MAKE_WEIGHTS = True

# Ranges for continuous fields (Stress_Level is ordinal -> excluded)
RANGE = {
    "Age": (10, 100),
    "Sleep_Hours": (0, 16),
    "Work_Hours": (0, 100),
    "Physical_Activity_Hours": (0, 20),
    "Social_Media_Usage": (0, 24),
}

# -----------------------
# Small utilities
# -----------------------
def read_csv_robust(path: Path) -> pd.DataFrame:
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
    return s.astype("string").str.strip().str.lower()

def norm_token(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.lower()
    return (x.str.replace(r"[^a-z0-9]+", " ", regex=True)
              .str.replace(r"\s+", " ", regex=True)
              .str.strip())

def find_col(df: pd.DataFrame, *names: str) -> str | None:
    norm_map = {c.lower().replace(" ", "_"): c for c in df.columns}
    for n in names:
        key = n.lower().replace(" ", "_")
        if key in norm_map: return norm_map[key]
    return None

def ensure_numeric(df: pd.DataFrame, cols: list[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def to_binary(s: pd.Series) -> pd.Series:
    YES = {"yes","y","true","t","1","ever","present","positive"}
    NO  = {"no","n","false","f","0","never","absent","negative"}
    x = norm_lower(s)
    y = pd.Series(pd.NA, index=s.index, dtype="Int8")
    return y.mask(x.isin(YES), 1).mask(x.isin(NO), 0)

def label_encode_with_unknown(s: pd.Series, unknown="Unknown"):
    s2 = s.replace(r"^\s*$", pd.NA, regex=True).fillna(unknown).astype("string")
    cats = pd.Categorical(s2)
    codes = pd.Series(cats.codes, index=s.index).astype("Int16")
    mapping = {str(cat): int(i) for i, cat in enumerate(cats.categories)}
    return codes, mapping

def drop_constant_lowvar(df: pd.DataFrame, min_unique=2):
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) < min_unique]
    return df.drop(columns=const_cols), const_cols

def clip_to_nan(df: pd.DataFrame, bounds: dict[str, tuple[float,float]]):
    for col,(lo,hi) in bounds.items():
        if col in df.columns:
            bad = df[col].lt(lo) | df[col].gt(hi)
            if bad.any():
                print(f"[INFO] clip -> {col}: {bad.sum()} out-of-range set to NA")
            df.loc[bad, col] = pd.NA

def k_anonymity_violations(df: pd.DataFrame, quasi_cols, k=5):
    g = df.groupby(quasi_cols, dropna=False).size()
    return g[g < k].sort_values()

def rr3_series(counts: pd.Series, seed=SEED):
    rng = np.random.default_rng(seed); base = 3; out = []
    for n in counts.astype(int):
        if n <= 0: out.append(0); continue
        lo = (n // base)*base; hi = lo + base
        if n == lo: out.append(n); continue
        p_hi = (n - lo) / base
        out.append(int(rng.choice([lo, hi], p=[1-p_hi, p_hi])))
    return pd.Series(out, index=counts.index, dtype=int)

def merge_low_freq(s: pd.Series, min_count=LOW_FREQ, other_label="Other"):
    vc = s.value_counts(dropna=False)
    rare = set(vc[vc < min_count].index.tolist())
    return s.where(~s.isin(rare), other_label)

def map_tokens(s: pd.Series, vocab: dict, name=""):
    x = norm_token(s)
    # minimal synonym expansion
    syn = {
        "non smoker":"non-smoker","nonsmoker":"non-smoker","ex smoker":"occasional smoker",
        "occasional":"occasional smoker","moderate":"regular smoker","regular":"regular smoker",
        "non drinker":"non-drinker","no alcohol":"non-drinker","teetotaler":"non-drinker",
        "social":"social drinker","socially":"social drinker","moderate drinker":"regular drinker",
        "poor":"unhealthy","good":"healthy","mild":"low","severe":"high",
    }
    x = x.replace(syn)
    bad = sorted(x[~x.isin(vocab) & x.notna()].unique().tolist())
    if bad and STRICT: raise ValueError(f"[{name}] unmapped: {bad[:10]} ...")
    if bad: print(f"[WARN] [{name}] unmapped->NA: {bad[:10]} ...")
    return x.where(x.isin(vocab), pd.NA).map(vocab)

# -----------------------
# Splits & CV (no sklearn)
# -----------------------
def target_from_full(clean_full: pd.DataFrame, label_maps: dict|None=None) -> np.ndarray|None:
    tcol = find_col(clean_full, "Mental_Health_Condition","MHC","Condition")
    if tcol:
        return (clean_full[tcol].astype("string").str.strip().str.lower()=="yes").astype(int).to_numpy()
    lbl = find_col(clean_full, "Mental_Health_Condition_lbl")
    if lbl and label_maps and "Mental_Health_Condition" in label_maps:
        inv = {int(v): str(k).lower() for k,v in label_maps["Mental_Health_Condition"].items()}
        arr = clean_full[lbl].to_numpy()
        return np.array([1 if inv.get(int(v),"")=="yes" else 0 for v in arr], dtype=int)
    return None

def stratified_70_15_15(y: np.ndarray, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y); tr = np.zeros(n, bool); va = np.zeros(n, bool); te = np.zeros(n, bool)
    for cls in [0,1]:
        idx = np.where(y==cls)[0]; rng.shuffle(idx); m = len(idx)
        ntr = int(np.floor(0.70*m)); nva = int(np.floor(0.15*m))
        tr[idx[:ntr]] = True; va[idx[ntr:ntr+nva]] = True; te[idx[ntr+nva:]] = True
    rest = ~tr & ~va & ~te
    te[rest] = True
    return tr, va, te

def stratified_kfold_on_train(y: np.ndarray, train_mask: np.ndarray, k=5, seed=SEED):
    rng = np.random.default_rng(seed)
    cv = np.full(len(y), -1, int)
    for cls in [0,1]:
        idx = np.where((y==cls) & train_mask)[0]; rng.shuffle(idx)
        for fid, chunk in enumerate(np.array_split(idx, k)):
            cv[chunk] = fid
    return cv

# -----------------------
# Blocks (privacy, fairness, smote prep)
# -----------------------
def make_privacy_tables(clean_full: pd.DataFrame, pub_dir: Path):
    pub = clean_full.copy()
    agec = find_col(pub, "Age")
    if agec:
        pub["AgeBand"] = pd.cut(pd.to_numeric(pub[agec], errors="coerce"),
                                bins=list(range(10,110,20)), right=False)
    if find_col(pub, "Occupation"):
        pub["Occupation_pub"] = merge_low_freq(pub[find_col(pub,"Occupation")])
    if find_col(pub, "Country"):
        pub["Country_pub"] = merge_low_freq(pub[find_col(pub,"Country")])
    qcols = [c for c in ["Gender_norm","Country_pub","Occupation_pub","AgeBand"] if c in pub.columns]
    viol_txt = ""
    if qcols:
        viol = k_anonymity_violations(pub, qcols, k=K_MIN)
        viol_txt = f"k-anonymity <{K_MIN}: {len(viol)} groups"
        print(("[WARN] " if len(viol)>0 else "[INFO] ") + viol_txt)
    if {"Gender_norm","Mental_Health_Condition"}.issubset(pub.columns):
        tab = pub.pivot_table(index="Gender_norm", columns="Mental_Health_Condition", aggfunc="size", fill_value=0)
        tab = tab.mask(tab < 3, 0)  # small cell suppression
        tab_rr3 = tab.apply(rr3_series, axis=0)
        pub_dir.mkdir(parents=True, exist_ok=True)
        (pub_dir / "mh_by_gender_rr3.csv").write_text(tab_rr3.to_csv(index=True))
        print(f"[INFO] RR3 table -> {pub_dir/'mh_by_gender_rr3.csv'}")
    return viol_txt

def make_model_view_and_weights(clean_full: pd.DataFrame,
                                clean_numeric: pd.DataFrame,
                                out_dir: Path,
                                label_maps: dict,
                                splits_path: Path,
                                lines_accum: list[str]):
    fair_lines = []

    # 1) model view without direct sensitive columns
    if MAKE_MODEL_VIEW:
        drop = []
        if SENSITIVE_GROUP_COL in clean_numeric.columns: drop.append(SENSITIVE_GROUP_COL)
        lbl = f"{SENSITIVE_GROUP_COL}_lbl"
        if lbl in clean_numeric.columns: drop.append(lbl)
        keep_cols = [c for c in clean_numeric.columns if c not in drop]
        model_view = clean_numeric[keep_cols].copy()
        if ADD_MISSING_FLAGS:
            miss_cols = [c for c in ["Age","Sleep_Hours","Work_Hours","Physical_Activity_Hours","Social_Media_Usage"]
                         if c in clean_full.columns]
            for c in miss_cols:
                model_view[f"{c}_isna"] = clean_full[c].isna().astype("Int8")
        model_path = out_dir / "clean_numeric_model.csv"
        model_view.to_csv(model_path, index=False)
        fair_lines.append("[FAIR] Model view: clean_numeric_model.csv (no direct sensitive columns)")

    # 2) group representation / prevalence in splits
    y_arr = target_from_full(clean_full, label_maps)
    if splits_path.exists() and y_arr is not None and SENSITIVE_GROUP_COL in clean_full.columns:
        sp = pd.read_csv(splits_path)
        for split_name in ["train","val","test"]:
            idx = sp.index[sp["split"]==split_name]
            sub = clean_full.loc[idx]
            if len(sub)==0:
                fair_lines.append(f"[FAIR] {split_name}: 0 rows"); continue
            grp = sub.groupby(SENSITIVE_GROUP_COL)["Mental_Health_Condition"].agg(["count"])
            prev = pd.Series(y_arr[idx], index=sub.index)
            rep = sub.assign(y=prev.values).groupby(SENSITIVE_GROUP_COL)["y"].mean().rename("pos_rate")
            rep = pd.concat([grp, rep], axis=1).sort_values("count", ascending=False)
            if split_name in ("val","test"):
                small = rep[rep["count"] < GROUP_MIN_PER_SPLIT]
                if len(small):
                    fair_lines.append(f"[FAIR][WARN] {split_name} small groups <{GROUP_MIN_PER_SPLIT}: " +
                                      ", ".join([f"{g}:{int(n)}" for g,n in small["count"].items()]))
            fair_lines.append(f"[FAIR] {split_name} by {SENSITIVE_GROUP_COL}: " +
                              ", ".join([f"{g}=n{int(n)}/pos{p:.2f}" for g,(n,p) in zip(rep.index, zip(rep['count'], rep['pos_rate']))]))

    # 3) missingness gaps by group
    if SENSITIVE_GROUP_COL in clean_full.columns:
        miss_cols = [c for c in ["Age","Sleep_Hours","Work_Hours","Physical_Activity_Hours","Social_Media_Usage"]
                     if c in clean_full.columns]
        gaps = []
        for c in miss_cols:
            rates = clean_full.groupby(SENSITIVE_GROUP_COL)[c].apply(lambda s: s.isna().mean())
            if len(rates) >= 2: gaps.append((c, float(rates.max()-rates.min())))
        if gaps:
            gaps.sort(key=lambda x: x[1], reverse=True)
            fair_lines.append("[FAIR] Missingness gap (top): " + ", ".join([f"{c}:{g:.2f}" for c,g in gaps[:5]]))

    # 4) optional sample weights (inverse label & group frequency)
    if MAKE_WEIGHTS and y_arr is not None:
        weights = pd.DataFrame({"row_id": np.arange(len(clean_full))})
        p1 = float(np.mean(y_arr)); p0 = 1.0 - p1
        w_label = np.where(y_arr==1, 1.0/max(p1,1e-6), 1.0/max(p0,1e-6))
        weights["w_label"] = w_label / np.mean(w_label)
        if SENSITIVE_GROUP_COL in clean_full.columns:
            g = clean_full[SENSITIVE_GROUP_COL].astype("string").fillna("Unknown")
            sizes = g.value_counts()
            w_group = g.map(lambda z: 1.0/max(float(sizes.get(z, len(clean_full))),1.0))
            weights["w_group"] = (w_group / w_group.mean()).values
            weights["w_combo"] = np.sqrt(weights["w_label"] * weights["w_group"])
        else:
            weights["w_group"] = 1.0; weights["w_combo"] = weights["w_label"]
        weights.to_csv(out_dir / "sample_weights.csv", index=False)
        fair_lines.append("[FAIR] Optional weights -> sample_weights.csv (w_label, w_group, w_combo)")

    lines_accum += fair_lines

def make_smote_prep(clean_full: pd.DataFrame,
                    clean_numeric: pd.DataFrame,
                    out_dir: Path,
                    label_maps: dict,
                    splits_path: Path,
                    lines_accum: list[str]):
    """
    Write two files for modeling stage:
      - smote_config.json: feature indices for SMOTENC + recommended knobs (do NOT oversample here)
      - imbalance_report.txt: human-readable summary
    """
    try:
        feat_path = out_dir / "clean_numeric_model.csv"
        if not feat_path.exists(): feat_path = out_dir / "clean_numeric.csv"
        cols_probe = pd.read_csv(feat_path, nrows=1).columns.tolist()

        cat_idx = [i for i,c in enumerate(cols_probe) if c.endswith("_lbl") or c.endswith("_bin")]
        num_idx = [i for i in range(len(cols_probe)) if i not in cat_idx]

        if not splits_path.exists():
            lines_accum.append("SMOTE prep: skipped (no splits file)")
            return

        y_vec = target_from_full(clean_full, label_maps)
        if y_vec is None:
            lines_accum.append("SMOTE prep: cannot infer target"); return
        sp = pd.read_csv(splits_path)
        train_rows = sp.index[sp["split"]=="train"].to_numpy()
        y_train = y_vec[train_rows]
        n_pos = int((y_train==1).sum()); n_neg = int((y_train==0).sum())
        minority = min(n_pos, n_neg); majority = max(n_pos, n_neg)
        ratio = float(minority/majority) if majority>0 else 0.0

        rec_k = max(1, min(5, minority-1))
        rec_strategy = 1.0 if ratio < 0.4 else "auto"
        rec_sampler = "SMOTENC" if len(cat_idx)>0 else "SMOTE"

        fold_ids = sp.loc[sp["split"]=="train", "cv_fold"].values
        fold_counts = []
        for k in range(int(sp["cv_fold"].max()+1)):
            mk = (fold_ids==k); yk = y_train[mk]
            fold_counts.append({"fold":int(k),"n_pos":int((yk==1).sum()),"n_neg":int((yk==0).sum())})

        cfg = {
            "features_path": str(feat_path),
            "feature_columns": cols_probe,
            "categorical_indices": cat_idx,
            "numeric_indices": num_idx,
            "target_name": "Mental_Health_Condition",
            "train_stats": {"n_pos": n_pos, "n_neg": n_neg, "minority_ratio": round(ratio,4)},
            "cv_folds": {"k": int(len(fold_counts)), "train_fold_counts": fold_counts},
            "recommended": {"sampler": rec_sampler, "sampling_strategy": rec_strategy,
                            "k_neighbors": int(rec_k), "random_state": int(SEED)},
            "notes": "Apply SMOTENC inside each training fold only; never touch val/test."
        }
        (out_dir / "smote_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = []
        lines.append(f"Feature table: {Path(cfg['features_path']).name}")
        lines.append(f"Categorical idx ({len(cat_idx)}): {cat_idx[:20]}{' ...' if len(cat_idx)>20 else ''}")
        lines.append(f"Numeric idx     ({len(num_idx)}): {num_idx[:20]}{' ...' if len(num_idx)>20 else ''}")
        lines.append(f"Train pos/neg: {n_pos}/{n_neg} (minority_ratio={ratio:.3f})")
        lines.append(f"Recommended: sampler={rec_sampler}, strategy={rec_strategy}, k_neighbors={rec_k}, random_state={SEED}")
        lines.append("Train folds: " + "; ".join([f"fold={d['fold']}, pos={d['n_pos']}, neg={d['n_neg']}" for d in fold_counts]))
        (out_dir / "imbalance_report.txt").write_text("\n".join(lines), encoding="utf-8")

        lines_accum.append(f"SMOTE prep: minority_ratio(train)={ratio:.3f}, cfg=smote_config.json, rpt=imbalance_report.txt")
    except Exception as e:
        lines_accum.append(f"SMOTE prep: error -> {e}")

# -----------------------
# Main pipeline
# -----------------------
def run(args):
    in_csv  = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    pub_dir = Path(args.pub_dir).resolve(); pub_dir.mkdir(parents=True, exist_ok=True)
    if not in_csv.exists():
        raise FileNotFoundError(f"CSV not found: {in_csv}\nHint: place raw file under {REPO_ROOT/'data'} or use --input")

    t0 = time.time()
    df = read_csv_robust(in_csv)
    df = df.replace(r"^\s*$", pd.NA, regex=True) \
           .replace(r"(?i)^(na|n/a|null|none|-|--)$", pd.NA, regex=True)
    df.columns = [c.strip().replace("  "," ").strip() for c in df.columns]
    n0 = len(df)

    # numeric coercion
    num_candidates = ["Age","Sleep_Hours","Work_Hours","Physical_Activity_Hours","Social_Media_Usage"]
    ensure_numeric(df, [find_col(df, x, x.replace("_"," ")) for x in num_candidates if find_col(df, x, x.replace("_"," "))])

    # normalize key text fields
    gcol = find_col(df, "Gender","Sex")
    if gcol:
        g = norm_lower(df[gcol])
        gender_map = {
            "male":"Male","m":"Male","man":"Male",
            "female":"Female","f":"Female","woman":"Female",
            "non-binary":"Other","nonbinary":"Other","nb":"Other","other":"Other",
            "prefer not to say":"Unknown","na":"Unknown","n/a":"Unknown","unknown":"Unknown","":"Unknown"
        }
        df["Gender_norm"] = g.map(gender_map).fillna("Unknown")

    rcol = find_col(df, "Relationship_Status","Relationship Status","Marital_Status","Marital Status","MaritalStatus")
    if rcol:
        rs = norm_lower(df[rcol])
        rs_map = {"single":"single","married":"married","divorced":"divorced","widowed":"widowed",
                  "in a relationship":"in_relationship","committed relationship":"in_relationship",
                  "prefer not to say":"unknown","unknown":"unknown","":"unknown"}
        df["Relationship_Status_norm"] = rs.map(rs_map).fillna("unknown")

    for base in ["Consultation_History","Medication_Usage"]:
        cname = find_col(df, base, base.replace("_"," "))
        if cname: df[f"{base}_bin"] = to_binary(df[cname])

    # ordinal mappings (compact)
    conf = [
        (find_col(df,"Smoking_Habit","Smoking Habit","Smoking","Smoking Status"),
         {"non-smoker":0,"occasional smoker":1,"regular smoker":2,"heavy smoker":3}, "Smoking_Habit"),
        (find_col(df,"Alcohol_Consumption","Alcohol Consumption","Alcohol","Alcohol Use","Alcohol Status"),
         {"non-drinker":0,"social drinker":1,"regular drinker":2,"heavy drinker":3}, "Alcohol_Consumption"),
        (find_col(df,"Diet_Quality","Diet Quality","Diet"),
         {"unhealthy":0,"average":1,"healthy":2}, "Diet_Quality"),
        (find_col(df,"Severity","Condition_Severity","Severity Level"),
         {"low":0,"medium":1,"high":2}, "Severity"),
        (find_col(df,"Stress_Level","Stress Level"),
         {"low":0,"medium":1,"high":2}, "Stress_Level"),
    ]
    for cname, vocab, name in conf:
        if cname:
            df[f"{name}_ord"] = map_tokens(df[cname], vocab, name=name).astype("Int8")
            if name in ("Smoking_Habit","Alcohol_Consumption"):
                df[f"{name}_bin"] = (df[f"{name}_ord"] >= 2).astype("Int8")

    # clip ranges
    bounds = {}
    for k,(lo,hi) in RANGE.items():
        cname = find_col(df, k, k.replace("_"," "))
        if cname:
            df[cname] = pd.to_numeric(df[cname], errors="coerce"); bounds[cname] = (lo,hi)
    clip_to_nan(df, bounds)

    # deduplicate
    idc = find_col(df, "ID")
    df = df.drop_duplicates(subset=[idc]) if idc else df.drop_duplicates()

    # full view (analysis)
    clean_full = df.copy()

    # label-encode for nominal fields
    label_maps = {}
    for (name, cname) in {
        "Gender_norm": find_col(df,"Gender_norm"),
        "Occupation":  find_col(df,"Occupation","Job","Employment"),
        "Country":     find_col(df,"Country","Location","Region","Nationality"),
        "Relationship_Status_norm": find_col(df,"Relationship_Status_norm"),
        "Mental_Health_Condition":  find_col(df,"Mental_Health_Condition","Mental Health Condition","Condition","MHC","MentalHealth"),
    }.items():
        if cname:
            codes, mapping = label_encode_with_unknown(df[cname])
            clean_full[f"{name}_lbl"] = codes; label_maps[name] = mapping

    # numeric view (modeling-ready)
    keep_suffix = ("_ord","_bin","_lbl")
    num_cols = [c for c in clean_full.columns
                if pd.api.types.is_numeric_dtype(clean_full[c]) or any(c.endswith(s) for s in keep_suffix)]
    clean_numeric = clean_full[num_cols].copy()

    clean_numeric, dropped = drop_constant_lowvar(clean_numeric, 2)
    for c in clean_numeric.columns:
        if pd.api.types.is_integer_dtype(clean_numeric[c]) or pd.api.types.is_bool_dtype(clean_numeric[c]):
            m = clean_numeric[c].mode(dropna=True)
            if len(m): clean_numeric[c] = clean_numeric[c].fillna(m.iloc[0])
        else:
            try:    clean_numeric[c] = clean_numeric[c].fillna(clean_numeric[c].median())
            except: clean_numeric[c] = clean_numeric[c].fillna(clean_numeric[c].mode(dropna=True).iloc[0])

    # privacy-safe public tables
    viol_txt = make_privacy_tables(clean_full, pub_dir)

    # write base artifacts
    out_full = out_dir / "clean_full.csv"
    out_num  = out_dir / "clean_numeric.csv"
    out_map  = out_dir / "label_mappings.json"
    clean_full.to_csv(out_full, index=False)
    clean_numeric.to_csv(out_num, index=False)
    out_map.write_text(json.dumps(label_maps, ensure_ascii=False, indent=2), encoding="utf-8")

    # splits + CV labels (optional)
    split_info = "Splits: skipped (--no_splits)"
    splits_path = out_dir / f"splits_70_15_15_k{args.kfolds}.csv"
    if not args.no_splits:
        y = target_from_full(clean_full, label_maps)
        if y is None: raise RuntimeError("Cannot locate Mental_Health_Condition (text or *_lbl via mapping).")
        tr, va, te = stratified_70_15_15(y, seed=SEED)
        cv_fold = stratified_kfold_on_train(y, tr, k=args.kfolds, seed=SEED)
        split = np.full(len(y), "unassigned", object)
        split[tr]="train"; split[va]="val"; split[te]="test"
        assert (split!="unassigned").all() and (cv_fold[split!="train"]==-1).all()
        pd.DataFrame({"row_id":np.arange(len(y),dtype=int),"split":split,"cv_fold":cv_fold}).to_csv(splits_path, index=False)
        counts = pd.Series(split).value_counts().to_dict()
        folds  = pd.Series(cv_fold[cv_fold>=0]).value_counts().sort_index().to_dict()
        split_info = f"Splits -> {splits_path.name} | counts={counts} | train folds={folds}"

    # fairness guardrails (model view / warnings / weights)
    lines = []
    make_model_view_and_weights(clean_full, clean_numeric, out_dir, label_maps, splits_path, lines)

    # SMOTE prep (config + imbalance report; NO oversampling here)
    make_smote_prep(clean_full, clean_numeric, out_dir, label_maps, splits_path, lines)

    # summary
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
        f"Privacy-safe tables: {DEFAULT_PUB}"
    ]
    summ.write_text("\n".join([r for r in rows if r]), encoding="utf-8")

    print("\n".join([r for r in rows if r]))
    print("Artifacts:")
    for p in [out_full, out_num, out_map, summ]:
        print(" -", p)
    if not args.no_splits: print(" -", splits_path)
    if (out_dir/"clean_numeric_model.csv").exists(): print(" -", out_dir/"clean_numeric_model.csv")
    if (out_dir/"sample_weights.csv").exists(): print(" -", out_dir/"sample_weights.csv")
    if (out_dir/"smote_config.json").exists(): print(" -", out_dir/"smote_config.json")
    if (out_dir/"imbalance_report.txt").exists(): print(" -", out_dir/"imbalance_report.txt")
    print(f"Done in {time.time()-t0:.2f}s.")

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Lean cleaning + privacy + splits + fairness guardrails + SMOTE prep")
    ap.add_argument("--input",   default=str(DEFAULT_IN),  help="path to raw CSV")
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT), help="output dir for processed artifacts")
    ap.add_argument("--pub_dir", default=str(DEFAULT_PUB), help="output dir for public (privacy-safe) tables")
    ap.add_argument("--no_splits", action="store_true", help="skip generating 70/15/15 + 5-fold labels")
    ap.add_argument("--kfolds", type=int, default=5,    help="CV folds on the training set")
    args = ap.parse_args()
    run(args)
