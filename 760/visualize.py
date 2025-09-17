# viz_key_plots.py — clean, presentation-ready charts (matplotlib only)
# One figure per chart; robust to missing columns; no explicit colors.
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Paths ----------
REPO = Path(__file__).resolve().parent
FULL = REPO / "data" / "processed" / "clean_full.csv"
NUM  = REPO / "data" / "processed" / "clean_numeric.csv"
OUT  = REPO / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- Load ----------
df_full = pd.read_csv(FULL)
df_num  = pd.read_csv(NUM)

# ---------- Style helpers ----------
def beautify(ax, title=None, xlabel=None, ylabel=None):
    ax.grid(True, ls="--", lw=0.6, alpha=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    if title: ax.set_title(title, fontsize=13, pad=10)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    plt.tight_layout()

def savefig(name):
    fp = OUT / name
    plt.tight_layout()
    plt.savefig(fp, dpi=160)
    print("saved:", fp)
    plt.close()

def bar_annot(ax, fmt="{:.0f}", offset=3):
    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h) and h > 0:
            ax.annotate(fmt.format(h), (p.get_x()+p.get_width()/2, h),
                        ha="center", va="bottom", fontsize=9, xytext=(0, offset),
                        textcoords="offset points")

def heatmap_with_text(mat, xlabels, ylabels, title):
    plt.figure(figsize=(6.5, 4.5))
    ax = plt.gca()
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels)
    beautify(ax, title=title, xlabel="", ylabel="")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # in-cell annotations
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(int(mat[i, j])), ha="center", va="center", fontsize=8)
    return ax

# ---------- Helpers ----------
def get_col(*cands, df=None):
    cols = df.columns if df is not None else df_full.columns
    for c in cands:
        if c in cols:
            return c
    return None

def ensure_binary_target(df):
    """Return y (0/1) from Mental_Health_Condition if available."""
    target = get_col("Mental_Health_Condition", "MHC", "Condition", df=df)
    if target is None:
        # fallback from label if present (NOT ideal, order arbitrary) -> skip
        y = None
    else:
        y = (df[target].astype("string").str.strip().str.lower() == "yes").astype(int)
    return y, target

# ===========================================================
# 1) Outcome distribution (Mental_Health_Condition)
# ===========================================================
y, target_col = ensure_binary_target(df_full)
if target_col:
    vc = df_full[target_col].fillna("NA").value_counts(dropna=False)
    order = ["No", "Yes", "NA"]
    order = [x for x in order if x in vc.index] + [x for x in vc.index if x not in order]
    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    vc.loc[order].plot(kind="bar", ax=ax)
    pos_rate = (y.mean()*100) if y is not None else np.nan
    beautify(ax, title=f"{target_col} distribution (pos={pos_rate:.1f}% if available)", xlabel=target_col, ylabel="Count")
    bar_annot(ax)
    savefig("01_target_distribution.png")

# ===========================================================
# 2) Severity_ord distribution (with NA)
# ===========================================================
sev = get_col("Severity_ord", "Severity", df=df_full)
if sev:
    s = df_full[sev]
    vc = s.value_counts(dropna=False).sort_index()
    vc.index = [("NA" if (isinstance(i, float) and np.isnan(i)) else i) for i in vc.index]
    plt.figure(figsize=(6.5, 4))
    ax = plt.gca()
    vc.plot(kind="bar", ax=ax)
    beautify(ax, title=f"Distribution of {sev}", xlabel=sev, ylabel="Count")
    bar_annot(ax)
    savefig("02_severity_distribution.png")

# ===========================================================
# 3) Smoking_Habit_ord × target (heatmap)
# ===========================================================
smk = get_col("Smoking_Habit_ord", df=df_full)
if smk and target_col:
    ct = pd.crosstab(df_full[smk], df_full[target_col])
    # Column order: No, Yes (if present)
    xlabels = [x for x in ["No","Yes"] if x in ct.columns] + [c for c in ct.columns if c not in ["No","Yes"]]
    mat = ct.reindex(columns=xlabels).values
    ax = heatmap_with_text(mat, xlabels=xlabels, ylabels=[str(i) for i in ct.index],
                           title=f"{smk} × {target_col} (counts)")
    savefig("03_smoking_by_target_heatmap.png")

# ===========================================================
# 4) Alcohol_Consumption_ord × target (heatmap)
# ===========================================================
alc = get_col("Alcohol_Consumption_ord", df=df_full)
if alc and target_col:
    ct = pd.crosstab(df_full[alc], df_full[target_col])
    xlabels = [x for x in ["No","Yes"] if x in ct.columns] + [c for c in ct.columns if c not in ["No","Yes"]]
    mat = ct.reindex(columns=xlabels).values
    ax = heatmap_with_text(mat, xlabels=xlabels, ylabels=[str(i) for i in ct.index],
                           title=f"{alc} × {target_col} (counts)")
    savefig("04_alcohol_by_target_heatmap.png")

# ===========================================================
# 5) Age histogram (overall)
# ===========================================================
age = get_col("Age", df=df_full)
if age:
    plt.figure(figsize=(6.5, 4))
    ax = plt.gca()
    df_full[age].dropna().astype(float).plot(kind="hist", bins=25, ax=ax)
    beautify(ax, title="Age distribution", xlabel="Age", ylabel="Frequency")
    savefig("05_age_hist.png")

# ===========================================================
# 6) Age × target (step histograms; linestyle to differentiate)
# ===========================================================
if age and target_col and y is not None:
    plt.fig
