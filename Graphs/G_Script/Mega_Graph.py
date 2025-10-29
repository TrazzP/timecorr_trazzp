#!/usr/bin/env python3
# Four horizontal panels (PCA, IncPCA, FA, TruncatedSVD)
# Averaged over levels; x=kernel width; lines=window profile; NO CI
# Panels are set taller-than-wide via set_box_aspect.

from pathlib import Path
import re, math
import pandas as pd
import matplotlib.pyplot as plt

INPUT_ROOT = Path("/app/Cluster_Data")
CONDS      = ["paragraph", "word", "intact", "rest"]
OUT_DIR    = Path("/app/Graphs/Method_Value_By_Level/Mega")
OUT_FIG    = OUT_DIR / "Method_HStack_ByWidth_NoCI.png"
FACTORS_REQUIRE = "700"

METHODS = ["PCA","IncrementalPCA","FactorAnalysis","TruncatedSVD"]
WP_MARKERS = {"Gaussian":"o","Laplace":"^","MexicanHat":"s"}

ALIAS_R = {
    "pca":"PCA",
    "incrementalpca":"IncrementalPCA","incremental_pca":"IncrementalPCA","ipca":"IncrementalPCA",
    "factoranalysis":"FactorAnalysis","fa":"FactorAnalysis",
    "truncatedsvd":"TruncatedSVD","truncsvd":"TruncatedSVD","svd":"TruncatedSVD",
}
ALIAS_WP = {
    "gaussian":"Gaussian",
    "laplace":"Laplace","laplacian":"Laplace",
    "mexicanhat":"MexicanHat","mexican_hat":"MexicanHat","mexican":"MexicanHat",
}

def nmethod(x:str)->str:
    s = re.sub(r"[^a-z0-9]+","",str(x).strip().lower())
    return ALIAS_R.get(s, str(x).strip())

def nwp(x:str)->str:
    s = re.sub(r"[^a-z0-9]+","",str(x).strip().lower())
    return ALIAS_WP.get(s, str(x).strip())

def parse_stem(stem:str):
    p = stem.split("_")
    if len(p)<8: raise ValueError(stem)
    cond,factors,level_from_name,reps,cfun,rfun,width = p[:7]
    wp = "_".join(p[7:])
    return cond,factors,level_from_name,reps,cfun,rfun,width,wp

def load_all():
    rows = []
    for cond in CONDS:
        folder = INPUT_ROOT / cond
        if not folder.is_dir(): continue
        for fp in folder.glob("*.csv"):
            try:
                c,factors,level_from_name,reps,cfun,rfun,width_str,wp = parse_stem(fp.stem)
            except ValueError:
                continue
            if factors != FACTORS_REQUIRE: continue
            try:
                df = pd.read_csv(fp)
            except Exception:
                continue
            if "level" not in df.columns or "accuracy" not in df.columns:
                continue
            df = df.copy()
            df["level"] = pd.to_numeric(df["level"], errors="coerce")
            df = df.dropna(subset=["level","accuracy"])
            try:
                width_val = int(str(width_str).lstrip("0") or "0")
            except Exception:
                try: width_val = int(float(width_str))
                except Exception: width_val = None
            df["cond"]   = c
            df["method"] = nmethod(rfun)
            df["wp"]     = nwp(wp)
            df["width"]  = width_val
            rows.append(df[["cond","method","wp","width","level","accuracy"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all()
    if df.empty:
        print("No usable data (factors==700)."); return 0

    # filter
    df = df[df["method"].isin(METHODS)]
    df = df[df["wp"].isin(WP_MARKERS.keys())]
    df = df.dropna(subset=["width"])
    if df.empty:
        print("No data after filtering."); return 0

    # per-level per-config average, then average across levels
    per_level = df.groupby(["cond","method","wp","width","level"], as_index=False)["accuracy"].mean()
    per_config = per_level.groupby(["cond","method","wp","width"], as_index=False)["accuracy"].mean()

    # mean across configs for plotting
    summary = (per_config
               .groupby(["method","wp","width"])["accuracy"]
               .mean().reset_index())

    # shared y
    y_min = summary["accuracy"].min()
    y_max = summary["accuracy"].max()
    pad = 0.02*(y_max - y_min if y_max>y_min else 1.0)
    y_lim = (y_min - pad, y_max + pad)

    # horizontal stack; each panel taller than wide via box aspect
    fig, axes = plt.subplots(1, 4, figsize=(16, 6), sharey=True, constrained_layout=True)

    for ax, method in zip(axes, METHODS):
        ms = summary[summary["method"]==method]
        if ms.empty:
            ax.text(0.5,0.5,f"No data: {method}", ha="center", va="center")
            ax.set_title(method); ax.set_ylim(*y_lim); ax.set_box_aspect(1.2); continue
        for wp_name, marker in WP_MARKERS.items():
            sub = ms[ms["wp"]==wp_name].sort_values("width")
            if sub.empty: continue
            ax.plot(sub["width"], sub["accuracy"], marker=marker, label=wp_name)
        ax.set_title(method)
        ax.set_xlabel("Kernel width")
        ax.set_ylim(*y_lim)
        ax.grid(True, alpha=0.3)
        ax.set_box_aspect(1.2)  # height > width

    axes[0].set_ylabel("Accuracy")

    # Single legend centered below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Window profile", loc="lower center", ncol=3, frameon=True)
    fig.suptitle("Averaged Over Levels • Accuracy vs Kernel Width (factors=700)", y=0.98, fontsize=13)
    fig.subplots_adjust(bottom=0.12, top=0.92)

    fig.savefig(OUT_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"→ Saved: {OUT_FIG}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
