#!/usr/bin/env python3
"""
Four horizontal panels by reduction method (PCA, IncPCA, FA, TruncatedSVD).
Each point summarizes a (method, window profile, kernel width) configuration:
  - x  = kernel width (numeric)
  - y  = mean decoding accuracy averaged across levels and conditions (as before)
  - marker shape = window profile (Gaussian, Laplace, MexicanHat)
  - marker size  = kernel width (scaled)
  - color = "plateau level" (the earliest level where accuracy vs level flattens)

Notes
-----
* Plateau detection is heuristic but tunable (EPS, CONSEC, SMOOTH_WIN).
* We compute accuracy vs level by averaging across conditions for a fixed (method, wp, width),
  then locate the earliest level where the smoothed first difference stays below EPS for CONSEC steps.
* If no plateau is found, we color with the max observed level for that series.
* Legend: a small proxy legend conveys marker shapes (window profiles). A colorbar indicates plateau level.
* No confidence intervals per your request. Panels are taller than wide.
"""

from pathlib import Path
import re
import math
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# -------------------- Configuration --------------------
INPUT_ROOT = Path("/app/Cluster_Data")
CONDS      = ["paragraph", "word", "intact", "rest"]
OUT_DIR    = Path("/app/Graphs/Method_Value_By_Level/Mega")
OUT_FIG = OUT_DIR / "Mega_Graph_Final.png"
FACTORS_REQUIRE = "700"

METHODS = ["PCA","IncrementalPCA","FactorAnalysis","TruncatedSVD"]
WP_MARKERS = {"Gaussian":"o","Laplace":"^","MexicanHat":"s"}

# Plateau detection hyperparameters (tweakable)
EPS        = 0.002      # minimal improvement considered "flat" (in accuracy units)
CONSEC     = 2          # how many consecutive diffs must be < EPS to call it a plateau
SMOOTH_WIN = 3          # moving-average window over levels before differencing (odd number recommended)

# Marker-size scaling (points^2)
SIZE_MIN = 30
SIZE_MAX = 220

# -------------------- Aliases / Normalizers --------------------
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

# -------------------- IO Helpers --------------------

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
        if not folder.is_dir():
            continue
        for fp in folder.glob("*.csv"):
            try:
                c,factors,level_from_name,reps,cfun,rfun,width_str,wp = parse_stem(fp.stem)
            except ValueError:
                continue
            if factors != FACTORS_REQUIRE:
                continue
            try:
                df = pd.read_csv(fp)
            except Exception:
                continue
            if "level" not in df.columns or "accuracy" not in df.columns:
                continue
            df = df.copy()
            df["level"] = pd.to_numeric(df["level"], errors="coerce")
            df = df.dropna(subset=["level","accuracy"])  # keep if both exist
            # Parse width to int if possible
            try:
                width_val = int(str(width_str).lstrip("0") or "0")
            except Exception:
                try:
                    width_val = int(float(width_str))
                except Exception:
                    width_val = None
            df["cond"]   = c
            df["method"] = nmethod(rfun)
            df["wp"]     = nwp(wp)
            df["width"]  = width_val
            rows.append(df[["cond","method","wp","width","level","accuracy"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

# -------------------- Plateau Detection --------------------

def moving_average(y: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(y) < 2:
        return y.copy()
    win = max(1, int(win))
    if win % 2 == 0:
        win += 1  # force odd window
    pad = win // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(win) / win
    return np.convolve(ypad, kernel, mode="valid")


def detect_inflection_and_plateau(
    levels: np.ndarray,
    acc: np.ndarray,
    eps: float = EPS,          # near-zero slope threshold for plateau
    consec: int = CONSEC,      # run length for plateau
    smooth_win: int = SMOOTH_WIN
) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (inflection_level, plateau_level_after_inflection) using ONLY:
      • Inflection = zero-crossing of 2nd derivative (of smoothed accuracy).
      • Plateau   = first post-inflection run where |1st derivative| < eps for `consec` steps.
    If no zero-crossing exists, inflection falls back to max |1st derivative| (still "rate greatest"),
    which is consistent with Lucy's wording and *does not* use an elbow/curvature proxy.
    """
    n = len(levels)
    if n < 3:
        if n: 
            L = float(np.max(levels))
            return (L, L)
        return (None, None)

    # Sort by level and smooth accuracy
    order = np.argsort(levels)
    lv = levels[order].astype(float)
    ya = acc[order].astype(float)
    ys = moving_average(ya, smooth_win)

    # If your smoother ever shortens length, align lv (moving_average here keeps length)
    if len(ys) != len(lv):
        trim = len(lv) - len(ys)
        left = trim // 2
        right = trim - left
        lv = lv[left: len(lv) - right]

    # Finite differences per unit level (handle uneven spacing)
    dlv = np.diff(lv)
    d1 = np.diff(ys) / dlv                  # first derivative (slope)
    d2 = np.diff(d1) / np.diff(lv[:-1])     # second derivative

    # ---- Inflection: first zero-crossing of the second derivative ----
    # Find indices k where d2[k] and d2[k+1] have opposite signs
    cross = np.where(np.sign(d2[:-1]) != np.sign(d2[1:]))[0]
    if cross.size:
        # choose the first zero-crossing (closest to the earliest bend)
        k = int(cross[0]) + 1   # d2[k] zero-crosses between levels ~ lv[k+2]
        inflection_level = float(lv[min(k + 1, len(lv) - 1)])
        # Map to starting point in slope space (d1 aligns with intervals [lv[i], lv[i+1]])
        start_idx_in_d1 = max(0, k)  # conservative: begin just after the crossing
    else:
        # No zero-crossing ⇒ fall back to "rate greatest": max |d1|
        i = int(np.argmax(np.abs(d1)))
        inflection_level = float(lv[i + 1])
        start_idx_in_d1 = i

    # ---- Plateau: first run after inflection where |slope| < eps for `consec` steps ----
    post = np.abs(d1[start_idx_in_d1:])
    plateau_level = float(lv.max())
    if post.size >= consec:
        run = np.convolve((post < eps).astype(int), np.ones(consec, dtype=int), mode="valid")
        j = np.argmax(run == consec) if np.any(run == consec) else -1
        if j >= 0:
            # d1[k] spans lv[k]→lv[k+1]; choose the right edge as the plateau "level"
            plateau_level = float(lv[start_idx_in_d1 + j + 1])

    return (inflection_level, plateau_level)



# -------------------- Main Plot --------------------

def build_plot_df_by_condition(per_level: pd.DataFrame,
                               eps: float, consec: int, smooth_win: int) -> dict:
    """Return a dict: cond -> plot_df with columns
       [method, wp, width, acc_mean, plateau_level, msize].
    Plateau and acc_mean are computed *within that condition*.
    """
    out = {}
    # Compute sizes globally so width scaling consistent across conditions
    widths = per_level["width"].dropna().to_numpy()
    wmin, wmax = (float(np.min(widths)), float(np.max(widths))) if widths.size else (0.0, 1.0)

    for cond, gcond in per_level.groupby("cond"):
        # A. y-value: mean accuracy across levels for each (method, wp, width) within this condition
        ydf = (gcond.groupby(["method","wp","width"], as_index=False)["accuracy"]
                    .mean().rename(columns={"accuracy":"acc_mean"}))

        # B. plateau detection within this condition
        plist = []
        for (method, wp, width), g in gcond.groupby(["method","wp","width"], as_index=False):
            series = (g.groupby("level", as_index=False)["accuracy"].mean().sort_values("level"))
            lv = series["level"].to_numpy()
            ya = series["accuracy"].to_numpy()
            infL, platL = detect_inflection_and_plateau(lv, ya, eps=eps, consec=consec, smooth_win=smooth_win)
            plist.append({"method":method, "wp":wp, "width":width,
                          "inflection_level":infL, "plateau_level": platL})

        pdf = pd.DataFrame(plist)

        plot_df = pd.merge(ydf, pdf, on=["method","wp","width"], how="inner")

        # Marker-size scaling shared globally
        if wmax == wmin:
            plot_df["msize"] = (SIZE_MIN + SIZE_MAX) / 2.0
        else:
            plot_df["msize"] = SIZE_MIN + (plot_df["width"] - wmin) * (SIZE_MAX - SIZE_MIN) / (wmax - wmin)
        out[cond] = plot_df
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all()
    if df.empty:
        print("No usable data (factors==700).")
        return 0

    # Filter to methods / profiles we care about and valid widths
    df = df[df["method"].isin(METHODS)]
    df = df[df["wp"].isin(WP_MARKERS.keys())]
    df = df.dropna(subset=["width"])  # ensure width is numeric
    if df.empty:
        print("No data after filtering.")
        return 0

    # Per-level mean across any duplicates (folds, reps)
    per_level = df.groupby(["cond","method","wp","width","level"], as_index=False)["accuracy"].mean()

    # Build condition-specific plot frames (acc_mean & plateau per condition)
    cond_plot = build_plot_df_by_condition(per_level, EPS, CONSEC, SMOOTH_WIN)

    # Establish shared y-limits across all 16 panels
    all_acc = np.concatenate([cond_plot[c]["acc_mean"].to_numpy() for c in cond_plot]) if cond_plot else np.array([])
    y_min = float(all_acc.min()) if all_acc.size else 0.0
    y_max = float(all_acc.max()) if all_acc.size else 1.0
    pad = 0.02*(y_max - y_min if y_max>y_min else 1.0)
    y_lim = (y_min - pad, y_max + pad)

    # Color normalization shared across conditions
    all_pl = np.concatenate([cond_plot[c]["plateau_level"].to_numpy() for c in cond_plot]) if cond_plot else np.array([])
    lmin = float(all_pl.min()) if all_pl.size else 0.0
    lmax = float(all_pl.max()) if all_pl.size else 1.0
    norm = Normalize(vmin=lmin, vmax=lmax)
    cmap = plt.cm.viridis

    # -------------------- Figure: 4x4 grid --------------------
    row_methods = METHODS                         # rows = dimensionality reduction methods
    col_conds   = ["intact","paragraph","word","rest"]  # columns = conditions

    fig, axes = plt.subplots(
        4, 4,
        figsize=(15, 14),
        sharex=True, sharey=True,
        constrained_layout=False,
        gridspec_kw={'hspace': 0.03, 'wspace': 0.12}
    )

    for r, method in enumerate(row_methods):
        for c, cond in enumerate(col_conds):
            ax = axes[r, c]
            ms_all = cond_plot.get(cond, pd.DataFrame())  # cond-specific df from earlier
            ms = ms_all[ms_all["method"] == method]

            if ms.empty:
                ax.text(0.5, 0.5, f"No data: {method} / {cond}", ha="center", va="center")
                ax.set_box_aspect(1.0)
            else:
                for wp_name, marker in WP_MARKERS.items():
                    sub = ms[ms["wp"]==wp_name].sort_values("width")
                    if sub.empty:
                        continue
                    ax.scatter(
                        sub["width"], sub["acc_mean"],
                        s=sub["msize"], marker=marker,
                        c=cmap(norm(sub["plateau_level"].to_numpy())),  # color = plateau level
                        edgecolors='white', linewidths=0.5, alpha=0.95
                    )
                ax.grid(True, alpha=0.25)
                ax.set_box_aspect(1.0)

            # Titles on top row: column headers = conditions
            if r == 0:
                ax.set_title(cond, fontsize=11, pad=6)

            # Y-labels on leftmost column: row headers = methods
            if c == 0:
                ax.set_ylabel(f"{method} — mean accuracy", fontsize=10)

            # Only the bottom row shows x-axis labels/ticks
            if r == len(row_methods) - 1:
                ax.set_xlabel("Kernel width", fontsize=10)
                ax.tick_params(axis='x', labelbottom=True, bottom=True)
            else:
                ax.tick_params(axis='x', labelbottom=False, bottom=False)

            # Custom y-limits for the "intact" column ONLY
            if cond == "intact":
                ax.set_ylim(0.2, 0.5)
            else:
                ax.set_ylim(*y_lim)

    # Tighten margins to reduce bottom whitespace
    plt.subplots_adjust(top=0.93, bottom=0.07, left=0.08, right=0.98, hspace=0.03, wspace=0.12)



    # Profile legend (marker shapes)
    proxy = [Line2D([0],[0], marker=m, linestyle='None', markersize=8, markerfacecolor='gray', markeredgecolor='gray')
             for m in WP_MARKERS.values()]
    labels = list(WP_MARKERS.keys())
    fig.legend(proxy, labels, title="Window profile", loc="lower center", ncol=3, frameon=True)

    # Colorbar for plateau level
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='vertical', pad=0.01, fraction=0.03)
    cbar.set_label("Plateau level (earliest)")

    fig.suptitle("Conditions × Methods ×  Kernel Width, Shape=WP, Color=Plateau Level, Factors=700", y=0.995, fontsize=14)

    fig.savefig(OUT_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"→ Saved: {OUT_FIG}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
