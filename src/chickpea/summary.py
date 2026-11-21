"""Batch uniformity: distribution of every trait across the seeds of a sample."""
import numpy as np
import pandas as pd
from skimage.color import deltaE_ciede2000

SKIP = {"seed_id", "x_mm", "y_mm", "depth_coverage"}


def summarise(seeds: pd.DataFrame) -> pd.DataFrame:
    traits = [c for c in seeds.columns if c not in SKIP and pd.api.types.is_numeric_dtype(seeds[c])]
    rows = []
    for t in traits:
        v = seeds[t].dropna()
        if v.empty:
            continue
        mean = v.mean()
        rows.append({
            "trait": t, "n": len(v), "mean": mean, "sd": v.std(ddof=1) if len(v) > 1 else np.nan,
            "cv_pct": 100 * v.std(ddof=1) / mean if len(v) > 1 and mean else np.nan,
            "min": v.min(), "p25": v.quantile(0.25), "median": v.median(),
            "p75": v.quantile(0.75), "max": v.max(),
        })
    out = pd.DataFrame(rows)

    # colour uniformity: perceptual distance of each seed from the batch's mean colour
    lab = seeds[["L_mean", "a_mean", "b_mean"]].to_numpy()
    if len(lab):
        de = deltaE_ciede2000(lab, np.broadcast_to(lab.mean(axis=0), lab.shape))
        out = pd.concat([out, pd.DataFrame([{
            "trait": "colour_deltaE_to_batch_mean", "n": len(de), "mean": de.mean(),
            "sd": de.std(ddof=1) if len(de) > 1 else np.nan, "cv_pct": np.nan,
            "min": de.min(), "p25": np.quantile(de, 0.25), "median": np.median(de),
            "p75": np.quantile(de, 0.75), "max": de.max(),
        }])], ignore_index=True)
    return out
