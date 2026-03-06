from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


FEATURE_COLS = [
    "rv_z",
    "spread_z",
    "intensity_z",
    "liq_rate_z",
    "impact_proxy",
    "micro_trend",
    "of_flow_persistence",
]


@dataclass(frozen=True)
class RegimeFitConfig:
    method: str = "hmm"
    n_regimes: int = 4
    seed: int = 42


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[sd == 0] = 1.0
    z = (x - mu) / sd
    z[~np.isfinite(z)] = 0.0
    return z, mu, sd


def _stable_label_map(labels: np.ndarray, score: np.ndarray) -> dict[int, int]:
    uniq = sorted({int(x) for x in labels})
    means = []
    for u in uniq:
        m = float(np.nanmean(score[labels == u])) if np.any(labels == u) else 0.0
        means.append((u, m))
    means.sort(key=lambda t: t[1])
    return {old: new for new, (old, _m) in enumerate(means)}


def fit_regimes(features_df: pd.DataFrame, cfg: RegimeFitConfig) -> pd.DataFrame:
    out = features_df.copy().sort_values("ts_ms").reset_index(drop=True)
    xdf = out[FEATURE_COLS].copy()
    x = xdf.to_numpy(dtype=float)
    xz, _mu, _sd = _standardize(x)

    method = str(cfg.method).lower().strip()
    n = int(max(2, cfg.n_regimes))

    if method == "kmeans":
        model = KMeans(n_clusters=n, random_state=int(cfg.seed), n_init=10)
        labels = model.fit_predict(xz)
        probs = np.ones(len(labels), dtype=float)
    else:
        # "hmm" and "gmm" both map to GMM here for deterministic/no-extra-deps flow.
        model = GaussianMixture(n_components=n, random_state=int(cfg.seed), covariance_type="full")
        model.fit(xz)
        labels = model.predict(xz)
        probs = model.predict_proba(xz).max(axis=1)

    # stable mapping by volatility proxy then intensity as tie-break component
    score = pd.to_numeric(out["rv_z"], errors="coerce").fillna(0.0).to_numpy(dtype=float) + 0.1 * pd.to_numeric(out["intensity_z"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    mapper = _stable_label_map(labels, score)
    mapped = np.array([mapper[int(l)] for l in labels], dtype=int)

    out["regime_id"] = mapped
    out["regime_name"] = out["regime_id"].map(lambda x: f"R{x}")
    out["regime_prob"] = probs.astype(float)
    out["regime_method"] = method
    return out
