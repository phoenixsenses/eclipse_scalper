from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


SHARED_FEATURES = [
    "rv_z",
    "spread_z",
    "intensity_z",
    "liq_rate_z",
    "impact_proxy",
    "micro_trend",
    "of_flow_persistence",
]


@dataclass(frozen=True)
class AlignmentConfig:
    method: str = "quantile_buckets"
    k: int = 6
    seed: int = 42
    sample_rows: int = 500_000


def _zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mu = float(x.mean()) if len(x) else 0.0
    sd = float(x.std(ddof=0)) if len(x) else 0.0
    if sd <= 0:
        return pd.Series(np.zeros(len(x), dtype=float), index=x.index)
    out = (x - mu) / sd
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ""


def _ensure_shared_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    warnings: List[str] = []

    if "rv_z" not in out.columns:
        src = _pick_col(out, ["rv_short", "rv"])
        if src:
            out["rv_z"] = _zscore(out[src])
        else:
            out["rv_z"] = 0.0
            warnings.append("rv_z:missing_source")

    if "spread_z" not in out.columns:
        src = _pick_col(out, ["spread", "spread_abs"])
        if src:
            out["spread_z"] = _zscore(out[src])
        else:
            out["spread_z"] = 0.0
            warnings.append("spread_z:missing_source")

    if "intensity_z" not in out.columns:
        src = _pick_col(out, ["F_intensity_z", "trade_intensity", "trade_intensity_qty_per_sec"])
        if src:
            out["intensity_z"] = _zscore(out[src])
        else:
            out["intensity_z"] = 0.0
            warnings.append("intensity_z:missing_source")

    if "liq_rate_z" not in out.columns:
        src = _pick_col(out, ["liq_rate", "liq_n_30s", "liq_qty_sum_30s"])
        if src:
            out["liq_rate_z"] = _zscore(out[src])
        else:
            out["liq_rate_z"] = 0.0
            warnings.append("liq_rate_z:missing_source")

    if "impact_proxy" not in out.columns:
        rv = pd.to_numeric(out.get("r_1", pd.Series(np.zeros(len(out), dtype=float), index=out.index)), errors="coerce").abs().fillna(0.0)
        q = pd.to_numeric(out.get("volume_proxy", pd.Series(np.zeros(len(out), dtype=float), index=out.index)), errors="coerce")
        if q.isna().all():
            q = pd.to_numeric(out.get("buy_qty", pd.Series(np.zeros(len(out), dtype=float), index=out.index)), errors="coerce").fillna(0.0) + pd.to_numeric(
                out.get("sell_qty", pd.Series(np.zeros(len(out), dtype=float), index=out.index)), errors="coerce"
            ).fillna(0.0)
        out["impact_proxy"] = (rv / np.sqrt(q.replace(0.0, np.nan))).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "micro_trend" not in out.columns:
        r1 = pd.to_numeric(out.get("r_1", pd.Series(np.zeros(len(out), dtype=float), index=out.index)), errors="coerce").fillna(0.0)
        out["micro_trend"] = r1.rolling(20, min_periods=5).mean().fillna(0.0)

    if "of_flow_persistence" not in out.columns:
        ofi = pd.to_numeric(out.get("F_ofi", pd.Series(np.zeros(len(out), dtype=float), index=out.index)), errors="coerce")
        if ofi.isna().all():
            ofi = pd.to_numeric(out.get("ofi", pd.Series(np.zeros(len(out), dtype=float), index=out.index)), errors="coerce")
        sign = np.sign(ofi.fillna(0.0))
        out["of_flow_persistence"] = (sign * sign.shift(1).fillna(0.0)).rolling(20, min_periods=5).mean().fillna(0.0)

    return out, warnings


def build_shared_alignment_frame(symbol_frames: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
    rows: List[pd.DataFrame] = []
    warnings: List[str] = []
    for symbol, frame in sorted(symbol_frames.items()):
        if frame.empty:
            warnings.append(f"{symbol}:empty_frame")
            continue
        cur = frame.copy().sort_values("ts_ms").reset_index(drop=True)
        cur["symbol"] = symbol
        cur, ww = _ensure_shared_features(cur)
        warnings.extend([f"{symbol}:{w}" for w in ww])
        # per-symbol standardization for comparability
        for f in SHARED_FEATURES:
            cur[f] = _zscore(cur[f])
        keep = [c for c in ["ts_ms", "ts_utc", "symbol"] + SHARED_FEATURES if c in cur.columns]
        rows.append(cur[keep])
    if not rows:
        return pd.DataFrame(), warnings
    out = pd.concat(rows, ignore_index=True).sort_values(["ts_ms", "symbol"]).reset_index(drop=True)
    return out, warnings


def _stable_label_map(labels: np.ndarray, rv: np.ndarray, intensity: np.ndarray) -> Dict[int, int]:
    uniq = sorted({int(x) for x in labels})
    scored: List[Tuple[int, float]] = []
    for u in uniq:
        m1 = float(np.nanmean(rv[labels == u])) if np.any(labels == u) else 0.0
        m2 = float(np.nanmean(intensity[labels == u])) if np.any(labels == u) else 0.0
        scored.append((u, m1 + 0.1 * m2))
    scored.sort(key=lambda t: t[1])
    return {old: new for new, (old, _s) in enumerate(scored)}


def assign_aligned_regimes(frame: pd.DataFrame, cfg: AlignmentConfig) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy().sort_values(["ts_ms", "symbol"]).reset_index(drop=True)
    x = out[SHARED_FEATURES].to_numpy(dtype=float)
    x[~np.isfinite(x)] = 0.0
    k = max(2, int(cfg.k))
    method = str(cfg.method).strip().lower()
    if method == "quantile_buckets":
        score = np.nanmean(x, axis=1)
        rank = pd.Series(score).rank(method="first", pct=True).to_numpy(dtype=float)
        labels = np.minimum(k - 1, np.floor(rank * k).astype(int))
    elif method == "kmeans_global":
        model = KMeans(n_clusters=k, random_state=int(cfg.seed), n_init=10)
        labels = model.fit_predict(x)
    elif method == "gmm_global":
        gm = GaussianMixture(n_components=k, random_state=int(cfg.seed), covariance_type="full")
        labels = gm.fit_predict(x)
    else:
        raise ValueError(f"unknown_alignment_method:{cfg.method}")

    mapper = _stable_label_map(labels, out["rv_z"].to_numpy(dtype=float), out["intensity_z"].to_numpy(dtype=float))
    out["aligned_regime_id"] = [int(mapper[int(x)]) for x in labels]
    return out


def describe_aligned_regimes(aligned: pd.DataFrame) -> pd.DataFrame:
    if aligned.empty:
        return pd.DataFrame(
            columns=["aligned_regime_id", "rows", "eth_frac", "btc_frac"] + [f"{c}_mean" for c in SHARED_FEATURES]
        )
    grp = aligned.groupby("aligned_regime_id", as_index=False)
    agg = grp.agg(rows=("ts_ms", "count"), **{f"{c}_mean": (c, "mean") for c in SHARED_FEATURES})
    sym_counts = (
        aligned.groupby(["aligned_regime_id", "symbol"], as_index=False)
        .size()
        .pivot(index="aligned_regime_id", columns="symbol", values="size")
        .fillna(0.0)
    )
    for sym in ["ETHUSDT", "BTCUSDT"]:
        if sym not in sym_counts.columns:
            sym_counts[sym] = 0.0
    sym_frac = sym_counts.div(sym_counts.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0).reset_index()
    sym_frac = sym_frac.rename(columns={"ETHUSDT": "eth_frac", "BTCUSDT": "btc_frac"})
    out = agg.merge(sym_frac[["aligned_regime_id", "eth_frac", "btc_frac"]], on="aligned_regime_id", how="left")
    out["eth_frac"] = out["eth_frac"].fillna(0.0)
    out["btc_frac"] = out["btc_frac"].fillna(0.0)
    return out.sort_values("aligned_regime_id").reset_index(drop=True)
