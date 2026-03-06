from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd


def infer_family(signal_name: str) -> str:
    s = str(signal_name).lower()
    if "liq" in s or "liquid" in s:
        return "liq"
    if "vac" in s:
        return "vacuum"
    if "comp" in s:
        return "compression"
    if "intensity" in s:
        return "intensity"
    if "ofi" in s:
        return "ofi"
    return "other"


def _spearman_like(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ra = a.rank(method="average")
    rb = b.rank(method="average")
    c = ra.corr(rb)
    return float(c) if pd.notna(c) else 0.0


@dataclass(frozen=True)
class FamilyGeneralization:
    family: str
    survival_frac_mean: float
    rank_consistency: float
    directional_consistency: float
    regime_similarity: float
    generalization_score: float


def compute_family_generalization(
    *,
    per_symbol: Dict[str, Dict[str, pd.DataFrame]],
    directional_per_symbol: Dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    symbols = sorted(per_symbol.keys())
    fam_rows: List[Dict[str, float | str]] = []
    families: set[str] = set()
    for sym in symbols:
        cand = per_symbol[sym].get("candidates", pd.DataFrame())
        for n in cand.get("signal", pd.Series([], dtype=str)).astype(str).tolist():
            families.add(infer_family(n))
    for family in sorted(families):
        survival_vals: List[float] = []
        regime_vals: List[float] = []
        for sym in symbols:
            cand = per_symbol[sym].get("candidates", pd.DataFrame())
            sel = per_symbol[sym].get("selected", pd.DataFrame())
            summ = per_symbol[sym].get("summary", pd.DataFrame())
            c = cand[cand["family"] == family] if not cand.empty else pd.DataFrame()
            s = sel[sel["family"] == family] if not sel.empty else pd.DataFrame()
            survival_vals.append(float(len(s) / max(1, len(c))))
            if not summ.empty:
                ss = summ[summ["family"] == family]
                regime_vals.append(float(pd.to_numeric(ss.get("regime_concentration"), errors="coerce").mean() if not ss.empty else 0.0))
        survival = float(sum(survival_vals) / max(1, len(survival_vals)))

        # Pairwise rank consistency over common signals in family.
        rank_pairs: List[float] = []
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                a = per_symbol[symbols[i]].get("summary", pd.DataFrame())
                b = per_symbol[symbols[j]].get("summary", pd.DataFrame())
                if a.empty or b.empty:
                    continue
                aa = a[a["family"] == family][["signal", "test_net_mean"]].rename(columns={"test_net_mean": "a"})
                bb = b[b["family"] == family][["signal", "test_net_mean"]].rename(columns={"test_net_mean": "b"})
                m = aa.merge(bb, on="signal", how="inner")
                if m.empty:
                    continue
                rank_pairs.append(_spearman_like(m["a"], m["b"]))
        rank_consistency = float(sum(rank_pairs) / max(1, len(rank_pairs))) if rank_pairs else 0.0
        rank_consistency = float(max(0.0, min(1.0, (rank_consistency + 1.0) / 2.0)))

        # Directional consistency from optional directional summaries.
        dir_cons = 1.0
        if directional_per_symbol:
            vals: List[float] = []
            for sym in symbols:
                d = directional_per_symbol.get(sym, pd.DataFrame())
                if d.empty:
                    continue
                dd = d[d["family"] == family]
                if dd.empty:
                    continue
                vals.append(float((pd.to_numeric(dd["win_rate"], errors="coerce") > 0.5).mean()))
            if vals:
                dir_cons = float(sum(vals) / len(vals))

        # Regime similarity from concentration closeness.
        if regime_vals:
            rmin = min(regime_vals)
            rmax = max(regime_vals)
            regime_sim = float(max(0.0, 1.0 - abs(rmax - rmin)))
        else:
            regime_sim = 0.0

        score = float(0.35 * survival + 0.35 * rank_consistency + 0.15 * dir_cons + 0.15 * regime_sim)
        fam_rows.append(
            {
                "family": family,
                "survival_frac_mean": survival,
                "rank_consistency": rank_consistency,
                "directional_consistency": float(dir_cons),
                "regime_similarity": regime_sim,
                "generalization_score": score,
            }
        )
    return pd.DataFrame(fam_rows).sort_values(["generalization_score", "family"], ascending=[False, True]).reset_index(drop=True)

