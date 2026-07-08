"""Faz 6A-R Drift Monitor — RESEARCH-ONLY.

Referans pencere vs guncel pencere: feature PSI, rejim occupancy, latent occupancy,
transition-matrix mesafesi, data-quality drift'i -> STABLE / WARNING / SHIFTED / UNUSABLE.

YETKI SINIRI: Monitor trade karari VEREMEZ, izin VEREMEZ/KALDIRAMAZ. Yalniz
Epistemic Governor'a ONERI uretir (applicability dusurme, confidence azaltma,
retest, shadow askiya alma). Nihai yetki governor'dadir. SHIFTED/UNUSABLE
durumunda oneri listesi BOS OLAMAZ (alarm susturma mutation'i bunu test eder).

Run: python -m ami.latent.drift_monitor
"""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ami.latent.dataset import load_dataset
from ami.latent.models import Standardizer, seeded_kmeans
from ami.latent.regime import DRIFT_PSI, psi, transition_matrix_within

OUT = ROOT / "reports" / "research" / "s34" / "AMI_DRIFT_MONITOR.md"


class DriftMonitor:
    STATUSES = ["STABLE", "WARNING", "SHIFTED", "UNUSABLE"]

    def assess(self, X_ref: np.ndarray, X_cur: np.ndarray, feats: list[str],
               lab_ref: np.ndarray | None = None, lab_cur: np.ndarray | None = None,
               k: int = 4) -> dict:
        per_feat = {}
        n_warn = n_shift = n_unusable = 0
        for i, f in enumerate(feats):
            pv = psi(X_ref[:, i], X_cur[:, i])
            miss_d = abs(float(np.isnan(X_cur[:, i]).mean()) - float(np.isnan(X_ref[:, i]).mean()))
            st = "STABLE"
            if miss_d > 0.30:
                st = "UNUSABLE"; n_unusable += 1
            elif np.isfinite(pv) and pv > DRIFT_PSI["unusable_feat"]:
                st = "SHIFTED"; n_shift += 1
            elif np.isfinite(pv) and pv > DRIFT_PSI["shifted"]:
                st = "SHIFTED"; n_shift += 1
            elif np.isfinite(pv) and pv > DRIFT_PSI["warning"]:
                st = "WARNING"; n_warn += 1
            per_feat[f] = {"psi": round(pv, 3) if np.isfinite(pv) else None,
                           "miss_delta": round(miss_d, 3), "status": st}
        occ_drift = trans_drift = None
        if lab_ref is not None and lab_cur is not None:
            o_r = np.bincount(lab_ref, minlength=k) / len(lab_ref)
            o_c = np.bincount(lab_cur, minlength=k) / len(lab_cur)
            occ_drift = round(float(np.abs(o_r - o_c).sum() / 2), 3)   # total variation
            A_r = transition_matrix_within(lab_ref, k, [])
            A_c = transition_matrix_within(lab_cur, k, [])
            trans_drift = round(float(np.abs(A_r - A_c).mean()), 3)
        if n_unusable > 0 or n_shift >= 3:
            status = "UNUSABLE"
        elif n_shift > 0 or (occ_drift is not None and occ_drift > 0.30):
            status = "SHIFTED"
        elif n_warn > 0 or (occ_drift is not None and occ_drift > 0.15):
            status = "WARNING"
        else:
            status = "STABLE"
        recs = self.recommendations(status)
        return {"status": status, "per_feature": per_feat,
                "latent_occupancy_drift_tv": occ_drift,
                "transition_matrix_drift": trans_drift,
                "recommendations": recs,
                "authority_note": "monitor izin degistiremez; oneriler Epistemic Governor'a gider"}

    def recommendations(self, status: str) -> list[str]:
        """SHIFTED/UNUSABLE'da bos liste DONULEMEZ (alarm susturma yasagi)."""
        if status == "STABLE":
            return []
        if status == "WARNING":
            return ["confidence_reduce(latent knowledge)"]
        if status == "SHIFTED":
            return ["applicability_restrict(latent knowledge)", "retest_request"]
        return ["applicability_restrict(latent knowledge)", "shadow_permission_suspend_suggest",
                "retest_request", "data_quality_investigation"]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    X, ts, miss, meta = load_dataset()
    feats = meta["features"]
    n = X.shape[0]
    ref, cur = X[: int(n * 0.8)], X[int(n * 0.8):]
    std = Standardizer().fit(ref, (int(ts[0]), int(ts[int(n * 0.8) - 1])))
    lab_r, C, _ = seeded_kmeans(std.transform(ref), 4, 11)
    d = ((std.transform(cur)[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    lab_c = d.argmin(axis=1)
    mon = DriftMonitor()
    rep = mon.assess(ref, cur, feats, lab_r, lab_c, k=4)
    print(f"STATUS: {rep['status']}  occ_drift_tv={rep['latent_occupancy_drift_tv']} "
          f"trans_drift={rep['transition_matrix_drift']}")
    for f, v in rep["per_feature"].items():
        if v["status"] != "STABLE":
            print(f"  {f:14s} psi={v['psi']} miss_d={v['miss_delta']} -> {v['status']}")
    print("recommendations:", rep["recommendations"])
    OUT.write_text("# AMI Drift Monitor (research-only)\n\n> " +
                   f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC\n\n```json\n" +
                   json.dumps(rep, indent=1) + "\n```\n\n*Monitor izin degistiremez; " +
                   "oneriler governor'a gider. Runner: `python -m ami.latent.drift_monitor`*",
                   encoding="utf-8")
    print(f"MD: {OUT}")


if __name__ == "__main__":
    main()
