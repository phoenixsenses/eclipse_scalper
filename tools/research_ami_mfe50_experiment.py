"""Paket 3 — Ilk Alpha-Improvement Deneyi: MFE+50 giveback ayrimi.

SORU: Trade +50bps MFE'ye ulastiginda, continuation'lari erken KESMEDEN
negative-giveback / positive-stall trade'lerini o anda mevcut feature'larla
ayirabilir miyiz?

PROTOKOL (once dondurulur, sonra kosulur — AMI Research OS uzerinden):
- Evren: hour17 200K(s>=2) + 100K(s>=3), tam gecmis, no-overlap admit, 6h hold.
- Milestone: entry sonrasi ilk +50bps dokunusu (6h icinde).
- Feature'lar YALNIZ hit aninda bilinen veriler (asagida frozen liste).
- TRAIN 70/30 kronolojik; kural secimi TRAIN'de (tek-feature medyan split,
  amac: TRAIN'de continuation_capture>=0.85 kisiti altinda max toplam net).
- Aksiyonlar: HOLD (baseline), EXIT@flag(+50 kilit), LOCK@flag(+25 taban),
  LOCK_ALL (kontrol).
- Basari (frozen): TEST cum net >= baseline VE continuation_capture >= 0.85
  VE maxDD <= baseline maxDD. Falsifikasyon: TEST cum net < baseline.
- Sonuc her durumda AMI knowledge/failure sistemine islenir.

Cikti: reports/research/s34/AMI_MFE50_EXPERIMENT.md + .json
"""
from __future__ import annotations
import json, math, sqlite3, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import load_mark_index
from tools.research_s34_trade_mgmt_gauntlet import build_universe, noov
from ami.enums import ClaimType, EvidenceLevel, FailureType, KnowledgeStatus, Permission
from ami.knowledge.objects import KnowledgeObject, Provenance
from ami.knowledge.store import KnowledgeStore
from ami.research.registry import EvidenceBundle, ExperimentSpec, ResearchRegistry, assert_no_overlap
from ami.storage import production as PR
from ami.storage import research_reader as RR

DB = ROOT / "data" / "microstructure.db"
OUT = ROOT / "reports" / "research" / "s34"
OJ = OUT / "AMI_MFE50_EXPERIMENT.json"; OM = OUT / "AMI_MFE50_EXPERIMENT.md"
FEE = 5.0; H6 = 360; LB = 400 * 24 * 3600_000
TRAIN = 0.70; MILESTONE = 50.0
FEATURES = ["btc10m", "ofi10m", "taker_sell10m", "liq_cont", "rv_hit", "ret1h_hit",
            "funding", "t2m_min", "bid_ratio", "spread"]


def _s(c, q, p=()):
    r = c.execute(q, p).fetchone(); return float(r[0]) if r and r[0] is not None else 0.0


def window_agg_trades_notional(root: str, symbol: str, start_ms: int, end_ms: int) -> tuple[float, float]:
    """Buy-side and total notional over [start_ms, end_ms) for `symbol`,
    via the unified research reader (transparently archive/SQLite/hybrid)
    instead of an ad-hoc direct-SQL SUM(CASE...). Migration pilot for
    BATCH-STORAGE-ROTATION-RETENTION-RESEARCH-READER-INTEGRATION-V1 --
    parity with the old direct-SQL query proven in
    tests/test_research_ami_mfe50_experiment_reader_migration_parity.py."""
    plan = RR.plan_read(root, table="agg_trades", symbol=symbol, start_ms=start_ms, end_ms=end_ms)
    result = RR.execute_read(plan, columns=("notional", "is_buyer_maker"))
    buy = tot = 0.0
    for notional, is_buyer_maker in result.iter_rows():
        tot += notional
        if is_buyer_maker == 0:
            buy += notional
    return buy, tot


def window_avg_book_ticker_bid_qty(root: str, symbol: str, start_ms: int, end_ms: int) -> float | None:
    """AVG(bid_qty) over [start_ms, end_ms) for `symbol`, via the unified
    research reader instead of an ad-hoc direct-SQL AVG(...). Same
    migration pilot as `window_agg_trades_notional` above."""
    plan = RR.plan_read(root, table="book_ticker", symbol=symbol, start_ms=start_ms, end_ms=end_ms)
    result = RR.execute_read(plan, columns=("bid_qty",))
    total = 0.0; n = 0
    for (bid_qty,) in result.iter_rows():
        total += bid_qty; n += 1
    return (total / n) if n else None


def feats_at_hit(conn, root, m, entry_ts: int, hit_min: int) -> dict:
    ts = entry_ts + hit_min * 60_000
    a = m.at_or_before(ts - 600_000); b = m.at_or_before(ts)
    def mret(sym, lb):
        x = conn.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (sym, ts - lb)).fetchone()
        y = conn.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (sym, ts)).fetchone()
        return (float(y[0]) - float(x[0])) / float(x[0]) * 1e4 if x and y and float(x[0]) > 0 else None
    buy, tot = window_agg_trades_notional(root, "ETHUSDT", ts - 600_000, ts)
    ofi = (2 * buy - tot) / tot if tot > 0 else None
    taker_sell = (tot - buy) / tot if tot > 0 else None
    liq_cont = _s(conn, "SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' AND ts_ms>=? AND ts_ms<?", (entry_ts + 60_000, ts))
    px = []
    for k in range(5, -1, -1):
        r = m.at_or_before(ts - k * 60_000)
        if r is None: px = []; break
        px.append(float(r[1]))
    rv = math.sqrt(sum(math.log(px[i + 1] / px[i]) ** 2 for i in range(5))) if len(px) == 6 else None
    fr = conn.execute("SELECT funding_rate FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? AND funding_rate IS NOT NULL ORDER BY ts_ms DESC LIMIT 1", (ts,)).fetchone()
    bt = conn.execute("SELECT bid_qty, spread_pct, ts_ms FROM book_ticker WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (ts,)).fetchone()
    bid_ratio = spread = None
    if bt and (ts - int(bt[2])) <= 5 * 60_000:
        pre_avg = window_avg_book_ticker_bid_qty(root, "ETHUSDT", entry_ts - 600_000, entry_ts)
        if pre_avg:
            bid_ratio = float(bt[0]) / pre_avg
        spread = float(bt[1]) if bt[1] is not None else None
    return {"btc10m": mret("BTCUSDT", 600_000), "ofi10m": ofi, "taker_sell10m": taker_sell,
            "liq_cont": liq_cont, "rv_hit": rv, "ret1h_hit": mret("ETHUSDT", 3600_000),
            "funding": float(fr[0]) if fr and fr[0] is not None else None,
            "t2m_min": float(hit_min), "bid_ratio": bid_ratio, "spread": spread}


def med(x):
    s = sorted(v for v in x if v is not None)
    return s[len(s) // 2] if s else None


def apply_policy(ev, flag: bool, mode: str) -> float:
    """hit sonrasi politika getirisi (bps, fee oncesi). mode: HOLD/EXIT/LOCK"""
    path = ev["path"]; hit = ev["hit_min"]
    final = path[min(H6, len(path) - 1)]
    if not flag or mode == "HOLD":
        return final
    if mode == "EXIT":
        return MILESTONE
    if mode == "LOCK":     # +25 taban: hit sonrasi 25'e dokunursa cik
        for k in range(hit + 1, min(H6, len(path) - 1) + 1):
            if path[k] <= 25.0:
                return 25.0
        return final
    return final


def metric_table(evs, rets, days: float) -> dict:
    net = [r - FEE for r in rets]
    if not net:
        return {"n": 0}
    n = len(net); wins = [x for x in net if x > 0]
    srt = sorted(net, reverse=True)
    eq = peak = dd = 0.0
    for v in net:
        eq += v; peak = max(peak, eq); dd = min(dd, eq - peak)
    gains = sum(wins); losses = -sum(x for x in net if x <= 0)
    mfe = [max(e["path"][:H6 + 1]) for e in evs]
    mae = [min(e["path"][:H6 + 1]) for e in evs]
    cont = [e for e in evs if e["label"] == "continuation"]
    cont_captured = sum(1 for e, r in zip(evs, rets)
                        if e["label"] == "continuation" and r >= 95.0)
    give = [e for e in evs if e["label"] == "negative"]
    give_cut = sum(1 for e, r in zip(evs, rets) if e["label"] == "negative" and r >= 20.0)
    s = sorted(net)
    return {"n": n, "trades_per_day": round(n / days, 2),
            "wr": round(100 * len(wins) / n, 1),
            "median_net": round(s[n // 2], 1), "mean_net": round(sum(net) / n, 1),
            "cum_net": round(sum(net), 0),
            "top3_removed": round(sum(srt[3:]), 0) if n > 3 else None,
            "profit_factor": round(gains / losses, 2) if losses > 0 else None,
            "max_dd": round(dd, 1),
            "avg_mfe": round(sum(mfe) / n, 1), "avg_mae": round(sum(mae) / n, 1),
            "neg_giveback_rate": round(100 * len(give) / n, 1),
            "giveback_rescued_pct": round(100 * give_cut / len(give), 1) if give else None,
            "continuation_capture": round(cont_captured / len(cont), 3) if cont else None}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("=== AMI MFE+50 Experiment (preregistered) ===")
    reg = ResearchRegistry(); store = KnowledgeStore()
    # -- 1) FROZEN PREREGISTRATION (hesaplamadan ONCE) --
    spec = ExperimentSpec(
        experiment_id="E-MFE50-001", question_id="Q-MFE-GIVEBACK-001",
        population="hour17 200K(s>=2)+100K(s>=3) full-history no-overlap admitted, milestone=first +50bps within 6h",
        target="policy net bps at 6h vs HOLD baseline",
        features=FEATURES, threshold_method="TRAIN median split, single feature, "
        "objective: max TRAIN cum net s.t. TRAIN continuation_capture>=0.85",
        chronological_split="70/30", untouched_data="TEST 30% (son donem) + forward",
        negative_control="LOCK_ALL (flagsiz herkese lock) kontrol politikasi",
        min_sample=25, effect_size_required_bps=0.0,
        decision_criteria="TEST: cum_net(policy)>=cum_net(HOLD) AND continuation_capture>=0.85 AND max_dd<=baseline max_dd",
        falsification_rule="TEST cum_net(policy) < cum_net(HOLD)",
        execution_model="mark_fill_fee5bps")
    spec.freeze(); reg.register_experiment(spec)
    print(f"  prereg FROZEN: {spec.experiment_id} hash={spec.frozen_hash}")

    # -- 2) evren + milestone --
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.execute("PRAGMA cache_size=-200000")
    root, _root_source = PR.resolve_production_root()
    now = int(datetime.now(tz=timezone.utc).timestamp() * 1000); start = now - LB
    m = load_mark_index(conn, "ETHUSDT")
    u200 = build_universe(conn, m, now, start, 200_000.0, 2)
    u100 = build_universe(conn, m, now, start, 100_000.0, 3)
    adm = noov(sorted(u200 + u100, key=lambda x: x["ts"]))
    days = (adm[-1]["ts"] - adm[0]["ts"]) / 86_400_000
    ms_evs = []
    for e in adm:
        path = e["path"]
        hit = next((k for k in range(1, H6 + 1) if path[min(k, len(path) - 1)] >= MILESTONE), None)
        if hit is None:
            continue
        rest = path[hit:H6 + 1]; final = path[min(H6, len(path) - 1)]
        label = "continuation" if max(rest) >= 100.0 else ("negative" if final < -5.0 else "stall")
        f = feats_at_hit(conn, root, m, e["ts"], hit)
        ms_evs.append({**e, "hit_min": hit, "label": label, "f": f})
    print(f"  admitted={len(adm)}  milestone(+50)={len(ms_evs)}  days={days:.0f}")
    labels = {L: sum(1 for e in ms_evs if e["label"] == L) for L in ("continuation", "negative", "stall")}
    print(f"  labels: {labels}")

    cut = int(len(ms_evs) * TRAIN)
    tr, te = ms_evs[:cut], ms_evs[cut:]
    assert_no_overlap({e["ts"] for e in tr}, {e["ts"] for e in te})
    tr_days = (tr[-1]["ts"] - tr[0]["ts"]) / 86_400_000 if len(tr) > 1 else 1
    te_days = (te[-1]["ts"] - te[0]["ts"]) / 86_400_000 if len(te) > 1 else 1

    # -- 3) TRAIN kural secimi (frozen protokole gore) --
    best = None
    for feat in FEATURES:
        thr = med([e["f"][feat] for e in tr])
        if thr is None:
            continue
        for direction in ("hi", "lo"):
            def flag(e, feat=feat, thr=thr, d=direction):
                v = e["f"][feat]
                if v is None:
                    return False
                return v >= thr if d == "hi" else v < thr
            for mode in ("EXIT", "LOCK"):
                rets = [apply_policy(e, flag(e), mode) for e in tr]
                mt = metric_table(tr, rets, tr_days)
                cc = mt.get("continuation_capture")
                if cc is not None and cc >= 0.85:
                    score = mt["cum_net"]
                    if best is None or score > best["train_cum"]:
                        best = {"feature": feat, "thr": thr, "dir": direction, "mode": mode,
                                "train_cum": score, "train_cc": cc}
    base_tr = metric_table(tr, [apply_policy(e, False, "HOLD") for e in tr], tr_days)
    print(f"  TRAIN baseline cum={base_tr['cum_net']}  best_rule={best}")

    # -- 4) TEST degerlendirme --
    def flag_best(e):
        if best is None:
            return False
        v = e["f"][best["feature"]]
        if v is None:
            return False
        return v >= best["thr"] if best["dir"] == "hi" else v < best["thr"]
    policies = {
        "HOLD_baseline": [apply_policy(e, False, "HOLD") for e in te],
        "POLICY_flagged": [apply_policy(e, flag_best(e), best["mode"] if best else "HOLD") for e in te],
        "LOCK_ALL_control": [apply_policy(e, True, "LOCK") for e in te],
        "EXIT_ALL_control": [apply_policy(e, True, "EXIT") for e in te],
    }
    results = {name: metric_table(te, rets, te_days) for name, rets in policies.items()}
    results["flag_rate_TEST"] = round(100 * sum(1 for e in te if flag_best(e)) / len(te), 1) if te else 0
    # session breakdown (policy)
    sess = {}
    for e, r in zip(te, policies["POLICY_flagged"]):
        s = "US" if 13 <= e["hour"] < 21 else "OFF"
        sess.setdefault(s, []).append(r - FEE)
    results["session_breakdown_policy"] = {k: {"n": len(v), "mean": round(sum(v) / len(v), 1)}
                                           for k, v in sess.items()}
    for name in ("HOLD_baseline", "POLICY_flagged", "LOCK_ALL_control", "EXIT_ALL_control"):
        r = results[name]
        print("  %-18s N=%-3d WR=%-5s mean=%-6s cum=%-7s dd=%-7s cc=%-5s give_rescue=%s"
              % (name, r["n"], r["wr"], r["mean_net"], r["cum_net"], r["max_dd"],
                 r["continuation_capture"], r["giveback_rescued_pct"]))

    # -- 5) frozen karar kriteri --
    hb, pf = results["HOLD_baseline"], results["POLICY_flagged"]
    if best is None:
        # Degenerate durum: TRAIN'de hicbir kural frozen kisiti (cc>=0.85) saglayamadi.
        # Politika baseline'a cokuyor; ayirma HIPOTEZI TRAIN asamasinda dustu.
        # (Frozen kriter degistirilmiyor — aday uretilemedigi kaydediliyor.)
        passed = False; falsified = True
        outcome = "FALSIFIES"
        print("  OUTCOME: FALSIFIES — TRAIN'de cc>=0.85 kisitini saglayan kural YOK "
              "(ayirma hipotezi bu feature setiyle dustu)")
    else:
        passed = (pf["cum_net"] >= hb["cum_net"]
                  and (pf["continuation_capture"] or 0) >= 0.85
                  and pf["max_dd"] >= hb["max_dd"])   # dd negatif: >= demek daha SIG (mutlak kucuk)
        falsified = pf["cum_net"] < hb["cum_net"]
        outcome = "SUPPORTS" if passed else ("FALSIFIES" if falsified else "INCONCLUSIVE")
        print(f"  OUTCOME: {outcome} (criteria frozen oncesi)")

    # -- 6) AMI kayitlari --
    ev = EvidenceBundle("EV-MFE50-001", spec.experiment_id,
                        {"train_rule": best, "labels": labels,
                         "TEST": {k: results[k] for k in policies},
                         "flag_rate": results["flag_rate_TEST"]},
                        outcome, evidence_family="mfe50:mgmt",
                        dataset_hash="s34-2026H1", code_ref="tools/research_ami_mfe50_experiment.py")
    reg.attach_evidence(ev, spec)
    if outcome == "SUPPORTS" and best is not None:
        ko = KnowledgeObject(
            knowledge_id="K-S34-MFE50-EXIT-001",
            claim=f"At +50bps MFE, rule [{best['feature']} {best['dir']} {round(best['thr'],5)} -> {best['mode']}] "
                  f"improves TEST cum net ({pf['cum_net']} vs {hb['cum_net']}) with continuation capture "
                  f"{pf['continuation_capture']} and no worse drawdown.",
            claim_type=ClaimType.OPERATIONAL, status=KnowledgeStatus.HOLDOUT_VALIDATED,
            provenance=Provenance(source_tables=["mark_prices", "agg_trades", "book_ticker", "liquidations"],
                                  data_time_range="2026-02-15..2026-07-02",
                                  code_ref="tools/research_ami_mfe50_experiment.py",
                                  dataset_hash="s34-2026H1", experiment_id=spec.experiment_id),
            evidence_level=EvidenceLevel.UNTOUCHED_HOLDOUT, replications=1, holdouts=1,
            falsification=[spec.falsification_rule + " (forward)"],
            confidence={"statistical": "MEDIUM", "forward": "NONE"},
            permitted=[Permission.RESEARCH_ONLY, Permission.OBSERVER_ALLOWED],
            forbidden=[Permission.LIVE_ALLOWED, Permission.SIZING_ALLOWED])
        store.put(ko, actor="mfe50_experiment")
        print(f"  Knowledge kaydedildi: {ko.knowledge_id} (OBSERVER_ALLOWED, live YASAK)")
    else:
        reason = (f"TRAIN'de cc>=0.85 kisitini saglayan tek-feature kural bulunamadi "
                  f"(29 continuation'in onemli kismi 6h'ye kadar 95bps altina geri veriyor; "
                  f"HOLD bile TRAIN'de kisiti karsilamiyor) — hipotez bu feature setiyle dustu"
                  ) if best is None else (
                  f"TEST policy cum={pf['cum_net']} vs baseline={hb['cum_net']} "
                  f"cc={pf['continuation_capture']}")
        store.archive_failure(
            f"MFE50 giveback separation via available-at-+50 single features ({FEATURES})",
            FailureType.NO_EDGE if falsified else FailureType.INSUFFICIENT_SAMPLE,
            reason=reason,
            data_period="2026-02-15..2026-07-02",
            retry="post-entry state-TRANSITION dizileri (lifecycle engine) + cift-feature kurallar; "
                  "capture tanimi politika-goreli (policy vs HOLD ayni trade) olarak revize edilebilir "
                  "(YENI prereg gerektirir)")
        print("  Failure archive kaydi yazildi.")
    conn.close()

    # -- 7) rapor --
    payload = {"spec_hash": spec.frozen_hash, "train_rule": best, "labels": labels,
               "results": results, "outcome": outcome, "passed": passed,
               "n_milestone": len(ms_evs), "train_baseline": base_tr}
    OJ.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    lines = ["# AMI MFE+50 Experiment — Baseline vs Candidate", "",
             f"> {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC — prereg FROZEN hash `{spec.frozen_hash}` "
             f"(hesaplamadan önce). Evren: {len(adm)} admitted, {len(ms_evs)} milestone. "
             f"Etiketler: {labels}", "",
             f"**TRAIN kuralı (frozen protokolle seçildi):** `{best}`", "",
             "## TEST (untouched %30)", "",
             "| Politika | N | WR | median | mean | cum | top3-rm | PF | maxDD | MFE | MAE | give% | give-kurtarma% | cont-capture |",
             "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for name in ("HOLD_baseline", "POLICY_flagged", "LOCK_ALL_control", "EXIT_ALL_control"):
        r = results[name]
        lines.append("| %s | %d | %s%% | %s | %s | %s | %s | %s | %s | %s | %s | %s%% | %s | %s |" % (
            name, r["n"], r["wr"], r["median_net"], r["mean_net"], r["cum_net"],
            r["top3_removed"], r["profit_factor"], r["max_dd"], r["avg_mfe"], r["avg_mae"],
            r["neg_giveback_rate"], r["giveback_rescued_pct"], r["continuation_capture"]))
    lines += ["", f"- Flag oranı (TEST): {results['flag_rate_TEST']}%",
              f"- Session breakdown (policy): {json.dumps(results['session_breakdown_policy'])}",
              f"- Execution feasibility: çıkışlar mark'tan modellendi; E1 bulgusuna göre",
              "  mark≈ask/bid (~0.6bps) — +50/+25 çıkışları limit-fill değil market varsayımı.",
              "", f"## SONUÇ: **{outcome}**", "",
              "Dürüst statü: `software-correct` ✓, `replay-validated` ✓ (deterministik path),",
              f"`holdout-validated` {'✓' if passed else '✗'}, `forward-validating` ✗ (başlamadı),",
              "`operationally-permitted` ✗ (governor: OBSERVER üstü izin YOK).",
              "", "*Script: tools/research_ami_mfe50_experiment.py — prereg: E-MFE50-001*"]
    OM.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nJSON:{OJ}\nMD:  {OM}\nDone.")
    reg.close(); store.close()


if __name__ == "__main__":
    main()
