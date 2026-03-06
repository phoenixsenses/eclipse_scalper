from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import rank_passive_pockets_forward as rp


def test_ranking_fee_priority_and_stability(monkeypatch) -> None:
    monkeypatch.setattr(
        rp,
        "_parse_candidates_from_md",
        lambda path, debug=False: (
            [
                {"symbol": "ETHUSDT", "horizon_sec": 60, "min_imbalance": 0.5, "min_trade_intensity": 2500.0, "max_spread": 0.00025},
                {"symbol": "BTCUSDT", "horizon_sec": 60, "min_imbalance": 0.5, "min_trade_intensity": 2500.0, "max_spread": 0.00025},
            ],
            {
                "total_rows_seen": 2,
                "table_rows_seen": 2,
                "rows_with_pass_yes": 2,
                "candidates_parsed": 2,
                "candidates_unique": 2,
                "rows_skipped_missing_fields": 0,
            },
        ),
    )

    def _fake_validate(**kwargs):
        sym = kwargs["symbol"]
        fee = float(kwargs["maker_fee_bps"])
        adv = float(kwargs.get("passive_adverse_mult", 1.0))
        rows = []
        for split in [1, 2]:
            for seed in [11, 22]:
                if sym == "ETHUSDT":
                    base = 0.00004 if fee == 1.0 and adv == 1.0 else 0.00001
                    if fee >= 1.5:
                        base = -0.00001
                else:
                    base = 0.00002 if fee == 0.5 else -0.00002
                # make BTC unstable
                if sym == "BTCUSDT" and seed == 22:
                    base -= 0.00003
                # net_per_attempt ≈ fill_rate * avg_net; val_attempts consistent with fill_rate=0.4
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "train_n": 100,
                        "val_n_rows": 100,
                        "effective_min_n": 50,
                        "filled_n": 80,
                        "filled_avg_net": base,
                        "filled_p90_net": 0.0001 if base > 0 else 0.0,
                        "filled_win_rate": 0.5,
                        "attempt_fill_rate": 0.4,
                        "net_per_attempt": base * 0.4,
                        "val_attempts": 200,
                        "val_filled": 80,
                        "attempts_per_min": 120.0,
                        "pass": bool(base >= 0.000005 and 0.0001 >= 0.00005),
                    }
                )
        pass_count = sum(1 for r in rows if r["pass"])
        return {
            "rows_total": len(rows),
            "pass_count": pass_count,
            "pass_rate": pass_count / len(rows),
            "insufficient_fill_rate": 0.0,
            "per_combo": rows,
        }

    monkeypatch.setattr(rp, "validate_pocket_forward", _fake_validate)
    out_md = Path("reports/test_rank_passive_pockets_forward.md")
    out_json = Path("reports/test_rank_passive_pockets_forward.json")
    argv = [
        "x",
        "--candidates-md",
        "reports/dummy.md",
        "--db",
        "data/microstructure.db",
        "--maker-fee-bps-grid",
        "0.5,1.0,1.5",
        "--min-n-frac",
        "0.0",
        "--passive-adverse-mult-grid",
        "0.8,1.0,1.2",
        "--out-md",
        str(out_md),
        "--out-json",
        str(out_json),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rc = rp.main()
    assert rc == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["run_summary"]["run_type"] == "rank_passive_pockets_forward"
    assert data["ranking"][0]["symbol"] == "ETHUSDT"
    assert "liquidation_scoring_impact" in data
    assert data["liquidation_scoring_impact"]["available"] is False


def test_liquidation_scoring_impact_summary(monkeypatch) -> None:
    monkeypatch.setattr(
        rp,
        "_parse_candidates_from_md",
        lambda path, debug=False: (
            [
                {"symbol": "ETHUSDT", "horizon_sec": 60, "min_imbalance": 0.5, "min_trade_intensity": 2500.0, "max_spread": 0.00025},
            ],
            {"total_rows_seen": 1, "table_rows_seen": 1, "rows_with_pass_yes": 1, "candidates_parsed": 1, "candidates_unique": 1, "rows_skipped_missing_fields": 0},
        ),
    )

    def _fake_validate(**kwargs):
        fee = float(kwargs["maker_fee_bps"])
        rule = str(kwargs["rule"])
        vol_quantile_reject = float(kwargs.get("vol_quantile_reject", 0.0) or 0.0)
        rows = []
        for split in [1, 2]:
            for seed in [11, 22]:
                base = 0.00001
                if rule == "micro_edge_v3_passive_alpha":
                    base = 0.00003 if fee <= 1.0 else 0.000015
                    if vol_quantile_reject > 0.0:
                        base += 0.00001
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "train_n": 100,
                        "val_n_rows": 100,
                        "effective_min_n": 20,
                        "filled_n": 60,
                        "filled_avg_net": base,
                        "filled_p90_net": 0.00010,
                        "filled_win_rate": 0.5,
                        "attempt_fill_rate": 0.4,
                        "net_per_attempt": base * 0.4,
                        "val_attempts": 150,
                        "val_filled": 60,
                        "attempts_per_min": 90.0,
                        "pass": True,
                    }
                )
        return {"rows_total": len(rows), "pass_count": len(rows), "pass_rate": 1.0, "insufficient_fill_rate": 0.0, "per_combo": rows}

    monkeypatch.setattr(rp, "validate_pocket_forward", _fake_validate)
    out_json = Path("reports/test_rank_liq_impact.json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--candidates-md",
            "reports/dummy.md",
            "--db",
            "data/microstructure.db",
            "--rule",
            "micro_edge_v3_passive_alpha",
            "--maker-fee-bps-grid",
            "1.0,1.5",
            "--passive-adverse-mult-grid",
            "1.0",
            "--min-attempt-fill-rate",
            "0.0",
            "--mitigation-profile",
            "anti_adverse_v3",
            "--out-md",
            "reports/test_rank_liq_impact.md",
            "--out-json",
            str(out_json),
        ],
    )
    rc = rp.main()
    assert rc == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    impact = data["liquidation_scoring_impact"]
    assert impact["available"] is True
    assert impact["count"] == 1
    assert impact["positive_delta_score_count"] == 1
    assert float(impact["avg_delta_score_raw_core"]) > 0.0


def test_anti_adverse_v5_passes_wait_and_scratch_overrides(monkeypatch) -> None:
    monkeypatch.setattr(
        rp,
        "_parse_candidates_from_md",
        lambda path, debug=False: (
            [
                {"symbol": "ETHUSDT", "horizon_sec": 60, "min_imbalance": 0.35, "min_trade_intensity": 200.0, "max_spread": 0.00035},
            ],
            {"total_rows_seen": 1, "table_rows_seen": 1, "rows_with_pass_yes": 1, "candidates_parsed": 1, "candidates_unique": 1, "rows_skipped_missing_fields": 0},
        ),
    )

    seen_waits: list[float] = []
    seen_scratch_bps: list[float] = []
    seen_scratch_windows: list[float] = []

    def _fake_validate(**kwargs):
        seen_waits.append(float(kwargs.get("passive_max_wait_buckets", -1)))
        seen_scratch_bps.append(float(kwargs.get("scratch_bps", -1.0)))
        seen_scratch_windows.append(float(kwargs.get("scratch_window_sec", -1)))
        return {
            "rows_total": 1,
            "pass_count": 1,
            "pass_rate": 1.0,
            "insufficient_fill_rate": 0.0,
            "per_combo": [
                {
                    "seed": 11,
                    "split": 1,
                    "train_n": 100,
                    "val_n_rows": 100,
                    "effective_min_n": 20,
                    "filled_n": 30,
                    "filled_avg_net": 0.00005,
                    "filled_p90_net": 0.00010,
                    "filled_win_rate": 0.5,
                    "attempt_fill_rate": 0.5,
                    "net_per_attempt": 0.000025,
                    "val_attempts": 60,
                    "val_filled": 30,
                    "attempts_per_min": 20.0,
                    "pass": True,
                    "fail_reason": "ok",
                }
            ],
            "failure_attribution_median": {},
        }

    monkeypatch.setattr(rp, "validate_pocket_forward", _fake_validate)
    out_json = Path("reports/test_rank_anti_adverse_v5.json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--candidates-md",
            "reports/dummy.md",
            "--db",
            "data/microstructure.db",
            "--rule",
            "high_liq_reversal_regime",
            "--maker-fee-bps-grid",
            "1.0",
            "--passive-adverse-mult-grid",
            "1.0",
            "--min-attempt-fill-rate",
            "0.0",
            "--mitigation-profile",
            "anti_adverse_v5",
            "--out-md",
            "reports/test_rank_anti_adverse_v5.md",
            "--out-json",
            str(out_json),
        ],
    )
    rc = rp.main()
    assert rc == 0
    assert 2.0 in seen_waits
    assert 4.0 in seen_scratch_bps
    assert 10.0 in seen_scratch_windows
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["gate_config"]["passive_max_wait_buckets"] == 2


def test_parse_candidates_md_v2_style() -> None:
    p = Path("reports/test_rank_candidates_v2_style.md")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "\n".join(
            [
                "# FILTER_SWEEP_PASSIVE_REALISTIC",
                "",
                "| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | v2_min_score | v2_min_persistence | filled_n | filled_avg_net | filled_p90_net | filled_win_rate | attempt_fill_rate | pass |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
                "| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | 0.000000 | 0.400000 | 67 | +0.000233 | +0.001711 | 47.76% | 48.20% | YES |",
                "| BTCUSDT | 60 | 0.40 | 1500 | 0.000300 | 0.500000 | 0.200000 | 80 | +0.000100 | +0.001000 | 51.00% | 40.00% | true |",
                "| BTCUSDT | 60 | 0.40 | 1500 | 0.000300 | 0.500000 | 0.200000 | 80 | +0.000100 | +0.001000 | 51.00% | 40.00% | 1 |",
                "| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | 0.000000 | 0.400000 | 67 | +0.000233 | +0.001711 | 47.76% | 48.20% | NO |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    candidates, stats = rp._parse_candidates_from_md(p, debug=False)
    assert len(candidates) > 0, f"expected >0 parsed candidates, got 0 with stats={stats}"
    # dedupe should collapse duplicated BTC row with YES/true/1
    assert len(candidates) == 2
    assert int(stats["rows_with_pass_yes"]) >= 3


def test_main_returns_error_when_no_candidates(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        rp,
        "_parse_candidates_from_md",
        lambda path, debug=False: (
            [],
            {
                "total_rows_seen": 10,
                "table_rows_seen": 2,
                "rows_with_pass_yes": 0,
                "candidates_parsed": 0,
                "candidates_unique": 0,
                "rows_skipped_missing_fields": 1,
            },
        ),
    )
    argv = [
        "x",
        "--candidates-md",
        "reports/none.md",
        "--db",
        "data/microstructure.db",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rc = rp.main()
    out = capsys.readouterr().out
    assert rc == 2
    assert "ERROR no candidates parsed" in out


def test_rank_skip_message_includes_effective_min_hint(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        rp,
        "_parse_candidates_from_md",
        lambda path, debug=False: (
            [
                {"symbol": "ETHUSDT", "horizon_sec": 60, "min_imbalance": 0.5, "min_trade_intensity": 2500.0, "max_spread": 0.00025},
            ],
            {"total_rows_seen": 1, "table_rows_seen": 1, "rows_with_pass_yes": 1, "candidates_parsed": 1, "candidates_unique": 1, "rows_skipped_missing_fields": 0},
        ),
    )

    def _fake_validate(**kwargs):
        rows = []
        for split in [1, 2]:
            for seed in [11, 22]:
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "filled_n": 40,
                        "filled_avg_net": 0.00002,
                        "filled_p90_net": 0.0001,
                        "attempt_fill_rate": 0.40,
                        "net_per_attempt": -0.00001,
                        "attempts_per_min": 1.2,
                        "effective_min_n": 212,
                        "pass": False,
                    }
                )
        return {"rows_total": len(rows), "pass_count": 0, "pass_rate": 0.0, "insufficient_fill_rate": 1.0, "per_combo": rows}

    monkeypatch.setattr(rp, "validate_pocket_forward", _fake_validate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--candidates-md",
            "reports/dummy.md",
            "--db",
            "data/microstructure.db",
            "--maker-fee-bps-grid",
            "1.0",
            "--passive-adverse-mult-grid",
            "1.0",
            "--max-insufficient-fill-rate",
            "0.50",
            "--min-n-frac",
            "0.00025",
            "--min-n",
            "50",
            "--out-md",
            "reports/test_rank_skip_hint.md",
            "--out-json",
            "reports/test_rank_skip_hint.json",
        ],
    )
    rc = rp.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "effective_min_n_median=212" in out
    assert "Try lower --min-n-frac" in out
    assert "current min_n_frac=0.000250" in out


def test_research_mode_sets_soft_threshold(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        rp,
        "_parse_candidates_from_md",
        lambda path, debug=False: (
            [
                {"symbol": "ETHUSDT", "horizon_sec": 60, "min_imbalance": 0.5, "min_trade_intensity": 2500.0, "max_spread": 0.00025},
            ],
            {"total_rows_seen": 1, "table_rows_seen": 1, "rows_with_pass_yes": 1, "candidates_parsed": 1, "candidates_unique": 1, "rows_skipped_missing_fields": 0},
        ),
    )

    def _fake_validate(**kwargs):
        rows = []
        for split in [1, 2]:
            for seed in [11, 22]:
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "filled_n": 80,
                        "filled_avg_net": 0.00002,
                        "filled_p90_net": 0.0001,
                        "attempt_fill_rate": 0.40,
                        "net_per_attempt": 0.00001,
                        "attempts_per_min": 1.2,
                        "effective_min_n": 30,
                        "pass": True if seed == 11 else False,
                    }
                )
        return {"rows_total": len(rows), "pass_count": 2, "pass_rate": 0.5, "insufficient_fill_rate": 0.0, "per_combo": rows}

    monkeypatch.setattr(rp, "validate_pocket_forward", _fake_validate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--candidates-md",
            "reports/dummy.md",
            "--db",
            "data/microstructure.db",
            "--maker-fee-bps-grid",
            "1.0",
            "--passive-adverse-mult-grid",
            "1.0,1.2",
            "--research-mode",
            "--out-md",
            "reports/test_rank_research.md",
            "--out-json",
            "reports/test_rank_research.json",
        ],
    )
    rc = rp.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "RESEARCH MODE enabled: pass_threshold=0.33" in out


def test_ranker_filters_low_capacity_pocket(monkeypatch) -> None:
    """Pocket whose core-eval attempt_fill_rate < --min-attempt-fill-rate must be excluded."""
    monkeypatch.setattr(
        rp,
        "_parse_candidates_from_md",
        lambda path, debug=False: (
            [{"symbol": "ETHUSDT", "horizon_sec": 60, "min_imbalance": 0.5, "min_trade_intensity": 2500.0, "max_spread": 0.00025}],
            {"total_rows_seen": 1, "table_rows_seen": 1, "rows_with_pass_yes": 1, "candidates_parsed": 1, "candidates_unique": 1, "rows_skipped_missing_fields": 0},
        ),
    )

    def _low_fill_validate(**kwargs):
        rows = []
        for split in [1, 2]:
            for seed in [11, 22]:
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "train_n": 100,
                        "val_n_rows": 100,
                        "effective_min_n": 50,
                        "filled_n": 5,
                        "filled_avg_net": 0.00005,
                        "filled_p90_net": 0.0001,
                        "filled_win_rate": 0.6,
                        "attempt_fill_rate": 0.04,   # well below default 0.10 threshold
                        "net_per_attempt": 0.000002,
                        "val_attempts": 100,
                        "val_filled": 5,
                        "attempts_per_min": 60.0,
                        "pass": True,
                    }
                )
        return {"rows_total": len(rows), "pass_count": len(rows), "pass_rate": 1.0, "insufficient_fill_rate": 0.0, "per_combo": rows}

    monkeypatch.setattr(rp, "validate_pocket_forward", _low_fill_validate)
    out_md = Path("reports/test_ranker_filter_cap.md")
    out_json = Path("reports/test_ranker_filter_cap.json")
    argv = [
        "x", "--candidates-md", "reports/dummy.md", "--db", "data/microstructure.db",
        "--maker-fee-bps-grid", "0.5,1.0,1.5",
        "--passive-adverse-mult-grid", "0.8,1.0,1.2",
        "--min-attempt-fill-rate", "0.10",
        "--out-md", str(out_md),
        "--out-json", str(out_json),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rc = rp.main()
    assert rc == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["count"] == 0, f"Expected pocket filtered out, got count={data['count']}"


def test_score_raw_fields_present_in_output(monkeypatch) -> None:
    """score_raw_core/stress/min must be present in JSON even when score is 0."""
    monkeypatch.setattr(
        rp,
        "_parse_candidates_from_md",
        lambda path, debug=False: (
            [{"symbol": "ETHUSDT", "horizon_sec": 60, "min_imbalance": 0.5, "min_trade_intensity": 2500.0, "max_spread": 0.00025}],
            {"total_rows_seen": 1, "table_rows_seen": 1, "rows_with_pass_yes": 1, "candidates_parsed": 1, "candidates_unique": 1, "rows_skipped_missing_fields": 0},
        ),
    )

    def _marginal_validate(**kwargs):
        fee = float(kwargs["maker_fee_bps"])
        base = 0.00002 if fee <= 0.5 else -0.00001   # positive only at lowest fee
        rows = []
        for split in [1, 2]:
            for seed in [11, 22]:
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "train_n": 100,
                        "val_n_rows": 100,
                        "effective_min_n": 50,
                        "filled_n": 60,
                        "filled_avg_net": base,
                        "filled_p90_net": 0.0001 if base > 0 else 0.0,
                        "filled_win_rate": 0.5,
                        "attempt_fill_rate": 0.4,
                        "net_per_attempt": base * 0.4,
                        "val_attempts": 150,
                        "val_filled": 60,
                        "attempts_per_min": 90.0,
                        "pass": base > 0,
                    }
                )
        return {"rows_total": len(rows), "pass_count": sum(1 for r in rows if r["pass"]), "pass_rate": 0.5, "insufficient_fill_rate": 0.0, "per_combo": rows}

    monkeypatch.setattr(rp, "validate_pocket_forward", _marginal_validate)
    out_md = Path("reports/test_score_raw.md")
    out_json = Path("reports/test_score_raw.json")
    argv = [
        "x", "--candidates-md", "reports/dummy.md", "--db", "data/microstructure.db",
        "--maker-fee-bps-grid", "0.5,1.0,1.5",
        "--passive-adverse-mult-grid", "0.8,1.0,1.2",
        "--min-attempt-fill-rate", "0.0",   # disable capacity filter so pocket appears in output
        "--out-md", str(out_md),
        "--out-json", str(out_json),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rc = rp.main()
    assert rc == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["count"] >= 1
    r = data["ranking"][0]
    assert "score_raw_core" in r
    assert "score_raw_stress" in r
    assert "score_raw_min" in r
    assert "net_per_attempt" in r
    assert "attempt_fill_rate" in r
    assert "attempts_per_min" in r
    # score_raw_core should be positive (eval at fee_min=0.5, adv=1.0 has base=0.00002)
    assert float(r["score_raw_core"]) > 0.0
    # score_raw_stress should be negative (eval at fee_max=1.5, adv_max=1.2 has base=-0.00001)
    assert float(r["score_raw_stress"]) < 0.0


def test_anti_adverse_v2_keeps_pass_core_within_tolerance(monkeypatch) -> None:
    """anti_adverse_v2 must remain light-touch for high-quality pocket."""
    monkeypatch.setattr(
        rp,
        "_parse_candidates_from_md",
        lambda path, debug=False: (
            [{"symbol": "ETHUSDT", "horizon_sec": 120, "min_imbalance": 0.50, "min_trade_intensity": 3500.0, "max_spread": 0.000300}],
            {"total_rows_seen": 1, "table_rows_seen": 1, "rows_with_pass_yes": 1, "candidates_parsed": 1, "candidates_unique": 1, "rows_skipped_missing_fields": 0},
        ),
    )

    def _fake_validate(**kwargs):
        # v2 gate present with looser threshold should remain light-touch.
        vol_thr = float(kwargs.get("max_volatility_extreme", 0.0) or 0.0)
        has_v2_gate = vol_thr > 0.0
        pass_rate = 0.50 if (not has_v2_gate or vol_thr >= 0.006) else 0.45
        rows = []
        for split in [1, 2]:
            for seed in [11, 22]:
                is_pass = (seed == 11) if not has_v2_gate else (split == 1 or seed == 11)
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "train_n": 100,
                        "val_n_rows": 100,
                        "effective_min_n": 30,
                        "filled_n": 60,
                        "filled_avg_net": 0.00002,
                        "filled_p90_net": 0.00010,
                        "filled_win_rate": 0.55,
                        "attempt_fill_rate": 0.45,
                        "net_per_attempt": 0.00001,
                        "val_attempts": 133,
                        "val_filled": 60,
                        "attempts_per_min": 80.0,
                        "pass": bool(is_pass),
                    }
                )
        return {"rows_total": len(rows), "pass_count": int(round(pass_rate * len(rows))), "pass_rate": pass_rate, "insufficient_fill_rate": 0.0, "per_combo": rows}

    monkeypatch.setattr(rp, "validate_pocket_forward", _fake_validate)

    out_baseline = Path("reports/test_rank_v2_tolerance_baseline.json")
    out_v2 = Path("reports/test_rank_v2_tolerance_v2.json")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x", "--candidates-md", "reports/dummy.md", "--db", "data/microstructure.db",
            "--maker-fee-bps-grid", "1.0", "--passive-adverse-mult-grid", "1.0",
            "--min-attempt-fill-rate", "0.0", "--out-md", "reports/test_rank_v2_tolerance_baseline.md",
            "--out-json", str(out_baseline), "--mitigation-profile", "baseline",
        ],
    )
    rc = rp.main()
    assert rc == 0
    base = json.loads(out_baseline.read_text(encoding="utf-8"))["ranking"][0]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x", "--candidates-md", "reports/dummy.md", "--db", "data/microstructure.db",
            "--maker-fee-bps-grid", "1.0", "--passive-adverse-mult-grid", "1.0",
            "--min-attempt-fill-rate", "0.0", "--out-md", "reports/test_rank_v2_tolerance_v2.md",
            "--out-json", str(out_v2), "--mitigation-profile", "anti_adverse_v2",
            "--max-volatility-extreme", "0.0060",
        ],
    )
    rc = rp.main()
    assert rc == 0
    v2 = json.loads(out_v2.read_text(encoding="utf-8"))["ranking"][0]

    base_pass_core = float(base.get("pass_rate_core", 0.0))
    v2_pass_core = float(v2.get("pass_rate_core", 0.0))
    assert (base_pass_core - v2_pass_core) <= 0.05


def test_help_includes_new_volatility_flags(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["x", "--help"])
    try:
        rp.main()
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "--max-volatility-extreme" in out
    assert "--vol-quantile-reject" in out
    assert "--horizon-sec" in out
    assert "--scratch-bps" in out
    assert "--scratch-window-sec" in out
    assert "--scratch-taker-fee-bps" in out
    assert "--scratch-slippage-bps" in out
    assert "--diagnostic-breakdown" in out
    assert "--emit-fee-cliff-summary" in out
    assert "--only-pocket" in out
    assert "--only-pocket-regex" in out
    assert "anti_adverse_v3" in out
    assert "anti_adverse_v4" in out


def test_only_pocket_exact_filters_candidates(monkeypatch) -> None:
    monkeypatch.setattr(
        rp,
        "_parse_candidates_from_md",
        lambda path, debug=False: (
            [
                {"symbol": "ETHUSDT", "horizon_sec": 120, "min_imbalance": 0.50, "min_trade_intensity": 2500.0, "max_spread": 0.000300},
                {"symbol": "BTCUSDT", "horizon_sec": 120, "min_imbalance": 0.50, "min_trade_intensity": 2500.0, "max_spread": 0.000300},
            ],
            {"total_rows_seen": 2, "table_rows_seen": 2, "rows_with_pass_yes": 2, "candidates_parsed": 2, "candidates_unique": 2, "rows_skipped_missing_fields": 0},
        ),
    )
    seen_symbols: list[str] = []

    def _fake_validate(**kwargs):
        seen_symbols.append(str(kwargs["symbol"]))
        rows = [{"seed": 11, "split": 1, "filled_n": 60, "filled_avg_net": 0.00002, "filled_p90_net": 0.00010, "attempt_fill_rate": 0.45, "net_per_attempt": 0.00001, "attempts_per_min": 80.0, "effective_min_n": 20, "pass": True}]
        return {"rows_total": 1, "pass_count": 1, "pass_rate": 1.0, "insufficient_fill_rate": 0.0, "per_combo": rows}

    monkeypatch.setattr(rp, "validate_pocket_forward", _fake_validate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x", "--candidates-md", "reports/dummy.md", "--db", "data/microstructure.db",
            "--maker-fee-bps-grid", "1.0", "--passive-adverse-mult-grid", "1.0",
            "--only-pocket", "ETHUSDT",
            "--out-md", "reports/test_rank_only_pocket_exact.md",
            "--out-json", "reports/test_rank_only_pocket_exact.json",
        ],
    )
    rc = rp.main()
    assert rc == 0
    assert seen_symbols and set(seen_symbols) == {"ETHUSDT"}


def test_only_pocket_regex_filters_candidates_and_empty_is_error(monkeypatch) -> None:
    monkeypatch.setattr(
        rp,
        "_parse_candidates_from_md",
        lambda path, debug=False: (
            [
                {"symbol": "ETHUSDT", "horizon_sec": 120, "min_imbalance": 0.50, "min_trade_intensity": 2500.0, "max_spread": 0.000300},
                {"symbol": "BTCUSDT", "horizon_sec": 60, "min_imbalance": 0.40, "min_trade_intensity": 1500.0, "max_spread": 0.000500},
            ],
            {"total_rows_seen": 2, "table_rows_seen": 2, "rows_with_pass_yes": 2, "candidates_parsed": 2, "candidates_unique": 2, "rows_skipped_missing_fields": 0},
        ),
    )
    seen_symbols: list[str] = []

    def _fake_validate(**kwargs):
        seen_symbols.append(str(kwargs["symbol"]))
        rows = [{"seed": 11, "split": 1, "filled_n": 60, "filled_avg_net": 0.00002, "filled_p90_net": 0.00010, "attempt_fill_rate": 0.45, "net_per_attempt": 0.00001, "attempts_per_min": 80.0, "effective_min_n": 20, "pass": True}]
        return {"rows_total": 1, "pass_count": 1, "pass_rate": 1.0, "insufficient_fill_rate": 0.0, "per_combo": rows}

    monkeypatch.setattr(rp, "validate_pocket_forward", _fake_validate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x", "--candidates-md", "reports/dummy.md", "--db", "data/microstructure.db",
            "--maker-fee-bps-grid", "1.0", "--passive-adverse-mult-grid", "1.0",
            "--only-pocket-regex", "BTCUSDT.*h=60",
            "--out-md", "reports/test_rank_only_pocket_regex.md",
            "--out-json", "reports/test_rank_only_pocket_regex.json",
        ],
    )
    rc = rp.main()
    assert rc == 0
    assert seen_symbols and set(seen_symbols) == {"BTCUSDT"}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x", "--candidates-md", "reports/dummy.md", "--db", "data/microstructure.db",
            "--only-pocket-regex", "DOES_NOT_MATCH",
        ],
    )
    rc2 = rp.main()
    assert rc2 == 2


def test_ranker_passes_max_volatility_extreme_to_validator(monkeypatch) -> None:
    monkeypatch.setattr(
        rp,
        "_parse_candidates_from_md",
        lambda path, debug=False: (
            [{"symbol": "ETHUSDT", "horizon_sec": 60, "min_imbalance": 0.5, "min_trade_intensity": 2500.0, "max_spread": 0.00025}],
            {"total_rows_seen": 1, "table_rows_seen": 1, "rows_with_pass_yes": 1, "candidates_parsed": 1, "candidates_unique": 1, "rows_skipped_missing_fields": 0},
        ),
    )
    seen = {"vol": []}

    def _fake_validate(**kwargs):
        seen["vol"].append(float(kwargs.get("max_volatility_extreme", 0.0) or 0.0))
        rows = []
        for split in [1, 2]:
            for seed in [11, 22]:
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "train_n": 100,
                        "val_n_rows": 100,
                        "effective_min_n": 30,
                        "filled_n": 60,
                        "filled_avg_net": 0.00002,
                        "filled_p90_net": 0.00010,
                        "filled_win_rate": 0.55,
                        "attempt_fill_rate": 0.45,
                        "net_per_attempt": 0.00001,
                        "val_attempts": 133,
                        "val_filled": 60,
                        "attempts_per_min": 80.0,
                        "pass": True,
                    }
                )
        return {"rows_total": len(rows), "pass_count": len(rows), "pass_rate": 1.0, "insufficient_fill_rate": 0.0, "per_combo": rows}

    monkeypatch.setattr(rp, "validate_pocket_forward", _fake_validate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--candidates-md",
            "reports/dummy.md",
            "--db",
            "data/microstructure.db",
            "--maker-fee-bps-grid",
            "1.0",
            "--passive-adverse-mult-grid",
            "1.0",
            "--mitigation-profile",
            "anti_adverse_v2",
            "--max-volatility-extreme",
            "0.0040",
            "--out-md",
            "reports/test_rank_v2_passthrough.md",
            "--out-json",
            "reports/test_rank_v2_passthrough.json",
        ],
    )
    rc = rp.main()
    assert rc == 0
    assert len(seen["vol"]) > 0
    assert all(abs(v - 0.0040) < 1e-12 for v in seen["vol"])


def test_ranker_passes_horizon_and_scratch_to_validator(monkeypatch) -> None:
    monkeypatch.setattr(
        rp,
        "_parse_candidates_from_md",
        lambda path, debug=False: (
            [{"symbol": "ETHUSDT", "horizon_sec": 120, "min_imbalance": 0.5, "min_trade_intensity": 2500.0, "max_spread": 0.00025}],
            {"total_rows_seen": 1, "table_rows_seen": 1, "rows_with_pass_yes": 1, "candidates_parsed": 1, "candidates_unique": 1, "rows_skipped_missing_fields": 0},
        ),
    )
    seen: dict[str, list[float | int]] = {"h": [], "sb": [], "sw": [], "stf": [], "ssl": []}

    def _fake_validate(**kwargs):
        seen["h"].append(int(kwargs.get("horizon_sec", 0)))
        seen["sb"].append(float(kwargs.get("scratch_bps", 0.0)))
        seen["sw"].append(int(kwargs.get("scratch_window_sec", 0)))
        seen["stf"].append(float(kwargs.get("scratch_taker_fee_bps", 0.0)))
        seen["ssl"].append(float(kwargs.get("scratch_slippage_bps", 0.0)))
        rows = []
        for split in [1, 2]:
            for seed in [11, 22]:
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "train_n": 100,
                        "val_n_rows": 100,
                        "effective_min_n": 30,
                        "filled_n": 60,
                        "filled_avg_net": 0.00002,
                        "filled_p90_net": 0.00010,
                        "filled_win_rate": 0.55,
                        "attempt_fill_rate": 0.45,
                        "net_per_attempt": 0.00001,
                        "val_attempts": 133,
                        "val_filled": 60,
                        "attempts_per_min": 80.0,
                        "pass": True,
                    }
                )
        return {"rows_total": len(rows), "pass_count": len(rows), "pass_rate": 1.0, "insufficient_fill_rate": 0.0, "per_combo": rows}

    monkeypatch.setattr(rp, "validate_pocket_forward", _fake_validate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--candidates-md",
            "reports/dummy.md",
            "--db",
            "data/microstructure.db",
            "--maker-fee-bps-grid",
            "1.0",
            "--passive-adverse-mult-grid",
            "1.0",
            "--horizon-sec",
            "60",
            "--scratch-bps",
            "4.0",
            "--scratch-window-sec",
            "10",
            "--scratch-taker-fee-bps",
            "1.0",
            "--scratch-slippage-bps",
            "0.5",
            "--out-md",
            "reports/test_rank_scratch_passthrough.md",
            "--out-json",
            "reports/test_rank_scratch_passthrough.json",
        ],
    )
    rc = rp.main()
    assert rc == 0
    assert seen["h"] and all(int(v) == 60 for v in seen["h"])
    assert seen["sb"] and all(abs(float(v) - 4.0) < 1e-12 for v in seen["sb"])
    assert seen["sw"] and all(int(v) == 10 for v in seen["sw"])
    assert seen["stf"] and all(abs(float(v) - 1.0) < 1e-12 for v in seen["stf"])
    assert seen["ssl"] and all(abs(float(v) - 0.5) < 1e-12 for v in seen["ssl"])


def test_emit_fee_cliff_summary_writes_sidecar(monkeypatch) -> None:
    monkeypatch.setattr(
        rp,
        "_parse_candidates_from_md",
        lambda path, debug=False: (
            [{"symbol": "ETHUSDT", "horizon_sec": 120, "min_imbalance": 0.50, "min_trade_intensity": 2500.0, "max_spread": 0.000300}],
            {"total_rows_seen": 1, "table_rows_seen": 1, "rows_with_pass_yes": 1, "candidates_parsed": 1, "candidates_unique": 1, "rows_skipped_missing_fields": 0},
        ),
    )

    def _fake_validate(**kwargs):
        rows = [{"seed": 11, "split": 1, "filled_n": 60, "filled_avg_net": 0.00002, "filled_p90_net": 0.00010, "attempt_fill_rate": 0.45, "net_per_attempt": 0.00001, "attempts_per_min": 80.0, "effective_min_n": 20, "pass": True}]
        return {
            "rows_total": 1,
            "pass_count": 1,
            "pass_rate": 1.0,
            "insufficient_fill_rate": 0.0,
            "per_combo": rows,
            "failure_attribution_median": {
                "n_events_total": 100,
                "n_rejected_attempt_gate": 10,
                "n_attempts_after_gate": 90,
                "n_filled": 45,
                "n_unfilled": 45,
                "avg_fill_prob": 0.5,
                "avg_adverse_bps_on_fills": 1.0,
                "avg_fee_bps": 0.8,
                "avg_scratch_bps_on_fills": 0.2,
                "avg_raw_return_bps_on_fills": 1.5,
                "avg_net_return_bps_on_fills": -0.5,
                "net_return_bps_p10": -2.0,
                "net_return_bps_p50": -0.5,
                "net_return_bps_p90": 1.0,
                "reject_vol_quantile_reject": 5,
                "reject_spread_too_wide": 3,
                "reject_imbalance_too_low": 1,
                "reject_intensity_too_low": 1,
                "reject_other_gate": 0,
            },
        }

    monkeypatch.setattr(rp, "validate_pocket_forward", _fake_validate)
    out_json = Path("reports/test_rank_emit_fee_cliff.json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--candidates-md",
            "reports/dummy.md",
            "--db",
            "data/microstructure.db",
            "--maker-fee-bps-grid",
            "0.8,1.0",
            "--passive-adverse-mult-grid",
            "1.0,1.2",
            "--emit-fee-cliff-summary",
            "--out-md",
            "reports/test_rank_emit_fee_cliff.md",
            "--out-json",
            str(out_json),
        ],
    )
    rc = rp.main()
    assert rc == 0
    sidecar = out_json.with_name(f"{out_json.stem}.fee_cliff.json")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert "rows" in payload and len(payload["rows"]) >= 1


def test_anti_adverse_v4_applies_default_scratch(monkeypatch) -> None:
    monkeypatch.setattr(
        rp,
        "_parse_candidates_from_md",
        lambda path, debug=False: (
            [{"symbol": "ETHUSDT", "horizon_sec": 120, "min_imbalance": 0.5, "min_trade_intensity": 2500.0, "max_spread": 0.00025}],
            {"total_rows_seen": 1, "table_rows_seen": 1, "rows_with_pass_yes": 1, "candidates_parsed": 1, "candidates_unique": 1, "rows_skipped_missing_fields": 0},
        ),
    )
    seen: dict[str, list[float]] = {"sb": [], "sw": [], "stf": [], "ssl": [], "vqr": []}

    def _fake_validate(**kwargs):
        seen["sb"].append(float(kwargs.get("scratch_bps", 0.0)))
        seen["sw"].append(float(kwargs.get("scratch_window_sec", 0.0)))
        seen["stf"].append(float(kwargs.get("scratch_taker_fee_bps", 0.0)))
        seen["ssl"].append(float(kwargs.get("scratch_slippage_bps", 0.0)))
        seen["vqr"].append(float(kwargs.get("vol_quantile_reject", 0.0)))
        rows = [{"seed": 11, "split": 1, "filled_n": 60, "filled_avg_net": 0.00002, "filled_p90_net": 0.00010, "attempt_fill_rate": 0.45, "net_per_attempt": 0.00001, "attempts_per_min": 80.0, "effective_min_n": 20, "pass": True}]
        return {"rows_total": 1, "pass_count": 1, "pass_rate": 1.0, "insufficient_fill_rate": 0.0, "per_combo": rows}

    monkeypatch.setattr(rp, "validate_pocket_forward", _fake_validate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--candidates-md",
            "reports/dummy.md",
            "--db",
            "data/microstructure.db",
            "--maker-fee-bps-grid",
            "1.0",
            "--passive-adverse-mult-grid",
            "1.0",
            "--mitigation-profile",
            "anti_adverse_v4",
            "--out-md",
            "reports/test_rank_v4_defaults.md",
            "--out-json",
            "reports/test_rank_v4_defaults.json",
        ],
    )
    rc = rp.main()
    assert rc == 0
    assert seen["sb"] and any(abs(v - 4.0) < 1e-12 for v in seen["sb"])
    assert seen["sw"] and any(abs(v - 10.0) < 1e-12 for v in seen["sw"])
    assert seen["stf"] and any(abs(v - 1.0) < 1e-12 for v in seen["stf"])
    assert seen["ssl"] and any(abs(v - 0.5) < 1e-12 for v in seen["ssl"])
    assert seen["vqr"] and any(abs(v - 0.01) < 1e-12 for v in seen["vqr"])


def test_failure_reason_top_enum(monkeypatch) -> None:
    monkeypatch.setattr(
        rp,
        "_parse_candidates_from_md",
        lambda path, debug=False: (
            [{"symbol": "ETHUSDT", "horizon_sec": 60, "min_imbalance": 0.5, "min_trade_intensity": 2500.0, "max_spread": 0.00025}],
            {"total_rows_seen": 1, "table_rows_seen": 1, "rows_with_pass_yes": 1, "candidates_parsed": 1, "candidates_unique": 1, "rows_skipped_missing_fields": 0},
        ),
    )

    def _fake_validate(**kwargs):
        rows = []
        for split in [1, 2]:
            for seed in [11, 22]:
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "filled_n": 60,
                        "filled_avg_net": -0.00002,
                        "filled_p90_net": -0.00001,
                        "attempt_fill_rate": 0.5,
                        "net_per_attempt": -0.00001,
                        "attempts_per_min": 1.0,
                        "effective_min_n": 20,
                        "pass": False,
                        "n_events_total": 100,
                        "n_rejected_attempt_gate": 10,
                        "n_attempts_after_gate": 90,
                        "n_filled": 60,
                        "n_unfilled": 30,
                        "avg_adverse_bps_on_fills": 2.5,
                        "avg_fee_bps": 1.0,
                        "avg_raw_return_bps_on_fills": -1.0,
                        "avg_net_return_bps_on_fills": -3.5,
                    }
                )
        return {"rows_total": len(rows), "pass_count": 0, "pass_rate": 0.0, "insufficient_fill_rate": 0.0, "per_combo": rows}

    monkeypatch.setattr(rp, "validate_pocket_forward", _fake_validate)
    out_json = Path("reports/test_rank_failure_reason_enum.json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--candidates-md",
            "reports/dummy.md",
            "--db",
            "data/microstructure.db",
            "--maker-fee-bps-grid",
            "1.0",
            "--passive-adverse-mult-grid",
            "1.2",
            "--min-attempt-fill-rate",
            "0.0",
            "--out-md",
            "reports/test_rank_failure_reason_enum.md",
            "--out-json",
            str(out_json),
        ],
    )
    rc = rp.main()
    assert rc == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["count"] >= 1
    reason = str(data["ranking"][0].get("failure_reason_top", ""))
    assert reason in {"gate_reject", "no_fills", "adverse_dominates", "fees_dominate", "mixed"}


def test_attribution_fields_merged_into_ranking_row(monkeypatch) -> None:
    monkeypatch.setattr(
        rp,
        "_parse_candidates_from_md",
        lambda path, debug=False: (
            [{"symbol": "ETHUSDT", "horizon_sec": 60, "min_imbalance": 0.5, "min_trade_intensity": 2500.0, "max_spread": 0.00025}],
            {"total_rows_seen": 1, "table_rows_seen": 1, "rows_with_pass_yes": 1, "candidates_parsed": 1, "candidates_unique": 1, "rows_skipped_missing_fields": 0},
        ),
    )

    def _fake_validate(**kwargs):
        rows = []
        for split in [1, 2]:
            for seed in [11, 22]:
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "filled_n": 90,
                        "filled_avg_net": -0.00001,
                        "filled_p90_net": 0.00001,
                        "attempt_fill_rate": 0.6,
                        "net_per_attempt": -0.000005,
                        "attempts_per_min": 1.2,
                        "effective_min_n": 20,
                        "pass": False,
                    }
                )
        return {
            "rows_total": len(rows),
            "pass_count": 0,
            "pass_rate": 0.0,
            "insufficient_fill_rate": 0.0,
            "per_combo": rows,
            "failure_attribution_median": {
                "n_events_total": 1000,
                "n_rejected_attempt_gate": 100,
                "n_attempts_after_gate": 900,
                "n_filled": 450,
                "n_unfilled": 450,
                "avg_fill_prob": 0.55,
                "avg_adverse_bps_on_fills": 1.2,
                "avg_fee_bps": 2.0,
                "avg_scratch_bps_on_fills": 0.5,
                "avg_raw_return_bps_on_fills": -0.5,
                "avg_net_return_bps_on_fills": -2.5,
                "net_return_bps_p10": -6.0,
                "net_return_bps_p50": -2.0,
                "net_return_bps_p90": 1.0,
                "reject_vol_quantile_reject": 12,
                "reject_spread_too_wide": 8,
                "reject_imbalance_too_low": 6,
                "reject_intensity_too_low": 4,
                "reject_other_gate": 2,
            },
        }

    monkeypatch.setattr(rp, "validate_pocket_forward", _fake_validate)
    out_json = Path("reports/test_rank_attribution_merge.json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--candidates-md",
            "reports/dummy.md",
            "--db",
            "data/microstructure.db",
            "--maker-fee-bps-grid",
            "1.0",
            "--passive-adverse-mult-grid",
            "1.2",
            "--min-attempt-fill-rate",
            "0.0",
            "--out-md",
            "reports/test_rank_attribution_merge.md",
            "--out-json",
            str(out_json),
        ],
    )
    rc = rp.main()
    assert rc == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["count"] >= 1
    row = data["ranking"][0]
    for k in [
        "n_events_total",
        "n_rejected_attempt_gate",
        "n_attempts_after_gate",
        "n_filled",
        "n_unfilled",
        "avg_fill_prob",
        "avg_adverse_bps_on_fills",
        "avg_fee_bps",
        "avg_scratch_bps_on_fills",
        "avg_raw_return_bps_on_fills",
        "avg_net_return_bps_on_fills",
        "net_return_bps_p10",
        "net_return_bps_p50",
        "net_return_bps_p90",
        "gate_reject_ratio",
        "fill_rate_after_gate",
    ]:
        assert k in row
        assert row[k] is not None
    assert abs(float(row["gate_reject_ratio"]) - 0.1) < 1e-12
    assert abs(float(row["fill_rate_after_gate"]) - 0.5) < 1e-12
    # net < 0 and fee (2.0) dominates adverse (1.2)
    assert row["failure_reason_top"] == "fees_dominate"
    assert "decomposition_core" in row
    dc = row["decomposition_core"]
    assert "gross_edge_npa" in dc
    assert "fee_cost_npa" in dc
    assert "adverse_cost_npa" in dc
    assert "scratch_cost_npa" in dc
    assert "net_npa" in dc
    assert "residual_npa" in dc
    lhs = float(dc["net_npa"])
    rhs = (
        float(dc["gross_edge_npa"])
        - float(dc["fee_cost_npa"])
        - float(dc["adverse_cost_npa"])
        - float(dc["scratch_cost_npa"])
    )
    assert abs(lhs - rhs) < 1e-12
    assert "reject_breakdown" in row
    assert set(row["reject_breakdown"].keys()) == {
        "vol_quantile_reject",
        "spread_too_wide",
        "imbalance_too_low",
        "intensity_too_low",
        "other_gate",
    }


def test_ranker_passes_regime_filter_to_validator(monkeypatch) -> None:
    monkeypatch.setattr(
        rp,
        "_parse_candidates_from_md",
        lambda path, debug=False: (
            [{"symbol": "ETHUSDT", "horizon_sec": 60, "min_imbalance": 0.5, "min_trade_intensity": 2500.0, "max_spread": 0.00025}],
            {"total_rows_seen": 1, "table_rows_seen": 1, "rows_with_pass_yes": 1, "candidates_parsed": 1, "candidates_unique": 1, "rows_skipped_missing_fields": 0},
        ),
    )
    seen = {"regime_filter": []}

    def _fake_validate(**kwargs):
        seen["regime_filter"].append(kwargs.get("regime_filter", ""))
        rows = []
        for split in [1, 2]:
            for seed in [11, 22]:
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "train_n": 100,
                        "val_n_rows": 100,
                        "effective_min_n": 30,
                        "filled_n": 60,
                        "filled_avg_net": 0.00002,
                        "filled_p90_net": 0.00010,
                        "filled_win_rate": 0.55,
                        "attempt_fill_rate": 0.45,
                        "net_per_attempt": 0.00001,
                        "val_attempts": 133,
                        "val_filled": 60,
                        "attempts_per_min": 80.0,
                        "pass": True,
                    }
                )
        return {"rows_total": len(rows), "pass_count": len(rows), "pass_rate": 1.0, "insufficient_fill_rate": 0.0, "per_combo": rows}

    monkeypatch.setattr(rp, "validate_pocket_forward", _fake_validate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--candidates-md",
            "reports/dummy.md",
            "--db",
            "data/microstructure.db",
            "--maker-fee-bps-grid",
            "1.0",
            "--passive-adverse-mult-grid",
            "1.0",
            "--regime",
            "up",
            "--out-md",
            "reports/test_regime_filter_passthrough.md",
            "--out-json",
            "reports/test_regime_filter_passthrough.json",
        ],
    )
    rc = rp.main()
    assert rc == 0
    assert len(seen["regime_filter"]) > 0
    assert all(v == "UP" for v in seen["regime_filter"])
