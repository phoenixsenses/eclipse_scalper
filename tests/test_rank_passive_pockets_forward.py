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
    assert data["ranking"][0]["symbol"] == "ETHUSDT"


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
    assert "Consider lowering --min-n-frac" in out


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
