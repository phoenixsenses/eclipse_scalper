from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import run_liq_reversal_e2e as e2e


def test_rank_summary_handles_empty() -> None:
    got = e2e._rank_summary({"count": 0, "ranking": []})
    assert got["count"] == 0
    assert got["top"] is None


def test_main_builds_end_to_end_payload(monkeypatch) -> None:
    out_root = Path("reports/test_run_liq_reversal_e2e")
    out_root.mkdir(parents=True, exist_ok=True)

    def _fake_run(module: str, args: list[str]) -> int:
        out_json = None
        out_md = None
        for i, tok in enumerate(args):
            if tok == "--out-json":
                out_json = Path(args[i + 1])
            if tok == "--out-md":
                out_md = Path(args[i + 1])
        assert out_json is not None
        assert out_md is not None
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        if module == "tools.liquidation_rule_coverage":
            out_json.write_text(
                json.dumps(
                    {
                        "symbol": "ETHUSDT",
                        "rule": "high_liq_reversal_regime",
                        "bucket_sec": 5,
                        "results": [{"lookback_min": 60, "rule_fire_count": 12, "rule_given_liq_rate": 0.5}],
                    }
                ),
                encoding="utf-8",
            )
        elif module == "tools.generate_liq_reversal_candidates":
            out_json.write_text(
                json.dumps(
                    {
                        "rule": "high_liq_reversal_regime",
                        "regime": "liq_reversal_research",
                        "symbols": ["ETHUSDT"],
                        "grid": {},
                        "count": 8,
                        "rows": [],
                    }
                ),
                encoding="utf-8",
            )
        elif module == "tools.rank_passive_pockets_forward" and "baseline" in args:
            out_json.write_text(json.dumps({"count": 0, "ranking": []}), encoding="utf-8")
        elif module == "tools.rank_passive_pockets_forward" and "anti_adverse_v5" in args:
            out_json.write_text(json.dumps({"count": 0, "ranking": []}), encoding="utf-8")
        elif module == "tools.rank_passive_pockets_forward":
            out_json.write_text(
                json.dumps(
                    {
                        "count": 1,
                        "ranking": [
                            {
                                "symbol": "ETHUSDT",
                                "rule": "high_liq_reversal_regime",
                                "horizon_sec": 60,
                                "score": 1e-4,
                                "score_raw_core": 2e-4,
                                "npa_core": 3e-5,
                                "pass_rate_core": 0.5,
                                "attempt_fill_rate": 0.25,
                                "failure_reason_top": "mixed",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
        out_md.write_text("# ok\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(e2e, "_run_tool", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--db",
            "data/microstructure.db",
            "--symbol",
            "ETHUSDT",
            "--reports-prefix",
            str(out_root / "LIQ_REVERSAL_E2E"),
        ],
    )
    rc = e2e.main()
    assert rc == 0
    payload = json.loads((out_root / "LIQ_REVERSAL_E2E.json").read_text(encoding="utf-8"))
    assert payload["summary"]["coverage"]["max_rule_fire_count"] == 12
    assert payload["summary"]["candidate_surface"]["count"] == 8
    assert payload["summary"]["rank_baseline"]["count"] == 0
    assert payload["summary"]["rank_v5"]["count"] == 0
    assert payload["summary"]["rank_v6"]["count"] == 1
    assert payload["summary"]["decision"]["next_step"] == "inspect_ranked_pockets"
    assert payload["run_summary"]["run_type"] == "run_liq_reversal_e2e"
