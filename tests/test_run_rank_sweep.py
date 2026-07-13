from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import run_rank_sweep as rrs


def test_run_rank_sweep_writes_registry_and_run_folders(monkeypatch, tmp_path) -> None:
    reports = tmp_path / "test_run_rank_sweep"
    reports.mkdir(parents=True, exist_ok=True)
    candidates = reports / "candidates.md"
    candidates.write_text(
        "\n".join(
            [
                "| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | pass |",
                "|---|---:|---:|---:|---:|---|",
                "| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | YES |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(rrs, "_git_commit", lambda: "deadbeef")

    def _fake_invoke(argv):
        out_json = None
        out_md = None
        for i, tok in enumerate(argv):
            if tok == "--out-json":
                out_json = Path(argv[i + 1])
            if tok == "--out-md":
                out_md = Path(argv[i + 1])
        assert out_json is not None
        assert out_md is not None
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(
                {
                    "count": 1,
                    "liquidation_scoring_impact": {
                        "available": True,
                        "count": 1,
                        "positive_delta_score_count": 1,
                        "avg_delta_score_raw_core": 1e-4,
                        "avg_delta_npa_core": 2e-5,
                        "avg_delta_pass_rate_core": 0.1,
                    },
                    "ranking": [
                        {
                            "symbol": "ETHUSDT",
                            "rule": "micro_edge_v3_passive_alpha",
                            "horizon_sec": 120,
                            "score": 1e-4,
                            "score_raw_core": 2e-4,
                            "npa_core": 1e-5,
                            "pass_rate_core": 0.5,
                            "failure_reason_top": "mixed",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        out_md.write_text("# md\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(rrs, "_invoke_rank", _fake_invoke)

    registry = reports / "RUN_RANK_SWEEP_REGISTRY.jsonl"
    registry.unlink(missing_ok=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--db",
            "data/microstructure.db",
            "--candidates-md",
            str(candidates),
            "--reports-dir",
            str(reports),
            "--registry",
            str(registry),
            "--maker-fee-bps-grid",
            "1.0,1.5",
            "--passive-adverse-mult-grid",
            "1.0",
            "--vol-quantile-reject-grid",
            "0.01",
        ],
    )
    rc = rrs.main()
    assert rc == 0
    assert registry.exists()
    lines = [ln for ln in registry.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["git_commit"] == "deadbeef"
    assert "run_id" in rec
    assert rec["outputs"]["json"].endswith("rank.json")
    assert rec["summary"]["count"] == 1
    assert rec["summary"]["liquidation_scoring_impact"]["available"] is True
    assert rec["summary"]["liquidation_scoring_impact"]["positive_delta_score_count"] == 1
    assert rec["run_summary"]["version"] == "v1"
    assert rec["run_summary"]["run_type"] == "run_rank_sweep"
    assert rec["run_summary"]["metrics"]["ranking_count"] == 1
    assert rec["run_summary"]["artifacts"]["registry"].endswith("RUN_RANK_SWEEP_REGISTRY.jsonl")
    run_dir = reports / "_runs" / rec["run_id"]
    assert (run_dir / "rank.json").exists()
    assert (run_dir / "rank.md").exists()


def test_run_rank_sweep_dry_run_no_registry_write(monkeypatch, capsys, tmp_path) -> None:
    reports = tmp_path / "test_run_rank_sweep_dry"
    reports.mkdir(parents=True, exist_ok=True)
    candidates = reports / "candidates.md"
    candidates.write_text("| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | pass |\n", encoding="utf-8")
    called = {"n": 0}
    monkeypatch.setattr(rrs, "_invoke_rank", lambda argv: called.__setitem__("n", called["n"] + 1) or 0)
    registry = reports / "RUN_RANK_SWEEP_REGISTRY.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--candidates-md",
            str(candidates),
            "--reports-dir",
            str(reports),
            "--registry",
            str(registry),
            "--dry-run",
        ],
    )
    rc = rrs.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "DRY_RUN" in out
    assert called["n"] == 0
    assert not registry.exists()
