from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.optimize_fill_timeout import build_recommendation, _upsert_env_key


def _mk_local_tmp() -> Path:
    p = Path("localtests") / f"fill_timeout_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def test_build_recommendation_uses_timeout_eval_best() -> None:
    payload = {
        "live_summary": {
            "recommended_timeout_sec": 13.0,
            "timeout_eval": [
                {"timeout_sec": 10, "filled_pnl_mean": 0.10, "filled_adverse_proxy_mean": 1.0, "fill_rate_within_timeout": 0.7, "eligible_frac": 0.8},
                {"timeout_sec": 15, "filled_pnl_mean": 0.12, "filled_adverse_proxy_mean": 0.9, "fill_rate_within_timeout": 0.6, "eligible_frac": 0.7},
                {"timeout_sec": 20, "filled_pnl_mean": 0.08, "filled_adverse_proxy_mean": 0.7, "fill_rate_within_timeout": 0.8, "eligible_frac": 0.9},
            ],
        }
    }
    rec = build_recommendation(payload, candidates=[5, 10, 15, 20, 30])
    assert int(rec.recommended) == 15
    assert rec.source == "timeout_eval"


def test_build_recommendation_fallback_to_live_summary() -> None:
    payload = {"live_summary": {"recommended_timeout_sec": 12.2, "timeout_eval": []}}
    rec = build_recommendation(payload, candidates=[5, 10, 15, 20, 30])
    assert int(rec.recommended) == 10
    assert rec.source == "live_summary"


def test_upsert_env_key_updates_existing_and_appends_new() -> None:
    tmp = _mk_local_tmp()
    try:
        env = tmp / ".env.paper"
        env.write_text("A=1\nENTRY_WATCH_MAX_AGE_SEC=10\n", encoding="utf-8")
        changed_existing = _upsert_env_key(env, "ENTRY_WATCH_MAX_AGE_SEC", 15)
        txt1 = env.read_text(encoding="utf-8")
        assert changed_existing is True
        assert "ENTRY_WATCH_MAX_AGE_SEC=15" in txt1

        changed_new = _upsert_env_key(env, "NEW_KEY", 7)
        txt2 = env.read_text(encoding="utf-8")
        assert changed_new is False
        assert "NEW_KEY=7" in txt2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_optimize_fill_timeout_roundtrip() -> None:
    tmp = _mk_local_tmp()
    try:
        analysis = tmp / "analysis.json"
        env = tmp / ".env.paper"
        out_md = tmp / "out.md"
        out_json = tmp / "out.json"
        analysis.write_text(
            json.dumps(
                {
                    "live_summary": {
                        "recommended_timeout_sec": 17.0,
                        "timeout_eval": [{"timeout_sec": 20, "filled_pnl_mean": 0.2, "filled_adverse_proxy_mean": 0.3, "fill_rate_within_timeout": 0.7, "eligible_frac": 0.8}],
                    }
                }
            ),
            encoding="utf-8",
        )
        env.write_text("ENTRY_WATCH_MAX_AGE_SEC=10\n", encoding="utf-8")
        rc = __import__("tools.optimize_fill_timeout", fromlist=["main"]).main  # type: ignore
        old = sys.argv
        try:
            sys.argv = [
                "optimize_fill_timeout",
                "--analysis-json",
                str(analysis),
                "--env-file",
                str(env),
                "--out-md",
                str(out_md),
                "--out-json",
                str(out_json),
                "--candidates",
                "5,10,20,30",
                "--apply",
            ]
            code = rc()
        finally:
            sys.argv = old
        assert int(code) == 0
        assert "ENTRY_WATCH_MAX_AGE_SEC=20" in env.read_text(encoding="utf-8")
        assert out_md.exists()
        assert out_json.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
