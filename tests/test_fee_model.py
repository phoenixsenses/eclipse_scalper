from __future__ import annotations

from core.fee_model import estimate_fee_bps_for_daily_volume, infer_vip_tier_from_30d_volume


def test_fee_model_tier_progression() -> None:
    t0 = infer_vip_tier_from_30d_volume(0.0)
    t1 = infer_vip_tier_from_30d_volume(100_000_000.0)
    assert int(t1.tier) >= int(t0.tier)
    assert float(t1.maker_bps) <= float(t0.maker_bps)


def test_estimate_fee_from_daily_volume() -> None:
    est = estimate_fee_bps_for_daily_volume(1_000_000.0)
    assert "maker_bps" in est and "taker_bps" in est
    assert float(est["maker_bps"]) >= 0.0

