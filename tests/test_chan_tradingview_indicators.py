from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPASS = ROOT / "tools/pine/chan_relative_regime_compass.pine"
JOURNEY = ROOT / "tools/pine/echo_journey_lens.pine"


def test_pine_v6_indicators_are_observation_only_and_non_lookahead() -> None:
    for path in (COMPASS, JOURNEY):
        source = path.read_text(encoding="utf-8")
        lowered = source.lower()
        assert source.startswith("//@version=6\nindicator(")
        assert "barmerge.lookahead_off" in source
        assert "strategy(" not in lowered
        assert "strategy.entry" not in lowered
        assert "strategy.exit" not in lowered
        assert "alert(" not in lowered


def test_compass_keeps_mechanisms_separate_and_beta_neutralizes_returns() -> None:
    source = COMPASS.read_text(encoding="utf-8")
    assert "relativeStep = coinRet - beta * btcRet" in source
    assert "relativeLog = ta.cum(nz(relativeStep, 0.0))" in source
    assert "lambda < 0" in source
    assert "varianceRatio < 1 - vrBand" in source
    assert "varianceRatio > 1 + vrBand" in source
    assert "residualShockZ >= shockZ and volumeZ >= volumeZMin" in source
    assert "confirmedOnly or barstate.isconfirmed" in source
    assert "score =" not in source.lower()
    assert 'primaryBenchmark = input.symbol("BINANCE:BTCUSDT.P"' in source
    assert 'btcChartFallback = input.symbol("BINANCE:ETHUSDT.P"' in source
    assert "effectiveBenchmark = syminfo.tickerid == primaryBenchmark ? btcChartFallback : primaryBenchmark" in source
    assert "sameAsBenchmark = syminfo.tickerid == effectiveBenchmark" in source
    assert "INVALID BENCHMARK" in source
    assert "WARMING UP / NEED HISTORY" in source


def test_journey_separates_wait_from_costed_paper_pnl() -> None:
    source = JOURNEY.read_text(encoding="utf-8")
    assert "inWait = time >= anchorTime and time < entryTime" in source
    assert "inTrade = time >= entryTime and time < exitTime" in source
    assert "netFromEntry = grossFromEntry - costBps" in source
    assert "btcNeutralNet = grossFromEntry - btcGrossFromEntry - costBps" in source
    assert "OBSERVE ONLY · NO ORDERS" in source
    assert 'timestamp("13 Aug 2026 20:35 +0000")' in source
    assert 'expectedChart = input.symbol("BINANCE:BANKUSDT.P"' in source
    assert "wrongChart = syminfo.tickerid != expectedChart" in source
    assert "wrongTimeframe = timeframe.in_seconds() != 60" in source
    assert "Use 1m for exact anchor/entry/exit prices" in source
