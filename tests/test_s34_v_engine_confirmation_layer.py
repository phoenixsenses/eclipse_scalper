from tools.s34_v_engine_confirmation_layer import condition_set, source_counts


def test_all3_confirmation_requires_anchor_btc_and_bull_reclaim():
    cond = condition_set()["all3"]

    assert cond(
        {
            "anchor_reclaimed_15m": True,
            "btc_context_bucket": "btc_down_then_stable",
            "candle15_pattern": "bull_reclaim",
        }
    )
    assert not cond(
        {
            "anchor_reclaimed_15m": False,
            "btc_context_bucket": "btc_down_then_stable",
            "candle15_pattern": "bull_reclaim",
        }
    )
    assert not cond(
        {
            "anchor_reclaimed_15m": True,
            "btc_context_bucket": "btc_down_continues",
            "candle15_pattern": "bull_reclaim",
        }
    )


def test_source_counts_is_stable_and_sorted():
    rows = [{"src": "b"}, {"src": "a"}, {"src": "b"}, {"src": None}]

    assert source_counts(rows, "src") == {"a": 1, "b": 2, "none": 1}
