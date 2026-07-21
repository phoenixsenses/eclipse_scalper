from tools.s34_v_engine_execution_frontier import anchor_mark_counterfactual, parse_float_tuple


class _Marks:
    def at_or_after(self, ts_ms):
        if ts_ms == 1000:
            return (1000, 100.0)
        if ts_ms > 1000:
            return (ts_ms, 101.0)
        return None


def test_parse_float_tuple():
    assert parse_float_tuple("0,5, 10") == (0.0, 5.0, 10.0)


def test_anchor_mark_counterfactual_long_net():
    out = anchor_mark_counterfactual(_Marks(), 1000, fee_bps=5.0)

    assert out == 95.0
