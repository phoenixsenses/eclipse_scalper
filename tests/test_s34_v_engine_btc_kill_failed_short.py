from tools.s34_v_engine_btc_kill_failed_short import parse_float_tuple, parse_int_tuple, source_counts


def test_parse_int_tuple():
    assert parse_int_tuple("5,10, 15") == (5, 10, 15)


def test_parse_float_tuple():
    assert parse_float_tuple("0,-10, -20") == (0.0, -10.0, -20.0)


def test_source_counts_sorts_none_key():
    rows = [{"src": "book"}, {"src": None}, {"src": "book"}]

    assert source_counts(rows, "src") == {"book": 2, "none": 1}
