"""Tests for ami/cvd/aggtrades_repair_rehearsal.py -- REST pagination law,
overlap dedup, missing-page/id detection, retry idempotency, rerun
determinism, immutable staging, cross-source fingerprint reconciliation
fail-closed behavior. Fully synthetic (mocked HTTP; no network)."""
import json
import sqlite3

import pytest

from ami.cvd import aggtrades_repair_rehearsal as rep


def _trade(a, T, p="100.0", q="1.0", m=False, f=None, l=None):
    return {"a": a, "p": p, "q": q, "T": T, "m": m,
            "f": f if f is not None else a, "l": l if l is not None else a}


def _mock_http(pages_by_url):
    calls = []

    def http_get(url, timeout=20.0):
        calls.append(url)
        body = pages_by_url[url]
        if isinstance(body, Exception):
            raise body
        return 200, json.dumps(body).encode()
    http_get.calls = calls
    return http_get


def _url_start(symbol, start, end):
    return f"{rep.BASE_URL}?symbol={symbol}&startTime={start}&endTime={end}&limit={rep.LIMIT}"


def _url_from(symbol, from_id):
    return f"{rep.BASE_URL}?symbol={symbol}&fromId={from_id}&limit={rep.LIMIT}"


def _full_pages(symbol, start, end, trades, page_size, sentinel=True):
    """Build a pages_by_url map emulating Binance startTime-then-fromId flow.
    A sentinel trade with T > end is appended to the final page so the
    extractor terminates via the past-end rule (the normal case for a
    bounded window inside continuous data), not the ambiguous short-page
    rule (which fail-closes as truncation)."""
    seq = list(trades)
    if sentinel and seq:
        seq = seq + [_trade(seq[-1]["a"] + 1, end + 1)]
    pages = {}
    pages[_url_start(symbol, start, end)] = seq[:page_size]
    idx = page_size
    while idx <= len(seq):
        from_id = seq[idx - 1]["a"] + 1
        pages[_url_from(symbol, from_id)] = seq[idx:idx + page_size]
        idx += page_size
    return pages


# 7. REST page overlap deduplication
def test_page_overlap_deduplicated():
    sym = "ETHUSDT"
    t = [_trade(10, 1000), _trade(11, 1100), _trade(12, 1200)]
    old = rep.LIMIT
    rep.LIMIT = 2
    try:
        # URLs must be built AFTER patching LIMIT (they embed limit=)
        pages = {
            _url_start(sym, 1000, 5000): [t[0], t[1]],
            # overlapping continuation page that re-serves id 11
            _url_from(sym, 12): [t[1], t[2]],
            # terminal page: sentinel past end -> clean termination
            _url_from(sym, 13): [_trade(13, 5001)],
        }
        res = rep.fetch_window(sym, 1000, 5000, http_get=_mock_http(pages), sleep=lambda s: None)
    finally:
        rep.LIMIT = old
    assert res["row_count"] == 3
    assert res["page_overlap_rows"] == 1
    assert res["missing_id_ranges"] == []
    assert rep.extraction_verdict(res, probe_only=False) == "EXACT_RECONSTRUCTED"


# 8. missing-page detection (id-range hole)
def test_missing_id_range_detected_and_never_exact():
    sym = "ETHUSDT"
    t = [_trade(10, 1000), _trade(11, 1100), _trade(14, 1400)]  # 12-13 missing
    old = rep.LIMIT
    rep.LIMIT = 2
    try:
        pages = {
            _url_start(sym, 1000, 5000): [t[0], t[1]],
            _url_from(sym, 12): [t[2]],
        }
        res = rep.fetch_window(sym, 1000, 5000, http_get=_mock_http(pages), sleep=lambda s: None)
    finally:
        rep.LIMIT = old
    assert res["missing_id_ranges"] == [[12, 13]]
    assert rep.extraction_verdict(res, probe_only=False) == "INCOMPLETE"


# 9. retry behavior + retry idempotency
def test_retry_then_success_is_idempotent_with_clean_run():
    sym = "ETHUSDT"
    trades = [_trade(10 + i, 1000 + i * 100) for i in range(3)]
    clean = _full_pages(sym, 1000, 5000, trades, page_size=1000)
    res_clean = rep.fetch_window(sym, 1000, 5000, http_get=_mock_http(clean),
                                 sleep=lambda s: None)

    flaky_state = {"failed": False}

    def flaky(url, timeout=20.0):
        if not flaky_state["failed"]:
            flaky_state["failed"] = True
            raise OSError("transient network error")
        return 200, json.dumps(clean[url]).encode()

    res_flaky = rep.fetch_window(sym, 1000, 5000, http_get=flaky, sleep=lambda s: None)
    assert res_flaky["content_sha256"] == res_clean["content_sha256"]
    assert res_flaky["request_errors"] == []  # retries within budget are not errors
    assert rep.extraction_verdict(res_flaky, probe_only=False) == "EXACT_RECONSTRUCTED"


# rerun determinism (Task 5)
def test_rerun_produces_identical_hashes_and_manifests():
    sym = "ETHUSDT"
    trades = [_trade(100 + i, 2000 + i * 50, p=f"{100 + i}.5", q="2.0", m=(i % 2 == 0))
              for i in range(5)]
    pages = _full_pages(sym, 2000, 9000, trades, page_size=1000)
    r1 = rep.fetch_window(sym, 2000, 9000, http_get=_mock_http(pages), sleep=lambda s: None)
    r2 = rep.fetch_window(sym, 2000, 9000, http_get=_mock_http(pages), sleep=lambda s: None)
    assert r1["content_sha256"] == r2["content_sha256"]
    assert r1["gap_manifest_sha256"] == r2["gap_manifest_sha256"]
    assert r1["duplicate_manifest_sha256"] == r2["duplicate_manifest_sha256"]
    assert r1["row_count"] == r2["row_count"]


def test_exhausted_retries_fail_closed():
    def dead(url, timeout=20.0):
        raise OSError("down")
    res = rep.fetch_window("ETHUSDT", 1000, 2000, http_get=dead, sleep=lambda s: None)
    assert res["failed"] is True
    assert rep.extraction_verdict(res, probe_only=False) == "FAILED"


def test_probe_only_verdict_never_claims_exact():
    sym = "ETHUSDT"
    trades = [_trade(10, 1000)]
    pages = _full_pages(sym, 1000, 2000, trades, page_size=1000)
    res = rep.fetch_window(sym, 1000, 2000, http_get=_mock_http(pages), sleep=lambda s: None)
    assert rep.extraction_verdict(res, probe_only=True) == "PROBE_ONLY"


def test_zero_rows_never_exact():
    sym = "ETHUSDT"
    pages = {_url_start(sym, 1000, 2000): []}
    res = rep.fetch_window(sym, 1000, 2000, http_get=_mock_http(pages), sleep=lambda s: None)
    assert res["row_count"] == 0
    assert rep.extraction_verdict(res, probe_only=False) == "INCOMPLETE"


# 10. conflicting duplicate rejection (immutable staging)
def _staged_result(trades, sym="ETHUSDT"):
    pages = _full_pages(sym, 1000, 9000, trades, page_size=1000)
    return rep.fetch_window(sym, 1000, 9000, http_get=_mock_http(pages), sleep=lambda s: None)


def test_stage_rows_immutable_conflict():
    conn = sqlite3.connect(":memory:")
    rep.init_schema(conn)
    res = _staged_result([_trade(10, 1000, p="100.0")])
    assert rep.stage_rows(conn, res, retrieval_batch_id="B1", source_regime_id="R2") == 1
    # identical restage: no new rows, no error
    assert rep.stage_rows(conn, res, retrieval_batch_id="B1", source_regime_id="R2") == 0
    # same identity, different content -> fail closed
    res2 = _staged_result([_trade(10, 1000, p="999.0")])
    with pytest.raises(rep.ImmutableRepairRowConflict):
        rep.stage_rows(conn, res2, retrieval_batch_id="B1", source_regime_id="R2")


def test_stage_schema_rejects_inconsistent_taker_side():
    conn = sqlite3.connect(":memory:")
    rep.init_schema(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO ami_agg_trades_repaired_stage (symbol, agg_trade_id, ts_ms,"
            " retrieved_at_ms, price, quantity, notional, signed_quantity, signed_notional,"
            " is_buyer_maker, taker_side, source_regime_id, retrieval_batch_id,"
            " retrieval_page_index, source_provenance, source_quality_status,"
            " legacy_match_status, data_version_id, created_ms)"
            " VALUES ('ETHUSDT',1,1000,0,'1','1',1.0,1.0,1.0,0,'SELL','R2','B',0,'t',"
            " 'EXACT_RECONSTRUCTABLE','NOT_ATTEMPTED','aggtrades-binance-fapi-repair-r1',0)")


# 11+12. cross-source fingerprint collision handling / ambiguous fail-closed
def test_reconcile_exact_one_to_one():
    rest = [_trade(10, 1000, p="100.0", q="1.0", m=False),
            _trade(11, 1100, p="101.0", q="2.0", m=True)]
    legacy = [(1000, 100.0, 1.0, 0), (1100, 101.0, 2.0, 1)]
    r = rep.reconcile_rest_vs_legacy(rest, legacy)
    assert r["exact_one_to_one"] == 2
    assert r["unmatched_rest"] == 0 and r["unmatched_legacy"] == 0
    assert r["deterministic_supersession_feasible"] is True


def test_reconcile_unmatched_rows():
    rest = [_trade(10, 1000, p="100.0", q="1.0", m=False)]
    legacy = [(2000, 55.0, 3.0, 1)]
    r = rep.reconcile_rest_vs_legacy(rest, legacy)
    assert r["unmatched_rest"] == 1
    assert r["unmatched_legacy"] == 1
    assert r["deterministic_supersession_feasible"] is False


def test_reconcile_collision_classes_fail_closed():
    # legacy has the same fingerprint TWICE (e.g. WS/REST double insert),
    # REST has it once -> one_to_many collision, not silently paired
    rest = [_trade(10, 1000, p="100.0", q="1.0", m=False)]
    legacy = [(1000, 100.0, 1.0, 0), (1000, 100.0, 1.0, 0)]
    r = rep.reconcile_rest_vs_legacy(rest, legacy)
    assert r["one_to_many_collisions"] == 1
    assert r["exact_one_to_one"] == 0
    assert r["duplicate_fingerprint_multiplicity_hist"] == {2: 1}
    assert r["deterministic_supersession_feasible"] is False


def test_reconcile_many_to_one_and_many_to_many():
    rest = [_trade(10, 1000, p="100.0", q="1.0", m=False),
            _trade(11, 1000, p="100.0", q="1.0", m=False)]  # genuine same-ms twins
    legacy = [(1000, 100.0, 1.0, 0)]
    r = rep.reconcile_rest_vs_legacy(rest, legacy)
    assert r["many_to_one_collisions"] == 1
    assert r["deterministic_supersession_feasible"] is False
    legacy2 = [(1000, 100.0, 1.0, 0), (1000, 100.0, 1.0, 0), (1000, 100.0, 1.0, 0)]
    r2 = rep.reconcile_rest_vs_legacy(rest, legacy2)
    assert r2["many_to_many_collisions"] == 1
    assert r2["deterministic_supersession_feasible"] is False


def test_reconcile_conflicting_side_flag():
    rest = [_trade(10, 1000, p="100.0", q="1.0", m=False)]
    legacy = [(1000, 100.0, 1.0, 1)]  # same (ts,p,q), opposite maker flag
    r = rep.reconcile_rest_vs_legacy(rest, legacy)
    assert r["conflicting_side_flag_rows"] == 2
    assert r["deterministic_supersession_feasible"] is False


def test_float_representation_equality_of_fingerprints():
    # REST serves strings; legacy stored REAL from float(str) at collect time.
    # Fingerprints must agree exactly under float() round-trip.
    rest = [_trade(10, 1000, p="1858.96", q="10.000", m=True)]
    legacy = [(1000, float("1858.96"), float("10.000"), 1)]
    r = rep.reconcile_rest_vs_legacy(rest, legacy)
    assert r["exact_one_to_one"] == 1
    assert r["deterministic_supersession_feasible"] is True
