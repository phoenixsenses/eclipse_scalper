"""Tests for the frozen-DB -> Parquet archival exporter (rotation Phase-3).

The point of the exporter is the proof gate, so the tests exercise what could
actually make the gate lie: a tampered archive, a truncated archive, a source
that changed after export, and SQLite's dynamic typing sneaking a wrong type
into a declared column.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from tools import frozen_db_parquet_export as fx


DAY_MS = 86_400_000
DAY0 = 1_784_000_000_000 // DAY_MS * DAY_MS  # aligned UTC midnight


def _build_source(path: Path, rows: list[tuple]) -> None:
    con = sqlite3.connect(path)
    con.execute(
        """
        CREATE TABLE mark_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            mark_price REAL NOT NULL,
            funding_rate REAL,
            next_funding_time_ms INTEGER
        )
        """
    )
    con.execute("CREATE INDEX idx_mark_symbol_ts ON mark_prices(symbol, ts_ms)")
    con.executemany(
        "INSERT INTO mark_prices (ts_ms, symbol, mark_price, funding_rate, next_funding_time_ms)"
        " VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    con.commit()
    con.close()


@pytest.fixture()
def source_db(tmp_path: Path) -> Path:
    rows: list[tuple] = []
    for i in range(250):
        rows.append((DAY0 + i * 1000, "BTCUSDT", 60000.0 + i, 0.0001 * i, DAY0 + DAY_MS))
    for i in range(120):
        # second day, and a NULL-bearing column to exercise null packing
        rows.append((DAY0 + DAY_MS + i * 1000, "BTCUSDT", 61000.5 + i, None, None))
    for i in range(40):
        rows.append((DAY0 + i * 1000, "ETHUSDT", 3000.25 + i, -0.00002, DAY0 + DAY_MS))
    path = tmp_path / "frozen.db"
    _build_source(path, rows)
    return path


def _export(source: Path, out_root: Path, **kw) -> int:
    argv = [
        "--table",
        "mark_prices",
        "--db",
        str(source),
        "--out-root",
        str(out_root),
        "--symbols",
        "BTCUSDT,ETHUSDT",
        "--batch-rows",
        str(kw.pop("batch_rows", 100)),
    ]
    for key, value in kw.items():
        flag = "--" + key.replace("_", "-")
        argv.append(flag)
        if value is not True:
            argv.append(str(value))
    return fx.main(argv)


def _verify(source: Path, out_root: Path, **kw) -> int:
    argv = [
        "--table",
        "mark_prices",
        "--db",
        str(source),
        "--out-root",
        str(out_root),
        "--verify",
    ]
    for key, value in kw.items():
        flag = "--" + key.replace("_", "-")
        argv.append(flag)
        if value is not True:
            argv.append(str(value))
    return fx.main(argv)


# --------------------------------------------------------------------------
# round trip + gate
# --------------------------------------------------------------------------


def test_export_then_verify_closes_the_gate(source_db: Path, tmp_path: Path) -> None:
    out_root = tmp_path / "parquet"
    assert _export(source_db, out_root) == 0
    assert _verify(source_db, out_root) == 0

    manifest = fx.read_manifest(fx.manifest_path(out_root, "mark_prices"))
    total_rows = sum(rec["rows"] for rec in manifest.values())
    assert total_rows == 410  # 250 + 120 + 40, nothing dropped, nothing duplicated


def test_parquet_holds_the_same_values_as_sqlite(source_db: Path, tmp_path: Path) -> None:
    out_root = tmp_path / "parquet"
    assert _export(source_db, out_root) == 0

    path = fx.partition_path(out_root, "mark_prices", "ETHUSDT", fx.day_label(DAY0))
    table = pq.read_table(path)
    assert table.num_rows == 40
    assert table.column("symbol").to_pylist()[0] == "ETHUSDT"
    assert table.column("mark_price").to_pylist()[0] == pytest.approx(3000.25)

    con = sqlite3.connect(source_db)
    expected = con.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol='ETHUSDT' ORDER BY ts_ms"
    ).fetchall()
    con.close()
    assert sorted(table.column("mark_price").to_pylist()) == sorted(r[0] for r in expected)


def test_nulls_survive_the_round_trip(source_db: Path, tmp_path: Path) -> None:
    out_root = tmp_path / "parquet"
    assert _export(source_db, out_root) == 0
    path = fx.partition_path(out_root, "mark_prices", "BTCUSDT", fx.day_label(DAY0 + DAY_MS))
    table = pq.read_table(path)
    assert table.column("funding_rate").to_pylist() == [None] * 120
    assert table.column("next_funding_time_ms").to_pylist() == [None] * 120


def test_one_row_group_per_batch(source_db: Path, tmp_path: Path) -> None:
    """The verify path re-reads row group by row group; alignment is load-bearing."""
    out_root = tmp_path / "parquet"
    assert _export(source_db, out_root, batch_rows=100) == 0
    path = fx.partition_path(out_root, "mark_prices", "BTCUSDT", fx.day_label(DAY0))
    pfile = pq.ParquetFile(path)
    assert pfile.num_row_groups == 3  # 250 rows at batch 100 -> 100/100/50
    assert [pfile.metadata.row_group(i).num_rows for i in range(3)] == [100, 100, 50]


# --------------------------------------------------------------------------
# the gate must actually fail
# --------------------------------------------------------------------------


def test_tampered_parquet_is_caught(source_db: Path, tmp_path: Path) -> None:
    out_root = tmp_path / "parquet"
    assert _export(source_db, out_root) == 0

    path = fx.partition_path(out_root, "mark_prices", "ETHUSDT", fx.day_label(DAY0))
    table = pq.read_table(path)
    prices = table.column("mark_price").to_pylist()
    prices[0] += 0.01  # one cent on one row out of 410
    import pyarrow as pa

    mutated = table.set_column(
        table.schema.get_field_index("mark_price"), "mark_price", pa.array(prices, pa.float64())
    )
    pq.write_table(mutated, path, compression="zstd")

    assert _verify(source_db, out_root) == 2


def test_truncated_parquet_is_caught(source_db: Path, tmp_path: Path) -> None:
    out_root = tmp_path / "parquet"
    assert _export(source_db, out_root) == 0

    path = fx.partition_path(out_root, "mark_prices", "ETHUSDT", fx.day_label(DAY0))
    table = pq.read_table(path)
    pq.write_table(table.slice(0, 39), path, compression="zstd")

    assert _verify(source_db, out_root) == 2


def test_missing_parquet_is_caught(source_db: Path, tmp_path: Path) -> None:
    out_root = tmp_path / "parquet"
    assert _export(source_db, out_root) == 0
    fx.partition_path(out_root, "mark_prices", "ETHUSDT", fx.day_label(DAY0)).unlink()
    assert _verify(source_db, out_root) == 2


def test_source_drift_is_caught(source_db: Path, tmp_path: Path) -> None:
    """If the frozen segment is not actually frozen, the gate must not say OK."""
    out_root = tmp_path / "parquet"
    assert _export(source_db, out_root) == 0
    assert _verify(source_db, out_root) == 0

    con = sqlite3.connect(source_db)
    con.execute(
        "INSERT INTO mark_prices (ts_ms, symbol, mark_price, funding_rate, next_funding_time_ms)"
        " VALUES (?, ?, ?, ?, ?)",
        (DAY0 + 999_000, "ETHUSDT", 1.0, None, None),
    )
    con.commit()
    con.close()

    assert _verify(source_db, out_root) == 2
    # ...but the archive itself is still internally intact
    assert _verify(source_db, out_root, skip_source_check=True) == 0


# --------------------------------------------------------------------------
# resumability
# --------------------------------------------------------------------------


def test_resume_skips_completed_partitions(source_db: Path, tmp_path: Path) -> None:
    out_root = tmp_path / "parquet"
    assert _export(source_db, out_root, max_partitions=1) == 0
    manifest = fx.manifest_path(out_root, "mark_prices")
    first = fx.read_manifest(manifest)
    assert len(first) == 1

    assert _export(source_db, out_root) == 0
    second = fx.read_manifest(manifest)
    assert len(second) > len(first)

    key = next(iter(first))
    assert second[key]["digest"] == first[key]["digest"]

    # a completed partition is written exactly once
    lines = [ln for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
    keys = [json.loads(ln)["key"] for ln in lines]
    assert len(keys) == len(set(keys))

    assert _verify(source_db, out_root) == 0


def test_rerun_after_completion_is_a_noop(source_db: Path, tmp_path: Path) -> None:
    out_root = tmp_path / "parquet"
    assert _export(source_db, out_root) == 0
    manifest = fx.manifest_path(out_root, "mark_prices")
    before = manifest.read_text(encoding="utf-8")
    assert _export(source_db, out_root) == 0
    assert manifest.read_text(encoding="utf-8") == before


# --------------------------------------------------------------------------
# source safety
# --------------------------------------------------------------------------


def test_live_db_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    live = tmp_path / "microstructure_02.db"
    _build_source(live, [(DAY0, "BTCUSDT", 1.0, None, None)])
    state = tmp_path / "rotation_state.json"
    state.write_text(json.dumps({"live_db_path": str(live)}), encoding="utf-8")
    monkeypatch.setattr(fx, "ROTATION_STATE", state)

    with pytest.raises(fx.ExportError, match="LIVE"):
        fx.open_source_ro(live)


def test_source_is_opened_read_only(source_db: Path) -> None:
    con = fx.open_source_ro(source_db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            con.execute("DELETE FROM mark_prices")
    finally:
        con.close()


def test_export_does_not_mutate_the_source(source_db: Path, tmp_path: Path) -> None:
    before = source_db.stat()
    assert _export(source_db, tmp_path / "parquet") == 0
    after = source_db.stat()
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)


# --------------------------------------------------------------------------
# canonical packing
# --------------------------------------------------------------------------


def _digest(rows: list[tuple], cols: list[tuple[str, str]]) -> bytes:
    return fx.digest_batch(fx.rows_to_arrow(rows, cols, fx.arrow_schema(cols)), cols)


def test_declared_type_drives_the_digest_not_the_python_type() -> None:
    """3000 stored in a REAL column must hash the same as 3000.0."""
    cols = [("mark_price", "REAL")]
    assert _digest([(3000,)], cols) == _digest([(3000.0,)], cols)


def test_int_and_float_do_not_collide() -> None:
    assert _digest([(3000,)], [("x", "INTEGER")]) != _digest([(3000.0,)], [("x", "REAL")])


def test_lossy_integer_coercion_is_refused() -> None:
    with pytest.raises(fx.ExportError, match="declared type"):
        _digest([(1.5,)], [("ts_ms", "INTEGER")])


def test_null_is_distinct_from_zero_and_empty_string() -> None:
    assert _digest([(None,)], [("x", "REAL")]) != _digest([(0.0,)], [("x", "REAL")])
    assert _digest([(None,)], [("s", "TEXT")]) != _digest([("",)], [("s", "TEXT")])


def test_null_slot_contents_cannot_leak_into_the_digest() -> None:
    """Arrow leaves null slots undefined; filling them is what makes the hash stable."""
    cols = [("x", "REAL")]
    import pyarrow as pa

    schema = fx.arrow_schema(cols)
    left = fx.rows_to_arrow([(None,), (2.0,)], cols, schema)
    # identical logical values, but a different byte sitting under the null slot
    right = pa.Table.from_arrays(
        [pa.array([7.5, 2.0], mask=[True, False], type=pa.float64())], schema=schema
    )
    assert fx.digest_batch(left, cols) == fx.digest_batch(right, cols)


def test_text_values_are_distinguished_by_content_and_order() -> None:
    cols = [("a", "TEXT"), ("b", "TEXT")]
    assert _digest([("AB", "C")], cols) != _digest([("A", "BC")], cols)
    one = [("s", "TEXT")]
    assert _digest([("X",), ("Y",)], one) != _digest([("Y",), ("X",)], one)


def test_row_order_changes_the_digest() -> None:
    cols = [("x", "INTEGER")]
    assert _digest([(1,), (2,)], cols) != _digest([(2,), (1,)], cols)


def test_chain_digest_is_order_sensitive() -> None:
    a = fx.chain_digest(fx.EMPTY_DIGEST, b"one" * 8)
    a = fx.chain_digest(a, b"two" * 8)
    b = fx.chain_digest(fx.EMPTY_DIGEST, b"two" * 8)
    b = fx.chain_digest(b, b"one" * 8)
    assert a != b


def test_stale_manifest_version_is_refused_not_reported_as_corruption(
    source_db: Path, tmp_path: Path
) -> None:
    out_root = tmp_path / "parquet"
    assert _export(source_db, out_root) == 0
    mpath = fx.manifest_path(out_root, "mark_prices")
    lines = [json.loads(ln) for ln in mpath.read_text(encoding="utf-8").splitlines() if ln.strip()]
    lines[0]["manifest_version"] = 1
    mpath.write_text(
        "\n".join(json.dumps(rec, sort_keys=True) for rec in lines) + "\n", encoding="utf-8"
    )
    assert _verify(source_db, out_root) == 2


def test_empty_partition_records_zero_rows_and_no_file(source_db: Path, tmp_path: Path) -> None:
    """Feed outages produce genuinely empty days; coverage must still be provable."""
    con = fx.open_source_ro(source_db)
    cols = fx.declared_types(con, "mark_prices")
    schema = fx.arrow_schema(cols)
    out_root = tmp_path / "parquet"
    empty_day = DAY0 + 10 * DAY_MS
    record = fx.export_partition(
        con,
        "mark_prices",
        cols,
        schema,
        {
            "symbol": "BTCUSDT",
            "dt": fx.day_label(empty_day),
            "start_ms": empty_day,
            "end_ms": empty_day + DAY_MS,
        },
        out_root,
        100,
        "zstd",
    )
    con.close()
    assert record["rows"] == 0
    assert record["path"] is None
    assert record["digest"] == fx.EMPTY_DIGEST
    assert not fx.partition_path(out_root, "mark_prices", "BTCUSDT", record["dt"]).exists()


def test_verification_resumes_instead_of_starting_over(
    source_db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A full pass takes hours and has died at 80% once; the second pass must carry."""
    out_root = tmp_path / "parquet"
    assert _export(source_db, out_root) == 0
    assert _verify(source_db, out_root) == 0
    first = capsys.readouterr().out
    assert "checked_now=3 carried=0" in first

    assert _verify(source_db, out_root) == 0
    second = capsys.readouterr().out
    assert "checked_now=0 carried=3" in second


def test_reverify_ignores_the_checkpoint(
    source_db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_root = tmp_path / "parquet"
    assert _export(source_db, out_root) == 0
    assert _verify(source_db, out_root) == 0
    capsys.readouterr()
    assert _verify(source_db, out_root, reverify=True) == 0
    assert "checked_now=3 carried=0" in capsys.readouterr().out


def test_a_carried_result_cannot_hide_a_parquet_changed_afterwards(
    source_db: Path, tmp_path: Path
) -> None:
    """The dangerous failure: pass once, tamper, then have the pass carried forward."""
    out_root = tmp_path / "parquet"
    assert _export(source_db, out_root) == 0
    assert _verify(source_db, out_root) == 0

    path = fx.partition_path(out_root, "mark_prices", "ETHUSDT", fx.day_label(DAY0))
    table = pq.read_table(path)
    prices = table.column("mark_price").to_pylist()
    prices[0] += 0.01
    import pyarrow as pa

    mutated = table.set_column(
        table.schema.get_field_index("mark_price"), "mark_price", pa.array(prices, pa.float64())
    )
    pq.write_table(mutated, path, compression="zstd")

    assert _verify(source_db, out_root) == 2


def test_parquet_only_checkpoint_does_not_satisfy_a_full_pass(
    source_db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_root = tmp_path / "parquet"
    assert _export(source_db, out_root) == 0
    assert _verify(source_db, out_root, skip_source_check=True) == 0
    capsys.readouterr()
    # the weaker pass must not let the stronger one skip the source re-query
    assert _verify(source_db, out_root) == 0
    assert "checked_now=3 carried=0" in capsys.readouterr().out


def test_failed_partition_is_not_checkpointed(source_db: Path, tmp_path: Path) -> None:
    out_root = tmp_path / "parquet"
    assert _export(source_db, out_root) == 0
    fx.partition_path(out_root, "mark_prices", "ETHUSDT", fx.day_label(DAY0)).unlink()
    assert _verify(source_db, out_root) == 2

    vpath = fx.verified_path(out_root, "mark_prices")
    carried = fx.read_manifest(vpath) if vpath.exists() else {}
    assert fx.partition_key("mark_prices", "ETHUSDT", fx.day_label(DAY0)) not in carried


def test_second_exporter_is_refused_while_one_is_running(
    source_db: Path, tmp_path: Path
) -> None:
    """Long runs get resumed by hand; a relaunch while one is live is a real accident."""
    out_root = tmp_path / "parquet"
    held = fx.ExportLock(out_root, "mark_prices")
    held.acquire()
    try:
        assert _export(source_db, out_root) == 2
        assert not fx.manifest_path(out_root, "mark_prices").exists()
    finally:
        held.release()
    assert _export(source_db, out_root) == 0


def test_lock_left_by_a_killed_run_is_reclaimed(source_db: Path, tmp_path: Path) -> None:
    out_root = tmp_path / "parquet"
    path = fx.lock_path(out_root, "mark_prices")
    path.parent.mkdir(parents=True, exist_ok=True)
    # a pid that is not running: exactly what a killed export leaves behind
    path.write_text(json.dumps({"pid": 999_999_999}), encoding="utf-8")
    assert _export(source_db, out_root) == 0
    assert not path.exists()


def test_lock_is_released_on_success(source_db: Path, tmp_path: Path) -> None:
    out_root = tmp_path / "parquet"
    assert _export(source_db, out_root) == 0
    assert not fx.lock_path(out_root, "mark_prices").exists()


def test_symbols_are_discovered_without_a_table_scan(source_db: Path) -> None:
    con = fx.open_source_ro(source_db)
    try:
        assert fx.discover_symbols(con, "mark_prices") == ["BTCUSDT", "ETHUSDT"]
    finally:
        con.close()


def test_discovery_finds_a_symbol_nobody_remembered(tmp_path: Path) -> None:
    """The failure mode this guards: an unknown symbol silently missing from the archive."""
    path = tmp_path / "frozen.db"
    _build_source(
        path,
        [
            (DAY0, "BTCUSDT", 1.0, None, None),
            (DAY0, "XRPUSDT", 2.0, None, None),
            (DAY0, "ETHUSDT", 3.0, None, None),
        ],
    )
    con = fx.open_source_ro(path)
    try:
        assert fx.discover_symbols(con, "mark_prices") == ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
    finally:
        con.close()


def test_unknown_symbol_argument_is_refused(source_db: Path, tmp_path: Path) -> None:
    argv = [
        "--table",
        "mark_prices",
        "--db",
        str(source_db),
        "--out-root",
        str(tmp_path / "parquet"),
        "--symbols",
        "BTCUSDT,DOGEUSDT",
    ]
    assert fx.main(argv) == 2  # typo'd symbol must not pass as "nothing to export"


def test_excluding_a_symbol_is_announced_not_silent(
    source_db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "--table",
        "mark_prices",
        "--db",
        str(source_db),
        "--out-root",
        str(tmp_path / "parquet"),
        "--symbols",
        "BTCUSDT",
        "--plan",
    ]
    assert fx.main(argv) == 0
    out = capsys.readouterr().out
    assert "SYMBOLS_EXCLUDED=ETHUSDT" in out


def test_coverage_check_catches_a_symbol_left_out_of_the_export(
    source_db: Path, tmp_path: Path
) -> None:
    """Per-partition digests cannot see rows that were never enumerated."""
    out_root = tmp_path / "parquet"
    argv = [
        "--table",
        "mark_prices",
        "--db",
        str(source_db),
        "--out-root",
        str(out_root),
        "--symbols",
        "BTCUSDT",  # ETHUSDT deliberately omitted
        "--batch-rows",
        "100",
    ]
    assert fx.main(argv) == 0
    # every exported partition is faithful...
    assert _verify(source_db, out_root) == 0
    # ...but the archive is still missing 40 rows, and only this catches it
    assert _verify(source_db, out_root, expect_rows=410) == 2
    assert _verify(source_db, out_root, expect_rows=370) == 0


def test_hive_key_does_not_collide_with_a_real_column(source_db: Path, tmp_path: Path) -> None:
    """`symbol=` as a directory key would make dataset readers refuse the schema."""
    out_root = tmp_path / "parquet"
    assert _export(source_db, out_root) == 0
    path = fx.partition_path(out_root, "mark_prices", "ETHUSDT", fx.day_label(DAY0))
    assert "sym=ETHUSDT" in str(path)
    assert "symbol=" not in str(path)
    # hive inference happens here; a colliding key raises ArrowTypeError instead
    table = pq.read_table(path)
    assert "symbol" in table.column_names


def test_partition_plan_covers_every_day_in_range(source_db: Path) -> None:
    con = fx.open_source_ro(source_db)
    plan = fx.plan_partitions(con, "mark_prices", ["BTCUSDT", "ETHUSDT"])
    con.close()
    btc = [p for p in plan if p["symbol"] == "BTCUSDT"]
    eth = [p for p in plan if p["symbol"] == "ETHUSDT"]
    assert len(btc) == 2
    assert len(eth) == 1
    assert btc[0]["end_ms"] == btc[1]["start_ms"]  # no gap, no overlap
