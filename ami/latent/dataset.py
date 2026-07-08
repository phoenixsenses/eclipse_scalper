"""Faz 6A dataset — 5dk cozunurluklu, outcome'suz sequence grid'i.

Kaynak: data/microstructure.db (salt-okunur). Cikti: data/ami/latent_dataset.npz
+ latent_dataset_meta.json (schema/feature version, availability, missingness,
data-health, session, sampling resolution).

KURAL: Outcome (ileri getiri) kolonlari BU DOSYAYA YAZILMAZ. Degerlendirme
katmani outcome'u freeze sonrasi mark index'ten ayrica hesaplar.
"""
from __future__ import annotations
import json, math, sqlite3, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "microstructure.db"
NPZ = ROOT / "data" / "ami" / "latent_dataset.npz"
META = ROOT / "data" / "ami" / "latent_dataset_meta.json"

SCHEMA_VERSION = "latent_ds_v1"
FEATURE_VERSION = "2026-07-02"
STEP_MS = 5 * 60_000
FEATURES = ["ret5m", "rv30m", "ofi10m", "stress10m", "buyliq10m",
            "fund_vel_1h", "spread5m", "trades10m", "ret1h"]
# ret1h modele girer (backward-looking); ayrica taxonomy-overlap icin de kullanilir.
FORBIDDEN_OUTCOME_FEATURES = ["fwd_", "y_", "label_", "outcome", "net_bps", "mfe", "mae",
                              "continuation", "giveback", "pnl"]


FORBIDDEN_IDENTITY_FEATURES = ["venue", "symbol", "source", "exchange", "feed_id"]


def assert_no_outcome(cols: list[str]) -> None:
    bad = [c for c in cols for f in FORBIDDEN_OUTCOME_FEATURES if f in c.lower()]
    if bad:
        raise ValueError(f"OUTCOME LEAKAGE: forbidden feature(s) in model input: {bad}")
    ident = [c for c in cols for f in FORBIDDEN_IDENTITY_FEATURES if f in c.lower()]
    if ident:
        raise ValueError(f"IDENTITY LEAKAGE: data-source identity in model input: {ident}")


def assert_backward_looking(sample_ts_ms: int, feature_data_max_ts_ms: int) -> None:
    """Future-feature guard: feature'in kullandigi veri sample anini asamaz."""
    if feature_data_max_ts_ms > sample_ts_ms:
        raise ValueError(f"FUTURE FEATURE TIMESTAMP: data {feature_data_max_ts_ms} > sample {sample_ts_ms}")


def era_missing_drop(miss: "np.ndarray", cut: int, expl_max: float = 0.30,
                     era_max: float = 0.90) -> list[int]:
    """Tutulacak feature indeksleri: exploration'da >%30 veya herhangi bir erada
    >%90 eksik olan feature model DISI (missingness'in sahte state olmasini engeller)."""
    keep = []
    for i in range(miss.shape[1]):
        e = float(miss[:cut, i].mean()); v = float(miss[cut:, i].mean()) if cut < miss.shape[0] else 0.0
        if e <= expl_max and e < era_max and v < era_max:
            keep.append(i)
    return keep


def build_dataset(db_path: Path = DB, out_npz: Path = NPZ, out_meta: Path = META,
                  step_ms: int = STEP_MS) -> dict:
    sys.path.insert(0, str(ROOT))
    try:
        from tools.research_s34_knowable_anchor_continuation import load_mark_index
    except ImportError as exc:
        raise ImportError(
            "build_dataset() requires the optional S34 research tool "
            "tools/research_s34_knowable_anchor_continuation.py, which is not "
            "part of the ami/ canonical core and is not present in this checkout."
        ) from exc
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA cache_size=-200000")
    now = int(time.time() * 1000)
    bs = conn.execute("SELECT MIN(ts_ms) FROM book_ticker WHERE symbol='ETHUSDT'").fetchone()[0]
    start = int(bs) + 3_600_000
    m = load_mark_index(conn, "ETHUSDT")

    def px_at(ts):
        r = m.at_or_before(ts)
        return float(r[1]) if r else None

    rows, ts_list, miss_list, sess_list, dq_list = [], [], [], [], []
    ts = start
    n = 0
    while ts <= now - 3_600_000:
        n += 1
        f = {}
        p0 = px_at(ts); p5 = px_at(ts - 300_000); p1h = px_at(ts - 3_600_000)
        f["ret5m"] = (p0 - p5) / p5 * 1e4 if p0 and p5 and p5 > 0 else None
        f["ret1h"] = (p0 - p1h) / p1h * 1e4 if p0 and p1h and p1h > 0 else None
        pxs = [px_at(ts - k * 300_000) for k in range(6, -1, -1)]
        if all(p and p > 0 for p in pxs):
            rets = [math.log(pxs[i + 1] / pxs[i]) for i in range(6)]
            f["rv30m"] = math.sqrt(sum(x * x for x in rets))
        else:
            f["rv30m"] = None
        o = conn.execute("SELECT SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END), SUM(notional), COUNT(*) FROM agg_trades WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?",
                         (ts - 600_000, ts)).fetchone()
        buy = float(o[0] or 0); tot = float(o[1] or 0)
        f["ofi10m"] = (2 * buy - tot) / tot if tot > 0 else None
        f["trades10m"] = float(o[2] or 0)
        st = conn.execute("SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol IN ('ETHUSDT','BTCUSDT') AND side='SELL' AND ts_ms>=? AND ts_ms<?",
                          (ts - 600_000, ts)).fetchone()
        f["stress10m"] = float(st[0] or 0)
        bl = conn.execute("SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol='ETHUSDT' AND side='BUY' AND ts_ms>=? AND ts_ms<?",
                          (ts - 600_000, ts)).fetchone()
        f["buyliq10m"] = float(bl[0] or 0)
        fr0 = conn.execute("SELECT funding_rate FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? AND funding_rate IS NOT NULL ORDER BY ts_ms DESC LIMIT 1", (ts,)).fetchone()
        fr1 = conn.execute("SELECT funding_rate FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? AND funding_rate IS NOT NULL ORDER BY ts_ms DESC LIMIT 1", (ts - 3_600_000,)).fetchone()
        f["fund_vel_1h"] = (float(fr0[0]) - float(fr1[0])) if fr0 and fr1 and fr0[0] is not None and fr1[0] is not None else None
        bt = conn.execute("SELECT spread_pct, ts_ms FROM book_ticker WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (ts,)).fetchone()
        f["spread5m"] = float(bt[0]) if bt and bt[0] is not None and (ts - int(bt[1])) <= 300_000 else None
        vec = [f[k] for k in FEATURES]
        rows.append([v if v is not None else np.nan for v in vec])
        miss_list.append([1 if v is None else 0 for v in vec])
        ts_list.append(ts)
        h = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).hour
        sess_list.append("EUROPE" if 7 <= h < 13 else ("US" if 13 <= h < 21 else "OFF"))
        dq_list.append(0 if (f["spread5m"] is None or f["ret5m"] is None) else 1)
        if n % 4000 == 0:
            print(f"    sample {n}", flush=True)
        ts += step_ms
    conn.close()
    X = np.array(rows, dtype=float)
    assert_no_outcome(FEATURES)
    np.savez_compressed(out_npz, X=X, ts=np.array(ts_list, dtype=np.int64),
                        miss=np.array(miss_list, dtype=np.int8))
    meta = {"schema_version": SCHEMA_VERSION, "feature_version": FEATURE_VERSION,
            "code_ref": "ami/latent/dataset.py", "built_utc": datetime.now(timezone.utc).isoformat(),
            "features": FEATURES, "n_samples": int(X.shape[0]),
            "sampling_resolution_ms": step_ms,
            "range": [int(ts_list[0]), int(ts_list[-1])],
            "symbol": "ETHUSDT", "venue": "BINANCE",
            "sessions": sess_list, "dq_ok": dq_list,
            "availability": "her feature yalniz t<=ts verisi kullanir (backward-looking)",
            "missing_policy": "NaN olarak saklanir; model katmani exploration-medyaniyla impute eder, "
                              ">%30 eksik feature model girdisinden CIKARILIR",
            "outcome_columns": "YOK (kural geregi bu dosyada outcome bulunamaz)"}
    out_meta.write_text(json.dumps(meta, default=str), encoding="utf-8")
    print(f"  dataset: {X.shape} -> {out_npz}")
    return meta


def load_dataset(npz: Path = NPZ, meta_path: Path = META):
    d = np.load(npz)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return d["X"], d["ts"], d["miss"], meta


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    build_dataset()
