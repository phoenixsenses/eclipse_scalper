r"""LANE C, round 46 -- the corpus's ONE relevant measurement disagrees with C-T41, so match its setup.

Read with tools/corpus_text_v1.py (NUL-safe, ligature- and hyphen-normalised), because an absence
claim made with a raw reader over this corpus is worthless and absence is what this round turns on.

WHAT THE CORPUS ACTUALLY CONTAINS ON THIS, MEASURED:

    multifractal        2 hits, BOTH IN A BIBLIOGRAPHY (Calvet & Fisher 2002; Lux 2008, in
                        Bouchaud's reference list).  Zero body treatment.
    monofractal         0
    scaling exponent    0
    structure function  0
    multiscaling        2, ECONOPHYS_ODM only (one body, one bibliography)
    anomalous scaling   5, ECONOPHYS_ODM only

So C-T41's central result -- that these prices are strongly multifractal, h(q) falling 14.6x to
54.8x the monofractal floor -- sits almost entirely OUTSIDE the corpus's coverage.  That matters
structurally: every form this lane has fitted from the corpus is a SINGLE-EXPONENT form.  The
propagator G(l) ~ l^-beta, the 11.4 collapse, the square-root law.  The corpus never treats the
multifractal case, so it never cautions against applying them to one, and that is the structural
reason four of this lane's exponents turned out to belong to their windows.

BUT THE ABSENCE MUST BE NARROWED, AND THE NARROWING IS THE POINT.  ECONOPHYS_ODM carries one BODY
passage that measures exactly this lane's quantity:

    "an analysis of the nonlinear moments m_zeta of the total return R(t,dt) ... shows that such a
     nonstationarity is accompanied by an anomalous scaling symmetry.  Indeed, to a good
     approximation one finds m_zeta(t,dt) ~ t^D in this range of t, where D ~ 0.364... is
     ESSENTIALLY INDEPENDENT OF zeta"

D independent of the moment order IS the monofractal answer.  On EUR/USD, at tens of minutes, in
CALENDAR time.  C-T41 measured trade time and volume time on crypto perpetuals and got the
opposite.  Two differences could carry that: the asset class, or the clock and scale.

THIS ROUND CHANGES ONLY THE CLOCK AND THE SCALE, and leaves everything else identical to C-T41 --
same estimator, same q grid, same monofractal floor calibrated on fractional Gaussian noise of the
same length.  If h(q) flattens on a calendar clock at tens of minutes, C-T41's multifractality is
a property of the clock this lane chose and not of the asset; if it survives, the disagreement
with the corpus's one measurement is about the market, not the method.

SAMPLE BY ARTIFACT PATH: data/microstructure_02.db :: agg_trades, first 2,000,000 rows by ts_ms
per symbol -- the same population as C-T33/C-T36/C-T37/C-T41.
THRESHOLD DECLARED AND SWEPT: the calendar bar width, at 1 s / 5 s / 30 s.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "microstructure_02.db"
OUT = ROOT / "reports" / "atlas"
SYMS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
NROWS = 2_000_000
QS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
BAR_SECONDS = (1, 5, 30)
SEED = 20260827

# C-T41, for the contrast; and the corpus's one relevant measurement
CT41_TRADE_FALL = {"BTCUSDT": 0.2232, "ETHUSDT": 0.1798, "SOLUSDT": 0.0980}
CT41_VOLUME_FALL = {"BTCUSDT": 0.3674, "ETHUSDT": 0.2975, "SOLUSDT": 0.2043}
CORPUS_D = 0.364


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def zeta_of_q(x, qs=QS, tmin=4, tmax=4000, npts=14):
    n = len(x)
    Ts = np.unique(np.round(np.geomspace(tmin, min(tmax, n // 300), npts)).astype(int))
    out = {}
    for q in qs:
        T, M = [], []
        for t in Ts:
            m = n // t
            if m < 300:
                continue
            s = np.abs(x[:m * t].reshape(m, t).sum(axis=1))
            mom = float(np.mean(s ** q))
            if mom <= 0:
                continue
            T.append(float(t))
            M.append(mom)
        if len(T) < 6:
            continue
        A = np.column_stack([np.ones(len(T)), np.log(T)])
        b, *_ = np.linalg.lstsq(A, np.log(M), rcond=None)
        out[q] = float(b[1])
    return out


def curvature(zq):
    qs = sorted(zq)
    h = [zq[q] / q for q in qs]
    return {"h_of_q": {str(q): round(zq[q] / q, 4) for q in qs},
            "total_fall": round(h[0] - h[-1], 4),
            "h_range": [round(min(h), 4), round(max(h), 4)]}


def fgn(n, H, rng):
    m = 1 << int(np.ceil(np.log2(2 * n)))
    k = np.arange(m // 2 + 1)
    f = np.maximum(k, 1) / m
    psd = f ** (1.0 - 2.0 * H)
    psd[0] = 0.0
    spec = np.sqrt(psd) * np.exp(1j * rng.uniform(0, 2 * np.pi, len(k)))
    spec[0] = 0.0
    spec[-1] = np.abs(spec[-1])
    x = np.fft.irfft(spec, m)[:n]
    return x / np.std(x)


def calendar_bars(ts, lp, sec):
    """last log-price in each equal-width calendar bar, then bar-to-bar returns"""
    t0 = ts[0]
    idx = ((ts - t0) // (sec * 1000)).astype(np.int64)
    nb = int(idx[-1]) + 1
    last = np.full(nb, np.nan)
    last[idx] = lp                      # later writes win -> last trade in the bar
    ok = ~np.isnan(last)
    filled = np.where(ok, last, np.nan)
    # forward-fill empty bars; a bar with no trade carries the previous price
    good = np.flatnonzero(ok)
    if len(good) == 0:
        return np.zeros(0), 0.0
    first = good[0]
    filled = filled[first:]
    m = np.isnan(filled)
    ix = np.where(~m, np.arange(len(filled)), 0)
    np.maximum.accumulate(ix, out=ix)
    filled = filled[ix]
    d = np.empty_like(filled)
    d[0] = 0.0
    d[1:] = np.diff(filled)
    empty_share = float(m.mean())
    return d, empty_share


def main() -> int:
    rng = np.random.default_rng(SEED)

    floor = {}
    for H in (0.5, 0.6, 0.75):
        floor["fgn_H_{0}".format(H)] = curvature(zeta_of_q(fgn(200_000, H, rng)))
    fall_floor = round(float(max(abs(v["total_fall"]) for v in floor.values())), 4)

    con = sqlite3.connect("file:{0}?mode=ro".format(DB.as_posix()), uri=True)
    per = {}
    try:
        for sym in SYMS:
            a = np.array(con.execute(
                "select ts_ms,price from agg_trades where symbol=? order by ts_ms limit ?",
                (sym, NROWS)).fetchall(), dtype=np.float64)
            ts, lp = a[:, 0], np.log(a[:, 1])
            span_h = (ts[-1] - ts[0]) / 3.6e6
            d = {}
            for sec in BAR_SECONDS:
                x, empty = calendar_bars(ts, lp, sec)
                if len(x) < 5000:
                    continue
                c = curvature(zeta_of_q(x))
                c["n_bars"] = int(len(x))
                c["empty_bar_share"] = round(empty, 4)
                c["scale_range_minutes"] = [round(4 * sec / 60.0, 3),
                                            round(min(4000, len(x) // 300) * sec / 60.0, 1)]
                c["fall_over_monofractal_floor"] = round(abs(c["total_fall"]) / fall_floor, 2)
                d["bar_{0}s".format(sec)] = c
            per[sym] = {"span_hours": round(span_h, 1),
                        "calendar": d,
                        "c_t41_trade_clock_fall": CT41_TRADE_FALL[sym],
                        "c_t41_volume_clock_fall": CT41_VOLUME_FALL[sym]}
            sys.stderr.write("{0} done\n".format(sym))
    finally:
        con.close()

    art = {"study": "C-T46", "lane": "C", "utc": _utc(),
           "corpus_reader": "tools/corpus_text_v1.py (NUL-safe, ligature/hyphen normalised)",
           "corpus_absence_measured": {"multifractal": "2, both bibliography",
                                       "monofractal": 0, "scaling exponent": 0,
                                       "structure function": 0,
                                       "multiscaling": "2, ECONOPHYS_ODM only",
                                       "anomalous scaling": "5, ECONOPHYS_ODM only"},
           "corpus_one_relevant_measurement": (
               "ECONOPHYS_ODM: m_zeta(t) ~ t^D on EUR/USD at tens of minutes, D ~ 0.364 "
               "'essentially independent of zeta' -- the MONOFRACTAL answer, in CALENDAR time"),
           "what_changed_from_C_T41": "the clock and the scale only; estimator and q grid identical",
           "monofractal_fall_floor": fall_floor, "floor_calibration": floor,
           "threshold_swept": "calendar bar width 1 s / 5 s / 30 s",
           "per_symbol": per}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C46_CALENDAR_MULTIFRACTAL_V1.json").write_text(json.dumps(art, indent=2),
                                                           encoding="utf-8")

    enc = sys.stdout.encoding or "utf-8"

    def w(s):
        sys.stdout.write(s.encode(enc, "replace").decode(enc, "replace") + "\n")

    w("monofractal floor for the fall of h(q): {0}".format(fall_floor))
    w("corpus's one relevant measurement: D ~ {0}, independent of the moment order "
      "(EUR/USD, calendar, tens of minutes)".format(CORPUS_D))
    w("")
    w("%-9s %-9s %8s %9s %11s %13s %11s %11s" % (
        "sym", "bar", "n_bars", "empty", "scale (min)", "h(q) range", "fall",
        "x floor"))
    for s in SYMS:
        for k, v in per[s]["calendar"].items():
            w("%-9s %-9s %8d %9s %11s %13s %11s %11s" % (
                s, k, v["n_bars"], v["empty_bar_share"], v["scale_range_minutes"],
                v["h_range"], v["total_fall"], v["fall_over_monofractal_floor"]))
        w("%-9s %-9s trade-clock fall %s | volume-clock fall %s   (C-T41)" % (
            s, "", per[s]["c_t41_trade_clock_fall"], per[s]["c_t41_volume_clock_fall"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
