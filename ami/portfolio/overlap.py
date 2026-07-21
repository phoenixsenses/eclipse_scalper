"""Portfolio overlap observer — whitepaper §73.1 / Part IX §48+§50 (read-only).

WHY THIS EXISTS
---------------
`OD-021` recorded an open operator question: the paper runner's
`max_open_trades=1` is enforced PER RULE (plus a same-symbol/same-direction
gate); there is NO global portfolio limit. The prompting incident was a
2026-07-10 ETH+SOL simultaneous stop-loss pair -- two rules, two symbols, one
correlated move, two losses.

That incident is not an outlier. Measured on the real paper ledger
(`data/s34_intelligence.db`, 265 closed trades): 138 concurrent-open pairs, of
which ETH+SOL is 64 (~46%), 52 pairs lost together for a combined -3815 bps.
Per-rule gating cannot see any of this, because each rule was individually
within its own limit the whole time.

This module measures that exposure. It does NOT propose, enforce, or size a
global cap -- `permitted` stays research/observation only, and OD-021's actual
decision (global cap yes/no) remains the operator's.

WHAT IS MEASURED VS WHAT IS REFUSED
-----------------------------------
  DERIVED
    concurrent pairs           two trades whose [entry, exit] intervals intersect
    symbol-pair concentration  which pairings dominate the overlap
    joint-loss / joint-stop    how often overlapping trades lost together
    episode-adjusted counts    pairs grouped into independent cascade episodes

  REFUSED
    "correlation" as a number   returns-correlation over 265 trades across 3
                                symbols is not estimable at this N, and §50
                                ("Correlation Beyond Returns") explicitly warns
                                against reducing it to one coefficient. This
                                module reports CO-OCCURRENCE, which is what the
                                ledger can actually support.
    significance / edge claims  none. See N SEMANTICS below.
    capital-path / ergodicity   §73.3 needs an equity path with real sizing;
                                the paper ledger's `net_bps` is per-trade and
                                size-agnostic.

N SEMANTICS (the same disease as OD-011, restated here on purpose)
------------------------------------------------------------------
A pair count is NOT an independent observation count. One cascade episode can
spawn many overlapping pairs, so `pair_n` is inflated the same way trade-N is
inflated relative to cycle-N (measured deflation elsewhere in this repo:
0.410-0.732). `episode_n` applies the canonical-v1 continuity rule (a >4h gap
starts a new episode) so the operator sees both numbers and never mistakes 138
pairs for 138 independent events.

TIMESTAMP SAFETY
----------------
Entry time comes from `s34_signals.signal_ts_ms`, NOT `s34_trades.entry_ts_ms`:
31 of 265 closed trades carry `entry_ts_ms = 0`, which silently chains them at
the epoch and manufactures a fake mega-episode. Rows whose timestamps are
unusable are EXCLUDED and counted, never defaulted to zero.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

# canonical-v1's continuity gate (ami/identity/cycle_resolver.py): a gap larger
# than this starts a new independent episode.
EPISODE_GAP_MS = 4 * 3600 * 1000

TRADE_QUERY = """
SELECT t.trade_id, t.rule_name, t.symbol, t.direction,
       s.signal_ts_ms AS entry_ts_ms, o.exit_ts_ms, o.net_bps, o.exit_reason
FROM s34_outcomes o
JOIN s34_trades  t ON t.trade_id  = o.trade_id
JOIN s34_signals s ON s.signal_id = t.signal_id
ORDER BY s.signal_ts_ms
"""


def load_closed_trades(conn) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return (usable trades, exclusion counts). A trade with a missing/zero
    timestamp or missing net_bps is EXCLUDED and counted -- never coerced."""
    cur = conn.execute(TRADE_QUERY)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    excl = Counter()
    out = []
    for r in rows:
        if not r.get("entry_ts_ms"):
            excl["MISSING_ENTRY_TS"] += 1
        elif not r.get("exit_ts_ms"):
            excl["MISSING_EXIT_TS"] += 1
        elif r.get("net_bps") is None:
            excl["MISSING_NET_BPS"] += 1
        elif r["exit_ts_ms"] <= r["entry_ts_ms"]:
            excl["NON_POSITIVE_DURATION"] += 1
        else:
            out.append(r)
    return out, dict(excl)


def find_concurrent_pairs(trades: list[dict[str, Any]]) -> list[tuple[dict, dict]]:
    """Every pair of trades whose open intervals intersect.

    Half-open intersection: a and b overlap iff a.entry < b.exit AND
    b.entry < a.exit. A trade closing exactly when another opens is NOT
    concurrent -- there is no instant of joint exposure.
    """
    ts = sorted(trades, key=lambda r: r["entry_ts_ms"])
    pairs: list[tuple[dict, dict]] = []
    for i, a in enumerate(ts):
        for b in ts[i + 1:]:
            if b["entry_ts_ms"] >= a["exit_ts_ms"]:
                break  # sorted by entry: nothing later can overlap a
            pairs.append((a, b))
    return pairs


def assign_episodes(trades: list[dict[str, Any]], gap_ms: int = EPISODE_GAP_MS) -> dict[str, int]:
    """Map trade_id -> episode index using canonical-v1's continuity gate,
    applied across the whole book (not per symbol): a portfolio episode is a
    stretch of market time with no >gap quiet period, which is exactly the
    window inside which two symbols can co-move."""
    ts = sorted(trades, key=lambda r: r["entry_ts_ms"])
    out: dict[str, int] = {}
    ep = 0
    for i, r in enumerate(ts):
        if i and r["entry_ts_ms"] - ts[i - 1]["entry_ts_ms"] > gap_ms:
            ep += 1
        out[r["trade_id"]] = ep
    return out


def analyze(conn, *, gap_ms: int = EPISODE_GAP_MS) -> dict[str, Any]:
    """Full read-only portfolio-overlap report."""
    trades, excluded = load_closed_trades(conn)
    if not trades:
        return {"status": "NO_USABLE_TRADES", "excluded": excluded,
                "note": "no usable trades -- NOT evidence of zero overlap"}

    pairs = find_concurrent_pairs(trades)
    episodes = assign_episodes(trades, gap_ms)

    sym_pairs = Counter()
    rule_pairs = Counter()
    joint_loss, joint_stop, joint_win = 0, 0, 0
    joint_loss_bps = 0.0
    pair_episodes = set()
    for a, b in pairs:
        sym_pairs[tuple(sorted((a["symbol"], b["symbol"])))] += 1
        rule_pairs[tuple(sorted((a["rule_name"], b["rule_name"])))] += 1
        pair_episodes.add(episodes[a["trade_id"]])
        if a["net_bps"] < 0 and b["net_bps"] < 0:
            joint_loss += 1
            joint_loss_bps += a["net_bps"] + b["net_bps"]
        if a["net_bps"] > 0 and b["net_bps"] > 0:
            joint_win += 1
        if a["exit_reason"] == "SL" and b["exit_reason"] == "SL":
            joint_stop += 1

    # Worst single episode by summed net across every trade in it.
    ep_net: dict[int, float] = defaultdict(float)
    ep_n: Counter = Counter()
    for r in trades:
        ep_net[episodes[r["trade_id"]]] += r["net_bps"]
        ep_n[episodes[r["trade_id"]]] += 1
    worst_ep = min(ep_net, key=ep_net.get) if ep_net else None

    n_pairs = len(pairs)
    return {
        "status": "OK",
        "trade_n": len(trades),
        "excluded": excluded,
        "excluded_n": sum(excluded.values()),
        "pair_n": n_pairs,
        "episode_n": len(set(episodes.values())),
        "episodes_with_overlap_n": len(pair_episodes),
        "symbol_pair_concentration": dict(sym_pairs.most_common()),
        "rule_pair_concentration": dict(rule_pairs.most_common(10)),
        "joint_loss_pairs": joint_loss,
        "joint_loss_rate": round(joint_loss / n_pairs, 3) if n_pairs else None,
        "joint_loss_total_bps": round(joint_loss_bps, 1),
        "joint_win_pairs": joint_win,
        "joint_stop_pairs": joint_stop,
        "worst_episode": {
            "episode": worst_ep,
            "trades": ep_n[worst_ep],
            "net_bps": round(ep_net[worst_ep], 1),
        } if worst_ep is not None else None,
        "n_semantics": (
            "pair_n counts overlapping trade PAIRS and is NOT an independent-observation "
            "count: one cascade episode can spawn many pairs. episodes_with_overlap_n is the "
            "episode-adjusted figure. No significance or edge claim is made from either."
        ),
        "refused": {
            "returns_correlation": "not estimable at this N across 3 symbols; §50 warns against "
                                   "reducing correlation to one coefficient -- co-occurrence reported instead",
            "capital_path_ergodicity": "§73.3 needs a real sized equity path; net_bps is size-agnostic",
        },
        "no_control_actions_available": True,
        "gating_note": (
            "the runner's max_open_trades=1 is PER RULE plus a same-symbol/same-direction gate; "
            "no global portfolio cap exists (OD-021), so every pair counted here was individually "
            "within its own rule's limit"
        ),
    }
