"""How sensitive the news layer is to the three numbers nobody has evidence for.

    python -m tools.news_intelligence_calibration
    python -m tools.news_intelligence_calibration --json reports/news_intelligence/sensitivity.json

Three parameters were chosen by judgement and have no measurement behind them:
the clustering similarity threshold (0.32), the clustering window (6h), and the
novelty memory (3 days). They decide how many independent observations exist,
which is the denominator of every significance test this layer will ever feed.

**This tool does not choose them and must never be extended to.** Picking the
value that maximises anything would be selecting a parameter on an outcome — the
same act the whole package is built to prevent, performed one level up. What it
does is show *where the answer changes*, so that a value chosen today is on the
record as having been chosen before any outcome existed, and so that a later
reader can see whether the choice sat on a cliff or on a plateau.

A parameter sitting mid-plateau is a weak assumption. A parameter sitting on a
cliff is a finding waiting to be blamed on the market.

No network, no database, no market data. Runs on the seven fixtures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from eclipse.news_intelligence.adapters.mock import fixture_events  # noqa: E402
from eclipse.news_intelligence.clustering.clusterer import LexicalClusterer  # noqa: E402
from eclipse.news_intelligence.novelty.engine import LexicalNoveltyEngine  # noqa: E402
from eclipse.news_intelligence.pipeline import NewsIntelligencePipeline  # noqa: E402

THRESHOLDS = (0.10, 0.20, 0.26, 0.32, 0.40, 0.50, 0.60, 0.75)
WINDOWS_HOURS = (0.5, 1, 3, 6, 12, 24)
MEMORY_DAYS = (0.25, 1, 3, 7)

#: What the fixtures are known to contain: four distinct real-world events, one
#: of which is reported four times. Stated here rather than inferred, so the
#: sweep is measured against an intent rather than against itself.
TRUTH_CLUSTERS = 4
TRUTH_ITEMS = 7


def _run(threshold: float, window_h: float, memory_d: float) -> dict:
    pipeline = NewsIntelligencePipeline()
    pipeline.clusterer = LexicalClusterer(threshold=threshold, window=timedelta(hours=window_h))
    pipeline.novelty = LexicalNoveltyEngine(memory=timedelta(days=memory_d))
    processed = pipeline.process_batch(fixture_events())
    independent = sum(1 for p in processed if p.is_independent_observation)
    novelties = [p.novelty.novelty_score for p in processed]
    return {
        "threshold": threshold,
        "window_hours": window_h,
        "memory_days": memory_d,
        "items": len(processed),
        "independent": independent,
        "matches_intent": independent == TRUTH_CLUSTERS,
        "min_novelty": round(min(novelties), 4),
        "max_novelty": round(max(novelties), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args()

    rows = []

    print(f"fixtures: {TRUTH_ITEMS} raw items describing {TRUTH_CLUSTERS} real-world events")
    print(f"          (one announcement carried by four sources)\n")

    print("similarity threshold          window 6h, memory 3d")
    print(f"{'threshold':>10} {'independent':>12} {'vs intent':>10}")
    for threshold in THRESHOLDS:
        row = _run(threshold, 6, 3)
        rows.append(row)
        mark = "ok" if row["matches_intent"] else ("SPLITS" if row["independent"] > TRUTH_CLUSTERS else "MERGES")
        print(f"{threshold:>10.2f} {row['independent']:>12} {mark:>10}")

    print("\ncluster window                threshold 0.32, memory 3d")
    print(f"{'hours':>10} {'independent':>12} {'vs intent':>10}")
    for window in WINDOWS_HOURS:
        row = _run(0.32, window, 3)
        rows.append(row)
        mark = "ok" if row["matches_intent"] else ("SPLITS" if row["independent"] > TRUTH_CLUSTERS else "MERGES")
        print(f"{window:>10} {row['independent']:>12} {mark:>10}")

    print("\nnovelty memory                threshold 0.32, window 6h")
    print(f"{'days':>10} {'min novelty':>12} {'max novelty':>12}")
    for memory in MEMORY_DAYS:
        row = _run(0.32, 6, memory)
        rows.append(row)
        print(f"{memory:>10} {row['min_novelty']:>12} {row['max_novelty']:>12}")

    stable = [r for r in rows if r["window_hours"] == 6 and r["memory_days"] == 3 and r["matches_intent"]]
    if stable:
        lo = min(r["threshold"] for r in stable)
        hi = max(r["threshold"] for r in stable)
        print(f"\nthreshold reproduces the intended grouping across [{lo:.2f}, {hi:.2f}]")
        chosen = 0.32
        margin = min(chosen - lo, hi - chosen)
        print(f"chosen 0.32 sits {margin:.2f} from the nearest edge of that range")
        if margin < 0.05:
            print("  -> that is a cliff, not a plateau: the grouping is fragile to this choice")
        else:
            print("  -> plateau: the grouping does not hinge on the exact value")
    else:
        print("\nno threshold in the swept range reproduces the intended grouping")

    print(
        "\nThis tool reports sensitivity. It does not select a value, and extending it "
        "\nto do so against any outcome would be parameter selection on the result."
    )

    if args.json_path:
        os.makedirs(os.path.dirname(args.json_path), exist_ok=True)
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "fixture_items": TRUTH_ITEMS,
                    "fixture_real_events": TRUTH_CLUSTERS,
                    "note": "sensitivity only; no parameter is selected here",
                    "rows": rows,
                },
                handle,
                indent=2,
            )
        print(f"\nwritten to {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
