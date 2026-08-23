"""Run the news layer over its fixtures and print what it produced.

    python -m tools.news_intelligence_demo
    python -m tools.news_intelligence_demo --json reports/news_intelligence/demo_payload.json

Deliberately tiny. No network, no database, no market store, no model: seven
synthetic items through the full path, in memory, in well under a second. It
exists so an operator can see the layer working without starting anything, while
the machine is busy with research.

Everything that would cost real resources is registered in
`eclipse.news_intelligence.deferred` and refuses to start. The last section of
the output lists those refusals, because "built but deliberately not running" and
"not built" must never look the same from the outside.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from eclipse.news_intelligence import deferred  # noqa: E402
from eclipse.news_intelligence.adapters.mock import MockAdapter  # noqa: E402
from eclipse.news_intelligence.pipeline import NewsIntelligencePipeline  # noqa: E402
from eclipse.news_intelligence.research.api import ResearchStore  # noqa: E402
from eclipse.news_intelligence.research.dashboard_contract import dashboard_payload  # noqa: E402
from eclipse.news_intelligence.validation.audit import explain  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_path", default=None,
                        help="also write the private dashboard payload here")
    parser.add_argument("--explain", dest="explain_index", type=int, default=None,
                        help="print the full audit trail for one event, by row number")
    args = parser.parse_args()

    pipeline = NewsIntelligencePipeline()
    raws = list(MockAdapter().poll())
    processed = [pipeline.process(raw) for raw in raws]

    store = ResearchStore()
    for item in processed:
        store.add_snapshot(item.snapshot, high_impact=item.is_high_impact)

    print(f"{'time':<6} {'entity':<17} {'type':<18} {'nov':>5} {'ampl':>5}  {'kind':<11} assets")
    print("-" * 96)
    for item in processed:
        event = item.event
        print(
            f"{event.first_seen_at.strftime('%H:%M'):<6} "
            f"{(event.entity or '—'):<17} "
            f"{event.event_type.value:<18} "
            f"{(event.novelty or 0):>5.2f} "
            f"{(event.amplification_score or 0):>5.2f}  "
            f"{('INDEPENDENT' if item.is_independent_observation else 'repeat'):<11} "
            f"{', '.join(event.asset_relevance.relevant(0.5)[:4])}"
        )

    counters = store.counters()
    print()
    print(f"raw items ................ {counters.raw_items}")
    print(f"independent clusters ..... {counters.independent_clusters}")
    print(f"duplication ratio ........ {counters.duplication_ratio}  "
          f"(quoting raw items as N would overstate the sample by this factor)")
    print(f"high impact .............. {counters.high_impact_events}")
    print(f"complete market labels ... {counters.events_with_complete_labels}  "
          f"(measurement is deferred; an empty count here is correct)")

    if args.explain_index is not None:
        item = processed[args.explain_index]
        raw = next(r for r in raws if r.raw_event_id == item.event.raw_event_id)
        print()
        print(json.dumps(explain(item.event, raw).as_dict(), indent=2, ensure_ascii=False))

    print()
    print("deferred while the machine is busy:")
    for row in deferred.register_report():
        print(f"  {row['key']:<30} {row['status']}  ({row['resource']})")

    if args.json_path:
        payload = dashboard_payload(
            processed,
            counters,
            market_state={},
            as_of=datetime.now(timezone.utc),
            complete_assets=(),
        )
        os.makedirs(os.path.dirname(args.json_path), exist_ok=True)
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print()
        print(f"private dashboard payload written to {args.json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
