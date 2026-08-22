"""Frozen E-DER V1 identity adapter.

This module does not redefine the detector.  It calls the canonical historical
implementation and exposes only the already-frozen identity projection needed
by reproduction tests and forward timing code.

Scientific language: forceOrder p*q is an observed forced-liquidation pressure
proxy, not true liquidation volume.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from tools import research_s34_echo_cross_asset_pooled as pooled
from tools import research_s34_echo_impact_elasticity_development as canonical


ROOT = Path(__file__).resolve().parents[1]
FROZEN_MANIFEST = ROOT / "reports/research/s34/S35_E_DER_EVENT_IDENTITY_MANIFEST_V1.csv"
REPORT_ROOT = ROOT / "reports/research/e_der_v1_implementation"
MINUTE_MS = 60_000
MULTISCALE_KEYS = ("i1_v30", "i3_v30", "i5_v30", "i10_v30")


@dataclass(frozen=True, order=True)
class FrozenIdentity:
    event_id: str
    symbol: str
    anchor_ts_ms: int
    base_ms: int
    entry_ms: int
    fixed_boundary_ms: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def frozen_timing(anchor_ts_ms: int) -> tuple[int, int, int]:
    """Return the frozen base, +31m entry and +240m boundary."""
    base_ms = (int(anchor_ts_ms) // MINUTE_MS) * MINUTE_MS + MINUTE_MS
    return base_ms, base_ms + 31 * MINUTE_MS, base_ms + 240 * MINUTE_MS


def identity_from_row(row: dict[str, Any]) -> FrozenIdentity:
    symbol = str(row["symbol"]).upper()
    anchor_ts_ms = int(row["anchor_ts_ms"])
    base_ms, entry_ms, boundary_ms = frozen_timing(anchor_ts_ms)
    return FrozenIdentity(
        event_id=f"E:{symbol}:{anchor_ts_ms}",
        symbol=symbol,
        anchor_ts_ms=anchor_ts_ms,
        base_ms=base_ms,
        entry_ms=entry_ms,
        fixed_boundary_ms=boundary_ms,
    )


def select_frozen_v1_rows(all_candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project the exact code-verified V1 selection chain without modification."""
    operational_pool = [row for row in all_candidates if bool(row["qualified_original"])]
    locked = canonical.lock_by_symbol(operational_pool, 4 * pooled.HOUR_MS)
    selected: list[dict[str, Any]] = []
    for row in locked:
        decision = canonical.multiscale_decision(row, list(MULTISCALE_KEYS))
        metric = row.get("i3_v30")
        if decision is None or decision[0] >= 0 or metric is None:
            continue
        if float(metric["q_echo"]) < float(metric["q_parent"]):
            continue
        if int(row["pre_parent_stress_count"]) < 2:
            continue
        selected.append(row)
    return sorted(selected, key=lambda row: (int(row["anchor_ts_ms"]), str(row["symbol"])))


def generate_canonical_identities() -> list[FrozenIdentity]:
    """Run the canonical historical builder and return identity fields only.

    The canonical builder reads its existing frozen source databases read-only.
    It calculates previously exposed development fields internally; this adapter
    neither emits nor uses those outcome fields when selecting V1 identities.
    """
    _, all_candidates, _ = canonical.make_rows()
    return [identity_from_row(row) for row in select_frozen_v1_rows(all_candidates)]


def load_frozen_manifest(path: Path = FROZEN_MANIFEST) -> list[FrozenIdentity]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [FrozenIdentity(
            event_id=row["event_id"],
            symbol=row["symbol"],
            anchor_ts_ms=int(row["anchor_ts_ms"]),
            base_ms=int(row["base_ms"]),
            entry_ms=int(row["entry_ms"]),
            fixed_boundary_ms=int(row["fixed_boundary_ms"]),
        ) for row in csv.DictReader(handle)]


def reproduction_receipt() -> dict[str, Any]:
    expected = load_frozen_manifest()
    actual = generate_canonical_identities()
    expected_set, actual_set = set(expected), set(actual)
    passed = actual == expected
    return {
        "gate": "FROZEN_25_EXACT_IDENTITY_REPRODUCTION",
        "status": "PASS" if passed else "HARD_STOP",
        "identity_fields": list(FrozenIdentity.__dataclass_fields__),
        "expected_count": len(expected),
        "actual_count": len(actual),
        "exact_ordered_match": passed,
        "missing": [asdict(item) for item in sorted(expected_set - actual_set)],
        "unexpected": [asdict(item) for item in sorted(actual_set - expected_set)],
        "frozen_manifest": str(FROZEN_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "frozen_manifest_sha256": sha256_file(FROZEN_MANIFEST),
        "canonical_module": "tools/research_s34_echo_impact_elasticity_development.py",
        "selection": {
            "qualified_original": True,
            "per_symbol_lock_minutes": 240,
            "multiscale_windows_minutes": [1, 3, 5, 10],
            "volume_normalization_minutes": 30,
            "direction": "SHORT",
            "q_echo_gte_q_parent": True,
            "minimum_prior_stress_anchors": 2,
            "entry": "exact OPEN at base_ms + 31 minutes",
            "boundary": "exact OPEN at base_ms + 240 minutes"
        },
        "outcome_exported_or_used_for_identity_selection": False,
        "source_database_access": "READ_ONLY_VIA_CANONICAL_MODE_RO",
    }


def main() -> int:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    receipt = reproduction_receipt()
    output = REPORT_ROOT / "FROZEN_25_REPRODUCTION_RECEIPT.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"],
        "expected_count": receipt["expected_count"],
        "actual_count": receipt["actual_count"],
        "receipt": str(output),
    }, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
