from __future__ import annotations

from pathlib import Path

try:
    from execution import entry_loop as el
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution import entry_loop as el


def test_effective_min_conf_respects_adaptive_guard_flag() -> None:
    assert el._effective_min_conf(0.10, 0.60, True) == 0.60
    assert el._effective_min_conf(0.10, 0.60, False) == 0.10
