"""The storage panel must report the db being written NOW, not the frozen segment.

`DashboardContext.microstructure_db` hard-coded `data/microstructure.db`. After the
2026-07-23 rotation that path is the FROZEN file, so the canonical :8770 dashboard's
storage-health panel showed 836 GiB of dead history instead of the live file -- and
once that segment is reclaimed it would have reported MISSING/MEDIUM forever: the
panel that exists to report on storage, broken by the storage operation itself.
"""

from __future__ import annotations

import json
from pathlib import Path

from dashboard.backend.sources import DashboardContext


def test_resolves_the_live_segment_not_the_frozen_one():
    ctx = DashboardContext(Path("D:/eclipse_scalper"))
    resolved = Path(ctx.microstructure_db)
    state = json.loads(Path("D:/eclipse_scalper/data/rotation_state.json").read_text(encoding="utf-8"))
    assert resolved == Path(state["live_db_path"])
    frozen = {Path(s["path"]) for s in state["frozen_segments"]}
    assert resolved not in frozen, "the dashboard is pointed at a frozen segment"


def test_falls_back_instead_of_taking_the_page_down(monkeypatch, tmp_path):
    """A resolver failure must degrade one panel, not raise through the request."""
    import ami.storage.union_reader as UR

    def _boom(*a, **k):
        raise RuntimeError("rotation state unreadable")

    monkeypatch.setattr(UR, "current_live_db_path", _boom)
    ctx = DashboardContext(tmp_path)
    assert Path(ctx.microstructure_db) == tmp_path / "data" / "microstructure.db"
