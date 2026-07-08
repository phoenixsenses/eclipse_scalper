"""Paket 2 — mutation suite pytest sarmalayicisi (kaynak: ami/mutation_suite.py)."""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ami.mutation_suite import SCENARIOS


@pytest.mark.parametrize("tag,fn", SCENARIOS, ids=[f"m{t}_{f.__name__}" for t, f in SCENARIOS])
def test_mutation(tag, fn, tmp_path):
    res = fn(tmp_path)
    assert res["passed"], (f"MUTATION NOT CAUGHT: {res['name']}\n"
                           f"  injected: {res['injected']}\n"
                           f"  expected: {res['expected']}\n"
                           f"  actual  : {res['actual']}\n"
                           f"  blocker : {res['blocked_by']}")
