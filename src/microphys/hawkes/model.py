from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HawkesModelStub:
    """TODO(Phase2+): self-exciting intensity modeling."""

    note: str = "hawkes intensity placeholder"

    def fit(self, _rows: Any) -> None:
        raise NotImplementedError("HawkesModelStub.fit is a placeholder")
