"""Is this new information, or the same information again?

The distinction the whole layer turns on. A market prices information once. The
fiftieth article about a tariff announcement is not fifty times the news, and a
study that treats it as news will find "news predicts nothing" for the excellent
reason that most of its sample was an echo.

Novelty is measured against what was already known **at the moment the item
arrived** — never against the corpus as it looks today. Scoring an old item
against a later corpus would let information from the future decide how
surprising something was at the time, which is lookahead wearing a lab coat.

The lexical implementation is a floor, and the interface is what matters: an
embedding index answers the same question better and is deferred while the
machine is busy. Swapping it in must not change a single caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

from ..clustering.clusterer import ClusterInput, jaccard


@dataclass(frozen=True, slots=True)
class NoveltyResult:
    novelty_score: float
    nearest_previous_event_id: str | None
    similarity: float
    time_since_similar_seconds: float | None
    method: str


@runtime_checkable
class NoveltyEngine(Protocol):
    method: str

    def score(self, item: ClusterInput) -> NoveltyResult:
        ...


class LexicalNoveltyEngine:
    """Novelty as one minus the best similarity to anything already seen.

    Only items strictly earlier than the one being scored participate, so the
    engine can be run over a historical stream and produce the same answer it
    would have produced live. Recency is a tiebreak, not a discount: something
    repeated after a long silence is genuinely more novel than the same thing
    repeated twice in a minute, and `time_since_similar_seconds` is exported so
    research can decide how much that is worth rather than having it baked in.
    """

    method = "lexical-jaccard@1"

    def __init__(self, memory: timedelta = timedelta(days=3)) -> None:
        self.memory = memory
        self._seen: list[tuple[datetime, str, frozenset[str], str]] = []

    def score(self, item: ClusterInput) -> NoveltyResult:
        tokens = item.tokens()
        cutoff = item.first_seen_at - self.memory
        best_similarity, best_id, best_time = 0.0, None, None

        for seen_at, event_id, seen_tokens, entity in self._seen:
            if seen_at >= item.first_seen_at or seen_at < cutoff:
                continue
            if entity and item.entity and entity != item.entity:
                # Cross-entity similarity is real but is a different question;
                # mixing it in here would make novelty depend on unrelated news
                # volume.
                continue
            similarity = jaccard(tokens, seen_tokens)
            if similarity > best_similarity:
                best_similarity, best_id, best_time = similarity, event_id, seen_at

        self._seen.append((item.first_seen_at, item.event_id, tokens, item.entity))
        # Anything older than the memory window can no longer be the nearest
        # match, so keeping it only makes the scan longer. Left unpruned this
        # list grew for the life of the process and the comparison went
        # quadratic in a stream that never ends. Pruned on every call rather
        # than past some count: the scan above is already linear in `_seen`, so
        # a size guard saves nothing and only makes the bound untestable.
        self._seen = [entry for entry in self._seen if entry[0] >= cutoff]

        return NoveltyResult(
            novelty_score=round(1.0 - best_similarity, 6),
            nearest_previous_event_id=best_id,
            similarity=round(best_similarity, 6),
            time_since_similar_seconds=(
                (item.first_seen_at - best_time).total_seconds() if best_time else None
            ),
            method=self.method,
        )

    def remembered(self) -> int:
        return len(self._seen)


__all__ = ["NoveltyEngine", "LexicalNoveltyEngine", "NoveltyResult"]
