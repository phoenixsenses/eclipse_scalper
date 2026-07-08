"""Research Marketplace — Part XII §25 priority formula + 60/25/15 portfolio."""
from __future__ import annotations
from ami.research.registry import ResearchQuestion


def priority_score(q: ResearchQuestion) -> float:
    """Priority = gain x relevance x falsifiability x readiness / (cost x MT-risk)."""
    num = (max(q.scientific_value, 0.01)
           * max(q.economic_value + q.risk_reduction_value, 0.01)
           * max(q.falsifiability, 0.01)
           * max(q.data_readiness, 0.01)
           * (0.5 + q.novelty / 2))
    den = max(q.estimated_cost, 0.05) * max(q.multiple_testing_risk, 0.05)
    return round(num / den, 3)


def rank_backlog(questions: list[ResearchQuestion]) -> dict:
    ranked = sorted(questions, key=priority_score, reverse=True)
    # 60/25/15 exploitation/exploration/curiosity split by novelty
    exploit = [q for q in ranked if q.novelty < 0.4]
    explore = [q for q in ranked if 0.4 <= q.novelty < 0.75]
    curiosity = [q for q in ranked if q.novelty >= 0.75]
    return {"ranked": [(q.question_id, priority_score(q), q.question[:80]) for q in ranked],
            "portfolio": {"exploitation_60pct": [q.question_id for q in exploit[:6]],
                          "exploration_25pct": [q.question_id for q in explore[:3]],
                          "curiosity_15pct": [q.question_id for q in curiosity[:2]]}}
