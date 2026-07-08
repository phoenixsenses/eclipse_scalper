"""Seed AMI stores with current S34 knowledge (whitepaper Part XVI) and the
research backlog (Part XX §109). Idempotent: re-running overwrites same IDs.

Run: python -m ami.seed_s34
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ami.enums import ClaimType, EvidenceLevel, FailureType, KnowledgeStatus, Permission
from ami.knowledge.objects import KnowledgeObject, Provenance
from ami.knowledge.store import KnowledgeStore
from ami.research.registry import Hypothesis, ResearchQuestion, ResearchRegistry

PERIOD = "2026-02-15..2026-07-02"


def _prov(code: str, tables: list[str] | None = None) -> Provenance:
    return Provenance(source_tables=tables or ["liquidations", "mark_prices", "agg_trades", "book_ticker"],
                      data_time_range=PERIOD, code_ref=code, dataset_hash="s34-2026H1")


def seed_knowledge(store: KnowledgeStore) -> list[str]:
    kos = [
        KnowledgeObject(
            knowledge_id="K-S34-HOUR17-001",
            claim="ETH SELL cascade >=200K with hour>=17 UTC + regime(btc4h<0 OR btc7d<0) followed by 6h hold produces positive LONG expectancy (~+40bps net, WR~62%).",
            claim_type=ClaimType.PREDICTIVE, status=KnowledgeStatus.FORWARD_VALIDATING,
            provenance=_prov("tools/research_s34_silence_predictor.py"),
            mechanism="US-afternoon retail overreaction reversion", direction="LONG",
            effect_size={"avg_net_bps": 40.8, "wr": 0.615},
            evidence_level=EvidenceLevel.UNTOUCHED_HOLDOUT, replications=3, holdouts=2,
            scope={"symbols": ["ETHUSDT"], "sessions": ["US", "OFF"], "timeframes": ["event", "6h"]},
            assumptions=["mark fill representative", "fees ~5bps", "Binance liquidation feed complete"],
            confidence={"statistical": "HIGH", "mechanism": "MEDIUM", "forward": "LOW"},
            falsification=["forward 40+ events avg <= 0", "edge absent under executable timestamps"],
            permitted=[Permission.RESEARCH_ONLY, Permission.SHADOW_ALLOWED, Permission.OBSERVER_ALLOWED,
                       Permission.PAPER_ALLOWED],
            forbidden=[Permission.SIZING_ALLOWED],
            decay_half_life_days=90),
        KnowledgeObject(
            knowledge_id="K-S34-BOOK-PULL-001",
            claim="Low pre-cascade bid-depth withdrawal (depth holds) is associated with higher post-cascade reversal outcomes in the gated ETH SELL universe (TEST delta ~ +70bps, WR 70.7%).",
            claim_type=ClaimType.PREDICTIVE, status=KnowledgeStatus.HOLDOUT_VALIDATED,
            provenance=_prov("tools/s34_mechanism_taxonomy.py", ["book_ticker", "liquidations", "mark_prices"]),
            mechanism="LIQUIDITY_REMAINS_AND_ABSORBS_FORCED_FLOW", direction="LONG",
            effect_size={"delta_bps": 70.2},
            evidence_level=EvidenceLevel.UNTOUCHED_HOLDOUT, replications=2, holdouts=1,
            scope={"symbols": ["ETHUSDT"], "timeframes": ["event", "6h"], "regimes": ["tested_gate_only"]},
            assumptions=["book data health HEALTHY", "feature definition frozen"],
            confidence={"statistical": "HIGH", "mechanism": "MEDIUM", "forward": "LOW", "generalization": "UNKNOWN"},
            falsification=["negative/null forward effect after minimum sample",
                           "effect disappears under executable timestamps"],
            permitted=[Permission.RESEARCH_ONLY, Permission.SHADOW_ALLOWED],
            decay_half_life_days=60),
        KnowledgeObject(
            knowledge_id="K-S34-REFILL-CTX-001",
            claim="bk_refill separator changed sign between ungated and gated universes; refill is NOT a universal mechanism.",
            claim_type=ClaimType.DESCRIPTIVE, status=KnowledgeStatus.REGIME_LIMITED,
            provenance=_prov("tools/s34_mechanism_taxonomy.py", ["book_ticker"]),
            mechanism="context-dependent maker return", direction="BOTH",
            evidence_level=EvidenceLevel.CHRONOLOGICAL, replications=1,
            falsification=["stable same-sign effect across both universes in forward data"],
            confidence={"statistical": "MEDIUM"},
            permitted=[Permission.RESEARCH_ONLY]),
        KnowledgeObject(
            knowledge_id="K-S34-FUNDING-LEVEL-001",
            claim="Funding LEVEL (<0) separates post-cascade LONG outcomes more strongly than funding velocity in the tested family (TEST delta ~+115bps).",
            claim_type=ClaimType.PREDICTIVE, status=KnowledgeStatus.HOLDOUT_VALIDATED,
            provenance=_prov("tools/research_s34_signal_discovery_v2.py", ["mark_prices"]),
            direction="LONG", effect_size={"delta_bps": 114.8},
            evidence_level=EvidenceLevel.UNTOUCHED_HOLDOUT, replications=2, holdouts=1,
            scope={"symbols": ["ETHUSDT"]},
            falsification=["forward delta <= 0 after 30+ events"],
            confidence={"statistical": "HIGH", "mechanism": "MEDIUM"},
            permitted=[Permission.RESEARCH_ONLY, Permission.SHADOW_ALLOWED]),
        KnowledgeObject(
            knowledge_id="K-S34-MONDAY-VETO-001",
            claim="Monday entries in the hour17 family are strongly negative across two universes and both chronological splits (WR 18-37%, avg -42..-47bps).",
            claim_type=ClaimType.PREDICTIVE, status=KnowledgeStatus.HOLDOUT_VALIDATED,
            provenance=_prov("tools/research_s34_refined_recipe_gauntlet.py"),
            direction="LONG", effect_size={"monday_avg_bps": -47.4},
            evidence_level=EvidenceLevel.UNTOUCHED_HOLDOUT, replications=2, holdouts=2,
            scope={"symbols": ["ETHUSDT"]},
            falsification=["Monday forward avg > 0 over 15+ events"],
            confidence={"statistical": "HIGH", "mechanism": "LOW"},
            permitted=[Permission.RESEARCH_ONLY, Permission.SHADOW_ALLOWED, Permission.OBSERVER_ALLOWED]),
        KnowledgeObject(
            knowledge_id="K-S34-MGMT-6H-001",
            claim="Uniform 6h hold with no stop and no early exit remains the strongest tested management baseline; sub-1h bar trailing destroys the edge.",
            claim_type=ClaimType.OPERATIONAL, status=KnowledgeStatus.HOLDOUT_VALIDATED,
            provenance=_prov("tools/research_s34_trade_mgmt_gauntlet.py"),
            direction="LONG",
            evidence_level=EvidenceLevel.UNTOUCHED_HOLDOUT, replications=3, holdouts=2,
            falsification=["a management variant beats baseline on TEST without higher mdd"],
            confidence={"statistical": "HIGH", "mechanism": "MEDIUM"},
            permitted=[Permission.RESEARCH_ONLY, Permission.SHADOW_ALLOWED, Permission.OBSERVER_ALLOWED,
                       Permission.PAPER_ALLOWED]),
        KnowledgeObject(
            knowledge_id="K-S34-SCALEIN-100-001",
            claim="Adding one unit at -100bps within first 2h improved per-unit expectancy and reduced mdd in-sample; requires forward observation before any sizing use.",
            claim_type=ClaimType.OPERATIONAL, status=KnowledgeStatus.PRELIMINARY,
            provenance=_prov("tools/research_s34_trade_mgmt_gauntlet.py"),
            direction="LONG", effect_size={"per_unit_bps": 86.3, "baseline_bps": 65.1},
            evidence_level=EvidenceLevel.IN_SAMPLE, replications=1,
            falsification=["forward combined per-unit <= baseline over 20+ adds"],
            confidence={"statistical": "MEDIUM", "forward": "NONE"},
            permitted=[Permission.RESEARCH_ONLY, Permission.OBSERVER_ALLOWED],
            forbidden=[Permission.LIVE_ALLOWED, Permission.SIZING_ALLOWED, Permission.PAPER_ALLOWED]),
        KnowledgeObject(
            knowledge_id="K-S34-PRECASCADE-001",
            claim="Pre-cascade state detection achieves ~2x lift on cascade probability but honest all-alert trading is negative both directions; early entry is navigation, not a trade rule.",
            claim_type=ClaimType.PREDICTIVE, status=KnowledgeStatus.HOLDOUT_VALIDATED,
            provenance=_prov("tools/s34_precascade_predictor.py", ["mark_prices", "agg_trades", "spot_prices", "liquidations"]),
            direction="NONE", effect_size={"lift10": 2.1, "long_at_trigger_bps": -30.4},
            evidence_level=EvidenceLevel.UNTOUCHED_HOLDOUT, replications=1, holdouts=1,
            falsification=["a trigger subset with positive honest EV over 50+ triggers"],
            confidence={"statistical": "HIGH"},
            permitted=[Permission.RESEARCH_ONLY]),
        KnowledgeObject(
            knowledge_id="K-S34-MECH-COMPOSITE-001",
            claim="Mechanism composite (funding<0, rv-hi, retail, ofi+, two-sided, refill, pull) >=4 on the gated universe: TEST WR 82.6%, avg +88bps; forward validation required.",
            claim_type=ClaimType.PREDICTIVE, status=KnowledgeStatus.HOLDOUT_VALIDATED,
            provenance=_prov("tools/s34_mechanism_taxonomy.py"),
            direction="LONG", effect_size={"test_avg_bps": 88.0, "test_wr": 0.826},
            evidence_level=EvidenceLevel.UNTOUCHED_HOLDOUT, replications=1, holdouts=1,
            scope={"symbols": ["ETHUSDT"], "regimes": ["hour17_gated"]},
            falsification=["forward avg <= 0 after 20+ events"],
            confidence={"statistical": "HIGH", "mechanism": "MEDIUM", "forward": "NONE"},
            permitted=[Permission.RESEARCH_ONLY, Permission.SHADOW_ALLOWED]),
    ]
    ids = []
    for ko in kos:
        store.put(ko, actor="seed_s34")
        ids.append(ko.knowledge_id)
    # genealogy + contradiction edges
    store.link("K-S34-MECH-COMPOSITE-001", "DEPENDS_ON", "K-S34-BOOK-PULL-001", actor="seed_s34")
    store.link("K-S34-MECH-COMPOSITE-001", "DEPENDS_ON", "K-S34-FUNDING-LEVEL-001", actor="seed_s34")
    store.link("K-S34-REFILL-CTX-001", "RESTRICTS", "K-S34-MECH-COMPOSITE-001", actor="seed_s34")
    store.link("K-S34-HOUR17-001", "SUPPORTS", "K-S34-MECH-COMPOSITE-001", actor="seed_s34")
    return ids


def seed_failures(store: KnowledgeStore) -> int:
    fails = [
        ("naive early pre-cascade entry (LONG/SHORT at trigger)", FailureType.NO_EDGE,
         "honest all-alert EV negative both directions (mc 0.88-0.99)", "retry with much stronger trigger precision"),
        ("broad micro-timing optimization 2s-15m", FailureType.NO_EDGE,
         "no delay creates alpha on ungated universe", ""),
        ("tight volatility-scaled stops", FailureType.EXECUTION_FAILURE,
         "rv-stops cut dip-recovery mechanism (TEST avg 3-4 vs 50.7)", "different mechanism family"),
        ("partial exits (half@+100/+150, thirds, half@3h)", FailureType.NO_EDGE,
         "all below 6h baseline without mdd benefit", ""),
        ("limit-entry improvement -10/-20/-30bps", FailureType.EXECUTION_FAILURE,
         "missed fills cost more than price improvement (EV/signal 24 vs 31)", "maker-fee VIP regime"),
        ("bk_refill as universal separator", FailureType.REGIME_LIMITED,
         "sign flip between ungated and gated universes", "forward data may resolve"),
        ("cascade gentleness/grind signal", FailureType.OVERFIT,
         "TRAIN/TEST inconsistent", ""),
        ("BUY-side short-squeeze fade (all variants)", FailureType.NO_EDGE,
         "graveyard: 5:159 positive ratio across reports", ""),
        ("cross-asset transfer (SOL/BTC own cascades)", FailureType.NO_EDGE,
         "edge is ETH-specific (mc 0.12-0.17)", ""),
        ("loser time-stops (1-3h x -25..-100)", FailureType.NO_EDGE,
         "losers recover; all grid cells below baseline", ""),
        ("clock-targeted exits (23:00/03:00/07:00 UTC)", FailureType.NO_EDGE,
         "exit-hour effect not convertible to policy", ""),
        ("silence as T0-tradeable main alpha", FailureType.LOOKAHEAD,
         "silence known only at T+30m; confirm-entry kills edge", ""),
    ]
    for idea, ft, reason, retry in fails:
        store.archive_failure(idea, ft, reason=reason, data_period=PERIOD, retry=retry)
    return len(fails)


def seed_backlog(reg: ResearchRegistry) -> int:
    qs = [
        ResearchQuestion("Q-MFE-GIVEBACK-001",
                         "Can post-entry state transitions identify profitable MFE giveback exits without damaging 6h swing expectancy?",
                         origin_observation="MFE +143.7 vs captured +65.1 (giveback 73.6bps)",
                         scientific_value=0.8, economic_value=0.9, novelty=0.35,
                         falsifiability=0.8, data_readiness=0.9, estimated_cost=0.3,
                         multiple_testing_risk=0.4, required_sample=60),
        ResearchQuestion("Q-SHORTNOISY-SUBTYPES-001",
                         "Does SHORT_NOISY split into scalp / no-trade / failed-short LONG transition / daily LONG subtypes?",
                         origin_observation="whitepaper §109.2; SHORT_NOISY MFE giveback pattern",
                         scientific_value=0.8, economic_value=0.7, novelty=0.55,
                         falsifiability=0.7, data_readiness=0.7, estimated_cost=0.4, required_sample=40),
        ResearchQuestion("Q-MECHCOMP-FORWARD-001",
                         "Does the mechanism composite survive forward shadow validation?",
                         origin_observation="gated mech>=4 TEST WR 82.6",
                         scientific_value=0.7, economic_value=0.9, novelty=0.2,
                         falsifiability=0.9, data_readiness=0.6, estimated_cost=0.2,
                         multiple_testing_risk=0.2, required_sample=20),
        ResearchQuestion("Q-BKPULL-FORWARD-001",
                         "Does bk_pull retain direction and effect under new forward data?",
                         origin_observation="pull robust across universes; refill flipped",
                         scientific_value=0.7, economic_value=0.6, novelty=0.3,
                         falsifiability=0.9, data_readiness=0.6, estimated_cost=0.2, required_sample=30),
        ResearchQuestion("Q-OI-EXHAUSTION-001",
                         "Does OI contraction vs expansion distinguish exhaustion from continuation after cascades?",
                         origin_observation="OI collection restarted 2026-07-02; no history",
                         scientific_value=0.9, economic_value=0.7, novelty=0.7,
                         falsifiability=0.8, data_readiness=0.2, estimated_cost=0.3, required_sample=30),
        ResearchQuestion("Q-FUND-OI-BOOK-LONG-001",
                         "Can funding/OI/book interactions create an independent LONG family (no cascade dependency)?",
                         origin_observation="whitepaper §109.6",
                         scientific_value=0.9, economic_value=0.8, novelty=0.8,
                         falsifiability=0.6, data_readiness=0.3, estimated_cost=0.6,
                         multiple_testing_risk=0.6, required_sample=50),
        ResearchQuestion("Q-TRANSITION-MODEL-001",
                         "Can a state-transition model outperform static rule selection?",
                         origin_observation="structure transition matrix now estimable",
                         scientific_value=0.9, economic_value=0.7, novelty=0.75,
                         falsifiability=0.6, data_readiness=0.5, estimated_cost=0.6, required_sample=100),
        ResearchQuestion("Q-META-VALIDATION-001",
                         "Which validation metrics best predict forward survival of S34 candidates?",
                         origin_observation="research-vs-paper gap (LONG_SILENCE research >> real 44%)",
                         scientific_value=0.9, economic_value=0.5, risk_reduction_value=0.8, novelty=0.6,
                         falsifiability=0.7, data_readiness=0.8, estimated_cost=0.4, required_sample=30),
        ResearchQuestion("Q-SIZING-EVIDENCE-001",
                         "What evidence threshold should permit sizing influence (double-trigger, conviction units)?",
                         origin_observation="sizing is the biggest lever but highest risk",
                         scientific_value=0.6, economic_value=0.9, risk_reduction_value=0.6, novelty=0.3,
                         falsifiability=0.7, data_readiness=0.7, estimated_cost=0.3, required_sample=40),
        ResearchQuestion("Q-EXPIRATION-RISK-001",
                         "Which current claims have the highest expiration risk (regime dependence)?",
                         origin_observation="whitepaper §109.10",
                         scientific_value=0.7, economic_value=0.4, risk_reduction_value=0.7, novelty=0.5,
                         falsifiability=0.5, data_readiness=0.9, estimated_cost=0.2, required_sample=0),
    ]
    for q in qs:
        reg.add_question(q)
    reg.add_hypothesis(Hypothesis("H-SHORTNOISY-1", "Q-SHORTNOISY-SUBTYPES-001",
                                  "SHORT_NOISY is scalp-only (edge < 30m)", "PRIMARY",
                                  prediction="30m EV > 0, 4h EV <= 0"))
    reg.add_hypothesis(Hypothesis("H-SHORTNOISY-2", "Q-SHORTNOISY-SUBTYPES-001",
                                  "Failed SHORT_NOISY transitions to 4h/1D LONG", "ALTERNATIVE",
                                  prediction="reclaim subset LONG 4h EV > 0"))
    return len(qs)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    store = KnowledgeStore()
    reg = ResearchRegistry()
    ids = seed_knowledge(store)
    nf = seed_failures(store)
    nq = seed_backlog(reg)
    print(f"knowledge objects: {len(ids)}")
    print(f"failure archive : {nf}")
    print(f"backlog questions: {nq}")
    print("stores:", store.path, "|", reg.path)
    store.close(); reg.close()


if __name__ == "__main__":
    main()
