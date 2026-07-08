"""AMI end-to-end phase validation on REAL data (read-only).

Whitepaper Part XXI 'Definition of Done' (early version) checks 1-10:
 1. observe with health/provenance      -> StateEngine.feed_ages + quality
 2. multi-timeframe states              -> StateBundle 1m..1W
 3. LONG/SHORT/NO-TRADE probabilities   -> decide() direction candidates
 4. trade lifecycle tracking            -> LifecycleEngine.replay_shadow_ledger
 5. governed Knowledge Objects          -> KnowledgeStore + seeds
 6. generate & rank research questions  -> marketplace ranking
 7. preregistered experiments           -> ResearchRegistry freeze demo
 8. contradictions & revision           -> governor.revise + graph
 9. explain recommendations             -> DecisionTrace packet
10. permission boundaries               -> authorize() outcomes

Run: python -m ami.run_phase_checks
Writes: reports/research/s34/AMI_PHASE_VALIDATION.md + data/ami/last_bundle.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ami.decision.trace import decide
from ami.enums import Action, KnowledgeStatus
from ami.governance.governor import EpistemicGovernor
from ami.knowledge.store import KnowledgeStore
from ami.lifecycle.engine import LifecycleEngine
from ami.research.marketplace import rank_backlog
from ami.research.registry import EvidenceBundle, ExperimentSpec, ResearchRegistry
from ami.seed_s34 import seed_backlog, seed_failures, seed_knowledge
from ami.states.engine import StateEngine
from ami.states.structure import estimate_transition_matrix

OUT_MD = ROOT / "reports" / "research" / "s34" / "AMI_PHASE_VALIDATION.md"
OUT_BUNDLE = ROOT / "data" / "ami" / "last_bundle.json"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    lines = ["# AMI Phase Validation (real data)", "",
             f"> {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}  ami v0.1.0", ""]
    ok = []

    # 5/6/7 stores + seeds
    store = KnowledgeStore(); reg = ResearchRegistry()
    ids = seed_knowledge(store); seed_failures(store); seed_backlog(reg)
    gov = EpistemicGovernor(store)
    print(f"[5] knowledge store: {len(store.all())} objects, {len(store.failures())} archived failures")
    ok.append(("5. governed knowledge objects", f"{len(store.all())} KO, {len(store.failures())} failures"))

    # 1/2 states
    eng = StateEngine()
    ages = eng.feed_ages_min(); dq = eng.data_quality(ages)
    bundle = eng.build_bundle("ETHUSDT")
    OUT_BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_BUNDLE.write_text(json.dumps(bundle.to_dict(), indent=2, default=str), encoding="utf-8")
    print(f"[1] feed ages (min): { {k: v for k, v in ages.items()} }")
    print(f"[1] quality: { {k: v.value for k, v in dq.items()} }")
    print(f"[2] bundle: {len(bundle.states)} states; conflict={bundle.conflict_report()}")
    ok.append(("1. observation health", json.dumps({k: v.value for k, v in dq.items()})))
    ok.append(("2. multi-TF states", f"{len(bundle.states)} states, conflict={bundle.conflict_report()}"))

    # structure transition matrix (Phase 2)
    tm = estimate_transition_matrix(eng, days=60)
    top_ph = list(tm["phase_freq"].items())[:4]
    print(f"[2b] structure 4h/60d: bars={tm['n_bars']} top phases={top_ph}")
    ok.append(("2b. structure transitions", f"bars={tm['n_bars']} phases={top_ph}"))

    # 3/9/10 decision + trace + permissions
    ctx = {"symbol": "ETHUSDT", "data_health": "HEALTHY"}
    trace = decide(bundle, gov, ["K-S34-HOUR17-001", "K-S34-MECH-COMPOSITE-001"], ctx)
    print(f"[3] direction: {trace.direction_candidates}  action={trace.action} -> {trace.result}")
    print(f"[9] trace persisted: {trace.decision_id} uncertainty={trace.uncertainty}")
    ok.append(("3. direction probabilities", json.dumps(trace.direction_candidates)))
    ok.append(("9. decision trace", f"{trace.decision_id} result={trace.result} unc={trace.uncertainty}"))
    live_try = gov.authorize(Action.OPEN_LONG, ["K-S34-MECH-COMPOSITE-001"], ctx)
    assert live_try.result != "GRANTED", "HOLDOUT_VALIDATED must not authorize live"
    print(f"[10] live attempt on HOLDOUT_VALIDATED -> {live_try.result} (correctly not GRANTED)")
    ok.append(("10. permission boundary", f"OPEN_LONG on holdout-KO -> {live_try.result}"))

    # 4 lifecycle on real shadow ledger
    lce = LifecycleEngine()
    lc = lce.replay_shadow_ledger(limit=120)
    print(f"[4] lifecycle: trades={lc['n_trades']} mfe50 n={lc['mfe50_n']} outcomes={lc['mfe50_outcomes']}")
    ok.append(("4. trade lifecycle", f"trades={lc['n_trades']} mfe50={lc['mfe50_outcomes']}"))

    # 6 marketplace
    ranked = rank_backlog(reg.questions())
    print("[6] top questions:", [r[0] for r in ranked["ranked"][:5]])
    ok.append(("6. research marketplace", str([r[0] for r in ranked["ranked"][:5]])))

    # 7 preregistration demo (freeze + evidence attach)
    spec = ExperimentSpec(
        experiment_id="E-MECHCOMP-FWD-001", question_id="Q-MECHCOMP-FORWARD-001",
        population="forward gated hour17 composite events from 2026-07-02",
        target="net_bps_6h", features=["mech_score"], threshold_method="frozen(>=4)",
        untouched_data="all events after 2026-07-02", negative_control="mech_score<=1 subset",
        decision_criteria="avg net > 0 and WR > 55% at n>=20",
        falsification_rule="avg net <= 0 at n>=20")
    spec.freeze(); reg.register_experiment(spec)
    reg.attach_evidence(EvidenceBundle("EV-0", "E-MECHCOMP-FWD-001",
                                       {"note": "registered; forward accumulating"}, "INCONCLUSIVE"), spec)
    print(f"[7] experiment frozen: {spec.experiment_id} hash={spec.frozen_hash}")
    ok.append(("7. preregistration", f"{spec.experiment_id} frozen={spec.frozen_hash}"))

    # 8 contradiction handling demo (graph edge -> promotion block)
    ko = store.get("K-S34-MECH-COMPOSITE-001")
    blocked = "no"
    try:
        gov.promote(ko, KnowledgeStatus.FORWARD_VALIDATING)
    except Exception as e:
        blocked = f"yes ({type(e).__name__})"
    print(f"[8] premature promotion blocked: {blocked}")
    ok.append(("8. contradiction/gates", f"premature promotion blocked: {blocked}"))

    eng.close(); store.close(); reg.close()

    lines += [f"- **{k}** — {v}" for k, v in ok]
    lines += ["", "Artifacts: `data/ami/knowledge.sqlite`, `data/ami/research.sqlite`, "
              "`data/ami/decisions.jsonl`, `data/ami/last_bundle.json`", "",
              "*Runner: `python -m ami.run_phase_checks`*"]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMD: {OUT_MD}\nDone. ({len(ok)}/10 checks emitted)")


if __name__ == "__main__":
    main()
