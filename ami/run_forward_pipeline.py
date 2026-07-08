"""Forward Evidence Pipeline gercek calistirici.

- E-HOUR17-FWD-001 (K-S34-HOUR17-001, LONG_HOUR17_HOLD6H) ve
  E-CONVCOMP-FWD-001 (K-S34-MECH-COMPOSITE-001 yerine conviction-composite'in
  gercekte loglanan karsiligi: LONG_HOUR17_COMPOSITE, conviction>=4) binding'lerini
  kurar (idempotent) ve gercek shadow ledger uzerinde run_once yapar.
- Rapor: reports/research/s34/AMI_FORWARD_EVIDENCE.md

Not (durustluk): E-MECHCOMP-FWD-001 mekanizma-kompoziti icin kayitli ama shadow
runner mech_score LOGLAMIYOR -> o deney BAGLANMADI (data_readiness eksik).
Bunun yerine conviction-composite icin ayri deney acildi. Mech-score loglamasi
eklendiginde E-MECHCOMP baglanabilir.

Run: python -m ami.run_forward_pipeline
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ami.governance.governor import EpistemicGovernor
from ami.knowledge.store import KnowledgeStore
from ami.research.forward_pipeline import ForwardEvidencePipeline
from ami.research.registry import ExperimentSpec, ResearchRegistry

OUT = ROOT / "reports" / "research" / "s34" / "AMI_FORWARD_EVIDENCE.md"


def ensure_binding(pipe: ForwardEvidencePipeline, reg: ResearchRegistry,
                   experiment_id: str, question_id: str, ko_id: str, signal: str,
                   min_conv: int | None, population: str) -> str:
    existing = {b.experiment_id for b in pipe.bindings()}
    if experiment_id in existing:
        return "already_bound"
    spec = ExperimentSpec(
        experiment_id=experiment_id, question_id=question_id,
        population=population, target="net_bps (shadow CLOSE)",
        features=["conviction_score"] if min_conv else ["route_gate"],
        threshold_method=f"frozen(conviction>={min_conv})" if min_conv else "frozen(route gate)",
        untouched_data="all shadow CLOSEs after binding freeze",
        negative_control="pre-freeze trades are rejected as PRE_FREEZE",
        min_sample=20,
        decision_criteria="avg net > 0 and WR > 55% at n>=20",
        falsification_rule="avg net <= 0 at n>=20",
        execution_model="mark_fill_fee5bps")
    spec.freeze()
    reg.register_experiment(spec)
    pipe.bind(spec, ko_id, signal, min_conviction=min_conv)
    return "bound"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    store = KnowledgeStore(); reg = ResearchRegistry()
    gov = EpistemicGovernor(store)
    pipe = ForwardEvidencePipeline(store, reg, gov)
    s1 = ensure_binding(pipe, reg, "E-HOUR17-FWD-001", "Q-MECHCOMP-FORWARD-001",
                        "K-S34-HOUR17-001", "LONG_HOUR17_HOLD6H", None,
                        "forward LONG_HOUR17_HOLD6H shadow CLOSEs")
    s2 = ensure_binding(pipe, reg, "E-CONVCOMP-FWD-001", "Q-MECHCOMP-FORWARD-001",
                        "K-S34-MECH-COMPOSITE-001", "LONG_HOUR17_COMPOSITE", 4,
                        "forward LONG_HOUR17_COMPOSITE shadow CLOSEs with conviction>=4")
    print(f"bindings: E-HOUR17={s1}  E-CONVCOMP={s2}")
    rep = pipe.run_once()
    lines = ["# AMI Forward Evidence Report", "",
             f"> {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} — otomatik pipeline (`ami/research/forward_pipeline.py`)", "",
             "| Experiment | Knowledge | Signal | Binding | Accepted | Rejected | Dup | Forward stats |",
             "|---|---|---|---|--:|--:|--:|---|"]
    for b in rep["bindings"]:
        lines.append("| %s | %s | %s | %s | %d | %d | %d | %s |" % (
            b["experiment_id"], b["knowledge_id"], b["signal"],
            "VALID" if b["valid"] else f"INVALID({b['why']})",
            b["accepted"], b["rejected"], b["duplicates"],
            json.dumps(b.get("forward_stats", {}))))
        print(f"  {b['experiment_id']}: acc={b['accepted']} rej={b['rejected']} "
              f"dup={b['duplicates']} stats={b.get('forward_stats')}")
    if rep["governor_actions"]:
        lines += ["", "## Governor kararlari", ""]
        for a in rep["governor_actions"]:
            lines.append(f"- **{a['knowledge_id']}**: {a['action']} ({a['reason']}) stats={a['stats']}")
            print(f"  GOVERNOR: {a}")
    else:
        lines += ["", "Governor kararı yok (n < min_sample — birikim sürüyor)."]
        print("  governor: karar yok (birikim suruyor)")
    lines += ["", "## Kurallar (aktif)",
              "- R1 freeze-öncesi trade = PRE_FREEZE reddi (lookahead sınırı)",
              "- R2 spec/candidate/dataset/execution değişimi = BINDING_INVALID",
              "- R3 trade başına tek evidence (kalıcı PK)",
              "- R4 provenance'sız evidence reddi",
              "- R5 pipeline izin VERMEZ — yalnız governor gate'lerine başvurur",
              "", "Not: E-MECHCOMP-FWD-001 kayıtlı fakat BAĞLANMADI — shadow runner mech_score",
              "loglamıyor (data_readiness eksik); conviction-composite için E-CONVCOMP-FWD-001 açıldı.",
              "", "*Runner: `python -m ami.run_forward_pipeline` (cron/oturum başına idempotent)*"]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"MD: {OUT}")
    store.close(); reg.close()


if __name__ == "__main__":
    main()
