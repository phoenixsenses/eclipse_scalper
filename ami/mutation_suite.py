"""Paket 2 — Adversarial / Mutation Suite (20 senaryo, tek dogruluk kaynagi).

Her senaryo BILEREK bir ihlal enjekte eder ve sistemin YAKALADIGINI kanitlar.
Donen kayit: {name, injected, expected, actual, blocked_by, passed, audit}.

Kullanim:
  pytest: tests/test_ami_mutation_suite.py (parametrize)
  rapor : python -m ami.run_mutation_report -> AMI_MUTATION_REPORT.md
"""
from __future__ import annotations
import json, sqlite3, time
from pathlib import Path
from typing import Callable

from ami.constitution import ConstitutionViolation
from ami.enums import (Action, ClaimType, DataQuality, EvidenceLevel, KnowledgeStatus,
                       Permission, StateFamily)
from ami.governance import epistemic_gates as _gates
from ami.governance.governor import EpistemicGovernor
from ami.knowledge.objects import KnowledgeObject, Provenance, now_ms
from ami.knowledge.store import KnowledgeStore
from ami.research.forward_pipeline import ForwardEvidencePipeline
from ami.research.registry import (EvidenceBundle, ExperimentSpec, ResearchRegistry,
                                   assert_no_overlap)
from ami.states.objects import StateBundle, StateObject
from ami.decision.trace import decide


# ---------- yardimcilar ----------
def _ko(kid="K-M-1", status=KnowledgeStatus.FORWARD_VALIDATING, exec_model="mark_fill_fee5bps",
        **kw) -> KnowledgeObject:
    ko = KnowledgeObject(
        knowledge_id=kid, claim="mutation test claim", claim_type=ClaimType.PREDICTIVE,
        status=status,
        provenance=Provenance(source_tables=["liquidations"], data_time_range="2026-01..2026-06",
                              code_ref="tests/mutation", dataset_hash="ds-1",
                              execution_model=exec_model),
        evidence_level=EvidenceLevel.UNTOUCHED_HOLDOUT, replications=1, holdouts=1,
        falsification=["forward avg <= 0"], **kw)
    ko.freeze()
    return ko


def _spec(eid="E-M-1", exec_model="mark_fill_fee5bps", min_sample=30) -> ExperimentSpec:
    s = ExperimentSpec(eid, "Q-M-1", population="forward trades", target="net_bps",
                       features=["conviction_score"], threshold_method="frozen",
                       untouched_data="post-freeze", negative_control="low conviction",
                       decision_criteria="avg>0 n>=20", falsification_rule="avg<=0 n>=min",
                       execution_model=exec_model, min_sample=min_sample)
    s.freeze()
    return s


def _ledger(path: Path, trades: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as f:
        for t in trades:
            f.write(json.dumps({"event": "CLOSE", **t}) + "\n")
    return path


def _trade(tid, entry_ms, net, signal="SIG_A", conv=5):
    return {"id": tid, "signal": signal, "entry_ts_ms": entry_ms, "net_bps": net,
            "conviction_score": conv}


def _env(tmp: Path, trades: list[dict]):
    store = KnowledgeStore(tmp / "k.sqlite")
    # BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1: explicit disposable
    # knowledge_path (same file the KnowledgeStore above already uses) --
    # without this, ResearchRegistry would default to the REAL
    # knowledge.sqlite for its M-0034 gate-receipt check, which these
    # synthetic mutation-suite specs (they never touch canonical.sqlite at
    # all -- this suite exercises ForwardEvidencePipeline/EpistemicGovernor,
    # not the graveyard/nullifier gate itself) could never satisfy.
    reg = ResearchRegistry(tmp / "r.sqlite", knowledge_path=tmp / "k.sqlite")
    gov = EpistemicGovernor(store)
    led = _ledger(tmp / "ledger.jsonl", trades)
    pipe = ForwardEvidencePipeline(store, reg, gov, led)
    return store, reg, gov, pipe


def _register_test_spec(reg: ResearchRegistry, spec: ExperimentSpec) -> None:
    """Test-harness pre-authorization: issues a direct M-0034 gate receipt
    for this synthetic spec's experiment_id before registering it. These
    specs are internal to this adversarial suite (never a real canonical.
    sqlite experiment), so the full register_experiment_with_gates()
    round-trip does not apply -- this calls the SAME
    ami.governance.epistemic_gates.issue_gate_receipt the real gate uses,
    directly, satisfying ResearchRegistry.register_experiment()'s new
    fail-closed requirement without inventing a bypass flag."""
    kconn = sqlite3.connect(str(reg.knowledge_path))
    _gates.init_gates_schema(kconn)
    _gates.issue_gate_receipt(
        kconn, experiment_id=spec.experiment_id, canonical_family_id="MUTATION_SUITE_TEST_HARNESS",
        split_version=None, nullifier=None, registry_result="TEST_HARNESS_DIRECT_RECEIPT")
    kconn.commit()
    kconn.close()
    reg.register_experiment(spec)


def _result(name, injected, expected, actual, blocked_by, passed, audit=""):
    return {"name": name, "injected": injected, "expected": expected,
            "actual": actual, "blocked_by": blocked_by, "passed": bool(passed),
            "audit": audit}


NOW = now_ms()
PAST = NOW - 3_600_000
FUTURE = NOW + 3_600_000


# ---------- 20 senaryo ----------
def m01_future_lookahead(tmp: Path):
    store, reg, gov, pipe = _env(tmp, [_trade("t-old", PAST, +50)])
    ko = _ko(); store.put(ko)
    spec = _spec(); _register_test_spec(reg, spec)
    pipe.bind(spec, ko.knowledge_id, "SIG_A")          # frozen_ms = simdi
    rep = pipe.run_once()
    b = rep["bindings"][0]
    row = reg.conn.execute("SELECT reject_reason FROM processed_trades WHERE trade_id='t-old'").fetchone()
    ok = b["accepted"] == 0 and b["rejected"] == 1 and row and row[0] == "PRE_FREEZE"
    return _result("01_future_lookahead",
                   "freeze ONCESI acilmis trade forward evidence olarak sunuldu",
                   "PRE_FREEZE reddi, evidence yazilmaz",
                   f"accepted={b['accepted']} rejected={b['rejected']} reason={row[0] if row else None}",
                   "ForwardEvidencePipeline.run_once (R1)", ok, "processed_trades + EVIDENCE_REJECTED")


def m02_train_test_leakage(tmp: Path):
    try:
        assert_no_overlap({1, 2, 3}, {3, 4})
        return _result("02_train_test_leakage", "train/test kesisen event", "ConstitutionViolation",
                       "gecti (HATA)", "registry.assert_no_overlap", False)
    except ConstitutionViolation as e:
        return _result("02_train_test_leakage", "train/test kesisen event (id=3)",
                       "ConstitutionViolation", f"raised: {e}", "registry.assert_no_overlap", True)


def m03_stale_sensor(tmp: Path):
    store = KnowledgeStore(tmp / "k.sqlite"); gov = EpistemicGovernor(store)
    q = gov.check_data_health({"mark_prices": 9999.0}, {"mark_prices": 5.0})
    ok = q["mark_prices"] == DataQuality.STALE and "data_health" in gov.circuit_open
    return _result("03_stale_sensor_healthy", "9999dk yasli feed HEALTHY diye sunuldu",
                   "STALE etiketi + circuit breaker", f"{q['mark_prices'].value}, breaker={list(gov.circuit_open)}",
                   "EpistemicGovernor.check_data_health", ok, "BREAKER_TRIP audit")


def m04_research_only_live(tmp: Path):
    store, reg, gov, pipe = _env(tmp, [])
    ko = _ko("K-RO", status=KnowledgeStatus.HOLDOUT_VALIDATED); store.put(ko)
    dec = gov.authorize(Action.OPEN_LONG, ["K-RO"], {"data_health": "HEALTHY"})
    ok = dec.result != "GRANTED"
    return _result("04_research_only_live_auth", "HOLDOUT_VALIDATED bilgiyle OPEN_LONG istendi",
                   "GRANTED degil", dec.result, "EpistemicGovernor.authorize (PERMISSION_MIN_STATUS)",
                   ok, "AUTHORIZE audit")


def m05_candidate_version_change(tmp: Path):
    store, reg, gov, pipe = _env(tmp, [_trade("t-new", FUTURE, +50)])
    ko = _ko(); store.put(ko)
    spec = _spec(); _register_test_spec(reg, spec)
    pipe.bind(spec, ko.knowledge_id, "SIG_A")
    ko.touch_version("threshold degisti"); store.put(ko)     # candidate v2
    rep = pipe.run_once()
    b = rep["bindings"][0]
    ok = (not b["valid"]) and "candidate_version_changed" in b["why"] and b["accepted"] == 0
    return _result("05_candidate_version_change", "bind sonrasi candidate versiyonu degisti, eski binding kullanildi",
                   "BINDING_INVALID, eski forward evidence yeni versiyona tasinamaz",
                   f"valid={b['valid']} why={b['why']}", "ForwardEvidencePipeline._validate_binding (R2)",
                   ok, "BINDING_INVALID audit")


def m06_dataset_hash_mismatch(tmp: Path):
    store, reg, gov, pipe = _env(tmp, [_trade("t-new", FUTURE, +50)])
    ko = _ko(); store.put(ko)
    spec = _spec(); _register_test_spec(reg, spec)
    pipe.bind(spec, ko.knowledge_id, "SIG_A")
    ko.provenance.dataset_hash = "ds-TAMPERED"; store.put(ko)
    rep = pipe.run_once(); b = rep["bindings"][0]
    ok = (not b["valid"]) and b["why"] == "dataset_hash_changed" and b["accepted"] == 0
    return _result("06_dataset_hash_mismatch", "KO dataset hash'i binding sonrasi degistirildi",
                   "BINDING_INVALID(dataset_hash_changed)", f"valid={b['valid']} why={b['why']}",
                   "ForwardEvidencePipeline._validate_binding", ok, "BINDING_INVALID audit")


def m07_execution_model_mismatch(tmp: Path):
    store, reg, gov, pipe = _env(tmp, [])
    ko = _ko(exec_model="mark_fill_fee5bps"); store.put(ko)
    spec = _spec(exec_model="ask_fill_fee10bps"); _register_test_spec(reg, spec)
    try:
        pipe.bind(spec, ko.knowledge_id, "SIG_A")
        return _result("07_execution_model_mismatch", "spec ve KO farkli execution model",
                       "bind reddi", "bind kabul edildi (HATA)", "pipeline.bind", False)
    except ConstitutionViolation as e:
        return _result("07_execution_model_mismatch", "spec=ask_fill_fee10bps, KO=mark_fill_fee5bps",
                       "ConstitutionViolation", f"raised: {e}", "ForwardEvidencePipeline.bind", True)


def m08_prereg_metric_changed(tmp: Path):
    reg = ResearchRegistry(tmp / "r.sqlite")
    spec = _spec("E-M-8"); _register_test_spec(reg, spec)
    spec.decision_criteria = "avg>-100 (sonuc gorulunce gevsetildi)"
    try:
        reg.attach_evidence(EvidenceBundle("EV-8", "E-M-8", {"avg": -50}, "SUPPORTS",
                                           dataset_hash="d", code_ref="c"), spec)
        return _result("08_prereg_metric_changed", "test sonrasi karar kriteri degistirildi",
                       "kanit reddi", "kabul edildi (HATA)", "registry.attach_evidence", False)
    except ConstitutionViolation as e:
        return _result("08_prereg_metric_changed", "decision_criteria post-hoc degistirildi",
                       "ConstitutionViolation (hash mismatch)", f"raised: {e}",
                       "ResearchRegistry.attach_evidence (§74)", True)


def m09_duplicate_trade_evidence(tmp: Path):
    store, reg, gov, pipe = _env(tmp, [_trade("t-dup", FUTURE, +40)])
    ko = _ko(); store.put(ko)
    spec = _spec(); _register_test_spec(reg, spec)
    pipe.bind(spec, ko.knowledge_id, "SIG_A")
    r1 = pipe.run_once(); r2 = pipe.run_once()
    n_ev = reg.conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
    ok = (r1["bindings"][0]["accepted"] == 1 and r2["bindings"][0]["accepted"] == 0
          and r2["bindings"][0]["duplicates"] == 1 and n_ev == 1)
    return _result("09_duplicate_trade_evidence", "ayni trade ikinci kez evidence olarak islendi",
                   "duplicate sayilir, evidence 1 kalir",
                   f"run1_acc={r1['bindings'][0]['accepted']} run2_dup={r2['bindings'][0]['duplicates']} evidence={n_ev}",
                   "processed_trades PK + evidence PK (R3)", ok)


def m10_assumption_permission_open(tmp: Path):
    store, reg, gov, pipe = _env(tmp, [])
    ko = _ko("K-ASM", status=KnowledgeStatus.OPERATIONAL_CANDIDATE)
    ko.permitted = [Permission.RESEARCH_ONLY, Permission.LIVE_ALLOWED]; ko.forbidden = []
    ko.assumptions = ["BookTicker fill is representative"]
    store.put(ko)
    gov.invalidate_assumption("bookticker fill")
    dec = gov.authorize(Action.OPEN_LONG, ["K-ASM"], {"data_health": "HEALTHY"})
    k2 = store.get("K-ASM")
    ok = dec.result != "GRANTED" and Permission.LIVE_ALLOWED in k2.forbidden \
        and k2.status == KnowledgeStatus.WEAKENED
    return _result("10_assumption_permission_open", "assumption gecersizlesti ama LIVE izni acik birakilmak istendi",
                   "izin otomatik sokulur, authorize GRANTED olmaz",
                   f"authorize={dec.result} status={k2.status.value} live_forbidden={Permission.LIVE_ALLOWED in k2.forbidden}",
                   "governor.invalidate_assumption + demote", ok, "DEMOTE audit")


def m11_contradiction_confidence(tmp: Path):
    store, reg, gov, pipe = _env(tmp, [])
    a = _ko("K-CA", status=KnowledgeStatus.OPERATIONAL_CANDIDATE)
    a.permitted = [Permission.RESEARCH_ONLY, Permission.LIVE_ALLOWED]; a.forbidden = []
    b = _ko("K-CB"); store.put(a); store.put(b)
    pre_ok, _ = a.is_permitted(Permission.LIVE_ALLOWED)
    store.link("K-CA", "CONTRADICTS", "K-CB")
    a2 = store.get("K-CA")
    post_ok, why = a2.is_permitted(Permission.LIVE_ALLOWED)
    ok = pre_ok and (not post_ok) and why == "unresolved_contradiction"
    return _result("11_contradiction_confidence", "celiskiye ragmen LIVE izni surdurulmek istendi",
                   "celiski LIVE/SIZING iznini otomatik dusurur",
                   f"pre={pre_ok} post={post_ok} why={why}",
                   "KnowledgeStore.link + KnowledgeObject.is_permitted", ok, "LINK audit")


def m12_restart_duplicate(tmp: Path):
    trades = [_trade("t-r", FUTURE, +30)]
    store, reg, gov, pipe = _env(tmp, trades)
    ko = _ko(); store.put(ko)
    spec = _spec(); _register_test_spec(reg, spec)
    pipe.bind(spec, ko.knowledge_id, "SIG_A")
    pipe.run_once()
    store.close(); reg.close()
    # RESTART: ayni dosyalar, yeni instance'lar
    store2 = KnowledgeStore(tmp / "k.sqlite"); reg2 = ResearchRegistry(tmp / "r.sqlite")
    pipe2 = ForwardEvidencePipeline(store2, reg2, EpistemicGovernor(store2), tmp / "ledger.jsonl")
    r2 = pipe2.run_once()
    n_ev = reg2.conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
    ok = r2["bindings"][0]["accepted"] == 0 and r2["bindings"][0]["duplicates"] == 1 and n_ev == 1
    return _result("12_restart_duplicate_processing", "proses restart sonrasi ayni ledger yeniden islendi",
                   "kalici processed_trades sayesinde 0 yeni evidence",
                   f"restart_accepted={r2['bindings'][0]['accepted']} evidence={n_ev}",
                   "processed_trades (persistent PK)", ok)


def m13_concurrent_writes(tmp: Path):
    s1 = KnowledgeStore(tmp / "k.sqlite")
    s2 = KnowledgeStore(tmp / "k.sqlite")
    for i in range(10):
        s1.put(_ko(f"K-C{i}"))
        s2.put(_ko(f"K-D{i}"))
    n = len(s1.all())
    integ = s1.conn.execute("PRAGMA integrity_check").fetchone()[0]
    s1.close(); s2.close()
    ok = n == 20 and integ == "ok"
    return _result("13_concurrent_sqlite", "iki baglanti ayni store'a esz. yazdi",
                   "20 kayit + integrity ok", f"n={n} integrity={integ}",
                   "KnowledgeStore (WAL + busy_timeout)", ok)


def m14_decision_replay(tmp: Path):
    store = KnowledgeStore(tmp / "k.sqlite"); gov = EpistemicGovernor(store)
    ko = _ko("K-DR", status=KnowledgeStatus.HOLDOUT_VALIDATED)
    ko.permitted = [Permission.RESEARCH_ONLY, Permission.SHADOW_ALLOWED]; store.put(ko)
    sts = [StateObject(f"S{i}", "ETHUSDT", tf, StateFamily.STRUCTURE_STATE, "RANGE",
                       start_ms=1_000_000, meta={"direction": d})
           for i, (tf, d) in enumerate([("1m", "UP"), ("5m", "UP"), ("15m", "UP"),
                                        ("1h", "UP"), ("4h", "UP"), ("1D", "UP"), ("1W", "UP")])]
    bundle = StateBundle(ts_ms=1_000_000, states=sts)
    ctx = {"data_health": "HEALTHY"}
    t1 = decide(bundle, gov, ["K-DR"], ctx)
    t2 = decide(bundle, gov, ["K-DR"], ctx)
    ok = (t1.direction_candidates == t2.direction_candidates and t1.action == t2.action
          and t1.result == t2.result and t1.uncertainty == t2.uncertainty)
    return _result("14_decision_replay", "ayni bundle+context iki kez karar",
                   "deterministik ayni cikti", f"a1={t1.action}/{t1.result} a2={t2.action}/{t2.result} eq={ok}",
                   "decide() (durum disi rastgelelik yok)", ok, "decisions.jsonl")


def m15_failed_not_archived(tmp: Path):
    trades = [_trade(f"t-l{i}", FUTURE + i, -30) for i in range(3)]
    store, reg, gov, pipe = _env(tmp, trades)
    ko = _ko(); store.put(ko)
    spec = _spec(min_sample=3); _register_test_spec(reg, spec)
    pipe.bind(spec, ko.knowledge_id, "SIG_A")
    rep = pipe.run_once()
    fails = store.failures()
    k2 = store.get(ko.knowledge_id)
    ok = (any("forward falsified" in f["idea"] for f in fails)
          and k2.status == KnowledgeStatus.WEAKENED
          and any(a["action"] == "DEMOTE" for a in rep["governor_actions"]))
    return _result("15_failed_experiment_archive", "falsifiye deney arsivlenmeden birakilmak istendi",
                   "otomatik DEMOTE + failure archive kaydi",
                   f"archived={any('forward falsified' in f['idea'] for f in fails)} status={k2.status.value}",
                   "pipeline._governor_review", ok, "ARCHIVE_FAILURE audit")


def m16_missing_provenance(tmp: Path):
    store = KnowledgeStore(tmp / "k.sqlite")
    ko = _ko("K-NP")
    ko.provenance.code_ref = ""
    try:
        store.put(ko)
        return _result("16_missing_provenance", "code_ref'siz KO", "red", "kabul (HATA)",
                       "KnowledgeObject.validate", False)
    except ConstitutionViolation as e:
        return _result("16_missing_provenance", "provenance.code_ref bos KO olusturuldu",
                       "ConstitutionViolation", f"raised: {e}", "KnowledgeObject.validate", True)


def m17_permission_escalation(tmp: Path):
    store = KnowledgeStore(tmp / "k.sqlite"); gov = EpistemicGovernor(store)
    ko = _ko("K-ESC", status=KnowledgeStatus.PRELIMINARY)
    ko.permitted = [Permission.RESEARCH_ONLY, Permission.LIVE_ALLOWED]   # elle eklendi
    ko.forbidden = []
    store.put(ko)
    okp, why = ko.is_permitted(Permission.LIVE_ALLOWED)
    dec = gov.authorize(Action.OPEN_LONG, ["K-ESC"], {"data_health": "HEALTHY"})
    ok = (not okp) and "status_too_low" in why and dec.result != "GRANTED"
    return _result("17_permission_escalation", "PRELIMINARY KO'ya elle LIVE_ALLOWED izni eklendi",
                   "statu min. kosulu izni gecersiz kilar",
                   f"is_permitted={okp} why={why} authorize={dec.result}",
                   "PERMISSION_MIN_STATUS + governor", ok)


def m18_exploration_in_holdout(tmp: Path):
    store, reg, gov, pipe = _env(tmp, [_trade("t-exp", PAST, +80)])
    ko = _ko(); store.put(ko)
    spec = _spec(); _register_test_spec(reg, spec)
    pipe.bind(spec, ko.knowledge_id, "SIG_A")
    pipe.run_once()
    audit = store.audit_tail(20)
    rejected = any(a[2] == "EVIDENCE_REJECTED" and "PRE_FREEZE" in (a[4] or "") for a in audit)
    n_ev = reg.conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
    ok = rejected and n_ev == 0
    return _result("18_exploration_in_holdout", "exploration donemi trade'i holdout/forward diye sunuldu",
                   "sinir ihlali reddi + audit kaydi", f"audit_rejected={rejected} evidence={n_ev}",
                   "pipeline freeze-boundary (R1) + audit", ok, "EVIDENCE_REJECTED")


def m19_top_winner_report_missing(tmp: Path):
    store, reg, gov, pipe = _env(tmp, [])
    ko = _ko(); ko.forward_events = 25; store.put(ko)
    spec = _spec(min_sample=3)
    stats_no_top3 = {"n": 25, "wr": 0.7, "avg_bps": 40.0}    # top3_removed_total YOK
    from ami.research.forward_pipeline import ForwardBinding
    b = ForwardBinding(spec.experiment_id, ko.knowledge_id, "SIG_A", None,
                       spec.frozen_hash, NOW, "ds-1", "tests/mutation", "mark_fill_fee5bps", 1)
    act = pipe._governor_review(b, spec, stats_no_top3)
    k2 = store.get(ko.knowledge_id)
    ok = (act is None or act.get("action") != "PROMOTE") and k2.status == KnowledgeStatus.FORWARD_VALIDATING
    return _result("19_top_winner_report_missing", "top-3-removed olmadan promotion istendi",
                   "promotion yapilmaz", f"action={act} status={k2.status.value}",
                   "pipeline promotion preconditions", ok)


def m20_dq_propagation(tmp: Path):
    from tests.test_ami_states_research import make_synth_db
    from ami.states.engine import StateEngine
    db = make_synth_db(tmp / "synth.sqlite", hours=24 * 5)
    eng = StateEngine(db)
    future_ts = int(time.time() * 1000) + 30 * 86_400_000    # veri bitiminden 30 gun sonra
    bundle = eng.build_bundle("ETHUSDT", ts_ms=future_ts)
    st = bundle.by_family(StateFamily.STRUCTURE_STATE)
    non_healthy = all(s.data_quality != DataQuality.HEALTHY for s in st)
    low_conf = all(s.confidence <= 0.45 for s in st)
    eng.close()
    ok = non_healthy and low_conf
    return _result("20_dq_propagation", "bayat mark feed'iyle state uretildi",
                   "tum structure state'ler non-HEALTHY + guven dusurulmus",
                   f"non_healthy={non_healthy} low_conf={low_conf}",
                   "StateEngine.build_bundle (dq propagation)", ok)


SCENARIOS: list[tuple[str, Callable]] = [
    ("01", m01_future_lookahead), ("02", m02_train_test_leakage), ("03", m03_stale_sensor),
    ("04", m04_research_only_live), ("05", m05_candidate_version_change),
    ("06", m06_dataset_hash_mismatch), ("07", m07_execution_model_mismatch),
    ("08", m08_prereg_metric_changed), ("09", m09_duplicate_trade_evidence),
    ("10", m10_assumption_permission_open), ("11", m11_contradiction_confidence),
    ("12", m12_restart_duplicate), ("13", m13_concurrent_writes), ("14", m14_decision_replay),
    ("15", m15_failed_not_archived), ("16", m16_missing_provenance),
    ("17", m17_permission_escalation), ("18", m18_exploration_in_holdout),
    ("19", m19_top_winner_report_missing), ("20", m20_dq_propagation),
]


def run_all(base: Path) -> list[dict]:
    out = []
    for tag, fn in SCENARIOS:
        d = base / f"m{tag}"
        d.mkdir(parents=True, exist_ok=True)
        try:
            out.append(fn(d))
        except Exception as e:  # senaryonun kendisi patlarsa = FAIL
            out.append(_result(fn.__name__, "senaryo calistirma", "kontrollu sonuc",
                               f"EXCEPTION {type(e).__name__}: {e}", "-", False))
    return out
