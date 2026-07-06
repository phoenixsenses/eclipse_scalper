"""Research OS — Phase 4 (whitepaper §94, Part X).

Question -> Hypotheses -> preregistered ExperimentSpec (frozen) -> EvidenceBundle.
Freeze is enforced: results cannot be attached to a spec whose frozen hash
changed (post-hoc criteria edits are structurally impossible, §16/§57/§74).

BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1 (Phase 4, research.sqlite
gap): `ResearchRegistry` is a separate, older projection store
(`data/ami/research.sqlite`, its OWN `experiments` table) with no relation to
canonical.sqlite's `experiment_registry`/`experiment_gate_receipts` -- before
this batch, `register_experiment()` was a blind `INSERT OR REPLACE`, callable
by anyone (`ami/latent/discovery.py`, `ami/latent/regime.py`,
`ami/latent/risk_applicability.py`, `ami/run_forward_pipeline.py`,
`ami/run_phase_checks.py`, `tools/research_ami_mfe50_experiment.py`,
`tools/research_s34_buyfade_reentry.py`,
`tools/research_s34_buyfade_silence_exit.py`,
`tools/research_s34_buyfade_structural.py`), with no gate participation at
all -- a second, wider bypass than the 10 canonical.sqlite modules.

Fix (this batch): `register_experiment()` now allows an identical-content
re-registration of an experiment_id ALREADY stored here unconditionally (pure
historical replay -- preserves every one of the callers above exactly, since
none of them are being asked to compute anything new by this batch), but
requires an M-0034 `experiment_gate_receipt` (`ami/governance/epistemic_gates
.issue_gate_receipt`, written only by `ami.warehouse.experiment_ledger
.register_experiment_with_gates`/`register_legacy_snapshot_with_gates`) for
any NEW experiment_id or any CHANGED frozen_hash -- fail-closed
(`ResearchRegistryUnauthorized`) otherwise. `research.sqlite` therefore can
no longer become an alternate, unenforced scientific registry: it can only
ever hold (a) already-known history, or (b) something that already passed
the canonical gate.
"""
from __future__ import annotations
import dataclasses, hashlib, json, sqlite3, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ami.constitution import ConstitutionViolation
from ami.enums import FailureType

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "ami" / "research.sqlite"


class ResearchRegistryUnauthorized(Exception):
    """`register_experiment()` was asked to write a NEW experiment_id, or
    CHANGE an existing one's frozen_hash, without a matching M-0034 gate
    receipt in knowledge.sqlite. Fail-closed: research.sqlite must not
    become an alternate, unenforced experiment registry -- register the
    experiment through `ami.warehouse.experiment_ledger
    .register_experiment_with_gates` (or `register_legacy_snapshot_with_gates`)
    against canonical.sqlite FIRST, then retry this call."""


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class ResearchQuestion:
    question_id: str
    question: str
    origin_observation: str = ""
    scientific_value: float = 0.5     # 0..1
    economic_value: float = 0.5
    risk_reduction_value: float = 0.0
    novelty: float = 0.5
    falsifiability: float = 0.5
    data_readiness: float = 0.5
    estimated_cost: float = 0.3       # 0..1 (1 = very expensive)
    multiple_testing_risk: float = 0.3
    required_sample: int = 30
    dependencies: list[str] = field(default_factory=list)
    status: str = "OPEN"
    created_ms: int = field(default_factory=now_ms)


@dataclass
class Hypothesis:
    hypothesis_id: str
    question_id: str
    statement: str
    kind: str = "PRIMARY"             # PRIMARY/NULL/ALTERNATIVE/CONFOUNDER/DATA_QUALITY/EXECUTION/REGIME
    prediction: str = ""
    falsified_by: str = ""


@dataclass
class ExperimentSpec:
    experiment_id: str
    question_id: str
    population: str
    target: str
    features: list[str]
    threshold_method: str
    chronological_split: str = "70/30"
    untouched_data: str = ""
    negative_control: str = ""
    min_sample: int = 30
    effect_size_required_bps: float = 20.0
    multiple_testing_control: str = "max-stat permutation"
    execution_model: str = "mark_fill_fee5bps"
    decision_criteria: str = ""
    falsification_rule: str = ""
    frozen_hash: str = ""
    created_ms: int = field(default_factory=now_ms)

    def compute_hash(self) -> str:
        d = dataclasses.asdict(self); d.pop("frozen_hash", None); d.pop("created_ms", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    def freeze(self) -> None:
        for f in ("population", "target", "decision_criteria", "falsification_rule", "untouched_data"):
            if not getattr(self, f):
                raise ConstitutionViolation(f"Preregistration incomplete: {f} missing (§57).")
        self.frozen_hash = self.compute_hash()


@dataclass
class EvidenceBundle:
    evidence_id: str
    experiment_id: str
    results: dict[str, Any]
    outcome: str                       # SUPPORTS / WEAKENS / FALSIFIES / INCONCLUSIVE
    evidence_family: str = ""
    dataset_hash: str = ""
    code_ref: str = ""
    created_ms: int = field(default_factory=now_ms)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (question_id TEXT PRIMARY KEY, payload TEXT, status TEXT, created_ms INTEGER);
CREATE TABLE IF NOT EXISTS hypotheses (hypothesis_id TEXT PRIMARY KEY, question_id TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS experiments (experiment_id TEXT PRIMARY KEY, question_id TEXT, payload TEXT, frozen_hash TEXT);
CREATE TABLE IF NOT EXISTS evidence (evidence_id TEXT PRIMARY KEY, experiment_id TEXT, payload TEXT, outcome TEXT, created_ms INTEGER);
CREATE TABLE IF NOT EXISTS forward_bindings (
    experiment_id TEXT PRIMARY KEY,
    knowledge_id TEXT NOT NULL,
    signal TEXT NOT NULL,
    min_conviction INTEGER,
    spec_hash TEXT NOT NULL,
    frozen_ms INTEGER NOT NULL,
    dataset_hash TEXT NOT NULL,
    code_ref TEXT NOT NULL,
    execution_model TEXT NOT NULL,
    candidate_version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS processed_trades (
    experiment_id TEXT NOT NULL,
    trade_id TEXT NOT NULL,
    entry_ts_ms INTEGER,
    net_bps REAL,
    accepted INTEGER NOT NULL,
    reject_reason TEXT,
    processed_ms INTEGER NOT NULL,
    PRIMARY KEY (experiment_id, trade_id)
);
"""


def assert_no_overlap(train_ids: set, test_ids: set) -> None:
    """§86 leakage guard: train/test event universes must be disjoint."""
    dup = set(train_ids) & set(test_ids)
    if dup:
        raise ConstitutionViolation(f"Train/test leakage: {len(dup)} duplicate events "
                                    f"(e.g. {sorted(list(dup))[:3]})")


class ResearchRegistry:
    def __init__(self, path: str | Path = DEFAULT_PATH, *, knowledge_path: str | Path | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        if knowledge_path is None:
            from ami.knowledge.store import DEFAULT_PATH as knowledge_path  # noqa: PLC0415
        self.knowledge_path = Path(knowledge_path)

    def add_question(self, q: ResearchQuestion) -> None:
        self.conn.execute("INSERT OR REPLACE INTO questions VALUES (?,?,?,?)",
                          (q.question_id, json.dumps(dataclasses.asdict(q)), q.status, q.created_ms))
        self.conn.commit()

    def add_hypothesis(self, h: Hypothesis) -> None:
        self.conn.execute("INSERT OR REPLACE INTO hypotheses VALUES (?,?,?)",
                          (h.hypothesis_id, h.question_id, json.dumps(dataclasses.asdict(h))))
        self.conn.commit()

    def register_experiment(self, spec: ExperimentSpec) -> None:
        """Historical replay (identical frozen_hash already stored here) is
        always allowed, unconditionally -- this is what every existing
        caller does today, and none of them are being asked to change
        anything by this batch. A NEW experiment_id, or a CHANGED frozen_hash
        for an existing one, requires a matching M-0034 gate receipt in
        knowledge.sqlite (`ami.governance.epistemic_gates.has_gate_receipt`)
        -- fails closed (`ResearchRegistryUnauthorized`) otherwise, and never
        invents or bypasses that requirement."""
        if not spec.frozen_hash:
            raise ConstitutionViolation("Experiment must be frozen before registration (§57).")
        existing = self.conn.execute(
            "SELECT frozen_hash FROM experiments WHERE experiment_id=?", (spec.experiment_id,)).fetchone()
        if existing is None or existing[0] != spec.frozen_hash:
            from ami.governance import epistemic_gates as _gates  # noqa: PLC0415
            kconn = sqlite3.connect(str(self.knowledge_path))
            try:
                _gates.init_gates_schema(kconn)  # idempotent; ensures the receipts table exists
                authorized = _gates.has_gate_receipt(kconn, spec.experiment_id)
            finally:
                kconn.close()
            if not authorized:
                raise ResearchRegistryUnauthorized(
                    f"RESEARCH_REGISTRY_UNAUTHORIZED: {spec.experiment_id!r} is "
                    f"{'a new experiment_id' if existing is None else 'a changed frozen_hash for an existing experiment_id'} "
                    "with no matching gate receipt in knowledge.sqlite. Register it through "
                    "ami.warehouse.experiment_ledger.register_experiment_with_gates (or "
                    "register_legacy_snapshot_with_gates) against canonical.sqlite first, then retry.")
        self.conn.execute("INSERT OR REPLACE INTO experiments VALUES (?,?,?,?)",
                          (spec.experiment_id, spec.question_id,
                           json.dumps(dataclasses.asdict(spec)), spec.frozen_hash))
        self.conn.commit()

    def attach_evidence(self, ev: EvidenceBundle, spec: ExperimentSpec) -> None:
        row = self.conn.execute("SELECT frozen_hash FROM experiments WHERE experiment_id=?",
                                (ev.experiment_id,)).fetchone()
        if not row:
            raise ConstitutionViolation("Evidence without registered experiment (§57).")
        if spec.compute_hash() != row[0]:
            raise ConstitutionViolation("Preregistration violated: spec changed after freeze (§74).")
        self.conn.execute("INSERT OR REPLACE INTO evidence VALUES (?,?,?,?,?)",
                          (ev.evidence_id, ev.experiment_id, json.dumps(dataclasses.asdict(ev)),
                           ev.outcome, ev.created_ms))
        self.conn.commit()

    def questions(self, status: str | None = None) -> list[ResearchQuestion]:
        q = "SELECT payload FROM questions"; args: tuple = ()
        if status:
            q += " WHERE status=?"; args = (status,)
        return [ResearchQuestion(**json.loads(r[0])) for r in self.conn.execute(q, args)]

    def evidence_for(self, experiment_id: str) -> list[EvidenceBundle]:
        return [EvidenceBundle(**json.loads(r[0])) for r in self.conn.execute(
            "SELECT payload FROM evidence WHERE experiment_id=?", (experiment_id,))]

    def close(self) -> None:
        self.conn.close()
