"""Paket 1 — Automated Forward Evidence Pipeline.

Shadow CLOSE -> frozen ExperimentSpec eslesmesi -> candidate version dogrulamasi
-> forward EvidenceBundle -> KnowledgeObject guncelleme -> governor karari -> audit.

ZORUNLU KURALLAR (operator paketi):
R1  Yalniz freeze SONRASI acilan trade'ler forward evidence sayilir (PRE_FREEZE reddi).
R2  Spec/candidate/feature/execution degisirse eski evidence yeni versiyona TASINMAZ:
    binding.spec_hash != guncel spec hash -> BINDING_INVALID, isleme durur.
R3  Ayni trade iki kez evidence yazilamaz (processed_trades PK + evidence PK).
R4  Provenance eksikse (dataset_hash/code_ref/execution_model) evidence REDDEDILIR.
R5  Research-only bilgi live permission uretemez (governor zaten zorlar; pipeline
    hicbir izin VERMEZ, yalniz governor'a promotion BASVURUSU yapar).
R6  Live executor/config/.env/leverage/sizing'e DOKUNMAZ (salt-okuma ledger + AMI store).
"""
from __future__ import annotations
import json, sqlite3, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ami.constitution import ConstitutionViolation
from ami.enums import EvidenceLevel, KnowledgeStatus, FailureType
from ami.governance.governor import EpistemicGovernor
from ami.knowledge.store import KnowledgeStore
from ami.research.registry import EvidenceBundle, ExperimentSpec, ResearchRegistry

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / "reports" / "shadow" / "s34_state_machine_shadow.jsonl"


@dataclass
class ForwardBinding:
    experiment_id: str
    knowledge_id: str
    signal: str
    min_conviction: int | None
    spec_hash: str
    frozen_ms: int
    dataset_hash: str
    code_ref: str
    execution_model: str
    candidate_version: int


class ForwardEvidencePipeline:
    def __init__(self, store: KnowledgeStore, registry: ResearchRegistry,
                 governor: EpistemicGovernor, ledger_path: str | Path = DEFAULT_LEDGER):
        self.store = store
        self.reg = registry
        self.gov = governor
        self.ledger_path = Path(ledger_path)

    # ---- binding yonetimi ---------------------------------------------------
    def bind(self, spec: ExperimentSpec, knowledge_id: str, signal: str,
             min_conviction: int | None = None) -> ForwardBinding:
        if not spec.frozen_hash:
            raise ConstitutionViolation("Binding requires frozen experiment (§74).")
        ko = self.store.get(knowledge_id)
        if ko is None:
            raise ConstitutionViolation(f"Binding target knowledge missing: {knowledge_id}")
        if not (spec.execution_model and ko.provenance.code_ref and ko.provenance.dataset_hash):
            raise ConstitutionViolation("No claim without provenance. (binding provenance incomplete)")
        if ko.provenance.execution_model != spec.execution_model:
            raise ConstitutionViolation(
                f"Execution model mismatch: KO={ko.provenance.execution_model} "
                f"spec={spec.execution_model}")
        b = ForwardBinding(spec.experiment_id, knowledge_id, signal, min_conviction,
                           spec.frozen_hash, int(time.time() * 1000),
                           ko.provenance.dataset_hash, ko.provenance.code_ref,
                           spec.execution_model, ko.version)
        self.reg.conn.execute(
            "INSERT OR REPLACE INTO forward_bindings VALUES (?,?,?,?,?,?,?,?,?,?)",
            (b.experiment_id, b.knowledge_id, b.signal, b.min_conviction, b.spec_hash,
             b.frozen_ms, b.dataset_hash, b.code_ref, b.execution_model, b.candidate_version))
        self.reg.conn.commit()
        self.store._audit("forward_pipeline", "BIND", knowledge_id,
                          f"{spec.experiment_id} signal={signal} frozen={b.frozen_ms}")
        self.store.conn.commit()
        return b

    def bindings(self) -> list[ForwardBinding]:
        rows = self.reg.conn.execute("SELECT * FROM forward_bindings").fetchall()
        return [ForwardBinding(*r) for r in rows]

    def _validate_binding(self, b: ForwardBinding) -> tuple[bool, str]:
        """R2: spec veya candidate degistiyse binding gecersiz."""
        row = self.reg.conn.execute("SELECT payload, frozen_hash FROM experiments WHERE experiment_id=?",
                                    (b.experiment_id,)).fetchone()
        if not row:
            return False, "experiment_missing"
        if row[1] != b.spec_hash:
            return False, "spec_hash_changed"
        ko = self.store.get(b.knowledge_id)
        if ko is None:
            return False, "knowledge_missing"
        if ko.version != b.candidate_version:
            return False, f"candidate_version_changed({b.candidate_version}->{ko.version})"
        if ko.provenance.execution_model != b.execution_model:
            return False, "execution_model_changed"
        if ko.provenance.dataset_hash != b.dataset_hash:
            return False, "dataset_hash_changed"
        return True, "ok"

    # ---- ledger okuma ---------------------------------------------------------
    def _closed_trades(self) -> list[dict]:
        out = []
        if not self.ledger_path.exists():
            return out
        with self.ledger_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("event") == "CLOSE" and r.get("net_bps") is not None:
                    out.append(r)
        return out

    # ---- ana akis ---------------------------------------------------------------
    def run_once(self, now_ms: int | None = None) -> dict[str, Any]:
        now = now_ms or int(time.time() * 1000)
        report: dict[str, Any] = {"ts_ms": now, "bindings": [], "governor_actions": []}
        trades = self._closed_trades()
        for b in self.bindings():
            ok, why = self._validate_binding(b)
            binfo: dict[str, Any] = {"experiment_id": b.experiment_id, "knowledge_id": b.knowledge_id,
                                     "signal": b.signal, "valid": ok, "why": why,
                                     "accepted": 0, "rejected": 0, "duplicates": 0}
            if not ok:
                self.store._audit("forward_pipeline", "BINDING_INVALID", b.knowledge_id,
                                  f"{b.experiment_id}: {why}")
                self.store.conn.commit()
                report["bindings"].append(binfo)
                continue
            spec_payload = json.loads(self.reg.conn.execute(
                "SELECT payload FROM experiments WHERE experiment_id=?", (b.experiment_id,)).fetchone()[0])
            spec = ExperimentSpec(**spec_payload)
            for t in trades:
                if t.get("signal") != b.signal:
                    continue
                tid = str(t.get("id") or f"{t.get('signal')}:{t.get('entry_ts_ms')}")
                dup = self.reg.conn.execute(
                    "SELECT 1 FROM processed_trades WHERE experiment_id=? AND trade_id=?",
                    (b.experiment_id, tid)).fetchone()
                if dup:
                    binfo["duplicates"] += 1
                    continue
                entry = int(t.get("entry_ts_ms") or 0)
                accepted, reason = True, ""
                if entry <= b.frozen_ms:
                    accepted, reason = False, "PRE_FREEZE"          # R1 / lookahead siniri
                elif b.min_conviction is not None and int(t.get("conviction_score") or 0) < b.min_conviction:
                    accepted, reason = False, "BELOW_MIN_CONVICTION"
                elif not (b.dataset_hash and b.code_ref and b.execution_model):
                    accepted, reason = False, "MISSING_PROVENANCE"   # R4
                self.reg.conn.execute(
                    "INSERT INTO processed_trades VALUES (?,?,?,?,?,?,?)",
                    (b.experiment_id, tid, entry, float(t.get("net_bps") or 0),
                     int(accepted), reason, now))
                if not accepted:
                    binfo["rejected"] += 1
                    self.store._audit("forward_pipeline", "EVIDENCE_REJECTED", b.knowledge_id,
                                      f"{tid}: {reason}")
                    continue
                ev = EvidenceBundle(
                    evidence_id=f"EV-{b.experiment_id}-{tid}",
                    experiment_id=b.experiment_id,
                    results={"trade_id": tid, "entry_ts_ms": entry,
                             "net_bps": float(t.get("net_bps")),
                             "conviction_score": t.get("conviction_score")},
                    outcome="SUPPORTS" if float(t.get("net_bps")) > 0 else "WEAKENS",
                    evidence_family=f"forward:{b.signal}",
                    dataset_hash=b.dataset_hash, code_ref=b.code_ref)
                self.reg.attach_evidence(ev, spec)                    # R3: PK + hash kontrol
                binfo["accepted"] += 1
            self.reg.conn.commit()
            # KO guncelle + governor degerlendirmesi
            stats = self._forward_stats(b.experiment_id)
            binfo["forward_stats"] = stats
            ko = self.store.get(b.knowledge_id)
            if ko and stats["n"] != ko.forward_events:
                ko.forward_events = stats["n"]
                if stats["n"] >= 5:
                    ko.evidence_level = max(ko.evidence_level, EvidenceLevel.FORWARD_SHADOW,
                                            key=int)
                ko.last_verified_ms = now
                self.store.put(ko, actor="forward_pipeline")
            act = self._governor_review(b, spec, stats)
            if act:
                report["governor_actions"].append(act)
            report["bindings"].append(binfo)
        self.store.conn.commit()
        return report

    def _forward_stats(self, experiment_id: str) -> dict[str, Any]:
        rows = self.reg.conn.execute(
            "SELECT net_bps FROM processed_trades WHERE experiment_id=? AND accepted=1 ORDER BY entry_ts_ms",
            (experiment_id,)).fetchall()
        vals = [float(r[0]) for r in rows]
        if not vals:
            return {"n": 0}
        n = len(vals); wins = sum(1 for v in vals if v > 0)
        srt = sorted(vals, reverse=True)
        top3_removed = sum(srt[3:]) if n > 3 else 0.0
        eq = peak = dd = 0.0
        for v in vals:
            eq += v; peak = max(peak, eq); dd = min(dd, eq - peak)
        return {"n": n, "wr": round(wins / n, 3), "avg_bps": round(sum(vals) / n, 1),
                "total_bps": round(sum(vals), 1), "top3_removed_total": round(top3_removed, 1),
                "mdd_bps": round(dd, 1), "worst": round(min(vals), 1)}

    def _governor_review(self, b: ForwardBinding, spec: ExperimentSpec,
                         stats: dict[str, Any]) -> dict | None:
        """Falsification / promotion degerlendirmesi. Izin VERMEZ; governor gate'lerini cagirir."""
        ko = self.store.get(b.knowledge_id)
        if ko is None or stats.get("n", 0) == 0:
            return None
        n = stats["n"]
        if n >= spec.min_sample and stats["avg_bps"] <= 0:
            self.gov.demote(ko, KnowledgeStatus.WEAKENED,
                            f"forward falsification: avg={stats['avg_bps']} at n={n}",
                            actor="forward_pipeline")
            self.store.archive_failure(
                f"forward falsified: {b.knowledge_id} via {b.experiment_id}",
                FailureType.NO_EDGE,
                reason=f"avg={stats['avg_bps']}bps n={n}",
                data_period=f"forward since {b.frozen_ms}")
            return {"knowledge_id": b.knowledge_id, "action": "DEMOTE",
                    "reason": "forward_falsification", "stats": stats}
        if (ko.status == KnowledgeStatus.FORWARD_VALIDATING
                and n >= max(spec.min_sample, 20)
                and stats["avg_bps"] > 0 and "top3_removed_total" in stats):
            try:
                self.gov.promote(ko, KnowledgeStatus.OPERATIONAL_CANDIDATE,
                                 actor="forward_pipeline")
                return {"knowledge_id": b.knowledge_id, "action": "PROMOTE",
                        "reason": "forward_gates_met", "stats": stats}
            except ConstitutionViolation as e:
                return {"knowledge_id": b.knowledge_id, "action": "PROMOTION_BLOCKED",
                        "reason": str(e), "stats": stats}
        return None
