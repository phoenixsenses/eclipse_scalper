from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import collector_supervisor as cs


def _mk_local_tmp() -> Path:
    p = Path("localtests") / f"collector_supervisor_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


class _FakeProc:
    def __init__(self, pid: int):
        self.pid = pid
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self._alive = False

    def wait(self, timeout=None):
        self._alive = False
        return 0

    def kill(self):
        self._alive = False


def test_managed_process_stop_cleans_owned_pid_state():
    tmp_path = _mk_local_tmp()
    log = cs._setup_logger(tmp_path / "logs" / "collector_supervisor.test.log")
    mp = cs.ManagedProcess(cs.PROCS[0], sys.executable, tmp_path, log)
    mp.proc = _FakeProc(43210)
    cs._write_pid_state(
        cwd=tmp_path,
        module=mp.module,
        pid=mp.proc.pid,
        launcher="test",
        python=sys.executable,
        args=mp.extra_args,
    )
    try:
        assert cs._pid_file(tmp_path, mp.module).exists()
        assert cs._meta_file(tmp_path, mp.module).exists()
        mp.stop()
        assert not cs._pid_file(tmp_path, mp.module).exists()
        assert not cs._meta_file(tmp_path, mp.module).exists()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
