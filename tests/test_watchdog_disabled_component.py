"""A component that is switched OFF must not pin the health light to YELLOW.

`liquidation_silence` is gated behind -EnableLiquidationSilenceScheduler (default
OFF). Its last artifact stays on disk and goes stale forever, because no producer
will ever write another one. Treating that as UNKNOWN kept overall.json at YELLOW
permanently -- and a health signal stuck on YELLOW is how a real YELLOW gets missed.

The distinction that matters: pid file `0` means "not requested" (start_eclipse's
convention); a non-zero pid means the scheduler WAS requested, so a stale artifact
there is a stuck producer and must still be loud.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import heartbeat_watchdog as W


def _pid(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "liquidation_silence_scheduler.pid"
    p.write_text(content, encoding="utf-8")
    return p


def test_zero_pid_means_deliberately_disabled(tmp_path):
    assert W._liq_silence_deliberately_disabled(_pid(tmp_path, "0")) is True


def test_a_running_scheduler_is_not_excused(tmp_path):
    """A requested-but-stuck producer must keep raising the alarm."""
    assert W._liq_silence_deliberately_disabled(_pid(tmp_path, "12345")) is False


def test_an_absent_pid_file_is_not_excused(tmp_path):
    """NOT_REQUESTED-with-no-file is a different state; do not guess it is off."""
    assert W._liq_silence_deliberately_disabled(tmp_path / "nope.pid") is False


@pytest.mark.parametrize("junk", ["", "   ", "not-a-pid", "0x0"])
def test_an_unreadable_pid_file_fails_towards_the_alarm(tmp_path, junk):
    """Ambiguity must not silence a component; only an explicit 0 does."""
    assert W._liq_silence_deliberately_disabled(_pid(tmp_path, junk)) is False


def test_the_disabled_state_is_reported_not_hidden():
    """GREEN-because-off must be distinguishable from GREEN-because-checked."""
    import inspect
    src = inspect.getsource(W.evaluate)
    assert "liquidation_silence_disabled" in src
