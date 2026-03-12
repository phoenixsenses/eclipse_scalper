from __future__ import annotations

import sys
from unittest.mock import patch


def test_main_refuses_paper_profile_without_dry_run(monkeypatch) -> None:
    import main

    monkeypatch.setenv("SCALPER_ENV_PROFILE", "paper")
    monkeypatch.delenv("SCALPER_DRY_RUN", raising=False)
    monkeypatch.setattr(sys, "argv", ["main.py"])

    with patch.object(main, "run_bot") as run_bot:
        rc = main.main()

    assert rc == 2
    run_bot.assert_not_called()


def test_main_refuses_preexisting_dry_run_without_flag(monkeypatch) -> None:
    import main

    monkeypatch.delenv("SCALPER_ENV_PROFILE", raising=False)
    monkeypatch.setenv("SCALPER_DRY_RUN", "1")
    monkeypatch.setattr(sys, "argv", ["main.py"])

    with patch.object(main, "run_bot") as run_bot:
        rc = main.main()

    assert rc == 2
    run_bot.assert_not_called()
