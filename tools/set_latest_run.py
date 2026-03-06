from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, List


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_latest_", dir=str(dst.parent))
    os.close(fd)
    try:
        shutil.copy2(str(src), tmp_name)
        os.replace(tmp_name, str(dst))
    finally:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except Exception:
            pass


def _atomic_write_text(dst: Path, text: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_latest_", dir=str(dst.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
        os.replace(tmp_name, str(dst))
    finally:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except Exception:
            pass


def _copy_or_link(src: Path, dst: Path, mode: str) -> None:
    if mode == "hardlink":
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst.unlink()
            os.link(str(src), str(dst))
            return
        except Exception:
            pass
    _atomic_copy(src, dst)


def _parse_extra(raw: str) -> List[str]:
    names: List[str] = []
    for part in str(raw or "").split(","):
        v = str(part).strip()
        if not v:
            continue
        names.append(Path(v).name)
    return names


def _parse_globs(raw_items: Iterable[str]) -> List[str]:
    out: List[str] = []
    for raw in raw_items:
        for part in str(raw or "").split(","):
            v = str(part).strip()
            if v:
                out.append(v)
    return out


def _collect_extra_sources(
    run_dir: Path,
    extra_names: List[str],
    include_globs: List[str],
    strict_extra: bool,
    warnings_out: List[str],
) -> List[Path]:
    picked: dict[str, Path] = {}

    for name in extra_names:
        src = run_dir / name
        if src.exists() and src.is_file():
            picked[name] = src
        else:
            msg = f"missing extra file: {name}"
            if strict_extra:
                raise FileNotFoundError(msg)
            warnings_out.append(msg)

    matched: List[Path] = []
    for pat in include_globs:
        matched.extend([p for p in run_dir.glob(pat) if p.is_file()])
    for src in sorted(matched, key=lambda p: p.name):
        if src.name not in picked:
            picked[src.name] = src

    return [picked[k] for k in sorted(picked.keys())]


def _safe_float(v: object) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except Exception:
        return 0.0


def _resolve_core_run_dir(run_dir: Path) -> Path:
    if (run_dir / "metrics.json").exists() and (run_dir / "config.json").exists():
        return run_dir
    index_csv = run_dir / "index.csv"
    runs_dir = run_dir / "runs"
    if not index_csv.exists() or not runs_dir.exists():
        return run_dir
    try:
        with index_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return run_dir
        rows_sorted = sorted(
            rows,
            key=lambda r: (
                -_safe_float(r.get("pnl_net_sum", 0.0)),
                str(r.get("slice_id", "")),
            ),
        )
        best = rows_sorted[0]
        rel = str(best.get("run_dir", "")).strip()
        if rel:
            candidate = run_dir / rel
            if (candidate / "metrics.json").exists() and (candidate / "config.json").exists():
                return candidate
    except Exception:
        return run_dir
    return run_dir


def _format_env_lines(latest_dir: Path, enable_alpha_gate: bool) -> List[str]:
    def _win(p: Path) -> str:
        return str(p).replace("/", "\\")

    lines = [f"set ALPHA_GATE_METRICS_PATH={_win(latest_dir / 'metrics.json')}"]
    if enable_alpha_gate:
        lines.append("set ALPHA_GATE_ENABLED=1")
    if (latest_dir / "stability.csv").exists():
        lines.append(f"set ALPHA_GATE_STABILITY_PATH={_win(latest_dir / 'stability.csv')}")
    if (latest_dir / "stability_up.csv").exists():
        lines.append(f"set ALPHA_GATE_STABILITY_UP_PATH={_win(latest_dir / 'stability_up.csv')}")
    if (latest_dir / "stability_down.csv").exists():
        lines.append(f"set ALPHA_GATE_STABILITY_DOWN_PATH={_win(latest_dir / 'stability_down.csv')}")
    return lines


def set_latest_run(
    run_dir: Path,
    latest_dir: Path,
    copy_mode: str = "copy",
    overwrite: bool = True,
    *,
    extra: str = "",
    include_glob: List[str] | None = None,
    strict_extra: bool = False,
    warnings_out: List[str] | None = None,
) -> Path:
    core_run_dir = _resolve_core_run_dir(run_dir)
    metrics = core_run_dir / "metrics.json"
    config = core_run_dir / "config.json"
    if not run_dir.exists():
        raise FileNotFoundError(f"run dir not found: {run_dir}")
    if not metrics.exists():
        raise FileNotFoundError(f"missing metrics.json in {run_dir}")
    if not config.exists():
        raise FileNotFoundError(f"missing config.json in {run_dir}")

    latest_dir.mkdir(parents=True, exist_ok=True)
    warnings_local: List[str] = []

    for src_name in ("metrics.json", "config.json", "summary.md"):
        src = core_run_dir / src_name
        if not src.exists():
            continue
        dst = latest_dir / src_name
        if dst.exists() and not overwrite:
            continue
        _copy_or_link(src, dst, copy_mode)

    extra_names = _parse_extra(extra)
    glob_patterns = _parse_globs(include_glob or [])
    extra_sources = _collect_extra_sources(
        run_dir=run_dir,
        extra_names=extra_names,
        include_globs=glob_patterns,
        strict_extra=bool(strict_extra),
        warnings_out=warnings_local,
    )
    for src in extra_sources:
        dst = latest_dir / src.name
        if dst.exists() and not overwrite:
            continue
        _copy_or_link(src, dst, copy_mode)

    stability_all_src = run_dir / "stability_all.csv"
    stability_src = run_dir / "stability.csv"
    if stability_all_src.exists() and stability_all_src.is_file():
        dst_all = latest_dir / "stability_all.csv"
        if overwrite or not dst_all.exists():
            _copy_or_link(stability_all_src, dst_all, copy_mode)
        dst_stability = latest_dir / "stability.csv"
        if (not stability_src.exists()) and (overwrite or not dst_stability.exists()):
            _copy_or_link(stability_all_src, dst_stability, copy_mode)

    _atomic_write_text(latest_dir / "run_dir.txt", str(run_dir))
    if warnings_out is not None:
        warnings_out.extend(warnings_local)
    return latest_dir


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Set selected run as runs/latest pointer artifacts.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--latest-dir", default="runs/latest")
    p.add_argument("--copy-mode", choices=("copy", "hardlink"), default="copy")
    p.add_argument("--print-env", action="store_true")
    p.add_argument("--enable-alpha-gate", action="store_true")
    p.add_argument("--overwrite", type=int, choices=(0, 1), default=1, help="1 overwrite (default), 0 keep existing files")
    p.add_argument("--extra", default="", help="Comma-separated extra filenames to copy from run-dir.")
    p.add_argument("--include-glob", action="append", default=[], help="Glob pattern(s) to include from run-dir. Can be repeated.")
    p.add_argument("--strict-extra", action="store_true", help="Fail if any --extra file is missing.")
    return p


def main() -> int:
    args = _parser().parse_args()
    try:
        warnings_list: List[str] = []
        latest = set_latest_run(
            run_dir=Path(str(args.run_dir)),
            latest_dir=Path(str(args.latest_dir)),
            copy_mode=str(args.copy_mode),
            overwrite=bool(int(args.overwrite)),
            extra=str(args.extra),
            include_glob=list(args.include_glob or []),
            strict_extra=bool(args.strict_extra),
            warnings_out=warnings_list,
        )
        print(f"Latest run set to: {Path(str(args.run_dir))}")
        for w in warnings_list:
            print(f"WARN: {w}")
        if bool(args.print_env):
            print("")
            for line in _format_env_lines(latest, bool(args.enable_alpha_gate)):
                print(line)
        return 0
    except Exception as e:
        print(f"set_latest_run error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
