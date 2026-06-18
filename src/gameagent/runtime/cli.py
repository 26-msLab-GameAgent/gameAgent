"""Terminal entrypoint for GameAgent."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from gameagent.models import load_config
from gameagent.runtime.factory import (
    build_capture,
    build_control,
    build_logger,
    build_model,
    build_validator,
)
from gameagent.runtime.runner import AgentRunner, RunnerOptions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gameagent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the frame/action loop")
    run_parser.add_argument("--config", required=True, help="YAML or JSON config path")
    run_parser.add_argument("--steps", type=int, default=None, help="override max episode steps")
    run_parser.add_argument("--dry-run", action="store_true", help="capture and decide, but do not act")

    doctor_parser = subparsers.add_parser("doctor", help="check local runtime dependencies")
    doctor_parser.add_argument("--adb-path", default="adb")

    args = parser.parse_args(argv)

    if args.command == "doctor":
        raise SystemExit(_doctor(args.adb_path))
    if args.command == "run":
        raise SystemExit(_run(args.config, args.steps, args.dry_run))


def _run(config_path: str, steps: int | None, dry_run: bool) -> int:
    config = load_config(config_path)
    if steps is not None:
        config.runtime["max_episode_steps"] = steps

    capture = build_capture(config)
    control = build_control(config)
    model = build_model(config)
    validator = build_validator(config)
    logger = build_logger(config)
    options = RunnerOptions(
        tick_interval_ms=int(config.runtime.get("tick_interval_ms", 800)),
        settle_after_action_ms=int(config.runtime.get("settle_after_action_ms", 800)),
        max_episode_steps=int(config.runtime.get("max_episode_steps", 1000)),
        emergency_stop_path=config.runtime.get("emergency_stop_path"),
        dry_run=dry_run,
    )
    runner = AgentRunner(capture, model, control, validator, logger, options)
    return runner.run()


def _doctor(adb_path: str) -> int:
    print("[gameagent] python package: ok")
    adb = shutil.which(adb_path)
    if not adb:
        print(f"[gameagent] adb: not found ({adb_path})")
        print("[gameagent] mock mode still works without adb")
        return 0
    print(f"[gameagent] adb: {adb}")
    proc = subprocess.run(
        [adb_path, "devices"],
        capture_output=True,
        text=True,
        check=False,
        env=_adb_env(),
    )
    if proc.returncode != 0:
        print(proc.stderr.strip())
        return proc.returncode
    print(proc.stdout.strip())
    return 0


def _adb_env() -> dict[str, str]:
    env = os.environ.copy()
    android_dir = Path.home() / ".android"
    try:
        android_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        fallback = Path.cwd() / ".android"
        fallback.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(Path.cwd())
        env["ANDROID_USER_HOME"] = str(fallback)
    return env


if __name__ == "__main__":
    main()
