"""Config loading with a small YAML fallback for minimal environments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AppConfig:
    runtime: dict[str, Any] = field(default_factory=dict)
    capture: dict[str, Any] = field(default_factory=dict)
    control: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    storage: dict[str, Any] = field(default_factory=dict)
    guards: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    raw = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        data = json.loads(raw)
    else:
        data = _load_yaml(raw)
    return AppConfig(
        runtime=dict(data.get("runtime", {})),
        capture=dict(data.get("capture", {})),
        control=dict(data.get("control", {})),
        model=dict(data.get("model", {})),
        storage=dict(data.get("storage", {})),
        guards=dict(data.get("guards", {})),
    )


def _load_yaml(raw: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(raw)
        return loaded or {}
    except ModuleNotFoundError:
        return _simple_yaml(raw)


def _simple_yaml(raw: str) -> dict[str, Any]:
    """Parse the simple nested YAML used by example configs if PyYAML is absent."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any] | list[Any]]] = [(-1, root)]

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if stripped.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError("List item found outside list context")
            parent.append(_coerce_scalar(stripped[2:].strip()))
            continue

        key, sep, value = stripped.partition(":")
        if not sep:
            raise ValueError(f"Invalid config line: {line}")
        key = key.strip()
        value = value.strip()

        if value == "":
            child: dict[str, Any] | list[Any]
            child = []
            if _next_meaningful_line_is_list(raw, line):
                child = []
            else:
                child = {}
            if isinstance(parent, dict):
                parent[key] = child
            else:
                raise ValueError("Nested mapping under list is not supported by fallback parser")
            stack.append((indent, child))
        else:
            if not isinstance(parent, dict):
                raise ValueError("Scalar mapping under list is not supported by fallback parser")
            parent[key] = _coerce_scalar(value)

    return root


def _next_meaningful_line_is_list(raw: str, current_line: str) -> bool:
    lines = raw.splitlines()
    try:
        idx = lines.index(current_line)
    except ValueError:
        return False
    current_indent = len(current_line) - len(current_line.lstrip(" "))
    for line in lines[idx + 1 :]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        return indent > current_indent and stripped.startswith("- ")
    return False


def _coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
