"""Optional per-user Hub settings. With no files present the Hub runs with built-in defaults.

Sources, later wins:
1. `~/.agents/workbench.json`, key "quota_hub" (written by the workbench installer; WORKBENCH_CONFIG overrides the path)
2. `<owner layer>/quota-hub.json`, where the owner layer is `~/.agents/owner-overrides` (OWNER_OVERRIDES_DIR overrides it)

Keys:
- "providers": list of enabled collectors, subset of ["agy", "claude", "codex"]; default all three.
  A disabled provider is reported as UNAVAILABLE and never queried.
- "live_activity_pattern": extra regex for process command lines counted as live work (e.g. your project folders).
- "tool_catalog": path to a Python file defining `recommend(task_request) -> dict` for task-based tool selection;
  default `<owner layer>/quota-hub/tool_catalog.py` when that file exists, otherwise no tools.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

KNOWN_PROVIDERS = ("agy", "claude", "codex")


def _agents_dir() -> Path:
    return Path.home() / ".agents"


def _owner_layer() -> Path:
    return Path(os.environ.get("OWNER_OVERRIDES_DIR") or _agents_dir() / "owner-overrides")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_hub_config() -> Dict[str, Any]:
    workbench = Path(os.environ.get("WORKBENCH_CONFIG") or _agents_dir() / "workbench.json")
    overrides = _owner_layer() / "quota-hub.json"
    cfg: Dict[str, Any] = {}
    section = _read_json(workbench).get("quota_hub")
    if isinstance(section, dict):
        cfg.update(section)
    cfg.update(_read_json(overrides))
    return cfg


def enabled_providers(cfg: Dict[str, Any] | None = None) -> set[str]:
    cfg = load_hub_config() if cfg is None else cfg
    wanted = cfg.get("providers")
    if not isinstance(wanted, list):
        return set(KNOWN_PROVIDERS)
    return {str(p).lower().strip() for p in wanted} & set(KNOWN_PROVIDERS)


def tool_catalog_path(cfg: Dict[str, Any] | None = None) -> Path | None:
    cfg = load_hub_config() if cfg is None else cfg
    raw = cfg.get("tool_catalog")
    path = Path(os.path.expandvars(os.path.expanduser(raw))) if raw else \
        _owner_layer() / "quota-hub" / "tool_catalog.py"
    return path if path.is_file() else None
