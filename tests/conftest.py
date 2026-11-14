"""
Test isolation for the Quota Hub: runs before any test module imports `main` / `orchestrator`.
- `QUOTA_HUB_HOME` points state + telemetry DB at a throwaway directory (never the live hub files).
- `ORCHESTRATOR_POLICY_PATH` points at a missing file so tests use the implementation defaults
  regardless of the owner's `~/.agents/policy.json`.
- `QUOTA_HUB_DISABLE_SUPERVISOR=1` keeps the autonomous supervisor from starting.
- `WORKBENCH_CONFIG` / `OWNER_OVERRIDES_DIR` point at missing paths so no personal Hub config or tool catalog leaks in.
"""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_HUB_HOME = tempfile.mkdtemp(prefix="quota-hub-test-")
os.environ["QUOTA_HUB_HOME"] = _HUB_HOME
os.environ["ORCHESTRATOR_POLICY_PATH"] = str(Path(_HUB_HOME) / "no-policy.json")
os.environ["QUOTA_HUB_DISABLE_SUPERVISOR"] = "1"
os.environ["WORKBENCH_CONFIG"] = str(Path(_HUB_HOME) / "no-workbench.json")
os.environ["OWNER_OVERRIDES_DIR"] = str(Path(_HUB_HOME) / "no-owner-overrides")
atexit.register(shutil.rmtree, _HUB_HOME, True)
