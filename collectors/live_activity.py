"""
Live Activity Monitor for AI Quota Hub (Multi-Provider: AGY, Claude Code, OpenAI Codex).

Discovers:
1. Active & Recent Coordinator Sessions across ALL 3 providers:
   - **AGY (`Antigravity`)**: Queries `GetAllCascadeTrajectories` on `language_server.exe` and inspects
     `~/.gemini/antigravity/brain/<cid>/.system_generated/logs/transcript.jsonl` to detect both actively
     generating turns (`CASCADE_RUN_STATUS_RUNNING`) and coordinators actively waiting on a running
     background task (`GENERIC` step with `status == "RUNNING"`).
   - **Claude (`Claude Code / Desktop`)**: Scans `~/.claude/sessions/*.json` with live OS PID verification.
   - **Codex (`OpenAI Codex Desktop / CLI`)**: Scans `~/.codex/state_5.sqlite` `threads` table for recent
     interactive orchestrator sessions.
2. Running Headless Worker Processes & Active Coordinator Background Tasks:
   - Headless CLI workers (`agy.exe -p=...`, `codex exec ...`, `claude -p ...` / `claude --print ...`).
   - Active background tasks (`run_command`) spawned by AGY coordinators (e.g. `validate.py --mode DEVICE`,
     `npx tsx ...`).
"""

from __future__ import annotations

import datetime
import json
import re
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hub_config import load_hub_config
from collectors.agy_collector import _discover_language_server, _query_rpc, get_active_agy_endpoint


DEFAULT_CLAUDE_SESSIONS_DIR = Path.home() / ".claude" / "sessions"
DEFAULT_AGY_BRAIN_DIR = Path.home() / ".gemini" / "antigravity" / "brain"
DEFAULT_CODEX_STATE_DB = Path.home() / ".codex" / "state_5.sqlite"
DEFAULT_BRAIN_SESSION_CTX = Path.home() / ".agents" / "brain" / "state" / "session-context.json"

_TR_FOLD = str.maketrans("İIıŞşĞğÜüÖöÇç\ufffd", "iiissgguuoocc ")


def _fold_title(text: str) -> str:
    return re.sub(r"[^a-z0-9/._\-]+", " ", (text or "").translate(_TR_FOLD).lower()).strip()


def _clean_display_title(title: str) -> str:
    """Repair common UTF-8 replacement characters in Claude session names (e.g. 'kinci beyin' -> 'İkinci beyin')."""
    if not title:
        return ""
    s = str(title)
    s = re.sub(r"^\ufffd?kinci beyin", "İkinci beyin", s)
    s = (
        s.replace("ajanlar\ufffd", "ajanları")
        .replace("Aray\ufffdz\ufffd", "Arayüzü")
        .replace("ArayA\ufffdzA\ufffd", "Arayüzü")
    )
    return s


def _resolve_effective_workspace(raw_cwd: str, title: str, session_id: str = "") -> str:
    """
    Resolve the effective project directory for a coordinator session when the user did not manually
    switch folders (e.g. a sticky default folder in Claude Desktop, `AGY Workspace` in AGY, or `Documents\\Codex`).
    """
    # 1. Check live Second Brain session-context map first (ignore meta/config directories)
    if session_id and DEFAULT_BRAIN_SESSION_CTX.is_file():
        try:
            ctx_map = json.loads(DEFAULT_BRAIN_SESSION_CTX.read_text(encoding="utf-8"))
            entry = ctx_map.get(session_id)
            if isinstance(entry, dict) and entry.get("effective_cwd"):
                eff_cand = str(entry["effective_cwd"])
                if not _is_meta_cwd_path(eff_cand):
                    return eff_cand
        except Exception:
            pass

    folded = _fold_title(title)
    home = Path.home()
    rules: List[Tuple[Tuple[str, ...], Path]] = [
        (("ikinci beyin", "kinci beyin", "second brain", "brain.py"), home / ".agents" / "brain"),
        (("agent-workbench", "agent workbench", "ai-agent-hub", "agent hub"), home / "ai-agent-hub"),
        (("quota hub", "ai quota hub", "ai-quota-orchestrator", "quota scheduler"), home / ".gemini" / "antigravity" / "scratch" / "ai-quota-orchestrator"),
        (("hermes", "hermes agent", "hermes ajan", "hermes ajanlari"), home / "AppData" / "Local" / "hermes" / "hermes-agent"),
        (("uaw", "uaw2", "ultimate ai workflow"), home / "uaw"),
        (("missionloop", "agentrelay", "agent relay"), home / ".gemini" / "antigravity" / "scratch" / "missionloop"),
    ]
    custom_aliases = load_hub_config().get("workspace_aliases")
    if isinstance(custom_aliases, dict):
        for kw, target_raw in custom_aliases.items():
            kw_clean = _fold_title(str(kw).strip())
            if kw_clean and target_raw:
                rules.append(((kw_clean,), Path(str(target_raw)).expanduser()))

    raw_leaf = Path(raw_cwd).name.lower() if raw_cwd and raw_cwd != "AGY Workspace" else ""
    raw_leaf_variants = {raw_leaf, raw_leaf.replace("-", " "), raw_leaf.replace("_", " ")} - {""}

    for keywords, target_path in rules:
        if any(re.search(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", folded) for kw in keywords):
            # If title explicitly names a target project and does NOT name raw_cwd's folder, use target_path
            raw_norm = (raw_cwd or "").lower().replace("\\", "/")
            if not raw_cwd or raw_cwd == "AGY Workspace" or target_path.name.lower() not in raw_norm:
                if raw_leaf_variants and any(k in folded for k in raw_leaf_variants):
                    return raw_cwd
                return str(target_path)

    return raw_cwd or "AGY Workspace"



def _cmdline_pattern() -> str:
    """Command-line regex for live work processes; hub_config "live_activity_pattern" adds e.g. project folders."""
    base = r"validate\.py|tsx"
    extra = str(load_hub_config().get("live_activity_pattern") or "").strip()
    try:
        if extra:
            re.compile(extra)
            return f"({extra}|{base})"
    except re.error:
        pass
    return f"({base})"


def _get_live_processes() -> List[Dict[str, Any]]:
    """Query Win32_Process for agy, claude, codex, pwsh, python, node, and java processes."""
    pattern = _cmdline_pattern().replace("'", "''")
    ps_script = r"""
    $procs = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match "^(agy|claude|codex|pwsh|python|node)" -or
        $_.CommandLine -match '__PATTERN__'
    }
    $out = @()
    foreach ($p in $procs) {
        $out += [PSCustomObject]@{
            pid = $p.ProcessId
            name = $p.Name
            cmdline = $p.CommandLine
        }
    }
    $out | ConvertTo-Json -Compress
    """
    ps_script = ps_script.replace("__PATTERN__", pattern)
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        raw = res.stdout.strip()
        if not raw:
            return []
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception:
        return []


def _read_tail_lines(path: Path, max_bytes: int = 131072) -> List[str]:
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            data = f.read().decode("utf-8", errors="replace")
        lines = data.splitlines()
        if size > max_bytes and len(lines) > 1:
            lines = lines[1:]
        return lines
    except Exception:
        return []


def _read_head_lines(path: Path, max_lines: int = 4, max_bytes: int = 16384) -> List[str]:
    try:
        with path.open("rb") as f:
            data = f.read(max_bytes).decode("utf-8", errors="replace")
        return data.splitlines()[:max_lines]
    except Exception:
        return []


def _is_meta_cwd_path(path_str: str) -> bool:
    """Return True if path_str points to global agent config/brain/skills/temp rather than a project workspace."""
    if not path_str:
        return True
    norm = path_str.replace("\\", "/").lower().rstrip("/")
    meta_markers = (
        "/.gemini/config",
        "/.gemini/antigravity/brain",
        "/.gemini/antigravity/builtin",
        "/.agents/skills",
        "/.agents/learning",
        "/appdata/local/temp",
    )
    if any(m in norm for m in meta_markers):
        return True
    if norm.endswith("/.agents") or norm.endswith("/.gemini") or norm.endswith("/.codex") or norm.endswith("/.claude"):
        return True
    return False


def _unescape_transcript_arg(val: Any) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if s.startswith('"') and s.endswith('"'):
        try:
            return str(json.loads(s))
        except Exception:
            s = s[1:-1]
    return s.replace('\\"', '"').replace("\\n", " ").replace("\\\\", "\\")


def _inspect_agy_transcript(cid: str, live_cmdlines: List[Tuple[int, str, str]]) -> Dict[str, Any]:
    """
    Inspect the head and tail of an AGY conversation's `transcript.jsonl` to determine:
    - Whether the trajectory is a spawned native subagent (`invoke_subagent`, `source == "SYSTEM_SDK"`)
    - Whether the trajectory is an active multi-agent Orchestrator vs. a normal Direct Chat
    - The primary workspace directory (`cwd`) from recent tool calls (ignoring global config/skill reads)
    - Any currently `RUNNING` background tasks (`GENERIC` step with `status == "RUNNING"`)
    """
    tfile = DEFAULT_AGY_BRAIN_DIR / cid / ".system_generated" / "logs" / "transcript.jsonl"
    info: Dict[str, Any] = {
        "cwd": "",
        "running_tasks": [],
        "last_step_age_sec": 999999,
        "is_subagent": False,
        "subagent_task": "",
        "subagent_finished": False,
        "parent_cid": "",
        "is_orchestrator": False,
    }
    if not tfile.exists():
        return info

    try:
        info["last_step_age_sec"] = max(0, int(time.time() - tfile.stat().st_mtime))
    except Exception:
        pass

    head_lines = _read_head_lines(tfile, max_lines=4)
    has_user_input_head = False
    first_source = ""
    for idx, hline in enumerate(head_lines):
        try:
            hobj = json.loads(hline)
        except Exception:
            continue
        if idx == 0:
            first_source = str(hobj.get("source") or "")
        if hobj.get("type") == "USER_INPUT":
            has_user_input_head = True
        if hobj.get("source") == "SYSTEM_SDK" and hobj.get("type") == "SYSTEM_MESSAGE":
            hcontent = str(hobj.get("content") or "")
            from_m = re.search(r"\[From\]\s*([0-9a-f\-]{36})", hcontent)
            if from_m:
                info["parent_cid"] = from_m.group(1)
            msg_m = re.search(r"\[Message\]\s*(.+)", hcontent, re.DOTALL)
            if msg_m:
                raw_msg = " ".join(msg_m.group(1).strip().splitlines()[:2]).strip()
                info["subagent_task"] = raw_msg[:140] + ("..." if len(raw_msg) > 140 else "")

    if first_source == "SYSTEM_SDK" and not has_user_input_head:
        info["is_subagent"] = True

    lines = _read_tail_lines(tfile, max_bytes=196608)
    cwd_found = ""
    fallback_meta_cwd = ""
    pending_cmd_by_step: Dict[int, Tuple[str, str]] = {}
    completed_task_ids: set[str] = set()
    last_planner_called_send_msg = False

    for line in lines:
        if not info["is_orchestrator"] and any(
            marker in line
            for marker in (
                "invoke_subagent",
                "manage_subagents",
                "dispatch_guard.py",
                "launch-resume-worker.py",
                "mission_loop.py",
                "claude -p",
                "codex exec",
                "-wt-",
                "skills/orchestrator/SKILL.md",
                "skills\\orchestrator\\SKILL.md",
            )
        ):
            info["is_orchestrator"] = True
        for done_m in re.finditer(r'Task id \\?"([^"\\]+)\\?" (?:finished|was canceled)', line):
            completed_task_ids.add(done_m.group(1))
        try:
            obj = json.loads(line)
        except Exception:
            continue
        step_idx = int(obj.get("step_index") or 0)
        stype = obj.get("type")

        if stype == "PLANNER_RESPONSE":
            called_send_msg = False
            cmd_offset = 0
            for tc in obj.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                tc_name = str(tc.get("name") or "")
                args = tc.get("args") or {}
                if tc_name in ("invoke_subagent", "define_subagent", "manage_subagents"):
                    info["is_orchestrator"] = True
                if tc_name == "send_message":
                    called_send_msg = True
                raw_cwd = _unescape_transcript_arg(
                    args.get("Cwd") or args.get("Path") or args.get("AbsolutePath") or args.get("TargetFile")
                )
                if raw_cwd:
                    norm_raw = raw_cwd.replace("\\", "/").lower()
                    if "skills/orchestrator/skill.md" in norm_raw or "skills/mission/skill.md" in norm_raw:
                        info["is_orchestrator"] = True
                    p_cwd = Path(raw_cwd)
                    if p_cwd.suffix:
                        p_cwd = p_cwd.parent
                    cand_str = str(p_cwd)
                    if _is_meta_cwd_path(cand_str):
                        fallback_meta_cwd = cand_str
                    else:
                        cwd_found = cand_str
                if tc_name == "run_command":
                    cmd_offset += 1
                    cmd_str = _unescape_transcript_arg(args.get("CommandLine"))
                    if any(
                        k in cmd_str
                        for k in (
                            "dispatch_guard.py",
                            "launch-resume-worker.py",
                            "mission_loop.py",
                            "claude -p",
                            "claude --print",
                            "codex exec",
                            "agy -p",
                            "agy.exe -p",
                            "git worktree",
                        )
                    ):
                        info["is_orchestrator"] = True
                    cmd_cwd = _unescape_transcript_arg(args.get("Cwd")) or cwd_found
                    pending_cmd_by_step[step_idx + cmd_offset] = (cmd_str, cmd_cwd)
            last_planner_called_send_msg = called_send_msg

    if info["is_subagent"] and last_planner_called_send_msg:
        info["subagent_finished"] = True

    # Check the last 4 steps of the conversation for a currently running background task
    for line in lines[-4:]:
        try:
            obj = json.loads(line)
        except Exception:
            continue
        step_idx = int(obj.get("step_index") or 0)
        stype = obj.get("type")
        sstatus = obj.get("status")

        if stype == "GENERIC" and sstatus == "RUNNING":
            content = str(obj.get("content") or "")
            tid_m = re.search(r"background task with task id:\s*([^\s\\n\"]+)", content)
            if tid_m:
                full_tid = tid_m.group(1).strip()
                if full_tid in completed_task_ids:
                    continue
                cmd_str, cmd_cwd = pending_cmd_by_step.get(step_idx, ("", cwd_found))
                if "start_dashboard.ps1" in cmd_str or "Get-CimInstance Win32_Process" in cmd_str:
                    continue
                matched_pid = 0
                for l_pid, l_name, l_cmd in live_cmdlines:
                    if "start_dashboard.ps1" in l_cmd or "Get-CimInstance Win32_Process" in l_cmd:
                        continue
                    if ("validate.py" in cmd_str and "validate.py" in l_cmd) or (
                        "playRun" in cmd_str and ("playRun" in l_cmd or "tsx" in l_cmd)
                    ) or (cmd_str[:22] and cmd_str[:22] in l_cmd):
                        matched_pid = l_pid
                        break
                # Even if child process spawned under a wrapper, if the step is in the last 4 lines and RUNNING
                # and its process is alive or recently started, report it
                if matched_pid > 0 or info["last_step_age_sec"] < 7200:
                    info["running_tasks"].append(
                        {
                            "pid": matched_pid,
                            "task_id": full_tid,
                            "cmd": cmd_str or content[:120],
                            "cwd": cmd_cwd or cwd_found,
                        }
                    )

    info["cwd"] = cwd_found or fallback_meta_cwd
    return info


def _collect_agy_coordinators_and_tasks(
    live_cmdlines: List[Tuple[int, str, str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Discover AGY coordinator/chat conversations and their active subagents & background tasks."""
    coordinators: List[Dict[str, Any]] = []
    bg_workers: List[Dict[str, Any]] = []

    cached_ep = get_active_agy_endpoint()
    servers = [cached_ep] if cached_ep else _discover_language_server()
    for srv in servers:
        ls_pid = int(srv.get("pid") or 0)
        token = srv.get("csrf_token", "")
        for port in srv.get("ports", []):
            res = _query_rpc(int(port), token, "GetAllCascadeTrajectories")
            if not res or "trajectorySummaries" not in res:
                continue
            summaries = res.get("trajectorySummaries") or {}
            for cid, tinfo in summaries.items():
                if not isinstance(tinfo, dict):
                    continue
                title = tinfo.get("summary") or (tinfo.get("annotations") or {}).get("title") or f"AGY Session {cid[:8]}"
                raw_status = str(tinfo.get("status") or "")
                t_meta = _inspect_agy_transcript(cid, live_cmdlines)

                running_tasks = t_meta.get("running_tasks") or []
                is_running = ("RUNNING" in raw_status) or (len(running_tasks) > 0)
                age_sec = int(t_meta.get("last_step_age_sec", 999999))

                clean_title = _clean_display_title(title)
                raw_cwd_str = t_meta.get("cwd") or "AGY Workspace"
                cwd_str = _resolve_effective_workspace(raw_cwd_str, clean_title, session_id=cid)

                # Native AGY Subagents (spawned via invoke_subagent): NEVER list as Coordinators!
                if t_meta.get("is_subagent"):
                    sub_active = is_running or (age_sec < 90 and not t_meta.get("subagent_finished"))
                    if sub_active:
                        sub_task = t_meta.get("subagent_task") or clean_title
                        bg_workers.append(
                            {
                                "pid": running_tasks[0]["pid"] if (running_tasks and running_tasks[0]["pid"]) else ls_pid,
                                "session_id": cid,
                                "provider": "agy",
                                "model": "AGY Subagent (invoke_subagent)",
                                "role": "subagent",
                                "workspace": cwd_str,
                                "workspace_short": Path(cwd_str).name if cwd_str and cwd_str != "AGY Workspace" else f"subagent-{cid[:6]}",
                                "task_summary": sub_task,
                                "native_subagent": True,
                            }
                        )
                    continue

                is_orch = bool(t_meta.get("is_orchestrator")) or bool(
                    re.search(r"orchestrat|takeover|koordinat|missionloop", clean_title, re.I)
                )
                has_live_child_worker = any(cid in l_cmd for _, _, l_cmd in live_cmdlines)
                if is_orch and (has_live_child_worker or age_sec < 180):
                    is_running = True
                # Filter out old idle sessions (keep orchestrators up to 24h, idle direct chats up to 6h)
                max_idle_sec = 86400 if is_orch else 21600
                if not is_running and age_sec > max_idle_sec:
                    continue

                status_label = "busy" if is_running else "idle"
                session_kind = "orchestrator" if is_orch else "chat"
                model_label = (
                    "🎯 Orkestratör (AGY · Argon / Opus)"
                    if is_orch
                    else "💬 Doğrudan Sohbet (AGY)"
                )

                coordinators.append(
                    {
                        "pid": running_tasks[0]["pid"] if (running_tasks and running_tasks[0]["pid"]) else ls_pid,
                        "session_id": cid,
                        "provider": "agy",
                        "name": clean_title,
                        "cwd": cwd_str,
                        "status": status_label,
                        "session_kind": session_kind,
                        "entrypoint": "antigravity",
                        "model": model_label,
                        "alive": True,
                        "sort_ts": time.time() - age_sec,
                    }
                )

                for rt in running_tasks:
                    w_cwd = rt.get("cwd") or cwd_str
                    cmd_clean = " ".join(str(rt.get("cmd") or "").split())
                    bg_workers.append(
                        {
                            "pid": rt.get("pid") or ls_pid,
                            "provider": "agy-task",
                            "model": f"BG Task ({clean_title[:22]})",
                            "role": "bg-task",
                            "workspace": w_cwd,
                            "workspace_short": Path(w_cwd).name if w_cwd else "agy-task",
                            "task_summary": cmd_clean[:140] + ("..." if len(cmd_clean) > 140 else ""),
                        }
                    )
            break
    return coordinators, bg_workers


def _collect_codex_coordinators(live_pids: set[int]) -> List[Dict[str, Any]]:
    """Discover recent unarchived interactive Codex coordinator/chat sessions from state_5.sqlite."""
    if not DEFAULT_CODEX_STATE_DB.exists():
        return []
    out: List[Dict[str, Any]] = []
    now_ts = int(time.time())
    try:
        conn = sqlite3.connect(f"file:{DEFAULT_CODEX_STATE_DB}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                """
                SELECT id, title, name, cwd, model, reasoning_effort, updated_at, source
                FROM threads
                WHERE archived = 0 AND source IN ('vscode', 'cli')
                ORDER BY updated_at DESC
                LIMIT 6
                """
            ).fetchall()
        finally:
            conn.close()
        for tid, title, name, cwd, model, effort, updated_at, source in rows:
            age_sec = max(0, now_ts - int(updated_at or 0))
            clean_cwd = str(cwd or "").replace("\\\\?\\", "")
            raw_title_str = str(title or "")
            display_name = name or (raw_title_str.splitlines()[0][:55] if raw_title_str else f"Codex {tid[:8]}")
            is_orch = bool(
                re.search(r"orchestrat|takeover|koordinat|missionloop", f"{display_name} {raw_title_str[:240]}", re.I)
            )
            max_idle_sec = 43200 if is_orch else 21600
            if age_sec > max_idle_sec:
                continue
            if display_name == "Use the Orchestrator skill" and clean_cwd:
                display_name = f"Codex Orchestrator ({Path(clean_cwd).name})"
            is_busy = age_sec < 90
            session_kind = "orchestrator" if is_orch else "chat"
            prefix = "🎯 Orkestratör" if is_orch else "💬 Doğrudan Sohbet"
            out.append(
                {
                    "pid": 0,
                    "session_id": tid,
                    "provider": "codex",
                    "name": display_name,
                    "cwd": clean_cwd,
                    "status": "busy" if is_busy else "idle",
                    "session_kind": session_kind,
                    "entrypoint": f"codex-{source}",
                    "model": f"{prefix} · {model or 'gpt-6.1-sol'} ({effort or 'medium'})",
                    "alive": is_busy,
                    "sort_ts": float(updated_at or 0),
                }
            )
    except Exception:
        pass
    return out


def _collect_codex_subagents() -> List[Dict[str, Any]]:
    """
    Discover active spawned Codex native subagent threads (`source LIKE '{"subagent":%'`) from state_5.sqlite.
    Checks both `updated_at` and the `rollout_path` file's mtime + terminal event markers (`task_complete` / `turn_aborted`)
    up to the 600s quiet threshold so long tool calls are not dropped prematurely while completed turns are excluded.
    """
    if not DEFAULT_CODEX_STATE_DB.exists():
        return []
    out: List[Dict[str, Any]] = []
    now_ts = int(time.time())
    try:
        conn = sqlite3.connect(f"file:{DEFAULT_CODEX_STATE_DB}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                """
                SELECT id, title, name, cwd, model, reasoning_effort, updated_at, source, agent_nickname, agent_role, rollout_path
                FROM threads
                WHERE archived = 0 AND source LIKE '{"subagent":%'
                ORDER BY updated_at DESC
                LIMIT 10
                """
            ).fetchall()
        finally:
            conn.close()
        for tid, title, name, cwd, model, effort, updated_at, source, nickname, role, rollout_path in rows:
            last_ts = float(updated_at or 0)
            r_path = Path(str(rollout_path or "").replace("\\\\?\\", "")) if rollout_path else None
            if r_path and r_path.is_file():
                try:
                    last_ts = max(last_ts, float(r_path.stat().st_mtime))
                    with r_path.open("rb") as fh:
                        fh.seek(0, 2)
                        size = fh.tell()
                        fh.seek(max(0, size - 4096))
                        tail = fh.read().decode("utf-8", errors="replace")
                    last_start = max(tail.rfind('"task_started"'), tail.rfind('"user_message"'))
                    last_end = max(tail.rfind('"task_complete"'), tail.rfind('"turn_aborted"'))
                    if last_end != -1 and last_end > last_start:
                        continue
                except OSError:
                    pass
            age_sec = max(0.0, float(now_ts) - last_ts)
            if age_sec > 600.0:  # Match witness quiet_threshold_sec (600s) for long-running tool calls
                continue
            clean_cwd = str(cwd or "").replace("\\\\?\\", "")
            summary = name or nickname or (str(title or "").splitlines()[0][:100] if title else f"Codex subagent {tid[:8]}")
            out.append(
                {
                    "pid": 0,
                    "session_id": str(tid),
                    "provider": "codex",
                    "model": f"{model or 'gpt-6.1-sol'} ({effort or 'medium'})",
                    "role": str(role or "subagent"),
                    "workspace": clean_cwd,
                    "workspace_short": Path(clean_cwd).name if clean_cwd else "codex-subagent",
                    "task_summary": summary,
                    "native_subagent": True,
                }
            )
    except Exception:
        pass
    return out


_CLAUDE_PRINT_FLAG = re.compile(r"(?:^|\s)(?:-p|--print)(?=\s|=|$)")


def _is_headless_claude(name: str, cmd: str) -> bool:
    """A `claude -p` / `claude --print` process (the CLI binary itself or node running the Claude Code CLI)."""
    lowered = cmd.lower()
    runs_claude_cli = "claude-code" in lowered or "claude.js" in lowered or ("cli.js" in lowered and "claude" in lowered)
    is_claude_proc = "claude" in name or ("node" in name and runs_claude_cli)
    return bool(is_claude_proc and _CLAUDE_PRINT_FLAG.search(cmd))


def _is_headless_codex(name: str, cmd: str) -> bool:
    """A `codex exec` process (either `codex.exe` or `node.exe ... codex.js exec`)."""
    lowered = cmd.lower()
    if "exec-server" in lowered:
        return False
    is_codex_proc = "codex" in name or ("node" in name and ("codex.js" in lowered or "@openai\\codex" in lowered or "@openai/codex" in lowered))
    return bool(is_codex_proc and " exec " in f" {lowered} ")


def collect_live_activity() -> Dict[str, Any]:
    """
    Return currently running Coordinator/Chat sessions (across AGY, Claude, and Codex)
    and Subagent / Headless Worker / Background Task processes on this machine.
    """
    procs = _get_live_processes()
    live_pids = {int(p["pid"]) for p in procs if p.get("pid")}
    live_cmdlines: List[Tuple[int, str, str]] = [
        (int(p.get("pid") or 0), str(p.get("name") or ""), str(p.get("cmdline") or ""))
        for p in procs
        if p.get("cmdline")
    ]

    coordinators: List[Dict[str, Any]] = []
    workers: List[Dict[str, Any]] = []

    # 1. AGY Coordinators, Active Subagents & Background Tasks
    agy_coords, agy_bg_workers = _collect_agy_coordinators_and_tasks(live_cmdlines)
    coordinators.extend(agy_coords)
    workers.extend(agy_bg_workers)

    headless_claude_pids: set[int] = {
        pid for pid, name_raw, cmd in live_cmdlines if _is_headless_claude(name_raw.lower(), cmd)
    }
    claude_session_by_pid: Dict[int, Dict[str, Any]] = {}

    # 2. Claude Code / Desktop Coordinators (excluding headless `claude -p` worker sessions)
    if DEFAULT_CLAUDE_SESSIONS_DIR.exists():
        for sfile in sorted(DEFAULT_CLAUDE_SESSIONS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                data = json.loads(sfile.read_text(encoding="utf-8", errors="replace"))
                pid = int(data.get("pid", 0))
                raw_status = str(data.get("status", "idle"))
                mtime = sfile.stat().st_mtime
                age_sec = max(0.0, time.time() - mtime)
                is_alive = pid in live_pids
                status = "busy" if (raw_status == "busy" and (is_alive or age_sec < 300)) else "idle"
                sid = str(data.get("sessionId") or "")
                cname = _clean_display_title(data.get("name") or "Claude Code Session")
                raw_ccwd = str(data.get("cwd") or "")
                eff_ccwd = _resolve_effective_workspace(raw_ccwd, cname, session_id=sid)
                if pid:
                    claude_session_by_pid[pid] = {"name": cname, "cwd": eff_ccwd, "session_id": sid}
                # Exclude headless `claude -p` worker processes or auto-named `<folder>-<2hex>` headless runs
                if pid in headless_claude_pids or re.match(r"^[a-z0-9_\-]+-[0-9a-f]{2}$", cname):
                    continue
                is_orch = bool(re.search(r"orchestrat|takeover|koordinat|missionloop", cname, re.I))
                if status != "busy" and not is_alive and age_sec > (86400 if is_orch else 21600):
                    continue
                session_kind = "orchestrator" if is_orch else "chat"
                prefix = "🎯 Orkestratör" if is_orch else "💬 Doğrudan Sohbet"
                coordinators.append(
                    {
                        "pid": pid,
                        "session_id": sid,
                        "provider": "claude",
                        "name": cname,
                        "cwd": eff_ccwd,
                        "status": status,
                        "session_kind": session_kind,
                        "entrypoint": data.get("entrypoint", "claude-desktop"),
                        "model": f"{prefix} · claude-opus-5-5",
                        "alive": is_alive,
                        "sort_ts": float(mtime),
                    }
                )
            except Exception:
                continue

    # 3. OpenAI Codex Interactive Coordinators & Active Native Subagent Threads
    coordinators.extend(_collect_codex_coordinators(live_pids))
    workers.extend(_collect_codex_subagents())

    # Sort sessions: busy first, then orchestrators before plain chats, then by recency
    coordinators.sort(
        key=lambda c: (
            1 if c.get("status") == "busy" else 0,
            1 if c.get("session_kind") == "orchestrator" else 0,
            float(c.get("sort_ts") or 0),
        ),
        reverse=True,
    )

    # 4. Headless CLI Workers (agy -p, codex exec, claude -p, and launch-resume-worker.py supervisors)
    for pid, name_raw, cmd in live_cmdlines:
        name = name_raw.lower()
        if "agy" in name and ("-p=" in cmd or "-p " in cmd or "--print" in cmd):
            if "-p=/usage" in cmd or '-p="/usage"' in cmd:
                continue
            model_m = re.search(r"--model\s+([^\s\"]+)", cmd)
            dir_m = re.search(r"--add-dir\s+\"?([^\"]+?)\"?(?:\s+--|\s+\"?-p)", cmd)
            idx = cmd.find("-p=")
            prompt_txt = cmd[idx + 3 :].strip(' "') if idx != -1 else ""
            worktree = dir_m.group(1).strip() if dir_m else ""
            role = "reviewer" if "reviewer" in prompt_txt.lower() else "worker"
            workers.append(
                {
                    "pid": pid,
                    "provider": "agy",
                    "model": model_m.group(1) if model_m else "gemini-4-hydrogen-medium",
                    "role": role,
                    "workspace": worktree,
                    "workspace_short": Path(worktree).name if worktree else "local",
                    "task_summary": prompt_txt[:140] + ("..." if len(prompt_txt) > 140 else ""),
                }
            )
        elif _is_headless_claude(name, cmd):
            model_m = re.search(r"--model[\s=]+\"?([^\s\"]+)", cmd)
            effort_m = re.search(r"--effort[\s=]+\"?([a-zA-Z]+)", cmd)
            model_name = model_m.group(1) if model_m else "claude"
            if effort_m:
                model_name = f"{model_name} ({effort_m.group(1)})"
            csess = claude_session_by_pid.get(pid) or {}
            c_cwd = str(csess.get("cwd") or "")
            c_name = str(csess.get("name") or "")
            p_idx = cmd.find(" -p ")
            prompt_excerpt = cmd[p_idx + 4 :].strip(' "') if p_idx != -1 else cmd
            if "node_modules" in prompt_excerpt or "WinGet\\Packages" in prompt_excerpt:
                prompt_excerpt = f"Headless Claude Worker ({c_name or f'PID {pid}'})"
            workers.append(
                {
                    "pid": pid,
                    "provider": "claude",
                    "model": model_name,
                    "role": "worker",
                    "workspace": c_cwd,
                    "workspace_short": Path(c_cwd).name if c_cwd else (c_name or "claude-print"),
                    "task_summary": prompt_excerpt[:140] + ("..." if len(prompt_excerpt) > 140 else ""),
                }
            )
        elif _is_headless_codex(name, cmd):
            model_m = re.search(r"-m\s+([^\s\"]+)", cmd)
            effort_m = re.search(r"model_reasoning_effort=([a-zA-Z]+)", cmd)
            model_name = model_m.group(1) if model_m else "gpt-6.1-sol"
            if effort_m:
                model_name = f"{model_name} ({effort_m.group(1)})"
            workers.append(
                {
                    "pid": pid,
                    "provider": "codex",
                    "model": model_name,
                    "role": "worker",
                    "workspace": "",
                    "workspace_short": "codex-exec",
                    "task_summary": cmd[:140] + ("..." if len(cmd) > 140 else ""),
                }
            )
        elif "python" in name and "launch-resume-worker.py" in cmd:
            task_m = re.search(r"--task-id\s+([^\s\"]+)", cmd)
            prov_m = re.search(r"--provider\s+([^\s\"]+)", cmd)
            model_m = re.search(r"--model\s+([^\s\"]+)", cmd)
            wt_m = re.search(r"--worktree\s+\"?([^\"]+?)\"?(?:\s+--|$)", cmd)
            task_id = task_m.group(1) if task_m else "worker"
            prov = prov_m.group(1) if prov_m else "worker"
            model_name = model_m.group(1) if model_m else "headless-worker"
            wt = wt_m.group(1).strip() if wt_m else ""
            workers.append(
                {
                    "pid": pid,
                    "provider": prov,
                    "model": model_name,
                    "role": "worker-supervisor",
                    "workspace": wt,
                    "workspace_short": Path(wt).name if wt else task_id[:24],
                    "task_summary": f"Supervised Task: {task_id}",
                }
            )

    active_orch_count = sum(
        1 for c in coordinators if c.get("status") == "busy" and c.get("session_kind") == "orchestrator"
    )
    active_chat_count = sum(
        1 for c in coordinators if c.get("status") == "busy" and c.get("session_kind") != "orchestrator"
    )

    return {
        "coordinators": coordinators,
        "workers": workers,
        "active_coordinator_count": sum(1 for c in coordinators if c["status"] == "busy"),
        "active_orchestrator_count": active_orch_count,
        "active_chat_count": active_chat_count,
        "active_worker_count": len(workers),
    }
