"""
AGY (Google Antigravity) Zero-Token Local Telemetry Collector.

Detects the local `language_server.exe` process, extracts its PID and
`--csrf_token`, discovers its listening loopback ports, and queries BOTH:
1. `POST http://127.0.0.1:[PORT]/exa.language_server_pb.LanguageServerService/RetrieveUserQuotaSummary`
   (the exact RPC used by `agy -p /usage` and the AGY Harness UI, returning separate
   `5h` and `weekly` buckets for `Gemini Models` and `Claude and GPT models`).
2. `POST http://127.0.0.1:[PORT]/exa.language_server_pb.LanguageServerService/GetUserStatus`
   (returning plan details, prompt/flow credits, and per-model catalog metadata).

Why earlier versions intermittently lost AGY quota on Windows:
- Go's `language_server.exe` binds loopback sockets (`127.0.0.1:PORT`, `remote=0.0.0.0:0`) that
  Windows `GetExtendedTcpTable` frequently reports with `dwState = 0` instead of `dwState = 2` (`LISTEN`).
- Both `netstat -ano` (which completely omits `dwState = 0` rows) and `Get-NetTCPConnection -State Listen`
  miss those sockets, returning `ports = []` even when `language_server.exe` is healthy and responding.

Permanent multi-layer port & endpoint discovery in this module:
1. Fast in-memory + disk-persisted endpoint cache (`pid`, `csrf_token`, `port`) verified via a ~2ms
   loopback RPC call (survives Quota Hub restarts without spawning PowerShell).
2. Direct Win32 kernel TCP table inspection via `ctypes.WinDLL("iphlpapi.dll").GetExtendedTcpTable`
   (both IPv4 `AF_INET` and IPv6 `AF_INET6` in <0.5ms), capturing all loopback/wildcard sockets owned by
   `pid` where `remote_port == 0` regardless of `dwState` (`0`, `2`, `100`, etc.).
3. Command-line `--host_bridge_url=http://127.0.0.1:<port>` port-range probing (`bridge_port + 1 .. + 16`)
   plus `netstat -ano` fallback.
4. Last-known-good snapshot retention marked with `confidence = "STALE"` on transient hiccups,
   `confidence = "VERIFIED"` for fresh authoritative RPC responses, and `confidence = "UNAVAILABLE"`
   (never fake `100% used`) when `language_server.exe` is genuinely not running.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import datetime
import json
from pathlib import Path
import re
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


_CACHE_FILE = Path(__file__).resolve().parent.parent / ".agy_endpoint_cache.json"

# Cached connection state to avoid spawning PowerShell on every poll
_CACHED_ENDPOINT: Dict[str, Any] = {
    "pid": None,
    "csrf_token": None,
    "port": None,
}

# Last-known-good result cache (prevents transient RPC hiccups from showing offline/full)
_LAST_GOOD_RESULT: Optional[Dict[str, Any]] = None
_LAST_GOOD_TS: float = 0.0
_CACHE_TTL_SEC: float = 600.0


def _load_disk_cache() -> None:
    """Load persisted `(pid, port, csrf_token)` and last good result from disk on startup."""
    global _LAST_GOOD_RESULT, _LAST_GOOD_TS
    if _CACHED_ENDPOINT.get("pid") and _CACHED_ENDPOINT.get("port") and _CACHED_ENDPOINT.get("csrf_token"):
        return
    try:
        if _CACHE_FILE.is_file():
            data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                ep = data.get("endpoint")
                if isinstance(ep, dict) and ep.get("pid") and ep.get("port") and ep.get("csrf_token"):
                    _CACHED_ENDPOINT["pid"] = int(ep["pid"])
                    _CACHED_ENDPOINT["port"] = int(ep["port"])
                    _CACHED_ENDPOINT["csrf_token"] = str(ep["csrf_token"])
                if _LAST_GOOD_RESULT is None and isinstance(data.get("last_good_result"), dict):
                    _LAST_GOOD_RESULT = data["last_good_result"]
                    _LAST_GOOD_TS = float(data.get("last_good_ts") or 0.0)
    except Exception:
        pass


def _save_disk_cache() -> None:
    """Persist verified `(pid, port, csrf_token)` and last good snapshot to disk."""
    try:
        payload = {
            "endpoint": dict(_CACHED_ENDPOINT),
            "last_good_ts": _LAST_GOOD_TS,
            "last_good_result": _LAST_GOOD_RESULT,
        }
        tmp = _CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(_CACHE_FILE)
    except Exception:
        pass


def _parse_iso_reset_seconds(reset_time_str: Optional[str], now: Optional[datetime.datetime] = None) -> int:
    """Parse ISO-8601 timestamp (e.g., '2026-10-13T23:30:15Z') and return seconds until reset."""
    if not reset_time_str:
        return 3600
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)
    try:
        cleaned = reset_time_str.strip()
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        dt = datetime.datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        diff = int((dt - now).total_seconds())
        return max(0, diff)
    except Exception:
        return 3600


class _MIB_TCPROW_OWNER_PID(ctypes.Structure):
    _fields_ = [
        ("dwState", wintypes.DWORD),
        ("dwLocalAddr", wintypes.DWORD),
        ("dwLocalPort", wintypes.DWORD),
        ("dwRemoteAddr", wintypes.DWORD),
        ("dwRemotePort", wintypes.DWORD),
        ("dwOwningPid", wintypes.DWORD),
    ]


class _MIB_TCP6ROW_OWNER_PID(ctypes.Structure):
    _fields_ = [
        ("ucLocalAddr", ctypes.c_ubyte * 16),
        ("dwLocalScopeId", wintypes.DWORD),
        ("dwLocalPort", wintypes.DWORD),
        ("ucRemoteAddr", ctypes.c_ubyte * 16),
        ("dwRemoteScopeId", wintypes.DWORD),
        ("dwRemotePort", wintypes.DWORD),
        ("dwState", wintypes.DWORD),
        ("dwOwningPid", wintypes.DWORD),
    ]


def _discover_ports_via_win32_tcp_table(pid: int) -> List[int]:
    """
    Directly query Windows `iphlpapi.dll` `GetExtendedTcpTable` (IPv4 + IPv6) in <0.5ms.
    Critically, this captures Go `language_server.exe` listening sockets where Windows reports
    `dwState == 0` (`remote_port == 0`), which `netstat -ano` and `Get-NetTCPConnection -State Listen`
    silently omit.
    """
    ports: List[int] = []
    established_loopback_ports: List[int] = []
    try:
        iphlpapi = ctypes.WinDLL("iphlpapi.dll")
    except Exception:
        return ports

    # 1. IPv4 table (AF_INET = 2, TCP_TABLE_OWNER_PID_ALL = 5)
    try:
        size = wintypes.DWORD(0)
        iphlpapi.GetExtendedTcpTable(None, ctypes.byref(size), False, 2, 5, 0)
        if size.value > 0:
            buf = ctypes.create_string_buffer(size.value)
            ret = iphlpapi.GetExtendedTcpTable(buf, ctypes.byref(size), False, 2, 5, 0)
            if ret == 0:
                num_entries = ctypes.cast(buf, ctypes.POINTER(wintypes.DWORD)).contents.value
                rows = ctypes.cast(
                    ctypes.addressof(buf) + 4,
                    ctypes.POINTER(_MIB_TCPROW_OWNER_PID * num_entries),
                ).contents
                for r in rows:
                    if int(r.dwOwningPid) != int(pid):
                        continue
                    laddr = socket.inet_ntoa(struct.pack("<I", r.dwLocalAddr))
                    lport = socket.ntohs(r.dwLocalPort & 0xFFFF)
                    rport = socket.ntohs(r.dwRemotePort & 0xFFFF)
                    if lport <= 0:
                        continue
                    # Any bound/listening socket has remote_port == 0 (even when dwState == 0 or 2)
                    if rport == 0 and laddr in ("127.0.0.1", "0.0.0.0"):
                        if lport not in ports:
                            ports.append(lport)
                    elif laddr == "127.0.0.1" and lport not in established_loopback_ports:
                        established_loopback_ports.append(lport)
    except Exception:
        pass

    # 2. IPv6 table (AF_INET6 = 23, TCP_TABLE_OWNER_PID_ALL = 5)
    try:
        size6 = wintypes.DWORD(0)
        iphlpapi.GetExtendedTcpTable(None, ctypes.byref(size6), False, 23, 5, 0)
        if size6.value > 0:
            buf6 = ctypes.create_string_buffer(size6.value)
            ret6 = iphlpapi.GetExtendedTcpTable(buf6, ctypes.byref(size6), False, 23, 5, 0)
            if ret6 == 0:
                num_entries6 = ctypes.cast(buf6, ctypes.POINTER(wintypes.DWORD)).contents.value
                rows6 = ctypes.cast(
                    ctypes.addressof(buf6) + 4,
                    ctypes.POINTER(_MIB_TCP6ROW_OWNER_PID * num_entries6),
                ).contents
                for r in rows6:
                    if int(r.dwOwningPid) != int(pid):
                        continue
                    lport = socket.ntohs(r.dwLocalPort & 0xFFFF)
                    rport = socket.ntohs(r.dwRemotePort & 0xFFFF)
                    if lport > 0 and rport == 0 and lport not in ports:
                        ports.append(lport)
    except Exception:
        pass

    # Try higher ports first (language_server binds gRPC first, then the HTTP Connect RPC port right above it)
    ports.sort(reverse=True)
    for ep in established_loopback_ports:
        if ep not in ports:
            ports.append(ep)
    return ports


def _discover_ports_via_netstat(pid: int) -> List[int]:
    """Fallback port discovery via `netstat -ano` (includes both LISTENING and local loopback ports)."""
    try:
        completed = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
        ports: List[int] = []
        pid_str = str(pid)
        for line in completed.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[-1] == pid_str:
                local_addr = parts[1]
                state_str = parts[3].upper()
                if "LISTENING" in state_str or local_addr.startswith("127.0.0.1:") or local_addr.startswith("[::1]:"):
                    if ":" in local_addr:
                        port_part = local_addr.rsplit(":", 1)[-1]
                        try:
                            p_int = int(port_part)
                            if p_int not in ports:
                                ports.append(p_int)
                        except ValueError:
                            pass
        return ports
    except Exception:
        return []


def _extract_bridge_candidate_ports(cmdline: str) -> List[int]:
    """
    Extract `--host_bridge_url=http://127.0.0.1:<port>` from `language_server.exe`'s command line
    and return the adjacent ephemeral ports (`bridge_port + 1 .. bridge_port + 12`) where
    `language_server.exe` binds its Connect RPC server.
    """
    if not cmdline:
        return []
    m = re.search(r"--host_bridge_url=https?://127\.0\.0\.1:(\d+)", cmdline)
    if not m:
        return []
    try:
        base_port = int(m.group(1))
        return [base_port + offset for offset in range(1, 13)]
    except ValueError:
        return []


def _discover_ports_for_pid(pid: int, cmdline: str = "") -> List[int]:
    """Combine Win32 `GetExtendedTcpTable`, `netstat`, and `--host_bridge_url` port candidates."""
    combined: List[int] = []
    for p in _discover_ports_via_win32_tcp_table(pid):
        if p not in combined:
            combined.append(p)
    for p in _discover_ports_via_netstat(pid):
        if p not in combined:
            combined.append(p)
    for p in _extract_bridge_candidate_ports(cmdline):
        if p not in combined:
            combined.append(p)
    return combined


def _discover_language_server() -> List[Dict[str, Any]]:
    """
    Find running `language_server.exe` processes on Windows, their PIDs, CSRF tokens,
    and listening loopback TCP ports using filtered WMI + Win32 `GetExtendedTcpTable`.
    """
    ps_script = r"""
    $procs = @(Get-CimInstance Win32_Process -Filter "Name LIKE 'language_server%'" -ErrorAction SilentlyContinue)
    $results = @()
    foreach ($p in $procs) {
        $pid_val = $p.ProcessId
        $cmd = $p.CommandLine
        $token = ""
        if ($cmd -match '--csrf_token[\s=]+([^\s"]+)') {
            $token = $matches[1]
        }
        $results += [PSCustomObject]@{
            pid = $pid_val
            csrf_token = $token
            cmdline = $cmd
        }
    }
    $results | ConvertTo-Json -Compress
    """
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        raw = completed.stdout.strip()
        if not raw:
            return []
        parsed = json.loads(raw)
        items = [parsed] if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])
        results: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            pid_val = item.get("pid")
            if not pid_val:
                continue
            cmdline = str(item.get("cmdline") or "")
            ports = _discover_ports_for_pid(int(pid_val), cmdline=cmdline)
            item["ports"] = ports
            results.append(item)
        return results
    except Exception:
        return []


def get_active_agy_endpoint() -> Optional[Dict[str, Any]]:
    """
    Return the active `(pid, port, csrf_token)` for `language_server.exe`, using the verified
    in-memory/disk cache first before falling back to process discovery.
    """
    _load_disk_cache()
    c_pid = _CACHED_ENDPOINT.get("pid")
    c_port = _CACHED_ENDPOINT.get("port")
    c_token = _CACHED_ENDPOINT.get("csrf_token")
    if c_pid and c_port and c_token:
        return {
            "pid": int(c_pid),
            "port": int(c_port),
            "csrf_token": str(c_token),
            "ports": [int(c_port)],
        }
    servers = _discover_language_server()
    return servers[0] if servers else None


def _query_rpc(port: int, csrf_token: str, method: str, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
    """Send Connect Protocol POST request to local language_server port with proxy bypass."""
    url = f"http://127.0.0.1:{port}/exa.language_server_pb.LanguageServerService/{method}"
    req = urllib.request.Request(
        url,
        data=b"{}",
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Connect-Protocol-Version": "1",
            "X-Codeium-Csrf-Token": csrf_token,
        },
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as resp:
            if resp.status == 200:
                body = resp.read().decode("utf-8", errors="replace")
                return json.loads(body)
    except Exception:
        return None
    return None


def _validate_fraction(val: Any) -> tuple[Optional[float], bool]:
    """Validate a remainingFraction in [0.0, 1.0]. Returns (fraction_or_None, is_malformed)."""
    if val is None:
        return None, False
    try:
        f = float(val)
        import math
        if math.isnan(f) or math.isinf(f) or f < -0.001 or f > 1.001:
            return None, True
        return round(max(0.0, min(1.0, f)), 4), False
    except Exception:
        return None, True


def parse_agy_quota_summary(
    summary_payload: Dict[str, Any],
    status_payload: Optional[Dict[str, Any]] = None,
    now: Optional[datetime.datetime] = None,
) -> Dict[str, Any]:
    """
    Parse `RetrieveUserQuotaSummary` response (which contains exact `gemini-5h`, `gemini-weekly`,
    `3p-5h`, and `3p-weekly` buckets matching the AGY Harness UI and `agy -p /usage`),
    and enrich with `GetUserStatus` plan metadata if provided.
    """
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)

    groups = summary_payload.get("response", {}).get("groups", []) if isinstance(summary_payload, dict) else []
    buckets_map: Dict[str, Dict[str, Any]] = {}
    has_malformed = False

    for grp in (groups if isinstance(groups, list) else []):
        if not isinstance(grp, dict):
            has_malformed = True
            continue
        g_name = str(grp.get("displayName", "")).lower()
        for b in (grp.get("buckets") or []):
            if not isinstance(b, dict):
                has_malformed = True
                continue
            b_id = str(b.get("bucketId", "")).lower()
            win = str(b.get("window", "")).lower()
            rem, malformed = _validate_fraction(b.get("remainingFraction"))
            if malformed:
                has_malformed = True
            r_iso = b.get("resetTime")
            r_sec = _parse_iso_reset_seconds(r_iso, now=now)
            entry = {
                "bucket_id": b_id,
                "remaining_fraction": rem,
                "used_percent": round((1.0 - rem) * 100.0, 1) if rem is not None else None,
                "reset_in_sec": r_sec,
                "resets_at": r_iso,
                "description": b.get("description", ""),
            }
            if b_id:
                buckets_map[b_id] = entry
            if "gemini" in g_name and win == "5h":
                buckets_map.setdefault("gemini-5h", entry)
            elif "gemini" in g_name and win == "weekly":
                buckets_map.setdefault("gemini-weekly", entry)
            elif ("claude" in g_name or "gpt" in g_name) and win == "5h":
                buckets_map.setdefault("3p-5h", entry)
            elif ("claude" in g_name or "gpt" in g_name) and win == "weekly":
                buckets_map.setdefault("3p-weekly", entry)

    gem_5h = dict(
        buckets_map.get(
            "gemini-5h",
            {"remaining_fraction": None, "used_percent": None, "reset_in_sec": 3600, "resets_at": None, "description": ""},
        )
    )
    gem_wk = dict(buckets_map.get("gemini-weekly", gem_5h))
    tp_5h = dict(
        buckets_map.get(
            "3p-5h",
            {"remaining_fraction": None, "used_percent": None, "reset_in_sec": 3600, "resets_at": None, "description": ""},
        )
    )
    tp_wk = dict(buckets_map.get("3p-weekly", tp_5h))

    gem_5h_rem = gem_5h["remaining_fraction"]
    gem_wk_rem = gem_wk["remaining_fraction"]
    tp_5h_rem = tp_5h["remaining_fraction"]
    tp_wk_rem = tp_wk["remaining_fraction"]

    gem_eff_rem = (
        round(min(float(gem_5h_rem), float(gem_wk_rem)), 2)
        if (gem_5h_rem is not None and gem_wk_rem is not None)
        else None
    )
    tp_eff_rem = (
        round(min(float(tp_5h_rem), float(tp_wk_rem)), 2)
        if (tp_5h_rem is not None and tp_wk_rem is not None)
        else None
    )

    if tp_wk_rem is not None and float(tp_wk_rem) <= 0.0 and "weekly limit" in str(tp_5h.get("description", "")).lower():
        tp_5h["blocked_by_weekly"] = True
    if gem_wk_rem is not None and float(gem_wk_rem) <= 0.0 and "weekly limit" in str(gem_5h.get("description", "")).lower():
        gem_5h["blocked_by_weekly"] = True

    if has_malformed:
        status = "unknown"
        confidence = "SUSPECT"
    elif gem_eff_rem is None and tp_eff_rem is None:
        status = "unknown"
        confidence = "UNKNOWN"
    elif (gem_eff_rem or 0.0) <= 0.05 and (tp_eff_rem or 0.0) <= 0.05:
        status = "exhausted"
        confidence = "VERIFIED"
    elif (gem_eff_rem if gem_eff_rem is not None else 1.0) < 0.20:
        status = "warning"
        confidence = "VERIFIED"
    else:
        status = "healthy"
        confidence = "VERIFIED"

    reset_in_sec = (
        gem_wk["reset_in_sec"]
        if (gem_wk_rem is not None and gem_5h_rem is not None and float(gem_wk_rem) <= 0.05 and float(gem_5h_rem) > 0.05)
        else gem_5h["reset_in_sec"]
    )

    plan_name = "Pro"
    prompt_credits = None
    flow_credits = None
    account_label = "Antigravity Local RPC"
    if status_payload and "userStatus" in status_payload:
        us = status_payload["userStatus"]
        ps = us.get("planStatus", {})
        plan_name = ps.get("planInfo", {}).get("planName", "Pro")
        prompt_credits = ps.get("availablePromptCredits")
        flow_credits = ps.get("availableFlowCredits")
        account_label = us.get("email") or us.get("name") or f"AGY {plan_name}"

    return {
        "status": status,
        "confidence": confidence,
        "freshness_age_sec": 0,
        "account": account_label,
        "gemini_flash": gem_eff_rem,
        "gemini_pro": gem_eff_rem,
        "gemini_argon": gem_eff_rem,
        "claude_sonnet": tp_eff_rem,
        "claude_opus": tp_eff_rem,
        "gpt_oss": tp_eff_rem,
        "reset_in_sec": int(reset_in_sec),
        "shared_pools": {
            "agy_gemini": {
                "pool_id": "agy_gemini",
                "models": ["gemini-4-hydrogen-low", "gemini-4-hydrogen-medium", "gemini-4-hydrogen-high", "gemini-3.8-flash-low", "gemini-3.8-flash-medium", "gemini-3.8-flash-high", "gemini-4-argon-low", "gemini-4-argon-medium", "gemini-4-argon-high"],
                "five_hour": gem_5h,
                "weekly": gem_wk,
                "effective_remaining": gem_eff_rem,
                "confidence": confidence,
            },
            "agy_claude_gpt": {
                "pool_id": "agy_claude_gpt",
                "models": ["claude-sonnet-5-5-medium", "claude-opus-5-5-medium", "claude-opus-5-5-high", "gpt-oss-120b-medium"],
                "five_hour": tp_5h,
                "weekly": tp_wk,
                "effective_remaining": tp_eff_rem,
                "confidence": confidence,
            },
        },
        "windows": {
            "five_hour": gem_5h,
            "weekly": gem_wk,
            "gemini_5h": gem_5h,
            "gemini_weekly": gem_wk,
            "claude_gpt_5h": tp_5h,
            "claude_gpt_weekly": tp_wk,
            "gemini_pool": gem_5h,
            "claude_gpt_pool": tp_5h,
        },
        "plan": plan_name,
        "available_prompt_credits": prompt_credits,
        "available_flow_credits": flow_credits,
        "models": [
            {
                "model_id": "gemini-4-hydrogen-medium",
                "label": "Gemini 4 Hydrogen (Medium)",
                "pool_id": "agy_gemini",
                "remaining_fraction": gem_eff_rem,
                "five_hour_rem": round(float(gem_5h_rem or 0.0), 2),
                "weekly_rem": round(float(gem_wk_rem or 0.0), 2),
            },
            {
                "model_id": "gemini-3.8-flash-high",
                "label": "Gemini 3.8 Flash (High)",
                "pool_id": "agy_gemini",
                "remaining_fraction": gem_eff_rem,
                "five_hour_rem": round(float(gem_5h_rem or 0.0), 2),
                "weekly_rem": round(float(gem_wk_rem or 0.0), 2),
            },
            {
                "model_id": "gemini-4-argon-medium",
                "label": "Gemini 4 Argon / Pro (Medium)",
                "pool_id": "agy_gemini",
                "remaining_fraction": gem_eff_rem,
                "five_hour_rem": round(float(gem_5h_rem or 0.0), 2),
                "weekly_rem": round(float(gem_wk_rem or 0.0), 2),
            },
            {
                "model_id": "claude-sonnet-5-5-medium",
                "label": "Claude Sonnet 5.5 (AGY)",
                "pool_id": "agy_claude_gpt",
                "remaining_fraction": tp_eff_rem,
                "five_hour_rem": round(float(tp_5h_rem or 0.0), 2),
                "weekly_rem": round(float(tp_wk_rem or 0.0), 2),
            },
            {
                "model_id": "claude-opus-5-5-medium",
                "label": "Claude Opus 5.5 (AGY)",
                "pool_id": "agy_claude_gpt",
                "remaining_fraction": tp_eff_rem,
                "five_hour_rem": round(float(tp_5h_rem or 0.0), 2),
                "weekly_rem": round(float(tp_wk_rem or 0.0), 2),
            },
            {
                "model_id": "gpt-oss-120b-medium",
                "label": "GPT-OSS 120B (AGY)",
                "pool_id": "agy_claude_gpt",
                "remaining_fraction": tp_eff_rem,
                "five_hour_rem": round(float(tp_5h_rem or 0.0), 2),
                "weekly_rem": round(float(tp_wk_rem or 0.0), 2),
            },
        ],
        "observed_at": now.isoformat(),
    }


def parse_agy_user_status(
    payload: Dict[str, Any],
    now: Optional[datetime.datetime] = None,
) -> Dict[str, Any]:
    """
    Fallback parser for raw `GetUserStatus` response when `RetrieveUserQuotaSummary` is not used.
    """
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)

    user_status = payload.get("userStatus", {})
    cascade_data = user_status.get("cascadeModelConfigData", {})
    configs = cascade_data.get("clientModelConfigs", [])
    plan_status = user_status.get("planStatus", {})
    plan_info = plan_status.get("planInfo", {})

    models_detail: List[Dict[str, Any]] = []
    gemini_flash_frac: Optional[float] = None
    gemini_pro_frac: Optional[float] = None
    claude_sonnet_frac: Optional[float] = None
    claude_opus_frac: Optional[float] = None

    reset_times_sec: List[int] = []
    gemini_reset_iso: Optional[str] = None
    claude_pool_reset_iso: Optional[str] = None
    has_malformed = False

    for cfg in configs:
        model_id = str(cfg.get("modelId", "")).lower()
        label = str(cfg.get("label", model_id))
        quota_info = cfg.get("quotaInfo", {})
        rem_raw = quota_info.get("remainingFraction")
        rem, malformed = _validate_fraction(rem_raw)
        if malformed:
            has_malformed = True
        if rem is None:
            continue
        reset_str = quota_info.get("resetTime")
        reset_sec = _parse_iso_reset_seconds(reset_str, now=now)
        if reset_str:
            reset_times_sec.append(reset_sec)

        models_detail.append(
            {
                "model_id": model_id,
                "label": label,
                "remaining_fraction": rem,
                "used_percent": round((1.0 - rem) * 100.0, 1),
                "reset_time": reset_str,
                "reset_in_sec": reset_sec,
            }
        )

        if "gemini" in model_id and ("flash" in model_id or "hydrogen" in model_id):
            gemini_flash_frac = rem if gemini_flash_frac is None else min(gemini_flash_frac, rem)
            gemini_reset_iso = gemini_reset_iso or reset_str
        elif "gemini" in model_id and ("pro" in model_id or "argon" in model_id or "carbon" in model_id):
            gemini_pro_frac = rem if gemini_pro_frac is None else min(gemini_pro_frac, rem)
            gemini_reset_iso = gemini_reset_iso or reset_str
        elif "claude" in model_id and "sonnet" in model_id:
            claude_sonnet_frac = rem if claude_sonnet_frac is None else min(claude_sonnet_frac, rem)
            claude_pool_reset_iso = claude_pool_reset_iso or reset_str
        elif "claude" in model_id and "opus" in model_id:
            claude_opus_frac = rem if claude_opus_frac is None else min(claude_opus_frac, rem)
            claude_pool_reset_iso = claude_pool_reset_iso or reset_str

    if gemini_flash_frac is None and gemini_pro_frac is not None:
        gemini_flash_frac = gemini_pro_frac
    if gemini_pro_frac is None and gemini_flash_frac is not None:
        gemini_pro_frac = gemini_flash_frac

    if has_malformed:
        return {
            "status": "unknown",
            "confidence": "SUSPECT",
            "freshness_age_sec": 0,
            "account": f"AGY {plan_info.get('planName', 'Pro')}",
            "gemini_flash": None,
            "gemini_pro": None,
            "gemini_argon": None,
            "claude_sonnet": None,
            "claude_opus": None,
            "gpt_oss": None,
            "reset_in_sec": 3600,
            "windows": {},
            "plan": plan_info.get("planName", "Pro"),
            "models": [],
            "observed_at": now.isoformat(),
        }

    g_flash = round(gemini_flash_frac if gemini_flash_frac is not None else 0.0, 2)
    g_pro = round(gemini_pro_frac if gemini_pro_frac is not None else 0.0, 2)
    c_sonnet = round(claude_sonnet_frac if claude_sonnet_frac is not None else 0.0, 2)
    c_opus = round(claude_opus_frac if claude_opus_frac is not None else c_sonnet, 2)

    primary_rem = min(g_flash, g_pro) if models_detail else 0.0
    if not models_detail:
        status = "offline"
        confidence = "UNAVAILABLE"
    elif primary_rem <= 0.05:
        status = "exhausted"
        confidence = "VERIFIED"
    elif primary_rem < 0.20:
        status = "warning"
        confidence = "VERIFIED"
    else:
        status = "healthy"
        confidence = "VERIFIED"

    reset_in_sec = min(reset_times_sec) if reset_times_sec else 3600
    gemini_reset_sec = _parse_iso_reset_seconds(gemini_reset_iso, now=now) if gemini_reset_iso else reset_in_sec
    claude_reset_sec = _parse_iso_reset_seconds(claude_pool_reset_iso, now=now) if claude_pool_reset_iso else reset_in_sec

    gem_win = {
        "remaining_fraction": primary_rem,
        "used_percent": round((1.0 - primary_rem) * 100.0, 1),
        "reset_in_sec": gemini_reset_sec,
        "resets_at": gemini_reset_iso,
    }
    claude_win = {
        "remaining_fraction": c_sonnet,
        "used_percent": round((1.0 - c_sonnet) * 100.0, 1),
        "reset_in_sec": claude_reset_sec,
        "resets_at": claude_pool_reset_iso,
    }

    return {
        "status": status,
        "confidence": confidence,
        "freshness_age_sec": 0,
        "account": f"AGY {plan_info.get('planName', 'Pro')}",
        "gemini_flash": g_flash,
        "gemini_pro": g_pro,
        "gemini_argon": g_pro,
        "claude_sonnet": c_sonnet,
        "claude_opus": c_opus,
        "gpt_oss": c_sonnet,
        "reset_in_sec": reset_in_sec,
        "windows": {
            "five_hour": gem_win,
            "weekly": gem_win,
            "gemini_5h": gem_win,
            "gemini_weekly": gem_win,
            "claude_gpt_5h": claude_win,
            "claude_gpt_weekly": claude_win,
            "gemini_pool": {
                "remaining_fraction": g_flash,
                "used_percent": round((1.0 - g_flash) * 100.0, 1),
                "reset_in_sec": gemini_reset_sec,
                "resets_at": gemini_reset_iso,
            },
            "claude_gpt_pool": claude_win,
        },
        "plan": plan_info.get("planName", "Pro"),
        "available_prompt_credits": plan_status.get("availablePromptCredits"),
        "available_flow_credits": plan_status.get("availableFlowCredits"),
        "models": models_detail,
        "observed_at": now.isoformat(),
    }


def _try_query_endpoint(pid: int, port_int: int, token: str, now: datetime.datetime) -> Optional[Dict[str, Any]]:
    """Try querying `RetrieveUserQuotaSummary` and `GetUserStatus` on a known `(pid, port, token)`."""
    summary_payload = _query_rpc(port_int, token, "RetrieveUserQuotaSummary")
    status_payload = _query_rpc(port_int, token, "GetUserStatus")

    if summary_payload and "response" in summary_payload:
        result = parse_agy_quota_summary(summary_payload, status_payload=status_payload, now=now)
        result["source"] = f"RetrieveUserQuotaSummary (PID {pid}, Port {port_int})"
        result["pid"] = pid
        result["port"] = port_int
        return result

    if status_payload and "userStatus" in status_payload:
        result = parse_agy_user_status(status_payload, now=now)
        result["source"] = f"GetUserStatus (PID {pid}, Port {port_int})"
        result["pid"] = pid
        result["port"] = port_int
        return result

    return None


def collect_agy_quota(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Discover local `language_server.exe` and query `RetrieveUserQuotaSummary` + `GetUserStatus`.
    Uses in-memory + disk-cached endpoint first (~2ms fast path), then Win32 `GetExtendedTcpTable`
    port discovery on cached PID, then full WMI + Win32 TCP table discovery if the process restarted.
    """
    global _LAST_GOOD_RESULT, _LAST_GOOD_TS
    now = datetime.datetime.now(datetime.timezone.utc)
    _load_disk_cache()

    # 1. Fast path: try cached (pid, port, csrf_token) first without spawning any subprocess
    c_pid = _CACHED_ENDPOINT.get("pid")
    c_port = _CACHED_ENDPOINT.get("port")
    c_token = _CACHED_ENDPOINT.get("csrf_token")
    if c_pid and c_port and c_token:
        fast_res = _try_query_endpoint(int(c_pid), int(c_port), str(c_token), now)
        if fast_res is not None:
            _LAST_GOOD_RESULT = fast_res
            _LAST_GOOD_TS = time.time()
            _save_disk_cache()
            return fast_res

        # 1b. If cached PID is still alive and only its port shifted, check Win32 TCP table for c_pid (<0.5ms)
        for cand_port in _discover_ports_via_win32_tcp_table(int(c_pid)):
            if int(cand_port) == int(c_port):
                continue
            res_shift = _try_query_endpoint(int(c_pid), int(cand_port), str(c_token), now)
            if res_shift is not None:
                _CACHED_ENDPOINT["port"] = int(cand_port)
                _LAST_GOOD_RESULT = res_shift
                _LAST_GOOD_TS = time.time()
                _save_disk_cache()
                return res_shift

    # 2. Slow path: discover language_server.exe via filtered WMI + Win32 GetExtendedTcpTable
    servers = _discover_language_server()
    for srv in servers:
        pid = srv.get("pid")
        token = srv.get("csrf_token", "")
        ports = srv.get("ports", [])
        if not pid or not token or not ports:
            continue
        for port in ports:
            try:
                port_int = int(port)
            except (ValueError, TypeError):
                continue

            res = _try_query_endpoint(int(pid), port_int, str(token), now)
            if res is not None:
                _CACHED_ENDPOINT["pid"] = int(pid)
                _CACHED_ENDPOINT["port"] = port_int
                _CACHED_ENDPOINT["csrf_token"] = str(token)
                _LAST_GOOD_RESULT = res
                _LAST_GOOD_TS = time.time()
                _save_disk_cache()
                return res

    # 3. If discovery momentarily failed due to system load, return recent last-known-good (<600s)
    age_sec = int(time.time() - _LAST_GOOD_TS) if _LAST_GOOD_TS > 0 else 999999
    if not force_refresh and _LAST_GOOD_RESULT is not None and age_sec <= _CACHE_TTL_SEC:
        cached_copy = dict(_LAST_GOOD_RESULT)
        cached_copy["cached"] = True
        cached_copy["confidence"] = "STALE"
        cached_copy["freshness_age_sec"] = age_sec
        cached_copy["source"] = f"{_LAST_GOOD_RESULT.get('source', 'AGY')} [STALE {age_sec}s]"
        return cached_copy

    # 4. Truly unreachable: return UNAVAILABLE / UNKNOWN (never fake 0.0 / 100% used!)
    return {
        "status": "unknown",
        "confidence": "UNAVAILABLE",
        "freshness_age_sec": age_sec,
        "account": "Unavailable",
        "gemini_flash": None,
        "gemini_pro": None,
        "gemini_argon": None,
        "claude_sonnet": None,
        "claude_opus": None,
        "gpt_oss": None,
        "reset_in_sec": 3600,
        "windows": {
            "five_hour": {"remaining_fraction": None, "used_percent": None, "reset_in_sec": 3600, "resets_at": None},
            "weekly": {"remaining_fraction": None, "used_percent": None, "reset_in_sec": 3600, "resets_at": None},
            "gemini_5h": {"remaining_fraction": None, "used_percent": None, "reset_in_sec": 3600, "resets_at": None},
            "gemini_weekly": {"remaining_fraction": None, "used_percent": None, "reset_in_sec": 3600, "resets_at": None},
            "claude_gpt_5h": {"remaining_fraction": None, "used_percent": None, "reset_in_sec": 3600, "resets_at": None},
            "claude_gpt_weekly": {"remaining_fraction": None, "used_percent": None, "reset_in_sec": 3600, "resets_at": None},
        },
        "plan": "Unknown",
        "models": [],
        "source": "language_server.exe not reachable (UNAVAILABLE)",
        "observed_at": now.isoformat(),
    }


if __name__ == "__main__":
    print(json.dumps(collect_agy_quota(), indent=2))
