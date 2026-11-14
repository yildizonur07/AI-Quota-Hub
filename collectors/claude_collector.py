"""
Claude Code Zero-Token Official Telemetry Collector.

Primary Source:
- Queries Claude's official OAuth usage endpoint (`GET https://api.anthropic.com/api/oauth/usage`)
  with `anthropic-beta: oauth-2025-04-20`.
- Resolves the active OAuth bearer token from BOTH:
  1. `%APPDATA%\\Claude\\config.json` (`oauth:tokenCacheV2` / `oauth:tokenCache`, decrypted via
     Windows DPAPI `CryptUnprotectData` + AES-256-GCM using `%APPDATA%\\Claude\\Local State`),
  2. `~/.claude/.credentials.json` (`claudeAiOauth.accessToken`).
- Reads exact `five_hour` (`utilization`, `resets_at`) and `seven_day` (`utilization`, `resets_at`)
  windows, plus optional `seven_day_opus` / `seven_day_sonnet` model-scoped caps if present.

Fallback Source:
- Only if completely offline, falls back to `plan-usage-history.json` (`confidence = "STALE"`) or
  returns `confidence = "UNAVAILABLE"` / `"UNKNOWN"` without fabricating 0% or 100% usage.
"""

from __future__ import annotations

import base64
import ctypes
import ctypes.wintypes
import datetime
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_CLAUDE_DIR = Path.home() / ".claude"
DEFAULT_CLAUDE_DESKTOP_DIR = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))) / "Claude"

FIVE_HOUR_SEC = 5 * 3600
SEVEN_DAY_SEC = 7 * 24 * 3600
FIVE_HOUR_COST_CEILING_USD = 35.0
WEEKLY_COST_CEILING_USD = 180.0

# 10-second in-memory cache for /api/oauth/usage
_OAUTH_USAGE_CACHE: Dict[str, Any] = {
    "timestamp": 0.0,
    "payload": None,
}


class _DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", ctypes.wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def _dpapi_decrypt(encrypted_bytes: bytes) -> Optional[bytes]:
    """Decrypt bytes protected with Windows DPAPI (CurrentUser scope)."""
    if os.name != "nt" or not encrypted_bytes:
        return None
    try:
        buf = ctypes.create_string_buffer(encrypted_bytes, len(encrypted_bytes))
        blob_in = _DATA_BLOB(len(encrypted_bytes), buf)
        blob_out = _DATA_BLOB()
        ok = ctypes.windll.crypt32.CryptUnprotectData(
            ctypes.byref(blob_in),
            None,
            None,
            None,
            None,
            0,
            ctypes.byref(blob_out),
        )
        if not ok:
            return None
        try:
            return ctypes.string_at(blob_out.pbData, blob_out.cbData)
        finally:
            ctypes.windll.kernel32.LocalFree(blob_out.pbData)
    except Exception:
        return None


def _extract_desktop_token_cache_candidates(desktop_dir: Path) -> List[Tuple[str, str]]:
    """
    Decrypt Electron `safeStorage` `oauth:tokenCacheV2` / `oauth:tokenCache` from
    `%APPDATA%\\Claude\\config.json` using the DPAPI master key in `Local State`.
    Returns a list of `(access_token, subscription_type)` pairs. Never logs tokens.
    """
    config_path = desktop_dir / "config.json"
    local_state_path = desktop_dir / "Local State"
    if not config_path.exists() or not local_state_path.exists():
        return []

    try:
        ls = json.loads(local_state_path.read_text(encoding="utf-8", errors="replace"))
        enc_key_b64 = ls.get("os_crypt", {}).get("encrypted_key")
        if not enc_key_b64:
            return []
        enc_key = base64.b64decode(enc_key_b64)
        if enc_key.startswith(b"DPAPI"):
            enc_key = enc_key[5:]
        master_key = _dpapi_decrypt(enc_key)
        if not master_key:
            return []

        cfg = json.loads(config_path.read_text(encoding="utf-8", errors="replace"))
        candidates: List[Tuple[float, str, str]] = []

        for cache_key in ("oauth:tokenCacheV2", "oauth:tokenCache"):
            b64_val = cfg.get(cache_key)
            if not b64_val or not isinstance(b64_val, str):
                continue
            try:
                raw = base64.b64decode(b64_val)
                if raw.startswith(b"v10") or raw.startswith(b"v11"):
                    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

                    nonce = raw[3:15]
                    ct = raw[15:]
                    pt = AESGCM(master_key).decrypt(nonce, ct, None).decode("utf-8", errors="replace")
                else:
                    dec = _dpapi_decrypt(raw)
                    if not dec:
                        continue
                    pt = dec.decode("utf-8", errors="replace")

                cache_obj = json.loads(pt)
                if not isinstance(cache_obj, dict):
                    continue
                for entry_key, entry_val in cache_obj.items():
                    if not isinstance(entry_val, dict):
                        continue
                    tok = entry_val.get("token") or entry_val.get("accessToken") or entry_val.get("access_token")
                    if not tok or not isinstance(tok, str):
                        continue
                    sub_type = str(entry_val.get("subscriptionType") or "pro")
                    exp = float(entry_val.get("expiresAt") or 0.0)
                    bonus = 1e12 if "user:inference" in str(entry_key) else 0.0
                    candidates.append((exp + bonus, tok, sub_type))
            except Exception:
                continue

        candidates.sort(key=lambda x: x[0], reverse=True)
        seen = set()
        result: List[Tuple[str, str]] = []
        for _, tok, sub_type in candidates:
            if tok not in seen:
                seen.add(tok)
                result.append((tok, sub_type))
        return result
    except Exception:
        return []


def _extract_cli_credentials_candidates(claude_dir: Path) -> List[Tuple[str, str]]:
    """Extract OAuth token from `~/.claude/.credentials.json` if present."""
    cred_file = claude_dir / ".credentials.json"
    if not cred_file.exists():
        return []
    try:
        cred_data = json.loads(cred_file.read_text(encoding="utf-8", errors="replace"))
        oauth_obj = cred_data.get("claudeAiOauth") or {}
        token = oauth_obj.get("accessToken")
        if token and isinstance(token, str) and len(token) > 10:
            sub_type = str(oauth_obj.get("subscriptionType") or "pro")
            return [(token, sub_type)]
    except Exception:
        pass
    return []


def _parse_iso_reset_seconds(reset_time_str: Optional[str], now: datetime.datetime) -> int:
    """Parse ISO-8601 timestamp and return seconds until reset."""
    if not reset_time_str:
        return FIVE_HOUR_SEC
    try:
        cleaned = str(reset_time_str).strip()
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        dt = datetime.datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return max(0, int((dt - now).total_seconds()))
    except Exception:
        return FIVE_HOUR_SEC


def _fetch_claude_oauth_usage(
    claude_dir: Path,
    desktop_dir: Optional[Path] = None,
    cache_ttl: float = 10.0,
    force_refresh: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Query `https://api.anthropic.com/api/oauth/usage` using candidate OAuth tokens
    from `~/.claude/.credentials.json` and `%APPDATA%\\Claude\\config.json` (`oauth:tokenCacheV2`).
    Never logs or exposes tokens.
    """
    now_mono = time.time()
    if (
        not force_refresh
        and _OAUTH_USAGE_CACHE["payload"] is not None
        and (now_mono - float(_OAUTH_USAGE_CACHE["timestamp"])) < cache_ttl
    ):
        return _OAUTH_USAGE_CACHE["payload"]

    d_dir = desktop_dir or DEFAULT_CLAUDE_DESKTOP_DIR
    candidates: List[Tuple[str, str]] = []
    candidates.extend(_extract_cli_credentials_candidates(claude_dir))
    candidates.extend(_extract_desktop_token_cache_candidates(d_dir))

    for token, sub_type in candidates:
        try:
            req = urllib.request.Request(
                "https://api.anthropic.com/api/oauth/usage",
                headers={
                    "Authorization": f"Bearer {token}",
                    "anthropic-beta": "oauth-2025-04-20",
                    "Accept": "application/json",
                    "User-Agent": "claude-code/2.1.295",
                },
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                if resp.status == 200:
                    payload = json.loads(resp.read().decode("utf-8", errors="replace"))
                    if isinstance(payload, dict) and ("five_hour" in payload or "seven_day" in payload):
                        payload["_subscription_type"] = sub_type
                        _OAUTH_USAGE_CACHE["timestamp"] = now_mono
                        _OAUTH_USAGE_CACHE["payload"] = payload
                        return payload
        except Exception:
            continue
    return None


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


def _scan_claude_session_logs(claude_dir: Path, now_ts: float) -> Dict[str, Any]:
    """Scan recent `.jsonl` files under `~/.claude/projects` for cost accounting & active sessions."""
    projects_dir = claude_dir / "projects"
    seven_days_ago = now_ts - SEVEN_DAY_SEC
    five_hours_ago = now_ts - FIVE_HOUR_SEC

    cost_5h_usd = 0.0
    cost_7d_usd = 0.0
    active_sessions_count = 0

    sessions_dir = claude_dir / "sessions"
    if sessions_dir.exists():
        for sfile in sessions_dir.glob("*.json"):
            try:
                if now_ts - sfile.stat().st_mtime < FIVE_HOUR_SEC:
                    active_sessions_count += 1
            except Exception:
                pass

    if not projects_dir.exists():
        return {
            "cost_5h_usd": 0.0,
            "cost_7d_usd": 0.0,
            "active_sessions_count": active_sessions_count,
        }

    jsonl_files: List[Tuple[float, Path]] = []
    for p in projects_dir.rglob("*.jsonl"):
        try:
            mtime = p.stat().st_mtime
            if mtime >= seven_days_ago:
                jsonl_files.append((mtime, p))
        except Exception:
            continue

    jsonl_files.sort(key=lambda x: x[0], reverse=True)
    for mtime, jpath in jsonl_files[:20]:
        tail_lines = _read_tail_lines(jpath, max_bytes=131072)
        for line in reversed(tail_lines):
            if '"type":"cost-state"' in line:
                try:
                    obj = json.loads(line)
                    if obj.get("type") == "cost-state":
                        c_usd = float(obj.get("totalCostUSD", 0.0))
                        cost_7d_usd += c_usd
                        if mtime >= five_hours_ago:
                            cost_5h_usd += c_usd
                        break
                except Exception:
                    pass

    return {
        "cost_5h_usd": round(cost_5h_usd, 4),
        "cost_7d_usd": round(cost_7d_usd, 4),
        "active_sessions_count": active_sessions_count,
    }


def _parse_plan_usage_history(history_file: Path, now_ts: float) -> Optional[Dict[str, Any]]:
    if not history_file.exists():
        return None
    try:
        raw = json.loads(history_file.read_text(encoding="utf-8", errors="replace"))
        entries = (
            raw
            if isinstance(raw, list)
            else raw.get("samples", raw.get("history", raw.get("entries", [])))
        )
        if not isinstance(entries, list) or not entries:
            return None
        last = entries[-1]
        t_ms = float(last.get("t", 0))
        if t_ms <= 0:
            return None
        t_sec = t_ms / 1000.0
        u = last.get("u", {})
        fh_used = float(u.get("fh", 0.0))
        sd_used = float(u.get("sd", 0.0))
        age_sec = max(0.0, now_ts - t_sec)
        fh_reset_in = 0 if age_sec >= FIVE_HOUR_SEC else int(max(0.0, FIVE_HOUR_SEC - age_sec))
        sd_reset_in = 0 if age_sec >= SEVEN_DAY_SEC else int(max(0.0, SEVEN_DAY_SEC - age_sec))
        return {
            "fh_used_percent": min(100.0, max(0.0, fh_used)),
            "sd_used_percent": min(100.0, max(0.0, sd_used)),
            "fh_reset_in_sec": fh_reset_in,
            "sd_reset_in_sec": sd_reset_in,
            "age_sec": int(age_sec),
            "post_reset_unverified": age_sec >= FIVE_HOUR_SEC,
        }
    except Exception:
        return None


def _validate_pct(val: Any) -> Tuple[Optional[float], bool]:
    """Validate a percentage in [0.0, 100.0]. Returns (pct_or_None, is_malformed)."""
    if val is None:
        return None, False
    try:
        import math
        f = float(val)
        if math.isnan(f) or math.isinf(f) or f < -0.1 or f > 100.1:
            return None, True
        return round(max(0.0, min(100.0, f)), 2), False
    except Exception:
        return None, True


def parse_claude_oauth_usage_payload(
    oauth_payload: Dict[str, Any],
    session_info: Optional[Dict[str, Any]] = None,
    now: Optional[datetime.datetime] = None,
) -> Dict[str, Any]:
    """
    Parse official `GET /api/oauth/usage` JSON into normalized Quota Hub schema.
    Validates that a HTTP 200 response actually contains valid numeric utilization percentages.
    """
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)
    if session_info is None:
        session_info = {"cost_5h_usd": 0.0, "cost_7d_usd": 0.0, "active_sessions_count": 0}

    if not isinstance(oauth_payload, dict):
        oauth_payload = {}

    five_h = oauth_payload.get("five_hour") if isinstance(oauth_payload.get("five_hour"), dict) else {}
    seven_d = oauth_payload.get("seven_day") if isinstance(oauth_payload.get("seven_day"), dict) else {}
    seven_d_opus = oauth_payload.get("seven_day_opus")
    seven_d_sonnet = oauth_payload.get("seven_day_sonnet")

    fh_used_pct, fh_malformed = _validate_pct(five_h.get("utilization"))
    sd_used_pct, sd_malformed = _validate_pct(seven_d.get("utilization"))
    has_malformed = fh_malformed or sd_malformed

    fh_resets_at = five_h.get("resets_at")
    fh_window_inactive = (fh_resets_at is None and fh_used_pct == 0.0)
    fh_reset_sec = _parse_iso_reset_seconds(fh_resets_at, now)

    sd_resets_at = seven_d.get("resets_at")
    sd_reset_sec = _parse_iso_reset_seconds(sd_resets_at, now)

    # Check if resets_at is already in the past on a live response (> 60s expired)
    post_reset_unverified = False
    for r_iso in (fh_resets_at, sd_resets_at):
        if r_iso:
            try:
                c_str = str(r_iso).strip()
                if c_str.endswith("Z"):
                    c_str = c_str[:-1] + "+00:00"
                dt = datetime.datetime.fromisoformat(c_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.timezone.utc)
                if (dt - now).total_seconds() < -60:
                    post_reset_unverified = True
            except Exception:
                has_malformed = True

    if has_malformed or (fh_used_pct is None and sd_used_pct is None):
        confidence = "SUSPECT" if has_malformed else "UNKNOWN"
        return {
            "status": "unknown",
            "confidence": confidence,
            "freshness_age_sec": 0,
            "account": f"Claude Code ({oauth_payload.get('_subscription_type', 'pro')})",
            "shared_pool": True,
            "sonnet": None,
            "opus": None,
            "haiku": None,
            "reset_in_sec": int(sd_reset_sec),
            "windows": {
                "five_hour": {"remaining_fraction": None, "used_percent": None, "reset_in_sec": int(fh_reset_sec), "resets_at": fh_resets_at},
                "weekly": {"remaining_fraction": None, "used_percent": None, "reset_in_sec": int(sd_reset_sec), "resets_at": sd_resets_at},
            },
            "accounting": {
                "cost_5h_usd": session_info["cost_5h_usd"],
                "cost_7d_usd": session_info["cost_7d_usd"],
                "active_sessions": session_info["active_sessions_count"],
                "five_hour_ceiling_usd": FIVE_HOUR_COST_CEILING_USD,
                "weekly_ceiling_usd": WEEKLY_COST_CEILING_USD,
            },
            "plan": str(oauth_payload.get("_subscription_type", "pro")),
            "models": [],
            "source": f"api.anthropic.com/api/oauth/usage ({confidence})",
            "observed_at": now.isoformat(),
        }

    # If one window is missing while the other is valid, mark SUSPECT and use the valid one conservatively
    if fh_used_pct is None or sd_used_pct is None:
        confidence = "SUSPECT"
        fh_used_pct = fh_used_pct if fh_used_pct is not None else sd_used_pct
        sd_used_pct = sd_used_pct if sd_used_pct is not None else fh_used_pct
    elif post_reset_unverified:
        confidence = "STALE"
    else:
        confidence = "VERIFIED"

    fh_rem = round(max(0.0, min(1.0, 1.0 - (float(fh_used_pct) / 100.0))), 4)
    sd_rem = round(max(0.0, min(1.0, 1.0 - (float(sd_used_pct) / 100.0))), 4)

    # Effective remaining fraction across the unified 5-hour and 7-day windows
    unified_rem = round(min(fh_rem, sd_rem), 2)

    if isinstance(seven_d_opus, dict) and seven_d_opus.get("utilization") is not None:
        op_pct, op_mal = _validate_pct(seven_d_opus.get("utilization"))
        if op_pct is not None and not op_mal:
            opus_7d_rem = max(0.0, min(1.0, 1.0 - (op_pct / 100.0)))
            opus_rem = round(min(fh_rem, opus_7d_rem), 2)
        else:
            opus_rem = unified_rem
    else:
        opus_rem = unified_rem

    if isinstance(seven_d_sonnet, dict) and seven_d_sonnet.get("utilization") is not None:
        sn_pct, sn_mal = _validate_pct(seven_d_sonnet.get("utilization"))
        if sn_pct is not None and not sn_mal:
            sonnet_7d_rem = max(0.0, min(1.0, 1.0 - (sn_pct / 100.0)))
            sonnet_rem = round(min(fh_rem, sonnet_7d_rem), 2)
        else:
            sonnet_rem = unified_rem
    else:
        sonnet_rem = unified_rem

    haiku_rem = unified_rem

    if unified_rem <= 0.05:
        status = "exhausted"
    elif unified_rem < 0.20:
        status = "warning"
    else:
        status = "healthy"

    # When 5h window is inactive (resets_at is None, 0% used), the active binding reset is the weekly window
    reset_in_sec = sd_reset_sec if (fh_window_inactive or (sd_rem <= 0.05 and fh_rem > 0.05)) else fh_reset_sec

    five_h_obj = {
        "remaining_fraction": fh_rem,
        "used_percent": round(float(fh_used_pct), 1),
        "reset_in_sec": int(fh_reset_sec),
        "resets_at": fh_resets_at,
        "window_inactive": fh_window_inactive,
        "post_reset_unverified": post_reset_unverified,
    }
    weekly_obj = {
        "remaining_fraction": sd_rem,
        "used_percent": round(float(sd_used_pct), 1),
        "reset_in_sec": int(sd_reset_sec),
        "resets_at": sd_resets_at,
        "post_reset_unverified": post_reset_unverified,
    }

    plan_str = str(oauth_payload.get("_subscription_type", "pro"))

    return {
        "status": status,
        "confidence": confidence,
        "freshness_age_sec": 0,
        "account": f"Claude OAuth ({plan_str.upper()})",
        "sonnet": sonnet_rem,
        "opus": opus_rem,
        "haiku": haiku_rem,
        "reset_in_sec": int(reset_in_sec),
        "windows": {
            "five_hour": five_h_obj,
            "weekly": weekly_obj,
        },
        "shared_pools": {
            "claude_shared": {
                "pool_id": "claude_shared",
                "models": ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"],
                "five_hour": five_h_obj,
                "weekly": weekly_obj,
                "effective_remaining": unified_rem,
                "confidence": "VERIFIED",
            }
        },
        "plan": plan_str,
        "shared_pool": (seven_d_opus is None and seven_d_sonnet is None),
        "accounting": {
            "cost_5h_usd": session_info.get("cost_5h_usd", 0.0),
            "cost_7d_usd": session_info.get("cost_7d_usd", 0.0),
            "active_sessions": session_info.get("active_sessions_count", 0),
        },
        "models": [
            {"model_id": "claude-haiku-5-5", "label": "Claude Haiku 5.5 (Low)", "pool_id": "claude_shared", "remaining_fraction": haiku_rem, "tier": "LOW"},
            {"model_id": "claude-sonnet-5-5", "label": "Claude Sonnet 5.5 (Mid)", "pool_id": "claude_shared", "remaining_fraction": sonnet_rem, "tier": "MID"},
            {"model_id": "claude-opus-5-5", "label": "Claude Opus 5.5 (High)", "pool_id": "claude_shared", "remaining_fraction": opus_rem, "tier": "HIGH"},
        ],
        "source": "api.anthropic.com/api/oauth/usage (live)",
        "observed_at": now.isoformat(),
    }


def collect_claude_quota(
    claude_dir: Optional[Path] = None,
    desktop_dir: Optional[Path] = None,
    now: Optional[datetime.datetime] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Collect and normalize Claude Code 5h and 7d quota status.
    Prioritizes official OAuth usage endpoint (`api.anthropic.com/api/oauth/usage`),
    falling back to `plan-usage-history.json` (`STALE`) or `UNAVAILABLE` if offline.
    """
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)
    now_ts = now.timestamp()

    c_dir = claude_dir or DEFAULT_CLAUDE_DIR
    d_dir = desktop_dir or DEFAULT_CLAUDE_DESKTOP_DIR

    session_info = _scan_claude_session_logs(c_dir, now_ts)

    # 1. Primary: Official OAuth usage endpoint (resolves token from CLI or Desktop safeStorage)
    oauth_payload = _fetch_claude_oauth_usage(c_dir, desktop_dir=d_dir, force_refresh=force_refresh)
    if oauth_payload is not None:
        return parse_claude_oauth_usage_payload(oauth_payload, session_info=session_info, now=now)

    # 2. Fallback: local plan-usage-history.json (marked STALE)
    history_info = _parse_plan_usage_history(d_dir / "plan-usage-history.json", now_ts)
    if history_info is not None:
        fh_used_pct = history_info["fh_used_percent"]
        sd_used_pct = history_info["sd_used_percent"]
        reset_5h_sec = history_info["fh_reset_in_sec"]
        reset_7d_sec = history_info["sd_reset_in_sec"]
        fh_rem = round(max(0.0, min(1.0, 1.0 - (fh_used_pct / 100.0))), 4)
        sd_rem = round(max(0.0, min(1.0, 1.0 - (sd_used_pct / 100.0))), 4)
        unified_rem = round(min(fh_rem, sd_rem), 2)

        if unified_rem <= 0.05:
            status = "exhausted"
        elif unified_rem < 0.20:
            status = "warning"
        else:
            status = "healthy"

        reset_in_sec = reset_7d_sec if (sd_rem <= 0.05 and fh_rem > 0.05) else reset_5h_sec
        resets_at_5h = (now + datetime.timedelta(seconds=reset_5h_sec)).isoformat()
        resets_at_7d = (now + datetime.timedelta(seconds=reset_7d_sec)).isoformat()

        return {
            "status": status,
            "confidence": "STALE",
            "freshness_age_sec": history_info["age_sec"],
            "post_reset_unverified": history_info["post_reset_unverified"],
            "account": "Claude (Offline Cache)",
            "sonnet": unified_rem,
            "opus": unified_rem,
            "haiku": unified_rem,
            "reset_in_sec": int(reset_in_sec),
            "windows": {
                "five_hour": {
                    "remaining_fraction": fh_rem,
                    "used_percent": round(fh_used_pct, 1),
                    "reset_in_sec": int(reset_5h_sec),
                    "resets_at": resets_at_5h,
                },
                "weekly": {
                    "remaining_fraction": sd_rem,
                    "used_percent": round(sd_used_pct, 1),
                    "reset_in_sec": int(reset_7d_sec),
                    "resets_at": resets_at_7d,
                },
            },
            "plan": "pro",
            "shared_pool": True,
            "accounting": {
                "cost_5h_usd": session_info["cost_5h_usd"],
                "cost_7d_usd": session_info["cost_7d_usd"],
                "active_sessions": session_info["active_sessions_count"],
            },
            "models": [
                {"model_id": "claude-haiku-5-5", "label": "Claude Haiku 5.5 (Low)", "pool_id": "claude_shared", "remaining_fraction": unified_rem, "tier": "LOW"},
                {"model_id": "claude-sonnet-5-5", "label": "Claude Sonnet 5.5 (Mid)", "pool_id": "claude_shared", "remaining_fraction": unified_rem, "tier": "MID"},
                {"model_id": "claude-opus-5-5", "label": "Claude Opus 5.5 (High)", "pool_id": "claude_shared", "remaining_fraction": unified_rem, "tier": "HIGH"},
            ],
            "source": "plan-usage-history.json [STALE offline fallback]",
            "observed_at": now.isoformat(),
        }

    # 3. Completely unavailable: never fabricate 0% or 100% remaining
    return {
        "status": "unknown",
        "confidence": "UNAVAILABLE",
        "freshness_age_sec": 999999,
        "account": "Unavailable",
        "sonnet": None,
        "opus": None,
        "haiku": None,
        "reset_in_sec": FIVE_HOUR_SEC,
        "windows": {
            "five_hour": {"remaining_fraction": None, "used_percent": None, "reset_in_sec": FIVE_HOUR_SEC, "resets_at": None},
            "weekly": {"remaining_fraction": None, "used_percent": None, "reset_in_sec": SEVEN_DAY_SEC, "resets_at": None},
        },
        "plan": "unknown",
        "shared_pool": True,
        "accounting": {
            "cost_5h_usd": session_info["cost_5h_usd"],
            "cost_7d_usd": session_info["cost_7d_usd"],
            "active_sessions": session_info["active_sessions_count"],
        },
        "models": [],
        "source": "Claude OAuth usage API unreachable (UNAVAILABLE)",
        "observed_at": now.isoformat(),
    }


if __name__ == "__main__":
    print(json.dumps(collect_claude_quota(), indent=2))
