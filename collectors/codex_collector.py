"""
OpenAI Codex Zero-Token Official & Local Telemetry Collector.

Primary Source:
- Queries OpenAI Codex's official backend usage endpoint (`GET https://chatgpt.com/backend-api/wham/usage`
  or `GET https://chatgpt.com/backend-api/codex/usage`) and banked reset entitlements endpoint
  (`GET https://chatgpt.com/backend-api/wham/rate-limit-reset-credits`) using the local OAuth bearer
  token and `ChatGPT-Account-Id` from `~/.codex/auth.json`. Never logs or exposes tokens.
- Reads exact real-time `rate_limit.primary_window` (5h, 18000s) and `rate_limit.secondary_window`
  (weekly, 604800s), plus `limit_reached`, `allowed`, and `rate_limit_reset_credits`.

Fallback Source:
- If offline (or when a custom `sessions_dir` is provided for testing), scans `~/.codex/sessions/`
  for recent `rollout-*.jsonl` files and selects the valid `token_count` event with the newest
  ISO `timestamp` across all recent files.
- Never fabricates 0% used when a reset timestamp passes in local logs without live API verification;
  marks crossed offline windows as `STALE` / `UNKNOWN` (`post_reset_unverified = True`).
- Never fabricates 0% remaining (`100% used`) when completely offline; returns `confidence = "UNAVAILABLE"`
  and `None` fractions.
"""

from __future__ import annotations

import datetime
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_CODEX_DIR = Path.home() / ".codex"
DEFAULT_CODEX_SESSIONS_DIR = DEFAULT_CODEX_DIR / "sessions"
DEFAULT_CODEX_AUTH_FILE = DEFAULT_CODEX_DIR / "auth.json"

FIVE_HOUR_SEC = 5 * 3600
SEVEN_DAY_SEC = 7 * 24 * 3600

_CODEX_USAGE_CACHE: Dict[str, Any] = {
    "timestamp": 0.0,
    "payload": None,
    "reset_credits_payload": None,
}


def _mask_email(email: Optional[str]) -> str:
    if not email or "@" not in str(email):
        return "Codex Authenticated Account"
    local, domain = str(email).split("@", 1)
    if len(local) <= 3:
        return f"{local[:1]}***@{domain}"
    return f"{local[:3]}***{local[-1:]}@{domain}"


def _fetch_codex_live_usage(
    auth_file: Path = DEFAULT_CODEX_AUTH_FILE,
    cache_ttl: float = 10.0,
    force_refresh: bool = False,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Query `https://chatgpt.com/backend-api/wham/usage` and
    `https://chatgpt.com/backend-api/wham/rate-limit-reset-credits` using local OAuth credentials
    from `~/.codex/auth.json`. Never logs or exposes tokens.
    """
    now_mono = time.time()
    if (
        not force_refresh
        and _CODEX_USAGE_CACHE["payload"] is not None
        and (now_mono - float(_CODEX_USAGE_CACHE["timestamp"])) < cache_ttl
    ):
        return _CODEX_USAGE_CACHE["payload"], _CODEX_USAGE_CACHE.get("reset_credits_payload")

    if not auth_file.exists():
        return None, None

    try:
        auth_data = json.loads(auth_file.read_text(encoding="utf-8", errors="replace"))
        tokens = auth_data.get("tokens") or {}
        access_token = tokens.get("access_token")
        account_id = tokens.get("account_id")
        if not access_token or not isinstance(access_token, str):
            return None, None

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "User-Agent": "codex-cli/0.1.0",
        }
        if account_id and isinstance(account_id, str):
            headers["ChatGPT-Account-Id"] = account_id

        usage_payload = None
        for url in (
            "https://chatgpt.com/backend-api/wham/usage",
            "https://chatgpt.com/backend-api/codex/usage",
        ):
            try:
                req = urllib.request.Request(url, headers=headers, method="GET")
                with urllib.request.urlopen(req, timeout=4.0) as resp:
                    if resp.status == 200:
                        payload = json.loads(resp.read().decode("utf-8", errors="replace"))
                        if isinstance(payload, dict) and "rate_limit" in payload:
                            usage_payload = payload
                            break
            except Exception:
                continue

        credits_payload = None
        if usage_payload is not None:
            for c_url in (
                "https://chatgpt.com/backend-api/wham/rate-limit-reset-credits",
                "https://chatgpt.com/backend-api/codex/rate-limit-reset-credits",
            ):
                try:
                    req_c = urllib.request.Request(c_url, headers=headers, method="GET")
                    with urllib.request.urlopen(req_c, timeout=4.0) as resp_c:
                        if resp_c.status == 200:
                            c_data = json.loads(resp_c.read().decode("utf-8", errors="replace"))
                            if isinstance(c_data, dict) and "credits" in c_data:
                                credits_payload = c_data
                                break
                except Exception:
                    continue

            _CODEX_USAGE_CACHE["timestamp"] = now_mono
            _CODEX_USAGE_CACHE["payload"] = usage_payload
            _CODEX_USAGE_CACHE["reset_credits_payload"] = credits_payload
            return usage_payload, credits_payload
    except Exception:
        return None, None
    return None, None


def parse_codex_banked_resets(
    usage_payload: Optional[Dict[str, Any]],
    credits_payload: Optional[Dict[str, Any]] = None,
    manual_override: Optional[Dict[str, Any]] = None,
    now: Optional[datetime.datetime] = None,
) -> Dict[str, Any]:
    """
    Parse and classify Codex banked/manual reset entitlements (`rate_limit_reset_credits`)
    per OpenAI documentation (`https://help.openai.com/en/articles/20001498-how-banked-codex-resets-work`).

    Distinguishes 5 categories:
    1. `automatic_window_reset`: Standard 5-hour / 7-day rolling window replenishment.
    2. `banked_manual_reset`: Promotional or banked one-time full reset (`Full reset (Weekly + 5 hr)`).
       Refreshes BOTH 5-hour and weekly allowances to 100% AND shifts the subsequent weekly reset
       schedule to `now + 7 days` (it is NOT an extra independent quota pool!).
    3. `global_promotional_reset`: Automatic/global promotional reset (`promo` field).
    4. `paid_instant_reset`: Paid instant reset (`immediate_reset_purchase_eligible` - NEVER auto-triggered).
    5. `additional_paid_credits`: Overage / paid credits (`credits.has_credits` / `credits.balance` - NEVER auto-consumed).
    """
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)

    usage_rl_credits = (usage_payload or {}).get("rate_limit_reset_credits") or {}
    credits_list_raw = (credits_payload or {}).get("credits") or []

    entitlements: List[Dict[str, Any]] = []
    earliest_expiry_iso: Optional[str] = None
    earliest_expiry_sec: Optional[int] = None

    for item in credits_list_raw:
        if not isinstance(item, dict):
            continue
        exp_iso = item.get("expires_at")
        exp_sec = None
        if exp_iso:
            try:
                dt = datetime.datetime.fromisoformat(str(exp_iso).replace("Z", "+00:00"))
                exp_sec = max(0, int((dt - now).total_seconds()))
                if earliest_expiry_sec is None or exp_sec < earliest_expiry_sec:
                    earliest_expiry_sec = exp_sec
                    earliest_expiry_iso = exp_iso
            except Exception:
                pass
        entitlements.append(
            {
                "id": item.get("id"),
                "category": "banked_manual_reset",
                "reset_type": item.get("reset_type", "codex_rate_limits"),
                "title": item.get("title", "Full reset (Weekly + 5 hr)"),
                "description": item.get("description", ""),
                "status": item.get("status", "available"),
                "is_supported_by_plan": bool(item.get("is_supported_by_plan", True)),
                "eligible_windows": ["five_hour", "weekly"],
                "shifts_weekly_schedule_to_7d_from_redemption": True,
                "granted_at": item.get("granted_at"),
                "expires_at": exp_iso,
                "expires_in_sec": exp_sec,
            }
        )

    entitlements.sort(key=lambda x: (x["expires_in_sec"] if x["expires_in_sec"] is not None else 999999999))

    if credits_payload is not None or "rate_limit_reset_credits" in (usage_payload or {}):
        avail_count = int(
            (credits_payload or {}).get(
                "available_count",
                usage_rl_credits.get("available_count", len(entitlements)),
            )
        )
        applicable_now = int(usage_rl_credits.get("applicable_available_count", 0))
        verification_status = "VERIFIED_API"
        source = "chatgpt.com/backend-api/wham/rate-limit-reset-credits"
    elif manual_override and manual_override.get("enabled"):
        avail_count = int(manual_override.get("available_count", 0))
        applicable_now = int(manual_override.get("applicable_available_count", 0))
        earliest_expiry_iso = manual_override.get("earliest_expires_at")
        verification_status = "MANUALLY_VERIFIED"
        source = f"Manual user verification ({manual_override.get('note', 'user input')})"
    else:
        avail_count = 0
        applicable_now = 0
        verification_status = "UNKNOWN"
        source = "Not available in offline session logs (use live API or manual override)"

    credits_info = (usage_payload or {}).get("credits") or {}
    promo_info = (usage_payload or {}).get("promo")

    return {
        "verification_status": verification_status,
        "source": source,
        "available_count": avail_count,
        "applicable_available_count": applicable_now,
        "earliest_expires_at": earliest_expiry_iso,
        "earliest_expires_in_sec": earliest_expiry_sec,
        "entitlements": entitlements,
        "effect_model": {
            "refreshes_windows": ["five_hour", "weekly"],
            "shifts_weekly_reset_anchor": True,
            "new_weekly_reset_offset_sec": SEVEN_DAY_SEC,
            "auto_redeem_allowed": False,
            "paid_purchase_allowed": False,
            "note": (
                "A full banked reset restores 5h and weekly to 100% AND resets the weekly 7-day cycle "
                "starting at redemption time. Never redeemed automatically without explicit user confirmation."
            ),
        },
        "distinguished_types": {
            "automatic_window_reset": "Active (5h & 7d rolling timers)",
            "banked_manual_reset": f"{avail_count} available ({applicable_now} currently applicable)",
            "global_promotional_reset": str(promo_info) if promo_info else "None active",
            "paid_instant_reset": (
                "Eligible (blocked by policy)"
                if (credits_payload or {}).get("immediate_reset_purchase_eligible")
                else "Not eligible / disabled"
            ),
            "additional_paid_credits": (
                f"Balance={credits_info.get('balance', '0')} (has_credits={bool(credits_info.get('has_credits'))}, blocked by policy)"
            ),
        },
    }


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


def parse_codex_live_usage_payload(
    live_payload: Dict[str, Any],
    token_usage: Optional[Dict[str, Any]] = None,
    credits_payload: Optional[Dict[str, Any]] = None,
    manual_reset_override: Optional[Dict[str, Any]] = None,
    now: Optional[datetime.datetime] = None,
) -> Dict[str, Any]:
    """
    Parse official `GET /backend-api/wham/usage` response into normalized Quota Hub telemetry.
    Validates that a HTTP 200 response actually contains valid numeric utilization percentages.
    """
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)
    now_ts = int(now.timestamp())

    if not isinstance(live_payload, dict):
        live_payload = {}

    rl = live_payload.get("rate_limit") if isinstance(live_payload.get("rate_limit"), dict) else {}
    p_win = rl.get("primary_window") if isinstance(rl.get("primary_window"), dict) else {}
    s_win = rl.get("secondary_window") if isinstance(rl.get("secondary_window"), dict) else {}

    p_used_opt, p_mal = _validate_pct(p_win.get("used_percent"))
    s_used_opt, s_mal = _validate_pct(s_win.get("used_percent"))
    has_malformed = p_mal or s_mal

    banked_resets = parse_codex_banked_resets(
        live_payload,
        credits_payload=credits_payload,
        manual_override=manual_reset_override,
        now=now,
    )

    if has_malformed or (p_used_opt is None and s_used_opt is None):
        confidence = "SUSPECT" if has_malformed else "UNKNOWN"
        return {
            "status": "unknown",
            "confidence": confidence,
            "freshness_age_sec": 0,
            "account": _mask_email(live_payload.get("email")),
            "account_id_short": str(live_payload.get("account_id") or "")[:8],
            "shared_pool": True,
            "luna": None,
            "terra": None,
            "sol": None,
            "reset_in_sec": FIVE_HOUR_SEC,
            "windows": {
                "five_hour": {"remaining_fraction": None, "used_percent": None, "reset_in_sec": FIVE_HOUR_SEC, "resets_at": None},
                "weekly": {"remaining_fraction": None, "used_percent": None, "reset_in_sec": SEVEN_DAY_SEC, "resets_at": None},
            },
            "banked_resets": banked_resets,
            "plan": str(live_payload.get("plan_type") or "plus"),
            "models": [],
            "source": f"chatgpt.com/backend-api/wham/usage ({confidence})",
            "observed_at": now.isoformat(),
        }

    p_used = float(p_used_opt if p_used_opt is not None else s_used_opt)
    s_used = float(s_used_opt if s_used_opt is not None else p_used_opt)
    partial_missing = (p_used_opt is None or s_used_opt is None)

    p_reset_at = int(p_win.get("reset_at") or (now_ts + int(p_win.get("reset_after_seconds", FIVE_HOUR_SEC))))
    p_reset_in_sec = max(0, int(p_win.get("reset_after_seconds", max(0, p_reset_at - now_ts))))
    p_resets_iso = datetime.datetime.fromtimestamp(p_reset_at, tz=datetime.timezone.utc).isoformat()

    s_reset_at = int(s_win.get("reset_at") or (now_ts + int(s_win.get("reset_after_seconds", SEVEN_DAY_SEC))))
    s_reset_in_sec = max(0, int(s_win.get("reset_after_seconds", max(0, s_reset_at - now_ts))))
    s_resets_iso = datetime.datetime.fromtimestamp(s_reset_at, tz=datetime.timezone.utc).isoformat()

    post_reset_unverified = (p_reset_at < now_ts - 60) or (s_reset_at < now_ts - 60)
    confidence = "SUSPECT" if partial_missing else ("STALE" if post_reset_unverified else "VERIFIED")

    if rl.get("limit_reached") or rl.get("allowed") is False:
        if p_used >= 99.0 or s_used < 99.0:
            p_used = 100.0

    p_rem = round(max(0.0, min(1.0, 1.0 - (p_used / 100.0))), 4)
    s_rem = round(max(0.0, min(1.0, 1.0 - (s_used / 100.0))), 4)
    effective_rem = round(min(p_rem, s_rem), 2)

    if effective_rem <= 0.05 or rl.get("limit_reached") or rl.get("allowed") is False:
        status = "exhausted"
    elif effective_rem < 0.20:
        status = "warning"
    else:
        status = "healthy"

    reset_in_sec = s_reset_in_sec if (s_rem <= 0.05 and p_rem > 0.05) else p_reset_in_sec

    if token_usage is None:
        token_usage = {
            "total_tokens": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
            "model_context_window": 475000,
        }

    five_h_obj = {
        "remaining_fraction": p_rem,
        "used_percent": round(p_used, 1),
        "reset_in_sec": int(p_reset_in_sec),
        "resets_at": p_resets_iso,
        "window_minutes": int(p_win.get("limit_window_seconds", FIVE_HOUR_SEC)) // 60,
        "post_reset_unverified": post_reset_unverified,
    }
    weekly_obj = {
        "remaining_fraction": s_rem,
        "used_percent": round(s_used, 1),
        "reset_in_sec": int(s_reset_in_sec),
        "resets_at": s_resets_iso,
        "window_minutes": int(s_win.get("limit_window_seconds", SEVEN_DAY_SEC)) // 60,
        "post_reset_unverified": post_reset_unverified,
    }

    return {
        "status": status,
        "confidence": confidence,
        "freshness_age_sec": 0,
        "account": _mask_email(live_payload.get("email")),
        "account_id_short": str(live_payload.get("account_id") or "")[:8],
        "shared_pool": True,
        "luna": effective_rem,
        "terra": effective_rem,
        "sol": effective_rem,
        "reset_in_sec": int(reset_in_sec),
        "windows": {
            "five_hour": five_h_obj,
            "weekly": weekly_obj,
        },
        "shared_pools": {
            "codex_shared": {
                "pool_id": "codex_shared",
                "models": ["gpt-6.1-sol", "gpt-6.1-sol-low", "gpt-6.1-sol-medium", "gpt-6.1-sol-high"],
                "five_hour": five_h_obj,
                "weekly": weekly_obj,
                "effective_remaining": effective_rem,
                "confidence": "VERIFIED",
            }
        },
        "banked_resets": banked_resets,
        "plan": live_payload.get("plan_type", "plus"),
        "token_usage": token_usage,
        "models": [
            {"model_id": "gpt-6.1-sol-low", "label": "GPT-6.1 Sol (Low)", "pool_id": "codex_shared", "remaining_fraction": effective_rem, "tier": "LOW"},
            {"model_id": "gpt-6.1-sol-medium", "label": "GPT-6.1 Sol (Medium)", "pool_id": "codex_shared", "remaining_fraction": effective_rem, "tier": "MID"},
            {"model_id": "gpt-6.1-sol-high", "label": "GPT-6.1 Sol (High)", "pool_id": "codex_shared", "remaining_fraction": effective_rem, "tier": "HIGH"},
        ],
        "source": "chatgpt.com/backend-api/wham/usage (live)",
        "observed_at": now.isoformat(),
    }


def _read_tail_lines(path: Path, max_bytes: int = 131072) -> List[str]:
    """Read last `max_bytes` of a file and return complete lines."""
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


def _find_recent_rollout_files(sessions_dir: Path, limit: int = 20) -> List[Path]:
    """Return the most recently modified `rollout-*.jsonl` files."""
    if not sessions_dir.exists():
        return []
    candidates: List[Tuple[float, Path]] = []
    for fpath in sessions_dir.rglob("rollout-*.jsonl"):
        try:
            candidates.append((fpath.stat().st_mtime, fpath))
        except Exception:
            continue
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in candidates[:limit]]


def parse_codex_token_count_event(
    event_obj: Dict[str, Any],
    now: Optional[datetime.datetime] = None,
) -> Dict[str, Any]:
    """
    Parse a Codex `event_msg` with `payload.type == 'token_count'` into normalized
    Quota Hub telemetry.
    IMPORTANT: Does NOT assume a reset happened merely because `resets_at <= now_ts` when reading
    offline session logs; marks `post_reset_unverified = True` and `confidence = "STALE"`.
    """
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)
    now_ts = int(now.timestamp())

    payload = event_obj.get("payload", {})
    rate_limits = payload.get("rate_limits", {})
    info = payload.get("info") or {}
    total_usage = info.get("total_token_usage") or {}

    primary = rate_limits.get("primary") or {}
    secondary = rate_limits.get("secondary") or {}

    post_reset_unverified = False

    p_used = float(primary.get("used_percent", 0.0))
    p_resets_at_epoch = int(primary.get("resets_at", now_ts + FIVE_HOUR_SEC))
    if p_resets_at_epoch <= now_ts:
        post_reset_unverified = True
        p_reset_in_sec = 0
        p_resets_iso = datetime.datetime.fromtimestamp(p_resets_at_epoch, tz=datetime.timezone.utc).isoformat()
    else:
        p_reset_in_sec = max(0, p_resets_at_epoch - now_ts)
        p_resets_iso = datetime.datetime.fromtimestamp(p_resets_at_epoch, tz=datetime.timezone.utc).isoformat()

    s_used = float(secondary.get("used_percent", 0.0))
    s_resets_at_epoch = int(secondary.get("resets_at", now_ts + SEVEN_DAY_SEC))
    if s_resets_at_epoch <= now_ts:
        post_reset_unverified = True
        s_reset_in_sec = 0
        s_resets_iso = datetime.datetime.fromtimestamp(s_resets_at_epoch, tz=datetime.timezone.utc).isoformat()
    else:
        s_reset_in_sec = max(0, s_resets_at_epoch - now_ts)
        s_resets_iso = datetime.datetime.fromtimestamp(s_resets_at_epoch, tz=datetime.timezone.utc).isoformat()

    p_rem = round(max(0.0, min(1.0, 1.0 - (p_used / 100.0))), 4)
    s_rem = round(max(0.0, min(1.0, 1.0 - (s_used / 100.0))), 4)

    effective_rem = round(min(p_rem, s_rem), 2)
    luna_rem = effective_rem
    terra_rem = effective_rem
    sol_rem = effective_rem

    if effective_rem <= 0.05 or rate_limits.get("rate_limit_reached_type"):
        status = "exhausted"
    elif effective_rem < 0.20:
        status = "warning"
    else:
        status = "healthy"

    confidence = "STALE" if post_reset_unverified else "VERIFIED"
    reset_in_sec = s_reset_in_sec if (s_rem <= 0.05 and p_rem > 0.05) else p_reset_in_sec

    return {
        "status": status,
        "confidence": confidence,
        "post_reset_unverified": post_reset_unverified,
        "shared_pool": True,
        "luna": luna_rem,
        "terra": terra_rem,
        "sol": sol_rem,
        "reset_in_sec": int(reset_in_sec),
        "windows": {
            "five_hour": {
                "remaining_fraction": p_rem,
                "used_percent": round(p_used, 1),
                "reset_in_sec": int(p_reset_in_sec),
                "resets_at": p_resets_iso,
                "window_minutes": primary.get("window_minutes", 300),
                "post_reset_unverified": p_resets_at_epoch <= now_ts,
            },
            "weekly": {
                "remaining_fraction": s_rem,
                "used_percent": round(s_used, 1),
                "reset_in_sec": int(s_reset_in_sec),
                "resets_at": s_resets_iso,
                "window_minutes": secondary.get("window_minutes", 10080),
                "post_reset_unverified": s_resets_at_epoch <= now_ts,
            },
        },
        "plan": rate_limits.get("plan_type", "plus"),
        "token_usage": {
            "total_tokens": total_usage.get("total_tokens", 0),
            "input_tokens": total_usage.get("input_tokens", 0),
            "cached_input_tokens": total_usage.get("cached_input_tokens", 0),
            "output_tokens": total_usage.get("output_tokens", 0),
            "reasoning_output_tokens": total_usage.get("reasoning_output_tokens", 0),
            "model_context_window": info.get("model_context_window", 475000),
        },
        "models": [
            {"model_id": "gpt-6.1-sol-low", "label": "GPT-6.1 Sol (Low)", "pool_id": "codex_shared", "remaining_fraction": luna_rem, "tier": "LOW"},
            {"model_id": "gpt-6.1-sol-medium", "label": "GPT-6.1 Sol (Medium)", "pool_id": "codex_shared", "remaining_fraction": terra_rem, "tier": "MID"},
            {"model_id": "gpt-6.1-sol-high", "label": "GPT-6.1 Sol (High)", "pool_id": "codex_shared", "remaining_fraction": sol_rem, "tier": "HIGH"},
        ],
        "event_timestamp": event_obj.get("timestamp"),
        "observed_at": now.isoformat(),
    }


def collect_codex_quota(
    sessions_dir: Optional[Path] = None,
    now: Optional[datetime.datetime] = None,
    manual_reset_override: Optional[Dict[str, Any]] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Collect and normalize OpenAI Codex 5h and 7d quota telemetry + banked reset entitlements.
    """
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)
    now_ts = int(now.timestamp())
    s_dir = sessions_dir or DEFAULT_CODEX_SESSIONS_DIR

    rollout_files = _find_recent_rollout_files(s_dir, limit=20)
    best_event_ts = ""
    best_event_obj: Optional[Dict[str, Any]] = None
    best_rfile_name = ""
    latest_limit_err_ts = ""

    for rfile in rollout_files:
        lines = _read_tail_lines(rfile, max_bytes=262144)
        for line in reversed(lines):
            if '"usage_limit_exceeded"' in line:
                try:
                    err_obj = json.loads(line)
                    ets = str(err_obj.get("timestamp") or "")
                    if ets > latest_limit_err_ts:
                        latest_limit_err_ts = ets
                except Exception:
                    pass
            if '"token_count"' not in line or '"rate_limits"' not in line:
                continue
            try:
                obj = json.loads(line)
                payload = obj.get("payload", {})
                if payload.get("type") != "token_count":
                    continue
                rl = payload.get("rate_limits") or {}
                if rl.get("primary") is None and rl.get("secondary") is None:
                    continue
                ev_ts = str(obj.get("timestamp") or "")
                if ev_ts >= best_event_ts:
                    best_event_ts = ev_ts
                    best_event_obj = obj
                    best_rfile_name = rfile.name
                break
            except Exception:
                continue

    latest_token_usage = None
    if best_event_obj is not None:
        parsed_rollout = parse_codex_token_count_event(best_event_obj, now=now)
        latest_token_usage = parsed_rollout.get("token_usage")

    # 1. Primary in production mode: live OpenAI Codex backend usage API + reset credits API
    if sessions_dir is None:
        live_payload, credits_payload = _fetch_codex_live_usage(force_refresh=force_refresh)
        if live_payload is not None:
            return parse_codex_live_usage_payload(
                live_payload,
                token_usage=latest_token_usage,
                credits_payload=credits_payload,
                manual_reset_override=manual_reset_override,
                now=now,
            )

    # 2. Fallback / test mode: newest rollout event across all recent rollout files
    if best_event_obj is not None:
        result = parse_codex_token_count_event(best_event_obj, now=now)
        rl = (best_event_obj.get("payload") or {}).get("rate_limits") or {}
        p_resets_at = int((rl.get("primary") or {}).get("resets_at") or 0)
        if latest_limit_err_ts and latest_limit_err_ts >= best_event_ts and p_resets_at > now_ts:
            result["windows"]["five_hour"]["used_percent"] = 100.0
            result["windows"]["five_hour"]["remaining_fraction"] = 0.0
            result["status"] = "exhausted"
            result["luna"] = 0.0
            result["terra"] = 0.0
            result["sol"] = 0.0
            for m in result.get("models", []):
                m["remaining_fraction"] = 0.0
        result["source"] = f"codex session ({best_rfile_name}) [offline fallback]"
        result["confidence"] = "STALE"
        result["banked_resets"] = parse_codex_banked_resets(
            None,
            credits_payload=None,
            manual_override=manual_reset_override,
            now=now,
        )
        return result

    return {
        "status": "unknown",
        "confidence": "UNAVAILABLE",
        "account": "Unavailable",
        "shared_pool": True,
        "luna": None,
        "terra": None,
        "sol": None,
        "reset_in_sec": 5400,
        "windows": {
            "five_hour": {"remaining_fraction": None, "used_percent": None, "reset_in_sec": 5400, "resets_at": None},
            "weekly": {"remaining_fraction": None, "used_percent": None, "reset_in_sec": 5400, "resets_at": None},
        },
        "banked_resets": parse_codex_banked_resets(None, None, manual_override=manual_reset_override, now=now),
        "plan": "unknown",
        "models": [],
        "source": "Codex API and rollout session files unavailable (UNAVAILABLE)",
        "observed_at": now.isoformat(),
    }


if __name__ == "__main__":
    print(json.dumps(collect_codex_quota(), indent=2))
