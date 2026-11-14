"""
Preference Sliders + Burn-Rate-Aware Multi-Provider Allocation & Smooth Weighted Round Robin (WP-ALLOCATION).

Implements:
1. `BurnRateLearner`:
   - SQLite-backed (`burn_rate_samples` table) pool-level burn-rate estimator across the 4 quota pools
     (`agy_gemini`, `agy_claude_gpt`, `claude_shared`, `codex_shared`) and both windows (`five_hour`, `weekly`).
   - Records a sample on quota refresh only when no reset occurred in between (`resets_at` unchanged and
     `d_used >= 0`), `agent_hours >= 0.05`, elapsed time is between 2 and 90 minutes (`120s..5400s`),
     and telemetry confidence is not `UNKNOWN` / `UNAVAILABLE` / `SUSPECT`.
   - Estimates `b[pool][window]` (fraction of window consumed per agent-hour) via EWMA (`alpha = 0.2`)
     with outlier clipping at `3x` the current median.
   - Uses a prior derived from `QuotaCostLearner.predict_cost` (`source="prior"`) until a pool/window
     accumulates `>= 5` valid samples (`source="ewma"`).
2. `compute_allocation`:
   - Pool capacity `A_pool = min_w usable_w / (b_w * max(T_w_hours, floor_w))` with floors
     `five_hour -> 0.25h`, `weekly -> 1.0h`. Unknown/unavailable telemetry or `usable <= 0` yields `A_pool = 0`.
   - Provider capacity `A_provider = sum_{pool in provider} A_pool`.
   - Target share `pi_provider = s_provider * A_provider / sum_q s_q * A_q` (`0` if denominator `0`; slider `0` = off).
   - Pool slack `sigma_pool = min_w remaining_w / max(0.02, T_w / L_w)` with `L_5h = 5h`, `L_weekly = 168h`;
     `sigma_provider = max_{pool in provider} sigma_pool`.
   - Sigma bands (`1.0 / 0.5` in `endurance`, `0.7 / 0.35` in `sprint`):
     * `normal`: standard capability effort, `max_workers = policy max_per_provider`
     * `economy`: non-critical packages one effort step lower, `max_workers = ceil(cap / 2)`
     * `saver`: non-critical packages `low` effort, `max_workers = 1`
     * Critical packages (`HIGH` floor + `design` or high-risk `review`, or explicitly flagged) keep their effort.
3. Classic Smooth Weighted Round Robin (`step_swrr` & `select_and_commit`, Amendments 3-4):
   - Persistent per-provider credit `C_p`. Among feasible `slider > 0` providers, sets effective weight
     `w_p = pi_p`, adds `w_p` to each `C_p`, selects `argmax_p C_p`, and subtracts
     `W = sum_p w_p` from the winner's credit (no clamping). Amendment 4: when any feasible provider is
     harvesting, SWRR runs over the harvesting providers only.
"""

from __future__ import annotations

import datetime
import math
import sqlite3
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


POOLS: Tuple[str, ...] = ("agy_gemini", "agy_claude_gpt", "claude_shared", "codex_shared")
WINDOWS: Tuple[str, ...] = ("five_hour", "weekly")
PROVIDERS: Tuple[str, ...] = ("agy", "claude", "codex")

POOL_TO_PROVIDER: Dict[str, str] = {
    "agy_gemini": "agy",
    "agy_claude_gpt": "agy",
    "claude_shared": "claude",
    "codex_shared": "codex",
}

PROVIDER_POOLS: Dict[str, Tuple[str, ...]] = {
    "agy": ("agy_gemini", "agy_claude_gpt"),
    "claude": ("claude_shared",),
    "codex": ("codex_shared",),
}

POOL_WINDOW_KEYS: Dict[str, Tuple[str, str]] = {
    "agy_gemini": ("gemini_5h", "gemini_weekly"),
    "agy_claude_gpt": ("claude_gpt_5h", "claude_gpt_weekly"),
    "claude_shared": ("five_hour", "weekly"),
    "codex_shared": ("five_hour", "weekly"),
}

POOL_FALLBACK_WINDOW_KEYS: Dict[str, Tuple[str, str]] = {
    "agy_gemini": ("five_hour", "weekly"),
    "agy_claude_gpt": ("claude_gpt_pool", "claude_gpt_pool"),
    "claude_shared": ("five_hour", "weekly"),
    "codex_shared": ("five_hour", "weekly"),
}

WINDOW_FLOORS_HOURS: Dict[str, float] = {
    "five_hour": 0.25,
    "weekly": 1.0,
}

WINDOW_LENGTH_HOURS: Dict[str, float] = {
    "five_hour": 5.0,
    "weekly": 168.0,
}

DEFAULT_PREFERENCE_WEIGHTS: Dict[str, int] = {
    "agy": 10,
    "codex": 8,
    "claude": 5,
}

DEFAULT_CREDITS: Dict[str, float] = {
    "agy": 0.0,
    "codex": 0.0,
    "claude": 0.0,
}

DEFAULT_MAX_PER_PROVIDER: Dict[str, int] = {
    "agy": 6,
    "claude": 6,
    "codex": 6,
}

# Fallback cold-start MID/implement baseline matching scheduler.py when no QuotaCostLearner instance is supplied
_FALLBACK_MID_IMPLEMENT = (0.045, 0.014, 180)  # (5h_mean, weekly_mean, est_duration_sec)
_FALLBACK_POOL_MULTIPLIERS: Dict[str, Tuple[float, float]] = {
    "agy_gemini": (0.75, 0.70),
    "agy_claude_gpt": (1.10, 1.15),
    "claude_shared": (1.05, 1.00),
    "codex_shared": (1.00, 0.95),
}

UNKNOWN_CONFIDENCE_STATES = {"UNKNOWN", "UNAVAILABLE", "SUSPECT"}
EWMA_ALPHA = 0.2
MIN_SAMPLES_FOR_EWMA = 5
# Realistic wall-clock duration of one MID/implement task in hours (20 minutes). Priors convert the
# `QuotaCostLearner` per-task mean cost into a per-agent-hour burn rate with this constant; the learner's
# own ~144-180 s `estimated_duration_sec` is a dispatch estimate and would inflate priors ~6-8x.
PRIOR_TASK_HOURS = 1.0 / 3.0
# An EWMA estimate is clamped to [prior / EWMA_CLAMP_FACTOR, prior * EWMA_CLAMP_FACTOR].
EWMA_CLAMP_FACTOR = 10.0
# Schema/version marker for `burn_rate_samples`; samples recorded before this version (counting idle
# coordinators as running workers, 144 s priors) are moved to `burn_rate_samples_pre_fix` on startup.
BURN_SAMPLES_SCHEMA_VERSION = "2"
RUNNING_STATUSES = {"busy", "running"}
MIN_AGENT_HOURS = 0.05
MIN_ELAPSED_SEC = 120.0
MAX_ELAPSED_SEC = 5400.0
OUTLIER_MEDIAN_MULT = 3.0
MIN_BURN_RATE = 1e-4


def normalize_window_name(window: str) -> str:
    """Normalize a window identifier to `'five_hour'` or `'weekly'`."""
    w = str(window or "").lower().strip()
    if w in {"five_hour", "5h", "gemini_5h", "claude_gpt_5h", "primary"}:
        return "five_hour"
    if w in {"weekly", "7d", "seven_day", "gemini_weekly", "claude_gpt_weekly", "secondary"}:
        return "weekly"
    raise ValueError(f"Unsupported window name: {window!r}")


def validate_preference_weights(raw: Any, allow_partial: bool = False) -> Dict[str, int]:
    """
    Validate provider preference sliders as 0..10 integers for `agy`, `codex`, and `claude`.
    Raises `ValueError` on missing keys (when `allow_partial=False`), booleans, non-integers, or out-of-range values.
    """
    if not isinstance(raw, dict):
        raise ValueError("preference_weights must be a JSON object mapping provider names to 0..10 integers")
    unknown = set(raw.keys()) - set(PROVIDERS)
    if unknown:
        raise ValueError(f"Unknown provider key(s) in preferences: {sorted(unknown)}")
    out = dict(DEFAULT_PREFERENCE_WEIGHTS)
    for prov in PROVIDERS:
        if prov not in raw:
            if not allow_partial:
                raise ValueError(f"Missing required provider preference slider: {prov!r}")
            continue
        val = raw[prov]
        if isinstance(val, bool) or val is None:
            raise ValueError(f"Preference slider for {prov!r} must be an integer in 0..10, got {val!r}")
        if isinstance(val, float):
            if not val.is_integer():
                raise ValueError(f"Preference slider for {prov!r} must be an integer in 0..10, got {val!r}")
            ival = int(val)
        elif isinstance(val, int):
            ival = int(val)
        else:
            raise ValueError(f"Preference slider for {prov!r} must be an integer in 0..10, got {val!r}")
        if ival < 0 or ival > 10:
            raise ValueError(f"Preference slider for {prov!r} must be between 0 and 10, got {ival}")
        out[prov] = ival
    return out


def coerce_preference_weights(raw: Any) -> Dict[str, int]:
    """Safely coerce persisted preference weights, falling back to `DEFAULT_PREFERENCE_WEIGHTS` per key."""
    out = dict(DEFAULT_PREFERENCE_WEIGHTS)
    if not isinstance(raw, dict):
        return out
    for prov in PROVIDERS:
        val = raw.get(prov)
        if isinstance(val, bool) or val is None:
            continue
        try:
            ival = int(val)
            if 0 <= ival <= 10:
                out[prov] = ival
        except (TypeError, ValueError):
            continue
    return out


def normalise_sliders(raw: Any) -> Dict[str, float]:
    """
    Normalise 0..10 provider preference sliders to sum to 1.0.
    If all sliders are 0, falls back to uniform (1/3, 1/3, 1/3).
    """
    weights = coerce_preference_weights(raw)
    total = sum(weights.values())
    if total <= 0:
        return {p: 1.0 / float(len(PROVIDERS)) for p in PROVIDERS}
    return {p: float(weights[p]) / float(total) for p in PROVIDERS}


normalize_sliders = normalise_sliders


def reset_credits() -> Dict[str, float]:
    """Return a fresh zeroed credit dictionary for all providers."""
    return {p: 0.0 for p in PROVIDERS}


def coerce_credits(raw: Any) -> Dict[str, float]:
    """Safely coerce persisted SWRR provider credits to floats."""
    out = reset_credits()
    if not isinstance(raw, dict):
        return out
    for prov in PROVIDERS:
        val = raw.get(prov)
        if isinstance(val, bool) or val is None:
            continue
        try:
            fval = float(val)
            if math.isfinite(fval):
                out[prov] = fval
        except (TypeError, ValueError):
            continue
    return out


def compute_prior_burn_rate(pool_id: str, window: str, cost_learner: Optional[Any] = None) -> float:
    """
    Compute the cold-start / empirical task prior burn rate (fraction of window consumed per agent-hour)
    derived from `QuotaCostLearner.predict_cost(pool_id, "implement", "MID", "med")` divided by
    `PRIOR_TASK_HOURS` (20 minutes per task).
    """
    w_norm = normalize_window_name(window)
    if cost_learner is not None and hasattr(cost_learner, "predict_cost"):
        try:
            pred = cost_learner.predict_cost(pool_id, "implement", "MID", "med")
            mean_cost = float(pred["five_hour_mean"] if w_norm == "five_hour" else pred["weekly_mean"])
            return round(max(MIN_BURN_RATE, mean_cost / PRIOR_TASK_HOURS), 6)
        except Exception:
            pass
    mult_5h, mult_wk = _FALLBACK_POOL_MULTIPLIERS.get(pool_id, (1.0, 1.0))
    base_5h, base_wk, _base_dur_sec = _FALLBACK_MID_IMPLEMENT
    mean_cost = (base_5h * mult_5h) if w_norm == "five_hour" else (base_wk * mult_wk)
    return round(max(MIN_BURN_RATE, mean_cost / PRIOR_TASK_HOURS), 6)


class BurnRateLearner:
    """
    Pool-level burn-rate learner backed by the `burn_rate_samples` SQLite table.

    - Tracks `b[pool][window]` = fraction of window consumed per agent-hour.
    - Filters out reset-crossing intervals, negative deltas, short/long intervals outside 2..90 minutes,
      intervals with `agent_hours < 0.05`, and untrusted (`UNKNOWN`/`UNAVAILABLE`/`SUSPECT`) telemetry.
    - Clips outliers at `3x` the current sample median and updates an EWMA (`alpha = 0.2`).
    - Returns the `QuotaCostLearner`-derived prior (`source="prior"`) until a pool/window has `>= 5` samples; EWMA is clamped to `[prior/10, prior*10]`.
    """

    def __init__(self, db_path: Path, cost_learner: Optional[Any] = None) -> None:
        self.db_path = Path(db_path)
        self.cost_learner = cost_learner
        self._prev_snapshots: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.db_timeout_sec = 5.0
        self.schema_ok = False
        self.unattributed_running = 0.0
        self.skipped_unattributed_intervals: Dict[str, int] = {p: 0 for p in PROVIDERS}
        self._init_db()

    def _init_db(self) -> None:
        """Create tables and run the guarded one-time migration; sets `self.schema_ok`."""
        self.schema_ok = False
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path), timeout=self.db_timeout_sec, isolation_level=None)
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS burn_rate_samples (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pool_id TEXT NOT NULL,
                        window_name TEXT NOT NULL,
                        recorded_at TEXT NOT NULL,
                        elapsed_sec REAL NOT NULL,
                        running_jobs REAL NOT NULL,
                        agent_hours REAL NOT NULL,
                        used_prev REAL NOT NULL,
                        used_now REAL NOT NULL,
                        d_used REAL NOT NULL,
                        raw_burn_rate REAL NOT NULL,
                        clipped_burn_rate REAL NOT NULL,
                        ewma_burn_rate REAL NOT NULL,
                        resets_at TEXT
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_burn_rate_pool_win ON burn_rate_samples (pool_id, window_name, id)"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS burn_rate_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                row = conn.execute("SELECT value FROM burn_rate_meta WHERE key = 'schema_version'").fetchone()
                if row is None or str(row[0]) != BURN_SAMPLES_SCHEMA_VERSION:
                    n_old = conn.execute("SELECT COUNT(*) FROM burn_rate_samples").fetchone()[0]
                    if n_old:
                        # Never delete silently: copy pre-fix samples to a backup table, then start fresh.
                        conn.execute(
                            "CREATE TABLE IF NOT EXISTS burn_rate_samples_pre_fix AS "
                            "SELECT * FROM burn_rate_samples WHERE 0"
                        )
                        conn.execute("INSERT INTO burn_rate_samples_pre_fix SELECT * FROM burn_rate_samples")
                        conn.execute("DELETE FROM burn_rate_samples")
                    conn.execute(
                        "INSERT OR REPLACE INTO burn_rate_meta (key, value) VALUES ('schema_version', ?)",
                        (BURN_SAMPLES_SCHEMA_VERSION,),
                    )
                conn.execute("COMMIT")
                self.schema_ok = True
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
            finally:
                conn.close()
        except Exception:
            pass

    def _marker_ok(self) -> bool:
        """True only when the stored schema marker equals the current version (samples are trustworthy)."""
        try:
            with sqlite3.connect(str(self.db_path), timeout=self.db_timeout_sec) as conn:
                row = conn.execute("SELECT value FROM burn_rate_meta WHERE key = 'schema_version'").fetchone()
            return row is not None and str(row[0]) == BURN_SAMPLES_SCHEMA_VERSION
        except Exception:
            return False

    def prior_burn_rate(self, pool_id: str, window: str) -> float:
        """Return the prior burn rate (fraction per agent-hour) for `(pool_id, window)`."""
        return compute_prior_burn_rate(pool_id, window, cost_learner=self.cost_learner)

    def record_sample(
        self,
        pool_id: str,
        window: str,
        used_prev: Optional[float] = None,
        used_now: Optional[float] = None,
        elapsed_sec: Optional[float] = None,
        running_jobs: Optional[float] = None,
        *,
        d_used: Optional[float] = None,
        agent_hours: Optional[float] = None,
        mean_active_workers: Optional[float] = None,
        elapsed_minutes: Optional[float] = None,
        elapsed_hours: Optional[float] = None,
        resets_at_prev: Optional[str] = None,
        resets_at_now: Optional[str] = None,
        pre_resets_at: Optional[str] = None,
        post_resets_at: Optional[str] = None,
        reset_crossed: bool = False,
        confidence: str = "VERIFIED",
        recorded_at: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Validate and record a pool/window burn-rate sample.
        Only records when:
        - `confidence` is not `UNKNOWN` / `UNAVAILABLE` / `SUSPECT`
        - no reset crossed (`resets_at` unchanged and `d_used >= 0`)
        - `elapsed_sec` is between 2 and 90 minutes (`120 <= elapsed_sec <= 5400`)
        - `agent_hours >= 0.05`
        """
        w_norm = normalize_window_name(window)
        if not self._marker_ok():
            return {
                "recorded": False,
                "rejected": 1,
                "reason": "schema_not_ready",
                "pool_id": pool_id,
                "window": w_norm,
            }
        conf_norm = str(confidence or "VERIFIED").upper().strip()
        if conf_norm in UNKNOWN_CONFIDENCE_STATES:
            return {
                "recorded": False,
                "rejected": 1,
                "reason": "untrusted_telemetry",
                "pool_id": pool_id,
                "window": w_norm,
            }

        if running_jobs is None and mean_active_workers is not None:
            running_jobs = float(mean_active_workers)

        # Resolve elapsed_sec
        if elapsed_sec is None:
            if elapsed_minutes is not None:
                elapsed_sec = float(elapsed_minutes) * 60.0
            elif elapsed_hours is not None:
                elapsed_sec = float(elapsed_hours) * 3600.0
            elif agent_hours is not None and running_jobs is not None and float(running_jobs) > 0:
                elapsed_sec = (float(agent_hours) / float(running_jobs)) * 3600.0
            elif agent_hours is not None:
                elapsed_sec = 1800.0
            else:
                return {
                    "recorded": False,
                    "rejected": 1,
                    "reason": "missing_elapsed_time",
                    "pool_id": pool_id,
                    "window": w_norm,
                }
        elapsed_val = float(elapsed_sec)
        if elapsed_val < (MIN_ELAPSED_SEC - 1e-6) or elapsed_val > (MAX_ELAPSED_SEC + 1e-6):
            return {
                "recorded": False,
                "rejected": 1,
                "reason": "elapsed_out_of_range",
                "pool_id": pool_id,
                "window": w_norm,
                "elapsed_sec": elapsed_val,
            }

        # Check reset crossing
        r_prev = resets_at_prev if resets_at_prev is not None else pre_resets_at
        r_now = resets_at_now if resets_at_now is not None else post_resets_at
        if reset_crossed or (r_prev is not None and r_now is not None and str(r_prev) != str(r_now)):
            return {
                "recorded": False,
                "rejected": 1,
                "reason": "reset_crossed",
                "pool_id": pool_id,
                "window": w_norm,
            }

        # Resolve d_used (in fraction 0..1)
        if d_used is None:
            if used_prev is None or used_now is None:
                return {
                    "recorded": False,
                    "rejected": 1,
                    "reason": "missing_usage",
                    "pool_id": pool_id,
                    "window": w_norm,
                }
            u_prev = float(used_prev)
            u_now = float(used_now)
            d_val = u_now - u_prev
        else:
            d_val = float(d_used)
            u_prev = float(used_prev) if used_prev is not None else 0.0
            u_now = float(used_now) if used_now is not None else (u_prev + d_val)

        if d_val < -1e-9:
            return {
                "recorded": False,
                "rejected": 1,
                "reason": "reset_or_negative_delta",
                "pool_id": pool_id,
                "window": w_norm,
                "d_used": d_val,
            }
        d_val = max(0.0, d_val)

        # Resolve agent_hours
        elapsed_hrs = elapsed_val / 3600.0
        if agent_hours is None:
            jobs_val = float(running_jobs or 0.0)
            ah_val = jobs_val * elapsed_hrs
        else:
            ah_val = float(agent_hours)
            jobs_val = float(running_jobs) if running_jobs is not None else (ah_val / elapsed_hrs if elapsed_hrs > 0 else 0.0)

        if ah_val < (MIN_AGENT_HOURS - 1e-9):
            return {
                "recorded": False,
                "rejected": 1,
                "reason": "insufficient_agent_hours",
                "pool_id": pool_id,
                "window": w_norm,
                "agent_hours": ah_val,
            }

        raw_rate = max(MIN_BURN_RATE, d_val / ah_val)
        if recorded_at:
            now_iso = str(recorded_at)
        elif ts is not None:
            now_iso = datetime.datetime.fromtimestamp(float(ts), tz=datetime.timezone.utc).isoformat()
        else:
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        prev_rates: List[float] = []
        prev_ewma: Optional[float] = None
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                rows = conn.execute(
                    """
                    SELECT raw_burn_rate, clipped_burn_rate, ewma_burn_rate
                    FROM burn_rate_samples
                    WHERE pool_id = ? AND window_name = ?
                    ORDER BY id ASC
                    """,
                    (pool_id, w_norm),
                ).fetchall()
                for r_raw, r_clip, r_ewma in rows:
                    prev_rates.append(float(r_clip if r_clip is not None else r_raw))
                    prev_ewma = float(r_ewma)

                if prev_rates:
                    med = statistics.median(prev_rates)
                    clipped_rate = min(raw_rate, OUTLIER_MEDIAN_MULT * med) if med > 0 else raw_rate
                else:
                    clipped_rate = raw_rate

                if prev_ewma is None:
                    ewma_rate = clipped_rate
                else:
                    ewma_rate = (EWMA_ALPHA * clipped_rate) + ((1.0 - EWMA_ALPHA) * prev_ewma)

                conn.execute(
                    """
                    INSERT INTO burn_rate_samples (
                        pool_id, window_name, recorded_at, elapsed_sec, running_jobs,
                        agent_hours, used_prev, used_now, d_used,
                        raw_burn_rate, clipped_burn_rate, ewma_burn_rate, resets_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pool_id,
                        w_norm,
                        now_iso,
                        round(elapsed_val, 3),
                        round(jobs_val, 4),
                        round(ah_val, 6),
                        round(u_prev, 6),
                        round(u_now, 6),
                        round(d_val, 6),
                        round(raw_rate, 6),
                        round(clipped_rate, 6),
                        round(ewma_rate, 6),
                        str(r_now) if r_now is not None else None,
                    ),
                )
                conn.commit()
        except Exception as exc:
            return {
                "recorded": False,
                "rejected": 1,
                "reason": f"db_error: {exc}",
                "pool_id": pool_id,
                "window": w_norm,
            }

        est = self.get_pool_window_estimate(pool_id, w_norm)
        return {
            "recorded": True,
            "rejected": 0,
            "pool_id": pool_id,
            "window": w_norm,
            "d_used": round(d_val, 6),
            "agent_hours": round(ah_val, 6),
            "raw_burn_rate": round(raw_rate, 6),
            "clipped_burn_rate": round(clipped_rate, 6),
            "ewma_burn_rate": round(ewma_rate, 6),
            "estimate": est,
        }

    @staticmethod
    def _extract_pool_window_usage(
        pool_id: str,
        window: str,
        providers_quota: Optional[Dict[str, Any]],
        pool_states: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[float], Optional[str], str]:
        """Return `(used_fraction_0_to_1, resets_at_str, confidence)` for `(pool_id, window)`."""
        w_norm = normalize_window_name(window)
        if isinstance(pool_states, dict) and pool_id in pool_states:
            ps = pool_states[pool_id] or {}
            conf = str(ps.get("confidence") or "VERIFIED").upper()
            w_obj = ps.get(w_norm) or {}
            rem = w_obj.get("remaining_fraction")
            used_pct = w_obj.get("used_percent")
            used_frac: Optional[float] = None
            if used_pct is not None:
                try:
                    used_frac = max(0.0, min(1.0, float(used_pct) / 100.0))
                except (TypeError, ValueError):
                    used_frac = None
            elif rem is not None:
                try:
                    used_frac = max(0.0, min(1.0, 1.0 - float(rem)))
                except (TypeError, ValueError):
                    used_frac = None
            resets_at = w_obj.get("resets_at")
            return used_frac, (str(resets_at) if resets_at is not None else None), conf

        prov = POOL_TO_PROVIDER.get(pool_id, "")
        p_data = (providers_quota or {}).get(prov) or {}
        if not p_data:
            return None, None, "UNKNOWN"
        conf = str(
            p_data.get("confidence")
            or ("UNAVAILABLE" if p_data.get("status") in {"offline", "unknown"} else "VERIFIED")
        ).upper()
        wins = p_data.get("windows") or {}
        k5, kw = POOL_WINDOW_KEYS[pool_id]
        fk5, fkw = POOL_FALLBACK_WINDOW_KEYS[pool_id]
        w_key = k5 if w_norm == "five_hour" else kw
        f_key = fk5 if w_norm == "five_hour" else fkw
        w_obj = wins.get(w_key) or wins.get(f_key) or {}
        used_pct = w_obj.get("used_percent")
        rem = w_obj.get("remaining_fraction")
        used_frac = None
        if used_pct is not None:
            try:
                used_frac = max(0.0, min(1.0, float(used_pct) / 100.0))
            except (TypeError, ValueError):
                used_frac = None
        elif rem is not None:
            try:
                used_frac = max(0.0, min(1.0, 1.0 - float(rem)))
            except (TypeError, ValueError):
                used_frac = None
        resets_at = w_obj.get("resets_at")
        return used_frac, (str(resets_at) if resets_at is not None else None), conf

    @staticmethod
    def _entry_pool_id(w: Dict[str, Any]) -> Optional[str]:
        """
        Map a live-activity entry to a quota pool, or `None` when it cannot be determined unambiguously
        (e.g. the AGY desktop label "AGY Coordinator (Gemini 4 Argon / Opus 5.5)" names both pools).
        """
        prov = str(w.get("provider") or "").lower()
        mod = str(w.get("model") or "").lower()
        if prov == "agy":
            claude_gpt = any(t in mod for t in ("claude", "gpt", "opus", "sonnet", "haiku"))
            gemini = any(t in mod for t in ("gemini", "flash", "argon"))
            if claude_gpt and not gemini:
                return "agy_claude_gpt"
            if gemini and not claude_gpt:
                return "agy_gemini"
            return None
        if prov == "claude":
            return "claude_shared"
        if prov == "codex":
            return "codex_shared"
        return None

    @staticmethod
    def _running_entries(live_activity: Optional[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Optional[str]]]:
        """
        Deduplicated busy/running entries as `(entry, pool_id_or_None)`. Coordinators are processed before
        workers. Dedupe key: `session_id` when present, otherwise a non-zero `pid`; pids seen on any
        counted entry also suppress later session-less entries (same process listed twice).
        """
        out: List[Tuple[Dict[str, Any], Optional[str]]] = []
        seen_sids: set = set()
        seen_pids: set = set()
        la = live_activity or {}
        entries = [(c, False) for c in (la.get("coordinators") or [])] + [
            (w, True) for w in (la.get("workers") or [])
        ]
        for w, is_worker in entries:
            if not isinstance(w, dict) or not w.get("alive", True):
                continue
            # A listed leaf worker without an explicit status is running by definition; coordinators must say so.
            status = str(w.get("status") or ("running" if is_worker else "")).lower()
            if status not in RUNNING_STATUSES:
                continue
            sid = w.get("session_id")
            try:
                pid = int(w.get("pid") or 0)
            except (TypeError, ValueError):
                pid = 0
            if sid:
                if sid in seen_sids:
                    continue
                seen_sids.add(sid)
            elif pid:
                if pid in seen_pids:
                    continue
            if pid:
                seen_pids.add(pid)
            out.append((w, BurnRateLearner._entry_pool_id(w)))
        return out

    @staticmethod
    def _count_pool_running_jobs(
        pool_id: str,
        live_activity: Optional[Dict[str, Any]],
        pool_states: Optional[Dict[str, Any]],
    ) -> float:
        """
        Return actively working agents on `pool_id`: alive entries with status in `{"busy", "running"}`,
        deduplicated (see `_running_entries`); entries whose pool is ambiguous count for no pool (see
        `_count_unattributed_running`). Falls back to `pool_states[pool_id]['running_jobs']` when no live
        activity is supplied.
        """
        if isinstance(live_activity, dict) and (
            live_activity.get("workers") is not None or live_activity.get("coordinators") is not None
        ):
            return float(sum(1 for _, pid_ in BurnRateLearner._running_entries(live_activity) if pid_ == pool_id))
        if isinstance(pool_states, dict) and pool_id in pool_states:
            ps = pool_states[pool_id] or {}
            if ps.get("running_jobs") is not None:
                return float(ps["running_jobs"])
        return 0.0

    @staticmethod
    def _unattributed_by_provider(live_activity: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Busy/running entries with an unknown pool, grouped by provider (`agy-task` belongs to `agy`)."""
        out: Dict[str, float] = {p: 0.0 for p in PROVIDERS}
        if not isinstance(live_activity, dict):
            return out
        for w, pool in BurnRateLearner._running_entries(live_activity):
            if pool is not None:
                continue
            prov = str(w.get("provider") or "").lower()
            if prov.startswith("agy"):
                prov = "agy"
            if prov in out:
                out[prov] += 1.0
        return out

    @staticmethod
    def _count_unattributed_running(live_activity: Optional[Dict[str, Any]]) -> float:
        """Busy/running entries whose pool could not be determined unambiguously."""
        if not isinstance(live_activity, dict):
            return 0.0
        return float(sum(1 for _, pid_ in BurnRateLearner._running_entries(live_activity) if pid_ is None))

    def observe_snapshot(
        self,
        providers_quota: Optional[Dict[str, Any]] = None,
        live_activity: Optional[Dict[str, Any]] = None,
        pool_states: Optional[Dict[str, Any]] = None,
        now_ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Poll hook called on every quota refresh.
        - Keeps the baseline snapshot (`used`, `resets_at`, `ts`) until a valid sample interval
          (`>= 120s`, within `120s..5400s`) has elapsed; short polls never discard the baseline.
        - Integrates `agent_hours += running_workers * dt` across every sub-interval `[last_ts, now_val]`
          using the worker count active during that sub-interval.
        """
        if not self.schema_ok or not self._marker_ok():
            self._init_db()  # retry a failed/locked migration
        self.unattributed_running = self._count_unattributed_running(live_activity)
        unattr_by_prov = self._unattributed_by_provider(live_activity)
        now_val = float(now_ts if now_ts is not None else time.time())
        recorded_iso = datetime.datetime.fromtimestamp(now_val, tz=datetime.timezone.utc).isoformat()
        results: Dict[str, Dict[str, Any]] = {}

        for pool_id in POOLS:
            results[pool_id] = {}
            running_now = self._count_pool_running_jobs(pool_id, live_activity, pool_states)
            prov_id = POOL_TO_PROVIDER.get(pool_id, "")
            taint_now = unattr_by_prov.get(prov_id, 0.0) > 0.0
            for w_name in WINDOWS:
                used_now, resets_now, conf = self._extract_pool_window_usage(
                    pool_id, w_name, providers_quota, pool_states
                )
                key = (pool_id, w_name)
                prev = self._prev_snapshots.get(key)
                if used_now is None or conf in UNKNOWN_CONFIDENCE_STATES:
                    results[pool_id][w_name] = {"recorded": False, "reason": "initial_or_untrusted_snapshot"}
                    continue

                if prev is None:
                    self._prev_snapshots[key] = {
                        "used": used_now,
                        "resets_at": resets_now,
                        "ts": now_val,
                        "last_ts": now_val,
                        "running_jobs": running_now,
                        "accum_agent_hours": 0.0,
                        "tainted": taint_now,
                    }
                    results[pool_id][w_name] = {"recorded": False, "reason": "initial_or_untrusted_snapshot"}
                    continue

                baseline_ts = float(prev["ts"])
                last_ts = float(prev.get("last_ts", baseline_ts))
                dt_sec = max(0.0, now_val - last_ts)
                elapsed_sec = now_val - baseline_ts
                last_jobs = float(prev.get("running_jobs", 0.0))
                accum_ah = float(prev.get("accum_agent_hours", 0.0)) + (last_jobs * (dt_sec / 3600.0))

                # Update integral state on the active baseline without discarding the baseline
                prev["last_ts"] = now_val
                prev["running_jobs"] = running_now
                prev["accum_agent_hours"] = accum_ah
                prev["tainted"] = bool(prev.get("tainted")) or taint_now

                r_prev = prev.get("resets_at")
                reset_crossed = (r_prev is not None and resets_now is not None and str(r_prev) != str(resets_now))
                d_used = used_now - float(prev["used"])

                if prev.get("tainted") and elapsed_sec >= (MIN_ELAPSED_SEC - 1e-6):
                    # Unattributed busy activity consumed quota without agent-hours: the interval is unusable.
                    self.skipped_unattributed_intervals[prov_id] = self.skipped_unattributed_intervals.get(prov_id, 0) + 1
                    results[pool_id][w_name] = {
                        "recorded": False,
                        "rejected": 1,
                        "reason": "unattributed_busy_activity",
                        "pool_id": pool_id,
                        "window": w_name,
                    }
                    self._prev_snapshots[key] = {
                        "used": used_now,
                        "resets_at": resets_now,
                        "ts": now_val,
                        "last_ts": now_val,
                        "running_jobs": running_now,
                        "accum_agent_hours": 0.0,
                        "tainted": taint_now,
                    }
                    continue

                if reset_crossed or d_used < -1e-9 or elapsed_sec > (MAX_ELAPSED_SEC + 1e-6):
                    mean_jobs = (accum_ah / (elapsed_sec / 3600.0)) if elapsed_sec > 0 else 0.0
                    res = self.record_sample(
                        pool_id=pool_id,
                        window=w_name,
                        used_prev=float(prev["used"]),
                        used_now=used_now,
                        elapsed_sec=elapsed_sec,
                        agent_hours=accum_ah,
                        running_jobs=mean_jobs,
                        resets_at_prev=r_prev,
                        resets_at_now=resets_now,
                        reset_crossed=reset_crossed,
                        confidence=conf,
                        recorded_at=recorded_iso,
                    )
                    results[pool_id][w_name] = res
                    # Baseline crossed a reset or exceeded the 90-minute cap; start a fresh baseline at now_val
                    self._prev_snapshots[key] = {
                        "used": used_now,
                        "resets_at": resets_now,
                        "ts": now_val,
                        "last_ts": now_val,
                        "running_jobs": running_now,
                        "accum_agent_hours": 0.0,
                        "tainted": taint_now,
                    }
                    continue

                if elapsed_sec < (MIN_ELAPSED_SEC - 1e-6):
                    # Short poll (< 120s): keep baseline snapshot intact and continue integrating agent_hours
                    results[pool_id][w_name] = {
                        "recorded": False,
                        "rejected": 1,
                        "reason": "elapsed_out_of_range",
                        "pool_id": pool_id,
                        "window": w_name,
                        "elapsed_sec": round(elapsed_sec, 3),
                        "agent_hours": round(accum_ah, 6),
                    }
                    continue

                mean_jobs = (accum_ah / (elapsed_sec / 3600.0)) if elapsed_sec > 0 else 0.0
                res = self.record_sample(
                    pool_id=pool_id,
                    window=w_name,
                    used_prev=float(prev["used"]),
                    used_now=used_now,
                    elapsed_sec=elapsed_sec,
                    agent_hours=accum_ah,
                    running_jobs=mean_jobs,
                    resets_at_prev=r_prev,
                    resets_at_now=resets_now,
                    confidence=conf,
                    recorded_at=recorded_iso,
                )
                results[pool_id][w_name] = res
                if res.get("recorded") or accum_ah <= 1e-9:
                    # Advance baseline after recording a valid sample or after an idle >=2m window
                    self._prev_snapshots[key] = {
                        "used": used_now,
                        "resets_at": resets_now,
                        "ts": now_val,
                        "last_ts": now_val,
                        "running_jobs": running_now,
                        "accum_agent_hours": 0.0,
                        "tainted": taint_now,
                    }
        return results

    def get_pool_window_estimate(self, pool_id: str, window: str) -> Dict[str, Any]:
        """
        Return `{b, source, samples, last_update, clamped}` for `(pool_id, window)`.
        Until `samples >= 5`, returns the `QuotaCostLearner` prior with `source="prior"`.
        """
        w_norm = normalize_window_name(window)
        samples = 0
        last_ewma: Optional[float] = None
        last_update: Optional[str] = None
        try:
            if not self._marker_ok():
                raise RuntimeError("burn schema marker missing; ignoring stored samples")
            with sqlite3.connect(str(self.db_path)) as conn:
                row = conn.execute(
                    """
                    SELECT COUNT(*),
                           (SELECT ewma_burn_rate FROM burn_rate_samples WHERE pool_id = ? AND window_name = ? ORDER BY id DESC LIMIT 1),
                           (SELECT recorded_at FROM burn_rate_samples WHERE pool_id = ? AND window_name = ? ORDER BY id DESC LIMIT 1)
                    FROM burn_rate_samples
                    WHERE pool_id = ? AND window_name = ?
                    """,
                    (pool_id, w_norm, pool_id, w_norm, pool_id, w_norm),
                ).fetchone()
                if row is not None:
                    samples = int(row[0] or 0)
                    last_ewma = float(row[1]) if row[1] is not None else None
                    last_update = str(row[2]) if row[2] is not None else None
        except Exception:
            pass

        prior = self.prior_burn_rate(pool_id, w_norm)
        clamped = False
        if samples >= MIN_SAMPLES_FOR_EWMA and last_ewma is not None:
            lo = max(MIN_BURN_RATE, prior / EWMA_CLAMP_FACTOR)
            hi = prior * EWMA_CLAMP_FACTOR
            raw_b = max(MIN_BURN_RATE, last_ewma)
            b_val = round(min(hi, max(lo, raw_b)), 6)
            clamped = abs(b_val - round(raw_b, 6)) > 1e-9
            source = "ewma"
        else:
            b_val = prior
            source = "prior"

        out = {
            "b": b_val,
            "source": source,
            "samples": samples,
            "last_update": last_update,
            "clamped": clamped,
        }
        return out

    def estimate_burn_rate(self, pool_id: str) -> Dict[str, Any]:
        """
        Return combined burn-rate estimate for `pool_id` across both windows:
        `{pool_id, b_5h, b_wk, source, sample_count, samples, last_update, five_hour, weekly}`.
        """
        e5 = self.get_pool_window_estimate(pool_id, "five_hour")
        ew = self.get_pool_window_estimate(pool_id, "weekly")
        sample_count = max(int(e5["samples"]), int(ew["samples"]))
        source = "ewma" if (e5["source"] == "ewma" or ew["source"] == "ewma") else "prior"
        last_update = e5.get("last_update") or ew.get("last_update")
        return {
            "pool_id": pool_id,
            "b_5h": float(e5["b"]),
            "b_wk": float(ew["b"]),
            "source": source,
            "sample_count": sample_count,
            "samples": sample_count,
            "last_update": last_update,
            "five_hour": e5,
            "weekly": ew,
        }

    def get_all_estimates(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Return `{pool_id: {window: {b, source, samples, last_update}}}` across all 4 pools."""
        return {
            pool_id: {
                w_name: self.get_pool_window_estimate(pool_id, w_name)
                for w_name in WINDOWS
            }
            for pool_id in POOLS
        }


def _normalize_burn_estimates(
    burn_estimates: Optional[Dict[str, Any]],
    burn_learner: Optional[BurnRateLearner] = None,
    cost_learner: Optional[Any] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Normalize caller-supplied `burn_estimates` or learner output into `{pool: {window: {b, source, samples, last_update}}}`."""
    base = (
        burn_learner.get_all_estimates()
        if burn_learner is not None
        else {
            pid: {
                w: {
                    "b": compute_prior_burn_rate(pid, w, cost_learner=cost_learner),
                    "source": "prior",
                    "samples": 0,
                    "last_update": None,
                }
                for w in WINDOWS
            }
            for pid in POOLS
        }
    )
    if not isinstance(burn_estimates, dict):
        return base

    for pid in POOLS:
        p_in = burn_estimates.get(pid)
        if not isinstance(p_in, dict):
            continue
        for w in WINDOWS:
            w_in = p_in.get(w)
            if w_in is None and w == "five_hour":
                w_in = p_in.get("5h") if p_in.get("5h") is not None else p_in.get("b_5h")
            if w_in is None and w == "weekly":
                w_in = (
                    p_in.get("7d")
                    if p_in.get("7d") is not None
                    else (p_in.get("b_wk") if p_in.get("b_wk") is not None else p_in.get("b_weekly"))
                )
            if isinstance(w_in, (int, float)) and not isinstance(w_in, bool):
                base[pid][w] = {
                    "b": max(MIN_BURN_RATE, float(w_in)),
                    "source": str(p_in.get("source") or "custom"),
                    "samples": int(p_in.get("samples", p_in.get("sample_count", MIN_SAMPLES_FOR_EWMA))),
                    "last_update": p_in.get("last_update"),
                }
            elif isinstance(w_in, dict):
                b_val = w_in.get("b", base[pid][w]["b"])
                base[pid][w] = {
                    "b": max(MIN_BURN_RATE, float(b_val)),
                    "source": str(w_in.get("source") or base[pid][w]["source"]),
                    "samples": int(w_in.get("samples", w_in.get("sample_count", base[pid][w]["samples"]))),
                    "last_update": w_in.get("last_update", base[pid][w].get("last_update")),
                }
    return base


def _extract_window_metrics(pool_state: Dict[str, Any], window: str) -> Tuple[Optional[float], Optional[float], float]:
    """
    Extract `(remaining_fraction, usable_fraction, T_hours)` from a pool state dict for `window`.
    Supports both full `build_pool_states` dicts and compact unit-test fixtures.
    """
    w_norm = normalize_window_name(window)
    w_obj = pool_state.get(w_norm)
    if not isinstance(w_obj, dict):
        alt_key = "5h" if w_norm == "five_hour" else "7d"
        w_obj = pool_state.get(alt_key) if isinstance(pool_state.get(alt_key), dict) else {}

    rem = w_obj.get("remaining_fraction")
    if rem is None:
        rem = w_obj.get("remaining")

    # Usable fraction after existing exclusion/reserve rules:
    # check explicit `usable` first; when Codex routine coding is blocked by reserve protection,
    # use `usable_after_orch_reserve`, otherwise `usable_before_reset`, then `usable_after_orch_reserve`, then `remaining`.
    usable = w_obj.get("usable")
    if usable is None:
        if pool_state.get("routine_coding_allowed_on_codex") is False and w_obj.get("usable_after_orch_reserve") is not None:
            usable = w_obj.get("usable_after_orch_reserve")
        else:
            usable = w_obj.get("usable_before_reset")
            if usable is None:
                usable = w_obj.get("usable_after_orch_reserve")
            if usable is None and rem is not None:
                usable = rem

    if rem is None and usable is not None:
        rem = usable

    rem_f = float(rem) if rem is not None else None
    usable_f = float(usable) if usable is not None else None

    if rem_f is not None and "exclude_at_used" in pool_state:
        min_rem = round(1.0 - float(pool_state["exclude_at_used"]), 6)
        if rem_f <= min_rem:
            usable_f = 0.0

    if w_obj.get("reset_hours") is not None:
        t_hours = float(w_obj["reset_hours"])
    elif w_obj.get("T_hours") is not None:
        t_hours = float(w_obj["T_hours"])
    else:
        t_sec = w_obj.get("reset_in_sec")
        if t_sec is None:
            t_sec = w_obj.get("time_until_reset_sec")
        if t_sec is not None:
            t_hours = max(0.0, float(t_sec)) / 3600.0
        else:
            t_hours = WINDOW_LENGTH_HOURS[w_norm]

    return rem_f, usable_f, t_hours


def compute_pool_capacity(
    pool_state: Dict[str, Any],
    burn_pool: Dict[str, Any],
    operating_mode: str = "endurance",
) -> Dict[str, Any]:
    """
    Compute pool capacity `A_pool = min over windows of usable_w / (b_w * max(T_w_hours, floor_w))`
    with floors `five_hour -> 0.25h`, `weekly -> 1.0h`.
    Unknown/UNAVAILABLE telemetry or `usable <= 0` yields `A_pool = 0`.
    """
    _ = operating_mode
    if not isinstance(pool_state, dict) or not pool_state:
        return {
            "A_pool": 0.0,
            "A_five_hour": 0.0,
            "A_weekly": 0.0,
            "c_5h": 0.0,
            "c_wk": 0.0,
            "capacity_tasks_per_hour": 0.0,
            "limiting_window": "unknown",
        }

    conf = str(pool_state.get("confidence") or "VERIFIED").upper().strip()
    if conf in UNKNOWN_CONFIDENCE_STATES:
        return {
            "A_pool": 0.0,
            "A_five_hour": 0.0,
            "A_weekly": 0.0,
            "c_5h": 0.0,
            "c_wk": 0.0,
            "capacity_tasks_per_hour": 0.0,
            "limiting_window": "unknown",
        }

    _r5, u5, t5_h = _extract_window_metrics(pool_state, "five_hour")
    _rw, uw, tw_h = _extract_window_metrics(pool_state, "weekly")

    if u5 is None or uw is None:
        return {
            "A_pool": 0.0,
            "A_five_hour": 0.0,
            "A_weekly": 0.0,
            "c_5h": 0.0,
            "c_wk": 0.0,
            "capacity_tasks_per_hour": 0.0,
            "limiting_window": "unknown",
        }

    if isinstance(burn_pool.get("five_hour"), dict):
        b5_raw = burn_pool["five_hour"].get("b", 0.30)
    else:
        b5_raw = burn_pool.get("b_5h", burn_pool.get("five_hour", 0.30))

    if isinstance(burn_pool.get("weekly"), dict):
        bw_raw = burn_pool["weekly"].get("b", 0.10)
    else:
        bw_raw = burn_pool.get("b_wk", burn_pool.get("b_weekly", burn_pool.get("weekly", 0.10)))

    b5 = max(MIN_BURN_RATE, float(b5_raw))
    bw = max(MIN_BURN_RATE, float(bw_raw))

    eff_t5 = max(t5_h, WINDOW_FLOORS_HOURS["five_hour"])
    eff_tw = max(tw_h, WINDOW_FLOORS_HOURS["weekly"])

    a5 = (u5 / (b5 * eff_t5)) if u5 > 0.0 else 0.0
    aw = (uw / (bw * eff_tw)) if uw > 0.0 else 0.0
    if u5 <= 0.0 or uw <= 0.0:
        a_pool = 0.0
        limiting = "five_hour" if u5 <= 0.0 else "weekly"
    elif _r5 is not None and _r5 <= 0.15:
        # 5-hour window acts as a brake only when running low (<= 15% remaining)
        a_pool = min(a5, aw)
        limiting = "five_hour" if a5 <= aw else "weekly"
    else:
        # Weekly Kalan % / Kalan Zaman is the primary capacity driver when 5h is healthy (> 15%)
        a_pool = aw
        limiting = "weekly"

    return {
        "A_pool": a_pool,
        "A_five_hour": a5,
        "A_weekly": aw,
        "c_5h": a5,
        "c_wk": aw,
        "capacity_tasks_per_hour": a_pool,
        "limiting_window": limiting,
    }


def compute_pool_sigma(pool_state: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute slack per pool:
    Weekly `remaining_wk / max(0.02, T_wk / 168h)` is the primary slack ratio;
    the 5-hour window only constrains `sigma` when 5h quota is low (`<= 15%` remaining).
    """
    if not isinstance(pool_state, dict) or not pool_state:
        return {"sigma": 0.0, "sigma_five_hour": 0.0, "sigma_weekly": 0.0}

    conf = str(pool_state.get("confidence") or "VERIFIED").upper().strip()
    if conf in UNKNOWN_CONFIDENCE_STATES:
        return {"sigma": 0.0, "sigma_five_hour": 0.0, "sigma_weekly": 0.0}

    r5, _u5, t5_h = _extract_window_metrics(pool_state, "five_hour")
    rw, _uw, tw_h = _extract_window_metrics(pool_state, "weekly")
    if r5 is None or rw is None:
        return {"sigma": 0.0, "sigma_five_hour": 0.0, "sigma_weekly": 0.0}

    s5 = max(0.0, r5) / max(0.02, t5_h / WINDOW_LENGTH_HOURS["five_hour"])
    sw = max(0.0, rw) / max(0.02, tw_h / WINDOW_LENGTH_HOURS["weekly"])
    return {
        "sigma": min(s5, sw),
        "sigma_five_hour": s5,
        "sigma_weekly": sw,
    }


def classify_sigma_band(
    sigma: float,
    operating_mode: str = "endurance",
    policy_cap: int = 6,
) -> Dict[str, Any]:
    """
    Map provider `sigma` and `operating_mode` to `(band, max_workers, effort_policy)`.
    - `endurance` thresholds: `normal >= 1.0`, `economy >= 0.5`, `saver < 0.5`
    - `sprint` thresholds: `normal >= 0.7`, `economy >= 0.35`, `saver < 0.35`
    The dynamic `max_workers` recommendation never exceeds `policy_cap`.
    """
    mode = str(operating_mode or "endurance").lower().strip()
    normal_thresh = 0.7 if mode == "sprint" else 1.0
    economy_thresh = 0.35 if mode == "sprint" else 0.5
    raw_cap = max(0, int(policy_cap))

    if sigma >= normal_thresh:
        band = "normal"
        max_workers = raw_cap
        effort_policy = "normal"
    elif sigma >= economy_thresh:
        band = "economy"
        max_workers = int(math.ceil(raw_cap / 2.0)) if raw_cap > 0 else 0
        effort_policy = "economy"
    else:
        band = "saver"
        max_workers = 1 if raw_cap > 0 else 0
        effort_policy = "saver"

    max_workers = min(raw_cap, max(0, max_workers))
    return {
        "band": band,
        "ceiling": raw_cap,
        "max_workers": max_workers,
        "effort_policy": effort_policy,
        "normal_threshold": normal_thresh,
        "economy_threshold": economy_thresh,
    }


def is_critical_package(
    floor_tier: str = "MID",
    kind: str = "implement",
    ambiguity: str = "low",
    blast_radius: str = "local",
    critical: bool = False,
    escalation_reason: Optional[str] = None,
) -> bool:
    """
    True when a work package is critical:
    - explicitly flagged (`critical=True` or non-empty `escalation_reason`), OR
    - capability floor is `HIGH` and `kind` is `design` or `review` of high-risk work.
    """
    if critical or (escalation_reason and str(escalation_reason).strip()):
        return True
    ft = str(floor_tier or "").upper().strip()
    k = str(kind or "").lower().strip()
    amb = str(ambiguity or "low").lower().strip()
    br = str(blast_radius or "local").lower().strip()
    if ft == "HIGH":
        if k in {"design", "review-of-high-risk", "review_of_high_risk"}:
            return True
        if k == "review" and (amb != "low" or br != "local"):
            return True
    return False


def apply_band_effort(
    base_effort: str,
    band: Any,
    floor_tier: str = "MID",
    kind: str = "implement",
    ambiguity: str = "low",
    blast_radius: str = "local",
    critical: bool = False,
    escalation_reason: Optional[str] = None,
    explicit_effort: Optional[str] = None,
) -> str:
    """
    Adjust reasoning effort for a package according to its provider's sigma `band`:
    - Explicit effort requests always win.
    - Critical packages keep their effort in every band.
    - `normal`: `base_effort` unchanged.
    - `economy`: non-critical packages one effort step lower (`high -> medium`, `medium -> low`, `low -> low`).
    - `saver` (or alias `critical`): non-critical packages `low`.
    """
    exp = str(explicit_effort or "").lower().strip()
    if exp in {"low", "medium", "high"}:
        return exp
    eff = str(base_effort or "medium").lower().strip()
    if eff not in {"low", "medium", "high"}:
        eff = "medium"
    if is_critical_package(
        floor_tier=floor_tier,
        kind=kind,
        ambiguity=ambiguity,
        blast_radius=blast_radius,
        critical=critical,
        escalation_reason=escalation_reason,
    ):
        return eff

    b_raw = band.get("band") if isinstance(band, dict) else band
    b = str(b_raw or "normal").lower().strip()
    if b == "normal":
        return eff
    if b == "economy":
        return {"high": "medium", "medium": "low", "low": "low"}[eff]
    if b in {"saver", "critical"}:
        return "low"
    if b == "surge" and str(floor_tier or "").upper().strip() == "HIGH" and eff == "medium":
        return "high"
    return eff


def compute_actual_share_24h(
    dispatch_history: Optional[List[Dict[str, Any]]] = None,
    db_path: Optional[Path] = None,
    now_ts: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute each provider's share of dispatched/claimed jobs over the last 24 hours
    from `dispatch_history` and the SQLite tables (`backlog_jobs`, `task_cost_observations`).
    """
    now_val = float(now_ts if now_ts is not None else time.time())
    cutoff_ts = now_val - 86400.0
    counts: Dict[str, int] = {p: 0 for p in PROVIDERS}
    seen_ids: set[str] = set()

    def _parse_iso(ts_raw: Any) -> Optional[float]:
        if ts_raw is None:
            return None
        if isinstance(ts_raw, (int, float)) and not isinstance(ts_raw, bool):
            return float(ts_raw)
        s = str(ts_raw).strip()
        if not s:
            return None
        try:
            dt = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt.timestamp()
        except Exception:
            return None

    for idx, entry in enumerate(dispatch_history or []):
        if not isinstance(entry, dict):
            continue
        ts = _parse_iso(entry.get("timestamp"))
        if ts is not None and ts < cutoff_ts:
            continue
        routing = entry.get("routing") or {}
        ar = routing.get("assignment_record") or {}
        prov = str(entry.get("provider") or ar.get("provider") or "").lower().strip()
        if prov not in counts:
            continue
        jid = str(ar.get("task_id") or entry.get("id") or f"hist-{idx}").strip()
        if jid in seen_ids and jid != "WP-PAUSED":
            continue
        seen_ids.add(jid)
        counts[prov] += 1

    if db_path is not None and Path(db_path).exists():
        try:
            with sqlite3.connect(str(db_path)) as conn:
                tables = {
                    str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                }
                if "backlog_jobs" in tables:
                    for jid, prov, started_at in conn.execute(
                        """
                        SELECT job_id, assigned_provider, started_at
                        FROM backlog_jobs
                        WHERE assigned_provider IS NOT NULL AND started_at IS NOT NULL AND started_at >= ?
                        """,
                        (cutoff_ts,),
                    ).fetchall():
                        p_str = str(prov or "").lower().strip()
                        j_str = str(jid or "").strip()
                        if p_str in counts and j_str and j_str not in seen_ids:
                            seen_ids.add(j_str)
                            counts[p_str] += 1
                if "task_cost_observations" in tables:
                    for tid, prov, started_at, ended_at in conn.execute(
                        """
                        SELECT task_id, provider, started_at, ended_at
                        FROM task_cost_observations
                        WHERE provider IS NOT NULL
                        ORDER BY rowid DESC
                        LIMIT 200
                        """
                    ).fetchall():
                        p_str = str(prov or "").lower().strip()
                        t_str = str(tid or "").strip()
                        if p_str not in counts or (t_str and t_str in seen_ids):
                            continue
                        ts = _parse_iso(ended_at) or _parse_iso(started_at)
                        if ts is not None and ts >= cutoff_ts:
                            if t_str:
                                seen_ids.add(t_str)
                            counts[p_str] += 1
        except Exception:
            pass

    total = sum(counts.values())
    if total <= 0:
        return {p: 0.0 for p in PROVIDERS}
    return {p: round(counts[p] / float(total), 4) for p in PROVIDERS}


def compute_allocation(
    pool_states: Dict[str, Any],
    burn_estimates: Optional[Dict[str, Any]] = None,
    sliders: Optional[Dict[str, int]] = None,
    operating_mode: str = "endurance",
    policy: Optional[Dict[str, Any]] = None,
    actual_share_24h: Optional[Dict[str, float]] = None,
    burn_learner: Optional[BurnRateLearner] = None,
    cost_learner: Optional[Any] = None,
    computed_at: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute burn-rate-normalised pool/provider capacities, slider-weighted target shares `pi_provider`,
    pool/provider slack `sigma`, sigma bands (`normal` | `economy` | `saver`), dynamic `max_workers`,
    and effort policy across `agy`, `claude`, and `codex`.
    """
    mode = str(operating_mode or "endurance").lower().strip()
    if mode not in {"sprint", "endurance"}:
        mode = "endurance"

    weights = coerce_preference_weights(sliders) if sliders is not None else dict(DEFAULT_PREFERENCE_WEIGHTS)
    norm_weights = normalise_sliders(weights)
    burn_map = _normalize_burn_estimates(burn_estimates, burn_learner=burn_learner, cost_learner=cost_learner)
    actual_24h = {p: round(float((actual_share_24h or {}).get(p, 0.0)), 4) for p in PROVIDERS}
    max_per_prov_cfg = dict(DEFAULT_MAX_PER_PROVIDER)
    if isinstance(policy, dict) and isinstance(policy.get("max_per_provider"), dict):
        for p in PROVIDERS:
            if policy["max_per_provider"].get(p) is not None:
                try:
                    max_per_prov_cfg[p] = max(0, int(policy["max_per_provider"][p]))
                except (TypeError, ValueError):
                    pass

    # Allow pool_states keyed by provider in minimal test fixtures as well as canonical 4-pool keys
    resolved_pools: Dict[str, Dict[str, Any]] = {}
    for pid in POOLS:
        if isinstance(pool_states, dict) and pid in pool_states and isinstance(pool_states[pid], dict):
            resolved_pools[pid] = pool_states[pid]
        else:
            prov = POOL_TO_PROVIDER[pid]
            if (
                isinstance(pool_states, dict)
                and prov in pool_states
                and isinstance(pool_states[prov], dict)
                and pid in {"agy_gemini", "claude_shared", "codex_shared"}
            ):
                resolved_pools[pid] = pool_states[prov]
            else:
                resolved_pools[pid] = {}

    pool_metrics: Dict[str, Dict[str, Any]] = {}
    for pid in POOLS:
        ps = resolved_pools[pid]
        prov = POOL_TO_PROVIDER[pid]
        cap_info = compute_pool_capacity(ps, burn_map[pid], operating_mode=mode)
        sig_info = compute_pool_sigma(ps)
        p_cap = int(ps.get("max_concurrent", max_per_prov_cfg.get(prov, 6)))
        pool_band_info = classify_sigma_band(sig_info["sigma"], operating_mode=mode, policy_cap=p_cap)
        stranded_frac = compute_pool_stranded_quota_fraction(ps)
        pool_metrics[pid] = {
            "pool_id": pid,
            "provider": prov,
            "A_pool": round(cap_info["A_pool"], 6),
            "A_five_hour": round(cap_info["A_five_hour"], 6),
            "A_weekly": round(cap_info["A_weekly"], 6),
            "c_5h": round(cap_info["A_five_hour"], 6),
            "c_wk": round(cap_info["A_weekly"], 6),
            "capacity_tasks_per_hour": round(cap_info["A_pool"], 6),
            "limiting_window": cap_info["limiting_window"],
            "sigma": round(sig_info["sigma"], 4),
            "sigma_five_hour": round(sig_info["sigma_five_hour"], 4),
            "sigma_5h": round(sig_info["sigma_five_hour"], 4),
            "sigma_weekly": round(sig_info["sigma_weekly"], 4),
            "stranded_quota_fraction": round(stranded_frac, 4),
            "band": pool_band_info["band"],
            "burn_rate": dict(burn_map[pid]),
        }

    provider_capacities: Dict[str, float] = {}
    provider_sigmas: Dict[str, float] = {}
    provider_stranded: Dict[str, float] = {}
    weighted_capacities: Dict[str, float] = {}

    for prov in PROVIDERS:
        p_pools = PROVIDER_POOLS[prov]
        a_prov = sum(pool_metrics[pid]["A_pool"] for pid in p_pools)
        # Provider sigma = max over its pools (for AGY, max over pools present in pool_states)
        present_sigmas = [
            pool_metrics[pid]["sigma"]
            for pid in p_pools
            if resolved_pools.get(pid)
        ]
        present_stranded = [
            pool_metrics[pid]["stranded_quota_fraction"]
            for pid in p_pools
            if resolved_pools.get(pid)
        ]
        sig_prov = max(present_sigmas) if present_sigmas else 0.0
        str_prov = max(present_stranded) if present_stranded else 0.0
        provider_capacities[prov] = a_prov
        provider_sigmas[prov] = sig_prov
        provider_stranded[prov] = str_prov
        s_prov = weights[prov]
        weighted_capacities[prov] = (float(s_prov) * a_prov) if s_prov > 0 else 0.0

    total_weighted = sum(weighted_capacities.values())
    now_iso = computed_at or datetime.datetime.now(datetime.timezone.utc).isoformat()

    providers_out: Dict[str, Dict[str, Any]] = {}
    for prov in PROVIDERS:
        s_prov = weights[prov]
        a_prov = provider_capacities[prov]
        sig_prov = provider_sigmas[prov]
        str_prov = provider_stranded[prov]
        pi_prov = (weighted_capacities[prov] / total_weighted) if total_weighted > 0.0 else 0.0

        # Resolve policy cap for provider (can also come from pool_states[pid]["max_per_provider"])
        pol_cap = max_per_prov_cfg[prov]
        if not (isinstance(policy, dict) and isinstance(policy.get("max_per_provider"), dict) and policy["max_per_provider"].get(prov) is not None):
            for pid in PROVIDER_POOLS[prov]:
                ps_obj = resolved_pools.get(pid) or {}
                ps_cap = ps_obj.get("policy_max_per_provider")
                if ps_cap is None:
                    ps_cap = ps_obj.get("max_per_provider")
                if ps_cap is not None:
                    try:
                        pol_cap = max(0, int(ps_cap))
                        break
                    except (TypeError, ValueError):
                        pass

        band_info = classify_sigma_band(sig_prov, operating_mode=mode, policy_cap=pol_cap)
        burn_prov = {
            pid: {
                w_name: dict(burn_map[pid][w_name])
                for w_name in WINDOWS
            }
            for pid in PROVIDER_POOLS[prov]
        }
        providers_out[prov] = {
            "slider": s_prov,
            "capacity_tasks_per_hour": round(a_prov, 6),
            "target_share": round(pi_prov, 4),
            "actual_share_24h": actual_24h[prov],
            "sigma": round(sig_prov, 4),
            "stranded_quota_fraction": round(str_prov, 4),
            "band": band_info["band"],
            "ceiling": pol_cap,
            "max_workers": band_info["max_workers"],
            "summary": f"ceiling {pol_cap} · now {band_info['max_workers']} ({band_info['band']} band)",
            "effort_policy": band_info["effort_policy"],
            "burn": burn_prov,
        }

    result: Dict[str, Any] = {
        "computed_at": now_iso,
        "operating_mode": mode,
        "sliders": dict(weights),
        "preference_weights": dict(weights),
        "normalised_weights": dict(norm_weights),
        "providers": providers_out,
        "pools": pool_metrics,
        "unattributed_running": float(getattr(burn_learner, "unattributed_running", 0.0) or 0.0),
        "skipped_unattributed_intervals": dict(
            getattr(burn_learner, "skipped_unattributed_intervals", None) or {p: 0 for p in PROVIDERS}
        ),
        "burn_schema_ok": bool(burn_learner._marker_ok()) if burn_learner is not None else True,
    }
    # Expose per-provider keys directly at top level of `allocation` as well as under `providers`
    for prov in PROVIDERS:
        result[prov] = providers_out[prov]
    return result


def compute_target_shares(
    sliders: Optional[Dict[str, Any]],
    capacities: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute per-provider target shares `pi_p = s_p * A_p / sum_q s_q * A_q`
    (`0.0` when `s_p <= 0` or the denominator is `0.0`).
    """
    weights = coerce_preference_weights(sliders)
    caps = capacities if isinstance(capacities, dict) else {}
    weighted: Dict[str, float] = {}
    for p in PROVIDERS:
        s_p = int(weights.get(p, 0))
        try:
            a_p = max(0.0, float(caps.get(p, 0.0) or 0.0))
        except (TypeError, ValueError):
            a_p = 0.0
        weighted[p] = (float(s_p) * a_p) if s_p > 0 else 0.0
    total = sum(weighted.values())
    return {
        p: (weighted[p] / total if total > 0.0 else 0.0)
        for p in PROVIDERS
    }


def _extract_capacities_from_pool_states(
    pool_states: Optional[Dict[str, Any]],
) -> Optional[Dict[str, float]]:
    """
    Extract per-provider capacities `A_p` from `pool_states` when pool/provider
    capacity fields (`provider_capacity_tasks_per_hour`, `capacity_tasks_per_hour`, or `A_pool`)
    are present. Returns `None` if no capacity fields are present.
    """
    if not isinstance(pool_states, dict) or not pool_states:
        return None
    found = False
    caps: Dict[str, float] = {p: 0.0 for p in PROVIDERS}
    prov_direct: Dict[str, float] = {}
    for key, val in pool_states.items():
        if not isinstance(val, dict):
            continue
        k_str = str(key or "").lower().strip()
        if k_str in POOL_TO_PROVIDER:
            prov = POOL_TO_PROVIDER[k_str]
        elif k_str in caps:
            prov = k_str
        else:
            continue
        if val.get("provider_capacity_tasks_per_hour") is not None:
            try:
                prov_direct[prov] = max(0.0, float(val["provider_capacity_tasks_per_hour"]))
                found = True
            except (TypeError, ValueError):
                pass
        cap_raw = val.get("capacity_tasks_per_hour")
        if cap_raw is None:
            cap_raw = val.get("A_pool")
        if cap_raw is not None:
            try:
                caps[prov] += max(0.0, float(cap_raw))
                found = True
            except (TypeError, ValueError):
                pass
    if not found:
        return None
    for prov, val in prov_direct.items():
        caps[prov] = val
    return caps


def _extract_window_reset_hours(w_dict: Dict[str, Any], default_hours: float) -> float:
    """Extract time until reset in hours from a pool window dict."""
    if not isinstance(w_dict, dict):
        return float(default_hours)
    for k_hr in ("reset_hours", "T_hours", "horizon_hours"):
        if w_dict.get(k_hr) is not None:
            try:
                return max(0.0, float(w_dict[k_hr]))
            except (TypeError, ValueError):
                pass
    for k_sec in ("reset_in_sec", "time_until_reset_sec"):
        if w_dict.get(k_sec) is not None:
            try:
                return max(0.0, float(w_dict[k_sec]) / 3600.0)
            except (TypeError, ValueError):
                pass
    return float(default_hours)


def compute_pool_harvest_boost(
    pool_state: Optional[Dict[str, Any]],
    pool_id: Optional[str] = None,
) -> float:
    """
    Compute the Amendment 2 / Amendment 3 pre-reset SWRR `harvest_boost` (`1.0` or `0.0`) for a single pool:
    `harvest_boost = 1.0` when the pool has verified telemetry and:
      (a) weekly reset in `<= 12 h` and at least 1 task fittable before that reset
          (`tasks_fittable_before_weekly_reset >= 1`), OR
      (b) 5-hour reset in `<= 1.5 h`, at least 1 task fittable before it (using the candidate's
          predicted 5-hour cost), and the weekly window not scarce (`weekly opportunity cost phi < 0.75`), OR
      (c) active Codex banked-reset harvest overlay (`banked_harvest_overlay_active == True`,
          `routine_coding_allowed_on_codex` not False, and at least 1 task fittable before reset);
    otherwise `0.0`. Same value in `sprint` and `endurance`.
    """
    if not isinstance(pool_state, dict) or not pool_state:
        return 0.0

    conf = str(pool_state.get("confidence") or "VERIFIED").upper().strip()
    if conf in UNKNOWN_CONFIDENCE_STATES:
        return 0.0

    if pool_state.get("harvest_boost") is not None:
        try:
            return 1.0 if float(pool_state["harvest_boost"]) > 0.0 else 0.0
        except (TypeError, ValueError):
            pass

    pid = str(pool_id or pool_state.get("pool_id") or "").lower().strip()
    m5, mw = _FALLBACK_POOL_MULTIPLIERS.get(pid, (1.0, 1.0))

    w5 = pool_state.get("five_hour") if isinstance(pool_state.get("five_hour"), dict) else {}
    ww = pool_state.get("weekly") if isinstance(pool_state.get("weekly"), dict) else {}
    pred_cost_dict = (
        pool_state.get("predicted_cost")
        if isinstance(pool_state.get("predicted_cost"), dict)
        else (
            pool_state.get("cost_prediction")
            if isinstance(pool_state.get("cost_prediction"), dict)
            else {}
        )
    )

    # Resolve candidate predicted weekly cost (falling back to cold-start prior only if no candidate cost is present)
    cw_raw = ww.get("predicted_weekly_cost")
    if cw_raw is None:
        cw_raw = ww.get("weekly_mean")
    if cw_raw is None:
        cw_raw = pool_state.get("predicted_weekly_cost")
    if cw_raw is None:
        cw_raw = pred_cost_dict.get("weekly_mean")
    if cw_raw is None:
        cw_raw = pred_cost_dict.get("predicted_weekly_cost")
    try:
        cw_mean = max(0.002, float(cw_raw)) if cw_raw is not None else max(0.002, 0.014 * mw)
    except (TypeError, ValueError):
        cw_mean = max(0.002, 0.014 * mw)

    # Resolve candidate predicted 5-hour cost (same source as weekly fittability; fallback only when absent)
    c5_raw = w5.get("predicted_5h_cost")
    if c5_raw is None:
        c5_raw = w5.get("five_hour_mean")
    if c5_raw is None:
        c5_raw = pool_state.get("predicted_5h_cost")
    if c5_raw is None:
        c5_raw = pred_cost_dict.get("five_hour_mean")
    if c5_raw is None:
        c5_raw = pred_cost_dict.get("predicted_5h_cost")
    try:
        c5_mean = max(0.005, float(c5_raw)) if c5_raw is not None else max(0.005, 0.045 * m5)
    except (TypeError, ValueError):
        c5_mean = max(0.005, 0.045 * m5)

    tw_hours = _extract_window_reset_hours(ww, 168.0)
    t5_hours = _extract_window_reset_hours(w5, 5.0)

    # Resolve tasks_fittable_before_weekly_reset
    fittable_wk_raw = ww.get("tasks_fittable_before_weekly_reset")
    if fittable_wk_raw is None:
        fittable_wk_raw = pool_state.get("tasks_fittable_before_weekly_reset")
    if fittable_wk_raw is not None:
        try:
            tasks_fittable_wk = max(0.0, float(fittable_wk_raw))
        except (TypeError, ValueError):
            tasks_fittable_wk = 0.0
    else:
        usable_wk_raw = ww.get("usable_before_reset")
        if usable_wk_raw is None:
            usable_wk_raw = ww.get("usable")
        if usable_wk_raw is None:
            usable_wk_raw = ww.get("remaining_fraction")
        try:
            usable_wk = max(0.0, float(usable_wk_raw)) if usable_wk_raw is not None else 0.0
        except (TypeError, ValueError):
            usable_wk = 0.0
        tasks_fittable_wk = usable_wk / cw_mean

    # Resolve tasks_fittable_before_5h_reset (using candidate's predicted 5-hour cost per Amendment 3 item 4)
    fittable_5h_raw = w5.get("tasks_fittable_before_5h_reset")
    if fittable_5h_raw is None:
        fittable_5h_raw = w5.get("tasks_fittable_before_reset")
    if fittable_5h_raw is None:
        fittable_5h_raw = pool_state.get("tasks_fittable_before_5h_reset")
    if fittable_5h_raw is not None:
        try:
            tasks_fittable_5h = max(0.0, float(fittable_5h_raw))
        except (TypeError, ValueError):
            tasks_fittable_5h = 0.0
    else:
        usable_5h_raw = w5.get("usable_before_reset")
        if usable_5h_raw is None:
            usable_5h_raw = w5.get("usable")
        if usable_5h_raw is None:
            usable_5h_raw = w5.get("remaining_fraction")
        try:
            usable_5h = max(0.0, float(usable_5h_raw)) if usable_5h_raw is not None else 0.0
        except (TypeError, ValueError):
            usable_5h = 0.0
        tasks_fittable_5h = usable_5h / c5_mean

    phi_wk_raw = ww.get("opportunity_cost_phi")
    if phi_wk_raw is None:
        phi_wk_raw = pool_state.get("opportunity_cost_phi", 1.0)
    try:
        phi_wk = float(phi_wk_raw)
    except (TypeError, ValueError):
        phi_wk = 1.0

    # Condition (a): weekly reset in <= 12 h and at least 1 task fittable before that reset
    if tw_hours <= 12.0 and tasks_fittable_wk >= 1.0:
        return 1.0

    # Condition (b): 5-hour reset in <= 1.5 h, at least 1 task fittable before it, and weekly not scarce (phi < 0.75)
    if t5_hours <= 1.5 and tasks_fittable_5h >= 1.0 and phi_wk < 0.75:
        return 1.0

    # Condition (c): active Codex banked-reset harvest overlay for that provider (Amendment 3 item 2)
    if (
        bool(pool_state.get("banked_harvest_overlay_active", False))
        and bool(pool_state.get("routine_coding_allowed_on_codex", True))
        and tasks_fittable_wk >= 1.0
    ):
        return 1.0

    return 0.0


def compute_provider_harvest_boosts(
    source: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Resolve per-provider Amendment 2 / Amendment 3 `harvest_boost_p` (`1.0` if any pool of `p` qualifies, else `0.0`)
    from a `pool_states` dict keyed by canonical pool IDs or provider IDs.
    """
    out: Dict[str, float] = {p: 0.0 for p in PROVIDERS}
    if not isinstance(source, dict) or not source:
        return out

    for key, val in source.items():
        k_str = str(key or "").lower().strip()
        if k_str in POOL_TO_PROVIDER:
            prov = POOL_TO_PROVIDER[k_str]
            pid = k_str
        elif k_str in out:
            prov = k_str
            pid = PROVIDER_POOLS[prov][0]
        else:
            continue

        if isinstance(val, (int, float)):
            boost = 1.0 if float(val) > 0.0 else 0.0
        elif isinstance(val, dict):
            boost = compute_pool_harvest_boost(val, pool_id=pid)
        else:
            boost = 0.0

        if boost > out[prov]:
            out[prov] = 1.0

    return out


def compute_pool_stranded_quota_fraction(pool_state: Optional[Dict[str, Any]]) -> float:
    """
    Compute the effective pre-reset `stranded_quota_fraction` in `[0.0, 1.0]` for a single pool.
    """
    if not isinstance(pool_state, dict) or not pool_state:
        return 0.0

    conf = str(pool_state.get("confidence") or "VERIFIED").upper().strip()
    if conf in UNKNOWN_CONFIDENCE_STATES:
        return 0.0

    if pool_state.get("stranded_quota_fraction") is not None:
        try:
            return min(1.0, max(0.0, float(pool_state["stranded_quota_fraction"])))
        except (TypeError, ValueError):
            pass

    w5 = pool_state.get("five_hour") if isinstance(pool_state.get("five_hour"), dict) else {}
    ww = pool_state.get("weekly") if isinstance(pool_state.get("weekly"), dict) else {}

    tw_hours = _extract_window_reset_hours(ww, 168.0)
    t5_hours = _extract_window_reset_hours(w5, 5.0)
    phi_wk = float(ww.get("opportunity_cost_phi", 1.0))

    s_wk = float(ww.get("stranded_quota_fraction") or 0.0) if tw_hours <= 12.0 else 0.0
    s_5h = float(w5.get("stranded_quota_fraction") or 0.0) if (t5_hours <= 1.5 and phi_wk < 0.75) else 0.0

    s_banked = 0.0
    if bool(pool_state.get("banked_harvest_overlay_active", False)) and bool(
        pool_state.get("routine_coding_allowed_on_codex", True)
    ):
        s_banked = max(
            float(w5.get("stranded_quota_fraction") or 0.0),
            float(pool_state.get("banked_harvest_allocation_boost") or 0.0),
        )

    return min(1.0, max(0.0, s_wk, s_5h, s_banked))


def compute_provider_stranded_fractions(
    source: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Resolve per-provider `max(stranded_quota_fraction over p's pools)` from a `pool_states` dict.
    """
    out: Dict[str, float] = {p: 0.0 for p in PROVIDERS}
    if not isinstance(source, dict) or not source:
        return out

    for key, val in source.items():
        k_str = str(key or "").lower().strip()
        if k_str in POOL_TO_PROVIDER:
            prov = POOL_TO_PROVIDER[k_str]
        elif k_str in out:
            prov = k_str
        else:
            continue

        if isinstance(val, (int, float)):
            frac = min(1.0, max(0.0, float(val)))
        elif isinstance(val, dict):
            frac = compute_pool_stranded_quota_fraction(val)
        else:
            frac = 0.0

        if frac > out[prov]:
            out[prov] = frac

    return out


def step_swrr(
    credits: Dict[str, float],
    target_shares: Dict[str, float],
    feasible_providers: Iterable[str],
    tie_break_scores: Optional[Dict[str, float]] = None,
    sliders: Optional[Dict[str, int]] = None,
    stranded_fractions: Optional[Dict[str, Any]] = None,
    pool_states: Optional[Dict[str, Any]] = None,
    operating_mode: str = "endurance",
) -> Tuple[Optional[str], Dict[str, float]]:
    """
    Execute one classic Smooth Weighted Round Robin (nginx SWRR) routing credit step
    (`docs/WP-ALLOCATION-CONTRACT.md` §3 + Amendments 3 and 4):
    - A provider with slider `0` or `target_share <= 0` is never chosen by SWRR.
    - H = feasible, slider > 0 providers whose harvest condition (a), (b) or (c) holds. If H is non-empty, SWRR runs
      over H only (only H's credits change); otherwise over all feasible, slider > 0 providers.
    - Weights are `pi_p` renormalised over the participating set (no boost term).
    - Add `w_p` to each participant's `C_p`, select the highest `C_p` (ties by `w_p`, `target_share`, candidate
      score) and subtract `W = sum(w_p)` from the winner. No clamping.
    - If only zero-share providers are feasible, returns `(None, new_credits)`.
    """
    feas_set = {str(p).lower().strip() for p in feasible_providers if p}
    feas = [p for p in PROVIDERS if p in feas_set]
    new_credits = coerce_credits(credits)
    if not feas:
        return None, new_credits

    pos_feas = [
        p
        for p in feas
        if float(target_shares.get(p, 0.0)) > 0.0
        and (sliders is None or int(sliders.get(p, 1)) > 0)
    ]
    scores = tie_break_scores or {}

    if not pos_feas:
        return None, new_credits

    prov_boosts = compute_provider_harvest_boosts(
        pool_states if pool_states is not None else stranded_fractions
    )
    if stranded_fractions is not None and pool_states is not None:
        override_boosts = compute_provider_harvest_boosts(stranded_fractions)
        for p in PROVIDERS:
            prov_boosts[p] = max(prov_boosts.get(p, 0.0), override_boosts.get(p, 0.0))

    # Amendment 4: harvesting providers (H) run classic SWRR among themselves only; otherwise all feasible ones do.
    harvesters = [p for p in pos_feas if float(prov_boosts.get(p, 0.0)) > 0.0]
    active = harvesters if harvesters else pos_feas
    total_pi = sum(float(target_shares[p]) for p in active)
    eff_weights: Dict[str, float] = {
        p: (float(target_shares[p]) / total_pi if total_pi > 0.0 else 0.0) for p in active
    }
    for p in active:
        new_credits[p] = round(new_credits[p] + eff_weights[p], 10)
    total_w = sum(eff_weights.values())
    chosen = max(
        active,
        key=lambda p: (
            round(new_credits[p], 9),
            round(eff_weights.get(p, 0.0), 9),
            float(target_shares.get(p, 0.0)),
            float(scores.get(p, 0.0)),
        ),
    )
    new_credits[chosen] = round(new_credits[chosen] - total_w, 10)
    return chosen, new_credits


_SWRR_LOCK = threading.RLock()  # re-entrant: orchestrator aliases it as _CREDIT_LOCK and nests calls
_SHARED_CREDITS: Dict[str, float] = reset_credits()
_PREFERENCE_GENERATION: int = 0
_PREFERENCE_WEIGHTS: Dict[str, int] = dict(DEFAULT_PREFERENCE_WEIGHTS)


def get_shared_preference_generation() -> int:
    """Return the current preference generation counter under `_SWRR_LOCK`."""
    with _SWRR_LOCK:
        return int(_PREFERENCE_GENERATION)


def get_shared_preference_weights() -> Dict[str, int]:
    """Return a copy of the active preference slider weights under `_SWRR_LOCK`."""
    with _SWRR_LOCK:
        return dict(_PREFERENCE_WEIGHTS)


def set_shared_preference_weights(
    weights: Dict[str, Any],
    reset_swrr: bool = True,
    bump_generation: bool = True,
) -> Dict[str, int]:
    """
    Validate and update shared preference slider weights under `_SWRR_LOCK`.
    Bumps `_PREFERENCE_GENERATION` and resets `_SHARED_CREDITS` to 0 when sliders change
    or `reset_swrr=True` (with `bump_generation=True`).
    """
    global _PREFERENCE_GENERATION
    cleaned = validate_preference_weights(weights)
    with _SWRR_LOCK:
        changed = cleaned != _PREFERENCE_WEIGHTS
        _PREFERENCE_WEIGHTS.clear()
        _PREFERENCE_WEIGHTS.update(cleaned)
        if reset_swrr or changed:
            if bump_generation:
                _PREFERENCE_GENERATION = int(_PREFERENCE_GENERATION) + 1
            fresh = reset_credits()
            _SHARED_CREDITS.clear()
            _SHARED_CREDITS.update(fresh)
        return dict(_PREFERENCE_WEIGHTS)


def get_shared_credits() -> Dict[str, float]:
    """Return a copy of the live shared SWRR credits dict under `_SWRR_LOCK`."""
    with _SWRR_LOCK:
        return dict(_SHARED_CREDITS)


def set_shared_credits(
    credits: Optional[Dict[str, Any]],
    expected_generation: Optional[int] = None,
) -> Dict[str, float]:
    """
    Set live shared SWRR credits under `_SWRR_LOCK` (used when loading persisted startup state).
    If `expected_generation` is given and does not match `_PREFERENCE_GENERATION`, the update is ignored.
    """
    cleaned = coerce_credits(credits)
    with _SWRR_LOCK:
        if expected_generation is not None and int(expected_generation) != int(_PREFERENCE_GENERATION):
            return dict(_SHARED_CREDITS)
        _SHARED_CREDITS.clear()
        _SHARED_CREDITS.update(cleaned)
        return dict(_SHARED_CREDITS)


def reset_shared_credits(bump_generation: bool = False) -> Dict[str, float]:
    """Reset live shared SWRR credits to 0.0 under `_SWRR_LOCK`."""
    global _PREFERENCE_GENERATION
    fresh = reset_credits()
    with _SWRR_LOCK:
        if bump_generation:
            _PREFERENCE_GENERATION = int(_PREFERENCE_GENERATION) + 1
        _SHARED_CREDITS.clear()
        _SHARED_CREDITS.update(fresh)
        return dict(_SHARED_CREDITS)


def select_and_commit(
    feasible: Iterable[str],
    shares: Optional[Dict[str, float]] = None,
    generation: Optional[int] = None,
    *,
    capacities: Optional[Dict[str, float]] = None,
    tie_break_scores: Optional[Dict[str, float]] = None,
    sliders: Optional[Dict[str, int]] = None,
    stranded_fractions: Optional[Dict[str, Any]] = None,
    pool_states: Optional[Dict[str, Any]] = None,
    operating_mode: str = "endurance",
) -> Optional[str]:
    """
    Atomically select a provider via classic SWRR and commit the credit update under `_SWRR_LOCK`
    (`docs/WP-ALLOCATION-CONTRACT.md` Amendment 3 item 3):
    - Inside `_SWRR_LOCK`, reads live `_PREFERENCE_WEIGHTS` and `_PREFERENCE_GENERATION`.
    - Recomputes the active provider set (`slider > 0`) and target shares `pi_p` from the live
      preference weights and the capacities passed in (or extracted from `pool_states`), so a route
      that started before a slider change can never pick a provider whose slider is now `0` and
      never commits against stale shares.
    - Computes the classic SWRR selection and updated credits via `step_swrr`.
    - Commits `updated_credits` to `_SHARED_CREDITS` when `chosen is not None` and
      `generation is None or int(generation) == _PREFERENCE_GENERATION`.
    """
    with _SWRR_LOCK:
        live_gen = int(_PREFERENCE_GENERATION)
        eff_gen = live_gen if generation is None else int(generation)
        # Shares always come from the LIVE preference weights (Amendment 3.3); a caller's `sliders` argument is
        # accepted for compatibility but never overrides them, so a stale snapshot cannot steer or commit credits.
        eff_sliders = dict(_PREFERENCE_WEIGHTS)

        resolved_caps = capacities if isinstance(capacities, dict) else _extract_capacities_from_pool_states(pool_states)
        if isinstance(resolved_caps, dict):
            live_shares = compute_target_shares(eff_sliders, resolved_caps)
        else:
            raw_shares = shares if isinstance(shares, dict) else {}
            live_shares = {
                p: (max(0.0, float(raw_shares.get(p, 0.0) or 0.0)) if int(eff_sliders.get(p, 0)) > 0 else 0.0)
                for p in PROVIDERS
            }

        live_feasible = [
            p
            for p in feasible
            if int(eff_sliders.get(str(p).lower().strip(), 0)) > 0
            and float(live_shares.get(str(p).lower().strip(), 0.0)) > 0.0
        ]
        chosen, updated_credits = step_swrr(
            credits=_SHARED_CREDITS,
            target_shares=live_shares,
            feasible_providers=live_feasible,
            tie_break_scores=tie_break_scores,
            sliders=eff_sliders,
            stranded_fractions=stranded_fractions,
            pool_states=pool_states,
            operating_mode=operating_mode,
        )
        if chosen is not None and eff_gen == live_gen:
            _SHARED_CREDITS.clear()
            _SHARED_CREDITS.update(updated_credits)
        return chosen

