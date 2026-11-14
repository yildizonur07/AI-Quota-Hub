# WP-ALLOCATION — preference sliders + burn-rate-aware allocation (contract)

Owner decision 2026-10-27: provider usage must follow (a) the owner's per-provider preference sliders and
(b) live quota of both windows (5h + weekly) normalised by each pool's *measured* burn rate. Fixed numbers such as
"AGY max 2 workers" or "Argon must be high effort" leave the rule files; the Hub computes them live.

Work only in this worktree: `<scratch>/hub-wt-allocation` (branch `wp/allocation`).

## edit_scope
- NEW `allocation.py` (all new math lives here; pure functions, no I/O except the learner's SQLite table)
- `scheduler.py` (only: call sites in `evaluate_candidate_schedule_score` / pool states that consume allocation)
- `orchestrator.py` (only: route selection using the allocation credit)
- `main.py` (state fields, endpoints, `/api/quotas` payload, poll hook)
- `static/index.html` (sliders + allocation panel)
- NEW `tests/test_allocation.py`; existing tests may be extended, never weakened or deleted
- `README.md` (short section)

## do_not_touch
`~/.agents/**`, `~/.claude/**`, `~/.codex/**`, `~/.gemini/config/**`, the live hub folder
`<scratch>/ai-quota-orchestrator` (do not start/stop/restart the server on port 8765),
`quota_hub_state.json` and `scheduler_telemetry.sqlite` of the live hub, any credentials. No git push, no commits on
`master`; commit only on `wp/allocation`. No paid API calls, no provider CLI calls (tests must not spend quota).

## Pools and windows
4 pools: `agy_gemini`, `agy_claude_gpt`, `claude_shared`, `codex_shared`; windows `five_hour`, `weekly`.
Sliders are per **provider** (`agy`, `claude`, `codex`); both AGY pools use the `agy` slider.

## 1. Pool-level burn-rate learner (`allocation.BurnRateLearner`)
- On every quota refresh, for each pool and window: `d_used = used_now - used_prev` (fractions 0..1) and
  `agent_hours = Σ running leaf jobs on that pool × elapsed hours` since the previous snapshot (use the same running-job
  source the scheduler already uses: live_activity / leases; include the coordinator only if it is recorded there).
- Record a sample only when: no reset happened in between (`resets_at` unchanged and `d_used >= 0`),
  `agent_hours >= 0.05`, elapsed between 2 and 90 minutes, telemetry confidence is not UNKNOWN/UNAVAILABLE.
- Estimate `b[pool][window] = fraction of window consumed per agent-hour` as an EWMA (alpha 0.2) over samples, with
  outlier clipping at 3× the current median. Persist samples/estimates in a NEW table in the hub SQLite DB
  (`burn_rate_samples`, created if missing). Expose sample count and last update.
- Until a pool/window has >= 3 samples, use a prior derived from the existing `QuotaCostLearner.predict_cost`
  (convert its per-task mean to per-agent-hour with its estimated duration) and mark `source="prior"`.

## 2. Allocation math (`allocation.compute_allocation`)
Inputs: pool states (remaining fraction, usable fraction after existing exclusion/reserve rules, reset_in_sec per
window, confidence), burn estimates b, sliders s_provider in 0..10, operating mode.
- `A_pool = min over windows of usable_w / (b_w * max(T_w_hours, floor_w))`, floors: 5h → 0.25 h, weekly → 1 h.
  Unknown/UNAVAILABLE telemetry or usable <= 0 → `A = 0` (unknown quota is never capacity).
- Provider capacity `A_provider = Σ A_pool` over its pools.
- Target share `pi_provider = s_provider * A_provider / Σ_q s_q * A_q` (0 if denominator 0). Slider 0 = off.
- Slack per pool `sigma = min over windows of remaining_w / max(0.02, T_w / L_w)` with `L_5h = 5 h`, `L_weekly = 168 h`.
  Provider sigma = max over its pools (AGY has two pools).
- Recommendations per provider:
  - `sigma >= 1.0` → band `normal`: effort per capability rules (strong models default `medium`, `high` allowed
    for critical packages), `max_workers = policy max_per_provider`.
  - `0.5 <= sigma < 1.0` → band `economy`: non-critical packages one effort step lower, `max_workers = ceil(cap/2)`.
  - `sigma < 0.5` → band `saver`: non-critical packages `low`, `max_workers = 1`.
  - Critical packages (capability floor HIGH and kind in design/review-of-high-risk, or explicitly flagged) keep their
    effort in every band. Capability floors are never lowered. Live explicit model/effort requests override.
  - In `sprint` mode use thresholds 0.7/0.35 instead of 1.0/0.5.

## 3. Route selection credit (Smooth Weighted Round Robin)
- Keep a persistent per-provider credit `C_p` in runtime state. For each routing decision: among **feasible**
  candidates (existing hard gates unchanged: concurrency, dual-window feasibility, Codex reserve, capability floor),
  for every provider that has a feasible candidate add `pi_p` (renormalised over those providers) to `C_p`; choose
  the provider with the highest `C_p`; then subtract 1 from the chosen provider's credit. Within the chosen provider
  pick the best existing-score candidate.
- Replace the old `pref_bonus = 0.12` term; keep `preferred_provider` / explicit model requests working (explicit match
  still wins). Existing harvest bonus / opportunity cost stay inside each provider's candidate ranking.
- Also enforce the dynamic `max_workers` from step 2 as an additional cap (never above the policy cap).
- Persist credits across restarts; reset credits to 0 when sliders change.

## 4. API + dashboard
- Runtime state `preference_weights = {"agy": 10, "codex": 8, "claude": 5}` (initial owner values), persisted in
  the state file, validated 0..10 integers.
- `POST /api/preferences {"agy": int, "codex": int, "claude": int}` → saves, resets credits, returns merged quotas.
- `/api/quotas` gains `allocation`: per provider `{slider, capacity_tasks_per_hour, target_share, actual_share_24h,
  sigma, band, max_workers, effort_policy, burn: {pool: {window: {b, source, samples}}}}` plus `computed_at`.
  `actual_share_24h` = share of dispatched/claimed jobs per provider in the last 24 h from existing history.
- `static/index.html`: a "Provider preference" card with three 0–10 sliders (live value label, saves on change via
  POST), and a table showing target share, actual 24h share, sigma band and max workers per provider. Keep the
  existing look; no new external dependencies.

## 5. Tests (`tests/test_allocation.py`, pytest, no network, no provider calls)
1. Equal capacity, sliders 10/8/5 → shares ≈ 0.435/0.348/0.217 (±0.01).
2. AGY burn 1/5 of others with equal remaining → AGY share rises above 0.70.
3. A provider whose 5h window is nearly empty but weekly is full is limited by the 5h term; weekly-limited case too.
4. Unknown telemetry → capacity 0, share 0.
5. SWRR: 1,000 simulated routing decisions with all providers feasible converge to target shares (±0.02); when one
   provider becomes infeasible its share flows to the others; credits persist and reset on slider change.
6. Burn learner: ignores reset-crossing samples, short intervals and zero agent-hours; EWMA moves toward new data;
   prior used until 3 samples.
7. Bands: sigma thresholds map to normal/economy/saver, critical package effort unchanged, sprint thresholds.
8. API: POST /api/preferences validates and persists; /api/quotas contains `allocation` with all fields.
Run the full suite: `python -m pytest -q -p no:cacheprovider --basetemp=<a temp dir>`; all 86 existing tests plus the
new ones must pass.

## Deliverable / return
Commit on `wp/allocation` (one or a few commits, clear messages). Then write `build/logs/WP-ALLOCATION-RESULT.md`:
status, changed files with line ranges, the formulas as implemented, test command + result, known limits,
`needs_parent_decision` (if any). Do not review your own work; a separate reviewer will.

## Amendment 2026-10-27 (coordinator): stranded-quota harvest through SWRR
Pre-reset harvest stays, but only through SWRR accounting (never a direct provider override, never a change to the
capacity formula). For each feasible provider p in a routing decision:
`harvest_boost_p = h * max(stranded_quota_fraction over p's pools)` where `stranded_quota_fraction` is the value the
Hub already computes in `pool_states` (quota that would expire unused before its reset at the learned burn rate),
`h = 1.0` in sprint mode and `0.6` in endurance mode. The boost is added to `C_p` together with `pi_p` in the same
locked step; the chosen provider is still debited 1. Slider 0 still disables a provider completely. Expected effect:
a provider whose quota is about to expire wins the next decisions until its stranded fraction falls, then the
long-run shares return to the slider/capacity targets.

## Amendment 2 (coordinator, replaces the boost formula of Amendment 1)
`harvest_boost_p = 1.0` when ANY pool of p has (a) weekly reset in <= 12 h and at least 1 task fittable before that reset
(`tasks_fittable_before_weekly_reset >= 1`), or (b) 5-hour reset in <= 1.5 h, at least 1 task fittable before it, and the
weekly window not scarce (weekly opportunity cost phi < 0.75); otherwise 0. Same value in sprint and endurance. Added to C_p
in the same locked SWRR step as pi_p; the winner is still debited 1. Credits are clamped to [-2, +2] after each step so a
harvested provider is not starved for long after its reset. Slider 0 still disables a provider.

## Amendment 3 (coordinator, final; replaces the clamp of Amendment 2)
1. Classic smooth weighted round robin (nginx): for the feasible, slider>0 providers of a decision, effective weight
   `w_p = pi_p + harvest_boost_p`; add `w_p` to every such `C_p`; choose the max; subtract `W = Σ w_p` from the winner.
   No clamping. Credits stay bounded and each provider gets decisions proportional to its weight.
2. harvest_boost_p = 1.0 also when (c) the Codex banked-reset harvest overlay is active for that provider (existing,
   owner-approved behaviour); otherwise only conditions (a)/(b) of Amendment 2.
3. Inside the single locked select-and-commit step, recompute the provider set and shares from the LIVE preference
   weights (and the capacities passed in), so a route that started before a slider change can never pick a provider whose
   slider is now 0 and never commits against stale shares.
4. 5-hour fittability for condition (b) uses the candidate's predicted 5-hour cost (same source as weekly fittability), not
   fixed fallback constants.

## Amendment 4 (coordinator, final harvest rule; supersedes the harvest parts of Amendments 1-3)
Let H = feasible, slider>0 providers whose harvest condition holds (Amendment 2 (a)/(b), Amendment 3 (c)).
- If H is non-empty: run classic SWRR (Amendment 3.1) over H only, with weights pi_p (renormalised over H); only H's
  credits change. A single harvesting provider therefore wins every decision while its condition holds; several share
  decisions in proportion to their weights (no starvation).
- If H is empty: classic SWRR over all feasible, slider>0 providers with weights pi_p.
- No boost term, no clamping. Amendment 3.3 (live preference weights inside the lock) and 3.4 (predicted 5h cost) stay.
