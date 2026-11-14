# WP-MODELS — model catalog file, discovery, price/performance review (contract)

Owner request 2026-10-27: new models should be found, compared on price/performance, and offered for assignment; very
expensive tiers (e.g. "Astra"-class) stay out unless the owner approves. Suggest-only: nothing joins routing without
the owner's click. Builds on WP-ALLOCATION (same Hub).

## edit_scope
NEW `model_catalog.json`, NEW `model_review.py`, `orchestrator.py` (load catalog from file instead of the hard-coded
`MODEL_CATALOG` list, keeping the same fields), `main.py` (endpoints, scheduler hook), `static/index.html` ("Model
review" card), NEW `tests/test_model_review.py`, `README.md`.

## 1. Catalog out of code
- Move the 16 entries of `MODEL_CATALOG` verbatim into `model_catalog.json` (`{"version": 1, "models": [...]}`), load it
  at startup with validation (required fields as today), atomic writes with a timestamped backup when it changes.
- New optional fields per model: `status` (`active` | `probation` | `retired`), `probation_share` (0..1),
  `added_at`, `source`, `price_in_per_mtok`, `price_out_per_mtok`, `coding_score`, `notes`.
- Routing uses `active` models normally; `probation` models only for packages with floor LOW/MID, kind not in
  {design, review}, and at most `probation_share` (default 0.10) of eligible routing decisions (deterministic counter);
  `retired` never.

## 2. Discovery (no LLM, no paid API)
- Local: parse `agy models` output (run with a 30 s timeout), `~/.codex/models_cache.json`, and a static Claude list
  (Claude Code has no list command) plus anything seen in recent hub dispatch history.
- Public metadata: `https://models.dev/api.json` and `https://openrouter.ai/api/v1/models` (GET, cached in the hub
  folder for 24 h, ETag when available). Extract price per Mtok in/out, context window, release date, and any coding
  benchmark score present. Network failure → keep the last cache, report `stale`.
- Match local model ids to public entries by normalised name; unmatched → `metadata: unknown`.

## 3. Review scoring and proposals
- For each locally available model not in the catalog (or whose public price changed > 20 %): compute
  `price_ratio = blended_price / blended_price(current active HIGH model of the same provider)` (blended = 0.25·in +
  0.75·out) and keep `coding_score` if known.
- Owner settings (persisted, editable in the dashboard): `max_price_ratio` (default 2.0), `review_interval_days`
  (default 7), `probation_share` (default 0.10), `auto_retire_after_days_unused` (default off).
- Proposal when `price_ratio <= max_price_ratio` (or unknown price, flagged); suggested tier from price band and
  benchmark relative to existing tiers; suggested `cost_weight` from price_ratio. Models above the ratio are listed as
  "excluded (too expensive)" with the reason, never proposed.
- `GET /api/models/review` (latest proposals + sources + last run + staleness), `POST /api/models/review/run`
  (manual run), `POST /api/models/decision {"model_id", "decision": "approve"|"probation"|"reject", "tier"?}`.
  Reject = 30-day cooldown for that model id. Approve/probation writes `model_catalog.json`.
- Scheduled: run once per `review_interval_days` from the hub's existing background loop (no new Windows task).
- After 14 days of probation, summarise the existing scorecard (acceptance, rework, measured quota cost) for that
  model and propose `promote` or `retire`; never auto-promote.

## 4. Dashboard
"Model review" card: table of proposals (model, provider, price ratio, coding score, context, suggested tier, sources
links) with Approve / Probation / Reject buttons; excluded-too-expensive list collapsed; settings inputs; last run time.

## 5. Tests (no network: inject fixture JSON for models.dev/OpenRouter and fake `agy models` output)
Catalog round-trip identical to the old list; validation errors; discovery diff; price ratio and exclusion; reject
cooldown; probation share respected over 1,000 routing decisions and never used for design/review; stale cache on
network error; approve writes the file atomically with backup.

## Additional coordinator notes
- WP-ALLOCATION (allocation.py, SWRR, sliders, limits) is merged on master; do not change its behaviour. Probation routing
  must apply after the SWRR provider choice, inside the chosen provider's candidate ranking.
- You cannot run commands in this environment; the coordinator runs `python -m pytest -q` after you finish and returns failures.
  Write tests carefully and keep fixtures self-contained (tmp paths, injected JSON, fake `agy models` text).
- Never write to the real `~/.agents`, the live hub folder, or the live `model_catalog.json` path from tests.
