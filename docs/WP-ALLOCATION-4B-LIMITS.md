# WP-ALLOCATION step 4b — hard ceilings editable in the dashboard (owner request 2026-10-27)

The hard ceilings stay the single source of truth in `~/.agents/policy.json` (`max_leaves`, `max_per_provider`,
`max_per_pool`); the Hub must make them visible and editable so the owner changes them in one place.

- `POST /api/limits {"max_leaves": int, "max_per_provider": {"agy","claude","codex": int}, "max_per_pool": {pool: int}}`
  Validation: integers, max_leaves 1..20, per provider 0..max_leaves, per pool 0..its provider value.
  Write `~/.agents/policy.json` atomically (temp file + os.replace), keep every other key unchanged, keep a timestamped
  backup copy next to it (`policy.json.bak-<ts>`, keep the newest 10), then reload the policy in the running process.
- `/api/quotas` shows the current ceilings plus the live dynamic `max_workers` per provider (from `allocation`), so the
  dashboard shows "ceiling 6 · now 3 (economy band)".
- Dashboard: a "Limits" card with number inputs for the three ceiling groups, a Save button, and the backup path after
  saving. The dynamic recommendation never exceeds the ceiling.
- Tests: validation errors (400), atomic write keeps other keys, backup created, reload effective without restart,
  dynamic max_workers capped by the new ceiling. Tests must use a temporary policy path (never the real file).
