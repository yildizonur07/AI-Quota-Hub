# Independent Acceptance Audit Review

verdict: pass

## Re-review — 2026-10-26

Reviewer: codex-audit-reviewer (OpenAI family), independent review performed directly; no subagents. Scope: the three original findings below and current working-tree diffs in scheduler.py, orchestrator.py and tests/test_hub_core.py. This verdict supersedes the original failure recorded in the historical review below.

| Finding | Re-review result | Evidence |
|---|---|---|
| P1 — Missing/unavailable workspace | Resolved | scheduler.py:3680 rejects an otherwise reviewable submission with ValueError when its specified workspace is not a directory. _compute_submission_snapshot records an existence/hash entry for every declared normalized path, including unavailable-workspace cases. _check_submission_drift returns declared paths, or `<missing_workspace>` for an empty declaration, when a specified workspace disappears or becomes a regular file. Integrity verification invalidates both PENDING_REVIEW and ACCEPTED; the supervisor checks integrity before evaluating accepted dependencies. |
| P2 — priority and satisfy_on | Resolved | scheduler.py:3159 forwards both values in req_dict; scheduler.py:3214 supplies them in ar when the router omits them. orchestrator.py:1189 preserves both in assignment_record, including direct routing callers. The constraint audit asserts priority=42 and satisfy_on='accepted' in both captured route requests and execution assignments across rate-limit pause and crash recovery. |
| P3 — Stale review findings | Resolved | scheduler.py:3572 clears review_findings together with reviewer_agent_id, review_session_id and reviewer_family in the integrity invalidation UPDATE. The audit and independent probes confirm stale findings are NULL after invalidation. |

All gates exercised by the requested audit suite passed: submission binding/invalidation and accepted-dependent blocking, dry-run database purity, constraint preservation, and atomic concurrency caps with recovery and bounded retries. The earlier scope-guard/token-bucket package results below are historical evidence and were not rerun in this targeted re-review. The original limitations concerning external activity snapshots and thread-based concurrency testing still apply.

Validation performed for this re-review:

- `pytest tests/test_hub_core.py -k test_audit_ -v -p no:cacheprovider` — **4 passed, 57 deselected**, exit 0. One Starlette/httpx deprecation warning; pytest also printed an ignored atexit temporary-directory cleanup PermissionError for pytest-current after the passing results.
- Additional independent in-memory-script probes against isolated temporary SQLite queues — **passed**, exit 0: missing and non-directory workspace snapshot completeness and empty-declaration sentinel; all four combinations of PENDING_REVIEW/ACCEPTED and renamed workspace/replacement regular file invalidate acceptance, block accepted dependents, clear all four review metadata fields, and reject resubmission while unavailable.

No unresolved finding remains within this re-review scope. Only this review document was edited; no source/test changes, live queue verdicts, commits, merges or worker dispatches were performed. Probe execution used fake executors and temporary databases.

## Original review — historical verdict: fail

Reviewer: codex-audit-reviewer (OpenAI family), independent review performed directly; no subagents.
Date: 2026-10-26 (Australia/Perth).
Scope: working-tree diffs in scheduler.py, orchestrator.py, tests/test_hub_core.py; live scope_guard.py and its tests; token_bucket.py, its tests, and submitted hash bindings.

## Findings

1. **P1 — Missing workspace bypasses acceptance integrity and releases accepted dependents.**
   `scheduler.py:2345-2349` returns no drift when the workspace is absent or is no longer a directory, before examining the saved existence/hash snapshot. This also affects submissions made with an unavailable workspace: snapshot construction can produce no binding while submission still becomes PENDING_REVIEW. Acceptance must fail closed when submitted content cannot be verified.

   Independently reproduced in an isolated temporary queue: complete UP, submit a.py, approve UP, then rename its workspace. `verify_acceptance_integrity()` returns `[]`; UP remains ACCEPTED; a live supervisor step dispatches DOWN with `depends_on=['UP']` and `satisfy_on='accepted'`. Removing the workspace removes every submitted path from its recorded location, so this violates the deletion/integrity requirement. Restore detection for missing/unavailable workspaces, require a complete snapshot for declared submitted files, and cover the unavailable-workspace case in tests.

2. **P2 — priority and satisfy_on are not preserved through routing.**
   The request built at `scheduler.py:3123-3147` omits both fields, and `orchestrator.py:1151-1192` omits them from assignment_record even when a direct caller supplies them. An isolated probe with priority=42 and satisfy_on='accepted' observed both values intact in SQLite but missing from the route request and execution assignment. A direct route_work_package call also dropped both values. Queue ordering and dependency evaluation still use the stored values correctly; the failure is the requested end-to-end field preservation. Forward both fields and assert them in route requests and assignments. The new constraint test checks these two fields only in persisted rows, so its passing result misses this gap.

3. **P3 — Invalidation retains review findings for obsolete content.**
   The invalidation UPDATE in `scheduler.py:3540-3551` clears reviewer_agent_id, review_session_id and reviewer_family but leaves review_findings unchanged. Reproduced by approving with findings='old approval', editing the file and verifying integrity: acceptance becomes INVALIDATED_BY_EDIT while review_findings remains 'old approval'. Clear review_findings with the other current-review fields, or explicitly move old findings into separately identified history. This does not independently unblock dependents, but leaves stale approval evidence on the invalidated submission.

## Required gate verification

| Requirement | Result | Evidence / limit |
|---|---|---|
| Refinery submission binding | Partial | submit_for_review persists per-file SHA-256/existence and git HEAD for valid workspaces. Both real live submissions contain matching source/test hashes and HEAD 3636ba56354c56a680e36dd008bf28c0b058f3bb. Missing workspace fails open (finding 1). HEAD is recorded as provenance; drift checks compare submitted file content/existence. |
| Automatic invalidation and dependent blocking | Fail overall | Normal file edits are invalidated before verdicts and live scheduling; preview evaluates drift in memory. Additional isolated probes passed modification, deletion and creation for both PENDING_REVIEW and ACCEPTED (6 cases). Missing workspace bypasses invalidation and accepted-dependent blocking. Reviewer identities clear, but findings remain (finding 3). |
| Dry-run purity | Pass | Preview skips integrity persistence, uses the guarded write helper, exits before claims/execution/observations, and requests a summary with verify_integrity=False. The audit test compared all backlog_jobs and task_cost_observations rows before/after previews, including in-memory drift and resumed jobs. |
| Constraint preservation | Fail overall | workspace, empty read-only edit_scope, read_only, explicit model/effort, cross-family requirement and implementer family survive the reviewed path; recovery updates execution state without replacing these fields. priority and satisfy_on remain in the queue but disappear at routing/assignment boundaries (finding 2). |
| Atomic global/provider/pool caps | Pass for database-managed claims and supplied live snapshot | The conditional SQLite UPDATE checks QUEUED and all three RUNNING counts in one statement, with _db_lock serializing same-process claim/commit. SQLite serializes writers across connections. _merge_running_into_live_activity includes persisted workers for routing; _external_leaf_counts adds workers not represented by RUNNING PIDs. Defaults are global 10/provider 6/pool 6. Threaded audit tests passed duplicate-claim, global, provider/pool, recovery and retry checks. External activity remains a supplied snapshot; the lock does not coordinate unrelated external launchers, and the suite is not a multiprocess/external-launcher stress test. |

## Test results

- Requested hub command: `pytest tests/test_hub_core.py -k test_audit_ -v -p no:cacheprovider` — **4 passed, 57 deselected**, exit 0. One Starlette/httpx deprecation warning and a pytest atexit temporary-directory PermissionError occurred after the passing results.
- Requested standalone scope command: `pytest ../orchestrator-live-verify/tests/test_scope_guard.py -v -p no:cacheprovider` — **4 passed**, exit 0.
- Combined live suites from orchestrator-live-verify: `pytest tests/test_scope_guard.py tests/test_token_bucket.py -v -p no:cacheprovider` — **25 passed** (4 scope guard + 21 token bucket), exit 0.
- Additional temporary-queue probes confirmed findings 1–3 and the six normal file-drift cases; final confirmation command exited 0. An earlier probe printed the same findings but encountered a Windows SQLite temporary-directory cleanup error; confirmation used garbage collection before cleanup and completed successfully.

## Live package review and recorded verdicts

**LIVE-AGY-01: pass.** Inspected scope_guard.py and all four tests. extract_literal_prefix now conservatively uses the directory preceding a wildcard-containing component, avoiding false disjointness for partial-component wildcards. Reviewed normalization, traversal rejection, conservative collision detection and documented glob matching. No blocking issue found within that documented contract.

Source SHA-256: `6f9b78372d9a0cab1ce8a93cf5e78e1a258fc063364201072c9d2bde6268c43c`.

**LIVE-CLAUDE-01: pass for the requested binding and suite verification.** Inspected token_bucket.py and its 21 tests; verified the submitted source and test hashes against disk and the dual-window gating behavior covered by the suite.

Source SHA-256: `ab8ee4d70e45e034dc2003fd2ab425dfe1ff15162155d56eaa9931e6834ca9ba`.

Executed both authorized cli.py verdict commands with the requested findings text. Both returned ACCEPTED / COMPLETED, reviewer codex-audit-reviewer, reviewer_family openai; implementer families were gemini and claude respectively. Re-read SQLite afterward and independently verified every submitted source/test hash still matched. Each command reported acceptance telemetry updated=false / updated_rows=0; the review verdicts themselves were persisted successfully. These package acceptances do not change the overall audit failure.

No source or test files were modified. No fixes, commits, merges or worker dispatches were performed against the live queue. Probe dispatches used fake executors and isolated temporary databases only.
