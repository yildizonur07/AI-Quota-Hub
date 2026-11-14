"""
CLI & Server Entrypoint for AI Quota Hub & Dynamic Multi-Provider Orchestrator.

Usage:
  python cli.py status | quotas            # Print normalized quota summary JSON
  python cli.py route --task "..." --kind implement --blast-radius module
  python cli.py dispatch --task "..." [--execute]   # Route + (dry-run or live) execute, logged like /api/dispatch
  python cli.py observe --task-id WP-1 --provider claude --model-id claude-sonnet-5-5 --pool-id claude_shared ...
  python cli.py policy                     # Effective policy (~/.agents/policy.json over defaults) + source
  python cli.py witness                    # Advisory worker-health snapshot (never kills/duplicates)
  python cli.py submit JOB-ID [--changed-file PATH ...] [--checks-failed] [--implementer ID]
  python cli.py verdict JOB-ID --reviewer ID --accept|--reject [--findings TEXT] [--reviewer-family F]
  python cli.py serve --port 8765          # Launch FastAPI + Web Dashboard server

Every command prints JSON; a refused or invalid operation exits with status 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _refuse(message: str) -> int:
    _print({"status": "refused", "error": message})
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Quota Hub & Dynamic Multi-Provider Orchestrator CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("status", help="Query real-time zero-token quotas across AGY, Claude Code, and Codex")
    sub.add_parser("quotas", help="Alias of status")

    def _add_route_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--task", required=True, help="Work package title / goal")
        p.add_argument("--prompt", default="", help="Worker prompt")
        p.add_argument("--kind", default="implement", help="scan|extract|mechanical|implement|debug|design|test-author|review|docs")
        p.add_argument("--ambiguity", default="low", choices=["low", "med", "high"])
        p.add_argument("--blast-radius", default="local", choices=["local", "module", "cross-module", "irreversible"])
        p.add_argument("--dependency-role", default="leaf", choices=["leaf", "upstream"])
        p.add_argument("--context-need", default="pointer", choices=["pointer", "module", "broad"])
        p.add_argument("--implementer-family", default="", help="gemini|claude|openai (review independence bonus)")
        p.add_argument("--require-cross-family", action="store_true", help="Hard-require a reviewer of another family")
        p.add_argument("--execute", action="store_true", help="Execute live headless CLI subprocess")

    _add_route_args(sub.add_parser("route", help="Route a work package to the healthiest eligible model"))
    p_disp = sub.add_parser("dispatch", help="Route and execute a work package, logging telemetry like /api/dispatch")
    _add_route_args(p_disp)
    p_disp.add_argument("--workspace", default=None, help="Worker workspace directory")
    p_disp.add_argument("--edit-scope", action="append", default=None, help="Declared edit_scope entry (repeatable)")
    p_disp.add_argument("--timeout-sec", type=int, default=60)

    p_obs = sub.add_parser("observe", help="Record a pre/post quota observation in the cost learner")
    p_obs.add_argument("--task-id", required=True)
    p_obs.add_argument("--provider", required=True, choices=["agy", "claude", "codex"])
    p_obs.add_argument("--model-id", required=True)
    p_obs.add_argument("--pool-id", required=True, choices=["agy_gemini", "agy_claude_gpt", "claude_shared", "codex_shared"])
    p_obs.add_argument("--task-kind", default="implement")
    p_obs.add_argument("--tier", default="MID", choices=["LOW", "MID", "HIGH"])
    p_obs.add_argument("--complexity", default="med", choices=["low", "med", "high"])
    p_obs.add_argument("--duration-sec", type=float, default=60.0)
    p_obs.add_argument("--failed", action="store_true", help="Mark the run as failed")
    p_obs.add_argument("--concurrent-pool-jobs", type=int, default=1)
    for name in ("pre-5h-used-pct", "post-5h-used-pct", "pre-weekly-used-pct", "post-weekly-used-pct"):
        p_obs.add_argument(f"--{name}", type=float, default=None)
    p_obs.add_argument("--dry-run", action="store_true")

    sub.add_parser("policy", help="Print the effective orchestrator policy and its source")
    sub.add_parser("witness", help="Print the advisory Witness worker-health snapshot")

    p_sync = sub.add_parser("sync", help="Check or automatically synchronize ~/.agents canonical rules and skills")
    p_sync.add_argument("--check", action="store_true", help="Check only; do not write target files")

    p_aud = sub.add_parser("audit", help="Audit unreviewed writer jobs and unapproved git workspace edits")
    p_aud.add_argument("--workspace", action="append", default=None, help="Workspace path to audit (repeatable)")

    p_claim = sub.add_parser("claim-lease", help="Atomically claim a concurrency + edit-scope lease for a native subagent")
    p_claim.add_argument("job_id")
    p_claim.add_argument("--provider", required=True, choices=["agy", "claude", "codex"])
    p_claim.add_argument("--pool-id", default=None)
    p_claim.add_argument("--model-id", default=None)
    p_claim.add_argument("--task", default="")
    p_claim.add_argument("--kind", default="implement")
    p_claim.add_argument("--workspace", default=None)
    p_claim.add_argument("--edit-scope", action="append", default=None)
    p_claim.add_argument("--read-only", action="store_true")
    p_claim.add_argument("--session-id", default=None)
    p_claim.add_argument("--worker-pid", type=int, default=None)

    p_rel = sub.add_parser("release-lease", help="Release a native subagent lease upon completion or failure")
    p_rel.add_argument("job_id")
    p_rel.add_argument("--run-id", default=None)
    p_rel.add_argument("--status", default="COMPLETED", choices=["COMPLETED", "FAILED_ESCALATED", "QUEUED"])
    p_rel.add_argument("--output-summary", default="")
    p_rel.add_argument("--session-id", default=None)

    p_sub = sub.add_parser("submit", help="Submit a COMPLETED job to the Refinery gate")
    p_sub.add_argument("job_id")
    p_sub.add_argument("--changed-file", action="append", default=None, help="Changed path (repeatable); omit to inspect via git")
    p_sub.add_argument("--checks-failed", action="store_true", help="Verification checks failed")
    p_sub.add_argument("--implementer", default=None, help="Implementer agent/session id (default: hub run id)")

    p_ver = sub.add_parser("verdict", help="Record an independent review verdict")
    p_ver.add_argument("job_id")
    p_ver.add_argument("--reviewer", required=True, help="Reviewer agent/session id (must differ from implementer)")
    grp = p_ver.add_mutually_exclusive_group(required=True)
    grp.add_argument("--accept", action="store_true")
    grp.add_argument("--reject", action="store_true")
    p_ver.add_argument("--findings", default="")
    p_ver.add_argument("--reviewer-family", default=None, choices=["gemini", "claude", "openai"])
    p_ver.add_argument("--review-session", default=None)
    p_ver.add_argument("--verify-session", action="store_true", help="Require review-session to resolve to a real distinct session")

    p_serve = sub.add_parser("serve", help="Start FastAPI Web Dashboard server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765)

    args = parser.parse_args()

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("main:app", host=args.host, port=args.port, reload=False)
        return 0

    if args.cmd == "sync":
        from scheduler import ensure_agents_synced

        res = ensure_agents_synced(auto_sync=not args.check, force_check=True)
        _print(res)
        return 0 if res.get("status") in {"in_sync", "synced"} else 1

    import os as _os
    if _os.environ.get("ORCHESTRATOR_AUTO_SYNC", "1") != "0" and not _os.environ.get("ORCHESTRATOR_POLICY_PATH"):
        from scheduler import ensure_agents_synced

        ensure_agents_synced(auto_sync=True)

    if args.cmd == "audit":
        from orchestrator import QUEUE_MANAGER

        res = QUEUE_MANAGER.audit_unreviewed_work(workspaces=args.workspace)
        _print(res)
        return 0 if res.get("review_gate_passed") else 1

    if args.cmd in {"claim-lease", "release-lease"}:
        from collectors import collect_live_activity
        from orchestrator import QUEUE_MANAGER

        try:
            if args.cmd == "claim-lease":
                res = QUEUE_MANAGER.claim_native_lease(
                    args.job_id,
                    provider=args.provider,
                    task=args.task,
                    pool_id=args.pool_id,
                    model_id=args.model_id,
                    workspace=args.workspace,
                    edit_scope=args.edit_scope,
                    read_only=True if args.read_only else None,
                    kind=args.kind,
                    session_id=args.session_id,
                    worker_pid=args.worker_pid,
                    live_activity=collect_live_activity(),
                )
                _print(res)
                return 0 if res.get("claimed") else 1
            res = QUEUE_MANAGER.release_native_lease(
                args.job_id,
                run_id=args.run_id,
                status=args.status,
                output_summary=args.output_summary,
                session_id=args.session_id,
            )
            _print(res)
            return 0
        except (KeyError, ValueError) as exc:
            return _refuse(str(exc))

    if args.cmd == "policy":
        from scheduler import load_orchestrator_policy

        _print(load_orchestrator_policy(auto_sync=True))
        return 0

    if args.cmd in {"submit", "verdict"}:
        from orchestrator import COST_LEARNER, QUEUE_MANAGER

        try:
            if args.cmd == "submit":
                res = QUEUE_MANAGER.submit_for_review(
                    args.job_id,
                    changed_files=args.changed_file,
                    checks_passed=not args.checks_failed,
                    implementer_agent_id=args.implementer,
                )
                _print(res)
                return 0 if res.get("acceptance_status") == "PENDING_REVIEW" else 1
            res = QUEUE_MANAGER.record_review_verdict(
                args.job_id,
                reviewer_agent_id=args.reviewer,
                accepted=bool(args.accept),
                findings=args.findings,
                reviewer_family=args.reviewer_family,
                review_session_id=args.review_session,
                require_verified_session=True if args.verify_session else None,
                cost_learner=COST_LEARNER,
            )
            _print(res)
            return 0
        except (KeyError, ValueError) as exc:
            return _refuse(str(exc))

    if args.cmd == "witness":
        from collectors import collect_live_activity
        from orchestrator import QUEUE_MANAGER
        from scheduler import load_orchestrator_policy

        _print(QUEUE_MANAGER.get_witness_report(live_activity=collect_live_activity(), policy=load_orchestrator_policy()))
        return 0

    from main import (
        TelemetryObservationRequest,
        WorkPackageDispatchRequest,
        api_dispatch_package,
        api_record_telemetry_observation,
        get_merged_quotas,
    )
    from orchestrator import execute_headless_dispatch, route_work_package

    if args.cmd == "observe":
        req = TelemetryObservationRequest(
            task_id=args.task_id,
            provider=args.provider,
            model_id=args.model_id,
            pool_id=args.pool_id,
            task_kind=args.task_kind,
            tier=args.tier,
            complexity=args.complexity,
            duration_sec=args.duration_sec,
            success=not args.failed,
            concurrent_pool_jobs=args.concurrent_pool_jobs,
            pre_5h_used_pct=args.pre_5h_used_pct,
            post_5h_used_pct=args.post_5h_used_pct,
            pre_weekly_used_pct=args.pre_weekly_used_pct,
            post_weekly_used_pct=args.post_weekly_used_pct,
            dry_run=args.dry_run,
        )
        _print(api_record_telemetry_observation(req)["recorded"])
        return 0

    if args.cmd == "dispatch":
        from fastapi import HTTPException

        req = WorkPackageDispatchRequest(
            task=args.task,
            prompt=args.prompt or args.task,
            kind=args.kind,
            ambiguity=args.ambiguity,
            blast_radius=args.blast_radius,
            dependency_role=args.dependency_role,
            context_need=args.context_need,
            implementer_family=args.implementer_family or None,
            require_cross_family=args.require_cross_family,
            workspace=args.workspace,
            execute_live=args.execute,
            timeout_sec=args.timeout_sec,
            **({"edit_scope": args.edit_scope} if args.edit_scope else {}),
        )
        try:
            entry = api_dispatch_package(req)
        except HTTPException as exc:
            return _refuse(str(exc.detail))
        _print(entry)
        return 0 if (entry.get("routing") or {}).get("status") == "routed" else 1

    snapshot = get_merged_quotas()
    if args.cmd in (None, "status", "quotas"):
        providers = snapshot["providers"]
        brief_out = {
            "providers": {
                "agy": {
                    "status": providers["agy"].get("status"),
                    "confidence": providers["agy"].get("confidence"),
                    "gemini_flash": providers["agy"].get("gemini_flash"),
                    "gemini_pro": providers["agy"].get("gemini_pro"),
                    "claude_sonnet": providers["agy"].get("claude_sonnet"),
                    "claude_opus": providers["agy"].get("claude_opus"),
                    "reset_in_sec": providers["agy"].get("reset_in_sec"),
                },
                "claude": {
                    "status": providers["claude"].get("status"),
                    "confidence": providers["claude"].get("confidence"),
                    "sonnet": providers["claude"].get("sonnet"),
                    "opus": providers["claude"].get("opus"),
                    "haiku": providers["claude"].get("haiku"),
                    "reset_in_sec": providers["claude"].get("reset_in_sec"),
                },
                "codex": {
                    "status": providers["codex"].get("status"),
                    "confidence": providers["codex"].get("confidence"),
                    "luna": providers["codex"].get("luna"),
                    "terra": providers["codex"].get("terra"),
                    "sol": providers["codex"].get("sol"),
                    "reset_in_sec": providers["codex"].get("reset_in_sec"),
                },
            },
            "orchestrator": {
                "preferred": snapshot["orchestrator"]["preferred"],
                "active": snapshot["orchestrator"]["active"],
            },
        }
        _print(brief_out)
        return 0

    if args.cmd == "route":
        req = {
            "task": args.task,
            "prompt": args.prompt or args.task,
            "kind": args.kind,
            "ambiguity": args.ambiguity,
            "blast_radius": args.blast_radius,
            "dependency_role": args.dependency_role,
            "context_need": args.context_need,
            "implementer_family": args.implementer_family,
            "require_cross_family": args.require_cross_family,
        }
        res = route_work_package(req, snapshot["providers"], snapshot["orchestrator"])
        if res.get("assignment_record"):
            res["execution"] = execute_headless_dispatch(res["assignment_record"], dry_run=not args.execute)
        _print(res)
        return 0 if res.get("status") == "routed" else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
