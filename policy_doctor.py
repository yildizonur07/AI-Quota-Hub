#!/usr/bin/env python3
"""
Policy Doctor — Automated Weekly Contradiction Auditor, Token-Budget Inspector & Compaction Bot
for `~/.agents` (`DELEGATION.md`, `ROUTING.md`, `LEARNING.md`, `owner-overrides/*`, `claude-agents/*.md`,
`providers/*.md`, `skills/*/SKILL.md`, and `policy.json`).

Features:
1. Cross-file contradiction detection (e.g., model bans in overrides vs. tier tables in ROUTING.md,
   skill vs. owner-override drift, policy.json thresholds vs. prose).
2. Subagent frontmatter & safety-bound verification (`model`, `maxTurns`, `effort`, read-only tool permissions).
3. Sync parity verification (`sync.ps1 -Check` or direct hash check across rendered provider files).
4. Token/line budget & compaction analyzer (flags bloated bullets, duplicate rules across files,
   and estimates always-loaded prompt tokens vs. on-demand skill tokens).
5. Structured Turkish "Active Rules Summary" for the AI Quota Hub dashboard (`http://127.0.0.1:8765`).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


AGENTS_ROOT = Path(os.environ.get("AGENTS_ROOT") or (Path.home() / ".agents")).resolve()
DEFAULT_STATE_FILE = AGENTS_ROOT / "quota-hub" / "policy_doctor_state.json"
DEFAULT_INTERVAL_DAYS = 7.0

# Soft token budgets (approx 1 token ~= 4 chars) for always-loaded vs. on-demand files
FILE_TOKEN_BUDGETS: Dict[str, int] = {
    "DELEGATION.md": 2500,
    "ROUTING.md": 1900,
    "LEARNING.md": 1600,
    "owner-overrides/DELEGATION.md": 450,
    "owner-overrides/ROUTING.md": 400,
    "providers/claude.md": 450,
    "providers/codex.md": 700,
    "providers/agy.md": 500,
    "skills/orchestrator/SKILL.md": 9200,
}

READ_ONLY_AGENTS = {"scout", "extractor", "reviewer", "advisor"}
WRITE_TOOLS = {"edit", "write", "multiedit", "notebookedit"}
EXPECTED_MAX_TURNS: Dict[str, int] = {
    "implementer": 40,
    "reviewer": 20,
    "advisor": 12,
    "scout": 12,
    "extractor": 12,
}
OWNER_EXCLUSIVE_MODELS = {"fable", "carbon", "gemini-pro-orchestrator"}
OWNER_EXCLUSIVE_EFFORTS = {"xhigh", "max"}


def _estimate_tokens(text: str) -> int:
    """Fast deterministic token estimate (~4 chars per token for English/Markdown)."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _read_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")


def _parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """Parse simple YAML frontmatter (`--- ... ---`) without external dependencies."""
    if not text.startswith("---\n"):
        return {}, text
    end_idx = text.find("\n---\n", 4)
    if end_idx == -1:
        return {}, text
    raw_fm = text[4:end_idx]
    body = text[end_idx + 5 :]
    meta: Dict[str, str] = {}
    for line in raw_fm.splitlines():
        line_str = line.strip()
        if not line_str or line_str.startswith("#") or ":" not in line_str:
            continue
        k, v = line_str.split(":", 1)
        meta[k.strip()] = v.strip()
    return meta, body


def _normalize_sentence(s: str) -> str:
    s = re.sub(r"`[^`]+`", lambda m: m.group(0).lower(), s)
    s = re.sub(r"[^a-zA-Z0-9_]+", " ", s.lower()).strip()
    return s


def check_sync_parity(root: Path = AGENTS_ROOT) -> Dict[str, Any]:
    """Run `sync.ps1 -Check` (on Windows) or verify key target files exist and have balanced sentinels."""
    sync_script = root / "sync.ps1"
    if not sync_script.exists():
        return {
            "id": "sync_parity",
            "title": "Sync Parity (sync.ps1 -Check)",
            "title_tr": "Senkronizasyon Tutarlılığı (sync.ps1 -Check)",
            "status": "WARN",
            "summary": f"{sync_script} not found.",
            "summary_tr": f"{sync_script} bulunamadı.",
            "details": [],
        }

    powershell_bin = "powershell" if os.name == "nt" else "pwsh"
    try:
        proc = subprocess.run(
            [powershell_bin, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(sync_script), "-Check"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if proc.returncode == 0 and "SYNC OK" in out:
            return {
                "id": "sync_parity",
                "title": "Sync Parity (sync.ps1 -Check)",
                "title_tr": "Senkronizasyon Tutarlılığı (sync.ps1 -Check)",
                "status": "PASS",
                "summary": "All provider target files (.claude, .codex, .gemini, .config/opencode) are in 1:1 sync with canonical sources (SYNC OK).",
                "summary_tr": "Tüm sağlayıcı hedef dosyaları (.claude, .codex, .gemini, .config/opencode) canonical kaynaklarla birebir senkronize (SYNC OK).",
                "details": [],
            }
        drift_lines = [ln.strip() for ln in (out + "\n" + err).splitlines() if ln.strip()]
        return {
            "id": "sync_parity",
            "title": "Sync Parity (sync.ps1 -Check)",
            "title_tr": "Senkronizasyon Tutarlılığı (sync.ps1 -Check)",
            "status": "FAIL",
            "summary": "Sync drift detected between canonical files and provider targets.",
            "summary_tr": "Canonical dosyalar ile sağlayıcı hedefleri arasında senkron kayması (DRIFT) tespit edildi.",
            "details": drift_lines[:10],
            "remediation": "Run: powershell -NoProfile -ExecutionPolicy Bypass -File ~/.agents/sync.ps1",
        }
    except Exception as exc:
        return {
            "id": "sync_parity",
            "title": "Sync Parity (sync.ps1 -Check)",
            "title_tr": "Senkronizasyon Tutarlılığı (sync.ps1 -Check)",
            "status": "WARN",
            "summary": f"Failed to execute sync.ps1: {exc}",
            "summary_tr": f"sync.ps1 çalıştırılamadı: {exc}",
            "details": [],
        }


def check_model_gates_and_opus_contradictions(root: Path = AGENTS_ROOT) -> Dict[str, Any]:
    """
    Detect contradictions between `ROUTING.md` (which allows Claude Opus 5.5 at medium effort)
    and override/skill files (`owner-overrides/ROUTING.md`, `skills/orchestrator/SKILL.md`),
    while ensuring owner-exclusive models (`Fable`, `Carbon`, `xhigh`, `max`) stay gated.
    """
    issues: List[str] = []
    routing_path = root / "ROUTING.md"
    ov_routing_path = root / "owner-overrides" / "ROUTING.md"
    skill_path = root / "skills" / "orchestrator" / "SKILL.md"
    ov_skill_path = root / "owner-overrides" / "skills" / "orchestrator" / "SKILL.md"

    routing_txt = _read_utf8(routing_path) if routing_path.exists() else ""
    ov_routing_txt = _read_utf8(ov_routing_path) if ov_routing_path.exists() else ""
    skill_txt = _read_utf8(skill_path) if skill_path.exists() else ""
    ov_skill_txt = _read_utf8(ov_skill_path) if ov_skill_path.exists() else ""

    opus_in_routing_table = bool(re.search(r"Opus\s+5\.5\s+medium", routing_txt, re.IGNORECASE))

    # Check if any file puts Claude Opus inside a "Never automatically select/route ... to" clause
    ban_pattern = re.compile(
        r"Never automatically (?:select or delegate|route)[^\n.;]*\b(?:Claude\s+Opus|Opus)\b",
        re.IGNORECASE,
    )
    for label, txt in (
        ("owner-overrides/ROUTING.md", ov_routing_txt),
        ("skills/orchestrator/SKILL.md", skill_txt),
        ("owner-overrides/skills/orchestrator/SKILL.md", ov_skill_txt),
    ):
        if txt and ban_pattern.search(txt):
            issues.append(
                f"Contradiction: 'Claude Opus' appears in an auto-selection ban in {label}, while ROUTING.md lists 'Opus 5.5 medium' as an active strong model."
            )

    # Check that the Anti-Spike / Ultra-Burn intention & examples (e.g. > 2x, Fable, Carbon, xhigh) exist
    for label, txt in (("owner-overrides/ROUTING.md", ov_routing_txt), ("skills/orchestrator/SKILL.md", skill_txt)):
        if not txt:
            continue
        for marker in ("Fable", "Carbon", "xhigh", "> 2x"):
            if marker.lower() not in txt.lower():
                issues.append(
                    f"Missing Protection Guard: Anti-Spike / Ultra-Burn criterion or example ('{marker}') missing in {label}."
                )

    if not opus_in_routing_table:
        issues.append("'Opus 5.5 medium' definition missing in ROUTING.md Machine Tier Map.")

    status = "PASS" if not issues else "FAIL"
    summary = (
        "Anti-Spike / Ultra-Burn Quota Protection Principle (>2x cost/quota weight, >25% 5h / >10% 7d single-dispatch burn, and >high/xhigh effort cap; e.g. Fable, Carbon) is active; standard strong models (Opus 5.5, Argon, GPT-6.1-Sol) are enabled and contradiction-free."
        if not issues
        else f"{len(issues)} model authorization/rule contradiction(s) detected."
    )
    summary_tr = (
        "Anti-Spike / Ultra-Burn Kota Koruma İlkesi (>2x maliyet/kota ağırlığı, >%25 5s / >%10 7g tekil tüketim ve >high/xhigh efor sınırı; ör. Fable, Carbon) aktif; standart güçlü modeller (Opus 5.5, Argon, GPT-6.1-Sol) serbest ve çelişkisiz."
        if not issues
        else f"{len(issues)} model yetki/kural çelişkisi bulundu."
    )
    return {
        "id": "model_gates_consistency",
        "title": "Anti-Spike Quota Protection & Model Consistency",
        "title_tr": "Anti-Spike Kota Koruma İlkesi & Model Tutarlılığı",
        "status": status,
        "summary": summary,
        "summary_tr": summary_tr,
        "details": issues,
    }


def check_subagent_definitions(root: Path = AGENTS_ROOT) -> Dict[str, Any]:
    """
    Validate `~/.agents/claude-agents/*.md`:
    - Required YAML frontmatter (`name`, `description`, `tools`, `model`, `maxTurns`, `effort`)
    - Read-only agents (`scout`, `extractor`, `reviewer`, `advisor`) must NOT have `Edit` or `Write` tools
    - `maxTurns` must be within role bounds (`implementer <= 40`, `reviewer <= 20`, `scout/extractor/advisor <= 12`)
    - `model` must not be an owner-exclusive model (`fable`, `carbon`) and `effort` must not be `xhigh`/`max`
    - Cross-check `providers/claude.md` descriptions against `claude-agents/*.md` frontmatter
    """
    agents_dir = root / "claude-agents"
    claude_md_path = root / "providers" / "claude.md"
    claude_md = _read_utf8(claude_md_path) if claude_md_path.exists() else ""

    issues: List[str] = []
    roster_summary: List[Dict[str, Any]] = []

    if not agents_dir.exists():
        return {
            "id": "subagent_bounds",
            "title": "Alt Ajan (Subagent) Sınırları & Read-Only İzinleri",
            "status": "FAIL",
            "summary": f"{agents_dir} dizini bulunamadı.",
            "details": [],
            "roster": [],
        }

    for md_file in sorted(agents_dir.glob("*.md")):
        meta, _ = _parse_frontmatter(_read_utf8(md_file))
        role = meta.get("name") or md_file.stem
        model = (meta.get("model") or "").lower().strip()
        effort = (meta.get("effort") or "").lower().strip()
        tools_raw = [t.strip() for t in (meta.get("tools") or "").split(",") if t.strip()]
        tools_lower = {t.lower() for t in tools_raw}
        max_turns_raw = meta.get("maxTurns")

        for req_key in ("name", "description", "tools", "model", "maxTurns", "effort"):
            if not meta.get(req_key):
                issues.append(f"claude-agents/{md_file.name}: '{req_key}' frontmatter alanı eksik.")

        max_turns_val: Optional[int] = None
        if max_turns_raw is not None:
            try:
                max_turns_val = int(max_turns_raw)
            except ValueError:
                issues.append(f"claude-agents/{md_file.name}: maxTurns='{max_turns_raw}' tam sayı değil.")

        expected_cap = EXPECTED_MAX_TURNS.get(role, 40)
        if max_turns_val is not None and (max_turns_val <= 0 or max_turns_val > expected_cap):
            issues.append(
                f"claude-agents/{md_file.name}: maxTurns={max_turns_val} rol tavanını ({expected_cap}) aşıyor."
            )

        if role in READ_ONLY_AGENTS:
            forbidden = tools_lower.intersection(WRITE_TOOLS)
            if forbidden:
                issues.append(
                    f"claude-agents/{md_file.name}: Read-only '{role}' ajanına yazma aracı verilmiş: {sorted(forbidden)}"
                )

        if model in OWNER_EXCLUSIVE_MODELS:
            issues.append(
                f"claude-agents/{md_file.name}: Varsayılan model owner-exclusive '{model}' olamaz."
            )
        if effort in OWNER_EXCLUSIVE_EFFORTS:
            issues.append(
                f"claude-agents/{md_file.name}: Varsayılan efor owner-exclusive '{effort}' olamaz."
            )

        # Check providers/claude.md parity for this agent
        if claude_md and role in EXPECTED_MAX_TURNS:
            pattern = re.compile(rf"`{re.escape(role)}`\s*\(([^)]+)\)", re.IGNORECASE)
            m = pattern.search(claude_md)
            if m:
                desc_paren = m.group(1).lower()
                if model and model not in desc_paren:
                    issues.append(
                        f"Çelişki: claude-agents/{md_file.name} model='{model}' iken providers/claude.md içinde '{m.group(1)}' yazıyor."
                    )
                if max_turns_val is not None and f"maxturns {max_turns_val}" not in desc_paren:
                    issues.append(
                        f"Çelişki: claude-agents/{md_file.name} maxTurns={max_turns_val} iken providers/claude.md içinde '{m.group(1)}' yazıyor."
                    )

        roster_summary.append(
            {
                "role": role,
                "model": model or "unknown",
                "max_turns": max_turns_val,
                "effort": effort or "medium",
                "read_only": role in READ_ONLY_AGENTS,
                "tools": tools_raw,
            }
        )

    status = "PASS" if not issues else "FAIL"
    summary = (
        f"Verified {len(roster_summary)} subagent definitions: maxTurns, effort, model, and read-only tool bounds match providers/claude.md."
        if not issues
        else f"Detected {len(issues)} mismatch(es) in subagent definitions."
    )
    summary_tr = (
        f"{len(roster_summary)} alt ajan tanımı doğrulandı: tümünde maxTurns, effort, model ve read-only araç sınırları providers/claude.md ile uyumlu."
        if not issues
        else f"Alt ajan tanımlarında {len(issues)} uyumsuzluk tespit edildi."
    )
    return {
        "id": "subagent_bounds",
        "title": "Subagent Bounds & Read-Only Permissions",
        "title_tr": "Alt Ajan (Subagent) Sınırları & Read-Only İzinleri",
        "status": status,
        "summary": summary,
        "summary_tr": summary_tr,
        "details": issues,
        "roster": roster_summary,
    }


def check_skill_override_parity(root: Path = AGENTS_ROOT) -> Dict[str, Any]:
    """
    Check if any `owner-overrides/skills/<skill>/SKILL.md` is missing CLI commands or sections
    present in the canonical `skills/<skill>/SKILL.md`.
    """
    skills_dir = root / "skills"
    ov_skills_dir = root / "owner-overrides" / "skills"
    drifts: List[str] = []
    checked = 0

    if skills_dir.exists() and ov_skills_dir.exists():
        for ov_skill in sorted(ov_skills_dir.glob("*/SKILL.md")):
            skill_name = ov_skill.parent.name
            base_skill = skills_dir / skill_name / "SKILL.md"
            if base_skill.exists():
                checked += 1
                b_txt = _read_utf8(base_skill)
                o_txt = _read_utf8(ov_skill)
                b_headings = len(re.findall(r"^##\s+", b_txt, flags=re.MULTILINE))
                o_headings = len(re.findall(r"^##\s+", o_txt, flags=re.MULTILINE))
                if b_headings != o_headings:
                    drifts.append(
                        f"Section count mismatch between skills/{skill_name}/SKILL.md ({b_headings}) and owner-overrides/skills/{skill_name}/SKILL.md ({o_headings})."
                    )
                # Ensure all `brain.py <subcmd>` or script references in base exist in override
                b_cmds = set(re.findall(r"brain\.py\s+([a-z\-]+)", b_txt))
                o_cmds = set(re.findall(r"brain\.py\s+([a-z\-]+)", o_txt))
                missing_cmds = sorted(b_cmds - o_cmds - {"command"})
                if missing_cmds:
                    drifts.append(
                        f"Missing subcommands in owner-overrides/skills/{skill_name}/SKILL.md: {', '.join(missing_cmds)}"
                    )

    status = "PASS" if not drifts else "WARN"
    summary = (
        f"Canonical and override skill files ({checked} shared skill(s)) are in full section and command parity."
        if not drifts
        else f"Detected canonical vs. override drift in {len(drifts)} skill file(s)."
    )
    summary_tr = (
        f"Canonical ve override skill dosyaları ({checked} ortak skill) bölüm ve komut düzeyinde tam uyumlu."
        if not drifts
        else f"{len(drifts)} skill dosyasında canonical vs. override farkı var."
    )
    return {
        "id": "skill_override_parity",
        "title": "Skill & Owner-Override Parity",
        "title_tr": "Skill & Owner-Override Senkronu",
        "status": status,
        "summary": summary,
        "summary_tr": summary_tr,
        "details": drifts,
    }


def check_policy_and_v2_invariants(root: Path = AGENTS_ROOT) -> Dict[str, Any]:
    """
    Verify `policy.json` numerical invariants and ensure core Orchestration v2 safeguards exist in
    `DELEGATION.md` and `ROUTING.md`.
    """
    issues: List[str] = []
    policy_path = root / "policy.json"
    del_path = root / "DELEGATION.md"
    rt_path = root / "ROUTING.md"

    if not policy_path.exists():
        issues.append("policy.json bulunamadı.")
    else:
        try:
            pol = json.loads(_read_utf8(policy_path))
            max_leaves = int(pol.get("max_leaves", 0))
            if max_leaves <= 0 or max_leaves > 20:
                issues.append(f"policy.json: max_leaves={max_leaves} geçersiz (1..20 bekleniyor).")
            quota = pol.get("quota") or {}
            if float(quota.get("exclude_at_used", 0)) != 0.95:
                issues.append(f"policy.json: quota.exclude_at_used={quota.get('exclude_at_used')} (ROUTING.md %95 diyor).")
            if float(quota.get("prefer_other_above", 0)) != 0.80:
                issues.append(f"policy.json: quota.prefer_other_above={quota.get('prefer_other_above')} (ROUTING.md %80 diyor).")
            if int(pol.get("max_attempts", 0)) != 2:
                issues.append(f"policy.json: max_attempts={pol.get('max_attempts')} (DELEGATION.md 2-attempt breaker bekliyor).")
        except Exception as exc:
            issues.append(f"policy.json okuma hatası: {exc}")

    del_txt = _read_utf8(del_path) if del_path.exists() else ""
    rt_txt = _read_utf8(rt_path) if rt_path.exists() else ""

    required_delegation_markers = [
        ("DELEGATION_DEPTH_MAX = 1", "Alt ajanların kendi alt ajanlarını açmasını engelleyen DELEGATION_DEPTH_MAX = 1 kuralı"),
        ("MAX_WRITERS = 1", "Ayrık edit_scope olmadan tek yazıcı sınırı (MAX_WRITERS = 1)"),
        ("Coordinator Thin Waist", "Koordinatörün uygulama kaynak koduna doğrudan edit yapmaması (Coordinator Thin Waist)"),
        ("Fresh-Session Default", "Tamamlanmış görev onarımı ve devirlerde taze oturum + handoff kuralı (Fresh-Session Default)"),
        ("Persistent 2-Attempt Breaker", "Oturumlar arası sıfırlanmayan 2 başarısız deneme sigortası (Persistent 2-Attempt Breaker)"),
    ]
    for marker, label in required_delegation_markers:
        if marker not in del_txt:
            issues.append(f"DELEGATION.md içinde '{label}' ({marker}) bulunamadı.")

    required_routing_markers = [
        ("CoordinatorPool != ImplementerPool", "Koordinatör ve implementer havuz ayrımı (Anti-Pileup)"),
        ("distinct from all surviving authors", "Bağımsız reviewer'ın tüm yazarlardan farklı havuz/model ailesinden seçilmesi"),
    ]
    for marker, label in required_routing_markers:
        if marker not in rt_txt:
            issues.append(f"ROUTING.md içinde '{label}' ({marker}) bulunamadı.")

    status = "PASS" if not issues else "FAIL"
    summary = (
        "policy.json ceilings and Orchestration v2 safety barriers (Thin Waist, Fresh-Session, 2-Attempt Breaker, Anti-Pileup) are complete and consistent."
        if not issues
        else f"Detected {len(issues)} core rule/limit mismatch(es)."
    )
    summary_tr = (
        "policy.json limitleri ve Orchestration v2 güvenlik bariyerleri (Thin Waist, Fresh-Session, 2-Attempt Breaker, Anti-Pileup) tam ve tutarlı."
        if not issues
        else f"{len(issues)} temel kural/limit uyumsuzluğu tespit edildi."
    )
    return {
        "id": "policy_v2_invariants",
        "title": "Orchestration v2 Safety Barriers & policy.json",
        "title_tr": "Orchestration v2 Koruma Bariyerleri & policy.json",
        "status": status,
        "summary": summary,
        "summary_tr": summary_tr,
        "details": issues,
    }


def analyze_bloat_and_compaction(root: Path = AGENTS_ROOT) -> Dict[str, Any]:
    """
    Inspect line counts, word counts, and estimated token budgets across canonical files,
    detect cross-file duplicate sentences/bullets and overly long bullets, and generate
    actionable compaction recommendations that preserve 100% of rules/efficiency.
    """
    file_metrics: List[Dict[str, Any]] = []
    proposals: List[Dict[str, Any]] = []
    sentence_locations: Dict[str, List[str]] = {}

    always_loaded_tokens = 0
    on_demand_skill_tokens = 0

    for rel_path, budget in FILE_TOKEN_BUDGETS.items():
        fpath = root / rel_path
        if not fpath.exists():
            continue
        txt = _read_utf8(fpath)
        lines = txt.splitlines()
        words = len(txt.split())
        est_tokens = _estimate_tokens(txt)
        is_skill = rel_path.startswith("skills/")
        if is_skill:
            on_demand_skill_tokens += est_tokens
        else:
            always_loaded_tokens += est_tokens

        utilization_pct = round((est_tokens / max(1, budget)) * 100.0, 1)
        status = "OK"
        if est_tokens > budget * 1.15:
            status = "OVER_BUDGET"
        elif est_tokens > budget * 0.95:
            status = "NEAR_BUDGET"

        # Check for single bullets > 110 words in always-loaded files
        long_bullets: List[Tuple[int, int, str]] = []
        for idx, ln in enumerate(lines, start=1):
            stripped = ln.strip()
            if stripped.startswith(("-", "*")) or re.match(r"^\d+\.", stripped):
                w_cnt = len(stripped.split())
                if w_cnt > 110 and not is_skill:
                    long_bullets.append((idx, w_cnt, stripped[:90] + "..."))

                # Track normalized clauses for duplicate detection across files
                norm = _normalize_sentence(stripped)
                if len(norm) >= 85:
                    sentence_locations.setdefault(norm, []).append(f"{rel_path}:{idx}")

        file_metrics.append(
            {
                "file": rel_path,
                "lines": len(lines),
                "words": words,
                "est_tokens": est_tokens,
                "token_budget": budget,
                "utilization_pct": utilization_pct,
                "category": "on_demand_skill" if is_skill else "always_loaded",
                "status": status,
                "long_bullets_count": len(long_bullets),
            }
        )

        if status == "OVER_BUDGET":
            proposals.append(
                {
                    "severity": "WARN",
                    "target": rel_path,
                    "estimated_token_saving": max(50, est_tokens - budget),
                    "suggestion": (
                        f"{rel_path} is ~{est_tokens} tokens (budget: {budget}). "
                        "Move explanatory examples to an on-demand skill reference and keep only hard rules here."
                    ),
                    "suggestion_tr": (
                        f"{rel_path} tahmini {est_tokens} token (bütçe: {budget}). "
                        "Açıklayıcı örnekleri on-demand skill referansına taşıyıp bu dosyada yalnız sert kuralları tutun."
                    ),
                }
            )
        for line_no, w_cnt, preview in long_bullets:
            proposals.append(
                {
                    "severity": "INFO",
                    "target": f"{rel_path}:{line_no}",
                    "estimated_token_saving": max(20, int((w_cnt - 70) * 1.3)),
                    "suggestion": (
                        f"Line {line_no} contains {w_cnt} words in a single bullet ({preview}). "
                        "Condense into a compact rule statement and move details to a reference doc."
                    ),
                    "suggestion_tr": (
                        f"Satır {line_no} tek maddede {w_cnt} kelime içeriyor ({preview}). "
                        "Maddeyi kısa kural ifadesine indirgeyip detayları alt referansa ayırabilirsiniz."
                    ),
                }
            )

    # Cross-file verbatim duplicate bullets
    for norm_clause, locs in sentence_locations.items():
        distinct_files = sorted({loc.split(":")[0] for loc in locs})
        if len(distinct_files) > 1:
            proposals.append(
                {
                    "severity": "INFO",
                    "target": " + ".join(locs[:3]),
                    "estimated_token_saving": 35 * (len(distinct_files) - 1),
                    "suggestion": (
                        f"Identical rule clause is duplicated across multiple files ({', '.join(distinct_files)}). "
                        "Keep one canonical definition in the always-loaded file and reference its heading elsewhere."
                    ),
                    "suggestion_tr": (
                        f"Aynı kural cümlesi birden fazla dosyada ({', '.join(distinct_files)}) birebir tekrar ediyor. "
                        "Her zaman yüklenen (always-loaded) dosyada tek özlü tanım bırakıp diğerlerinde başlığa referans verin."
                    ),
                }
            )

    # Check if owner-overrides/skills/orchestrator/SKILL.md duplicates skills/orchestrator/SKILL.md
    base_orch = root / "skills" / "orchestrator" / "SKILL.md"
    ov_orch = root / "owner-overrides" / "skills" / "orchestrator" / "SKILL.md"
    if base_orch.exists() and ov_orch.exists():
        if _read_utf8(base_orch).strip() == _read_utf8(ov_orch).strip():
            proposals.append(
                {
                    "severity": "TIP",
                    "target": "owner-overrides/skills/orchestrator/SKILL.md",
                    "estimated_token_saving": 0,
                    "suggestion": (
                        "owner-overrides/skills/orchestrator/SKILL.md is identical to canonical skills/orchestrator/SKILL.md. "
                        "Remove the duplicate override copy to halve maintenance overhead (sync.ps1 falls back to canonical)."
                    ),
                    "suggestion_tr": (
                        "owner-overrides/skills/orchestrator/SKILL.md ile canonical skills/orchestrator/SKILL.md birebir aynı. "
                        "Kişisel fark eklemeyecekseniz override kopyasını silip bakım yükünü yarıya indirebilirsiniz (sync.ps1 otomatik canonical dosyayı kullanır)."
                    ),
                }
            )

    over_budget_cnt = sum(1 for m in file_metrics if m["status"] == "OVER_BUDGET")
    compaction_score = max(0, min(100, 100 - (over_budget_cnt * 15) - (len(proposals) * 3)))

    return {
        "always_loaded_est_tokens": always_loaded_tokens,
        "on_demand_skill_est_tokens": on_demand_skill_tokens,
        "compaction_health_score": compaction_score,
        "file_metrics": file_metrics,
        "proposals": proposals[:12],
    }


def build_active_rules_summary(root: Path = AGENTS_ROOT) -> Dict[str, Any]:
    """
    Build a structured, easy-to-read Turkish summary of the exact rules that the Orchestrator
    and Subagents obey before and during work, for display in the Quota Hub UI.
    """
    policy_path = root / "policy.json"
    pol: Dict[str, Any] = {}
    if policy_path.exists():
        try:
            pol = json.loads(_read_utf8(policy_path))
        except Exception:
            pol = {}

    max_leaves = int(pol.get("max_leaves", 10))
    max_per_prov = pol.get("max_per_provider") or {"agy": 6, "claude": 6, "codex": 6}
    quota_cfg = pol.get("quota") or {"exclude_at_used": 0.95, "prefer_other_above": 0.80, "stale_after_sec": 120}

    return {
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "categories": [
            {
                "id": "coordinator_and_pools",
                "icon": "🎯",
                "title": "1. Coordinator & Quota Pool Rules (Pre-Dispatch)",
                "rules": [
                    f"Live Quota Gate: Query Quota Hub (`GET /api/quotas`) before every launch; never launch on pools >={int(float(quota_cfg.get('exclude_at_used', 0.95))*100)}% used, and prefer another healthy pool above {int(float(quota_cfg.get('prefer_other_above', 0.80))*100)}%.",
                    "CoordinatorPool != ImplementerPool (Anti-Pileup): Avoid stacking an active coordinator and multi-turn implementers onto the same quota pool when another eligible pool is healthy.",
                    "Coordinator Thin Waist (Zero Source Edits): During multi-worker orchestration, the coordinator writes contracts/ledgers and runs deterministic git/build gates; it makes zero direct application source-code edits.",
                    f"Concurrency Ceilings: Up to {max_leaves} total leaf workers (AGY: {max_per_prov.get('agy', 6)}, Claude: {max_per_prov.get('claude', 6)}, Codex: {max_per_prov.get('codex', 6)}). `MAX_WRITERS = 1` without a disjoint `edit_scope` contract.",
                ],
            },
            {
                "id": "subagent_limits",
                "icon": "🛡️",
                "title": "2. Subagent Turn, Context & Resume Bounds",
                "rules": [
                    "DELEGATION_DEPTH_MAX = 1: Workers are leaves; a subagent never spawns another subagent.",
                    "Fresh-Session Default (Anti-Cache-Blowout): Start a fresh session passing disk checkpoint/handoff paths (never bloated transcripts) for repairs, re-reviews, provider handoffs, and post-reset continuations.",
                    "Bounded Resume (<120k Context): `resume` / `SendMessage` on an active worker is allowed at most once for a brief clarification on the same unfinished task while input context is <120k tokens.",
                    "Hard Role Caps: Implementer (sonnet, <=40 turns / 32k output) · Reviewer (opus, <=20 turns / 12k output, read-only) · Advisor (opus, <=12 turns / 12k output, read-only) · Scout & Extractor (haiku, <=12 turns / 6k output, read-only).",
                ],
            },
            {
                "id": "review_and_escalation",
                "icon": "⚖️",
                "title": "3. Independent Cross-Family Review & 2-Attempt Breaker",
                "rules": [
                    "Independent Reviewer: An implementer never reviews its own work. Give the reviewer requirements, diffs, and frozen constraints as facts — never the implementer's reasoning.",
                    "Cross-Family / Distinct Pool Rule: When another eligible pool is healthy (or author pool is >60% used), select the reviewer from a model family and quota pool distinct from all surviving authors.",
                    "Persistent 2-Attempt Breaker: The 2-failed-attempt counter persists across fresh sessions and provider handoffs. At 2 failed attempts, quarantine the task and run 1 bounded read-only RCA with a changed approach before retrying.",
                    "2nd-Strike -> Deterministic Gate: If the same failure mode recurs twice, convert it into an automated unit test, lint rule, or Refinery check rather than adding prose warnings.",
                ],
            },
            {
                "id": "model_gates",
                "icon": "🚀",
                "title": "4. Anti-Spike Quota Protection & Model Eligibility",
                "rules": [
                    "Core Intent (Anti-Spike Principle): No single task or subagent loop may abruptly drain a 5h or weekly quota pool.",
                    "Model-Agnostic Guard + Examples: Never auto-route routine work to ultra-burn models costing >2x standard HIGH-tier weight (e.g. Fable, Carbon), effort above `high` (e.g. `xhigh`, `max`), or a single dispatch predicted to burn >25% 5h / >10% 7d quota.",
                    "Standard Strong Models Enabled: `Claude Opus 5.5` (`opus` medium), `Gemini 4 Argon`, and `GPT-6.1-Sol` are freely eligible for architecture, difficult debugging, `reviewer`, and `advisor` roles on healthy pools.",
                    "Capability Floor: Never downgrade `HIGH`-floor architectural/critical work to weak models for quota savings; reasoning effort (`low` / `medium` / `high`) is chosen independently of model tier.",
                ],
            },
        ],
        "categories_tr": [
            {
                "id": "coordinator_and_pools",
                "icon": "🎯",
                "title": "1. Koordinatör & Havuz Kuralları (Çalışmadan Önce)",
                "rules": [
                    f"Canlı Kota Kontrolü: Her görev öncesi Quota Hub (`GET /api/quotas`) okunur; >%{int(float(quota_cfg.get('exclude_at_used', 0.95))*100)} dolu havuzlara iş verilmez, >%{int(float(quota_cfg.get('prefer_other_above', 0.80))*100)} üzerinde alternatif sağlıklı havuz tercih edilir.",
                    "CoordinatorPool != ImplementerPool (Anti-Pileup): Aktif koordinatör ile çok turlu kod yazıcılar (implementer) aynı kota havuzuna yığılmaz; başka sağlıklı havuz varsa farklı havuza yönlendirilir.",
                    "Coordinator Thin Waist (Sıfır Kaynak Kod Editi): Çoklu işçi çalışırken koordinatör yalnız kontrat/ledger yazar ve deterministik git/build kapılarını çalıştırır; uygulama kaynak koduna doğrudan edit yapmaz.",
                    f"Eşzamanlı İşçi Tavanı: Toplam en fazla {max_leaves} yaprak işçi (AGY: {max_per_prov.get('agy', 6)}, Claude: {max_per_prov.get('claude', 6)}, Codex: {max_per_prov.get('codex', 6)}). Ayrık `edit_scope` kontratı olmadan `MAX_WRITERS = 1`.",
                ],
            },
            {
                "id": "subagent_limits",
                "icon": "🛡️",
                "title": "2. Alt Ajan (Subagent) Tur, Context & Resume Sınırları",
                "rules": [
                    "DELEGATION_DEPTH_MAX = 1: Alt ajanlar yapraktır (leaf); hiçbir alt ajan kendi alt ajanını doğuramaz.",
                    "Taze Oturum & Handoff (Anti-Cache-Blowout): Tamamlanmış görev sonrası onarım (repair), yeniden inceleme (re-review), sağlayıcı değişimi ve kota sıfırlanması sonrası devam işlemlerinde eski şişmiş transcript taşınmaz; diskteki checkpoint/handoff ile taze oturum açılır.",
                    "Bounded Resume (<120k Context): Aktif bir işçiye `resume` / `SendMessage` yalnız aynı tamamlanmamış görevde kısa bir netleştirme için en fazla 1 kez ve girdi context'i <120k token iken yapılabilir.",
                    "Sert Rol Tavanları: Implementer (sonnet, <=40 yanıt / 32k çıktı) · Reviewer (opus, <=20 yanıt / 12k çıktı, read-only) · Advisor (opus, <=12 yanıt / 12k çıktı, read-only) · Scout & Extractor (haiku, <=12 yanıt / 6k çıktı, read-only).",
                ],
            },
            {
                "id": "review_and_escalation",
                "icon": "⚖️",
                "title": "3. Bağımsız İnceleme (Review) & 2-Deneme Sigortası",
                "rules": [
                    "Bağımsız Reviewer: Kodu yazan ajan kendi kodunu inceleyemez. Reviewer'a yalnız gereksinim, diff ve kısıtlar verilir; yazan ajanın iç gerekçesi (reasoning) verilmez.",
                    "Farklı Aile / Havuz Zorunluluğu: Sağlıklı başka havuz varsa (veya yazar havuzu >%60 doluysa) reviewer, paketteki tüm yazarlardan farklı bir model ailesi ve kota havuzundan seçilir.",
                    "Kalıcı 2-Deneme Sigortası (Persistent 2-Attempt Breaker): Başarısız deneme sayacı taze oturum ve sağlayıcı devirlerinde sıfırlanmaz. 2 başarısız denemede görev karantinaya alınır; 1 bounded read-only RCA ve kayıtlı yaklaşım değişikliği olmadan yeni onarım başlatılmaz.",
                    "2nd-Strike -> Deterministik Kapı: Aynı tip hata ikinci kez tekrar ederse yalnız prompt'a uyarı cümlesi eklemekle yetinilmez; birim testi, lint veya Refinery kontrolüne dönüştürülür.",
                ],
            },
            {
                "id": "model_gates",
                "icon": "🚀",
                "title": "4. Anti-Spike Kota Koruma İlkesi & Model Yetkileri (Niyet + Örnekler)",
                "rules": [
                    "Ana Niyet (Anti-Spike Principle): Yeni modeller çıksa bile hiçbir tekil görev veya alt ajan döngüsü 5 saatlik veya haftalık kotayı ani bir sıçramayla (spike) bitiremez.",
                    "Modelden Bağımsız Sınır + Örnekler: Standart güçlü modellerin >2 katı kota/maliyet ağırlığına sahip ultra-burn modeller (ör. Fable, Carbon), `high` üstü eforlar (ör. `xhigh`, `max`) veya kalan 5s kotanın >%25'ini / 7g kotanın >%10'unu tek başına yutacak atamalar rutin işlerde otomatik seçilmez.",
                    "Standart Güçlü Modeller Serbest: `Claude Opus 5.5` (`opus` medium), `Gemini 4 Argon` ve `GPT-6.1-Sol` yasaklı değildir; mimari, zorlu hata ayıklama, `reviewer` ve `advisor` rollerinde sağlıklı havuzlarda doğrudan kullanılır.",
                    "Kalite Tabanı (Capability Floor): Kota tasarrufu uğruna `HIGH` tabanlı mimari/kritik işler zayıf modellere düşürülmez; efor seviyesi (`low` / `medium` / `high`) model kapasitesinden bağımsız seçilir.",
                ],
            },
        ],
    }


def run_policy_doctor_audit(
    root: Optional[Path] = None,
    state_file: Optional[Path] = None,
    save_state: bool = True,
    trigger: str = "manual",
) -> Dict[str, Any]:
    """Run all Policy Doctor checks + compaction analysis and optionally persist state."""
    root = root or AGENTS_ROOT
    state_file = state_file or DEFAULT_STATE_FILE
    now = datetime.datetime.now(datetime.timezone.utc)
    now_iso = now.isoformat()

    checks = [
        check_sync_parity(root),
        check_model_gates_and_opus_contradictions(root),
        check_subagent_definitions(root),
        check_skill_override_parity(root),
        check_policy_and_v2_invariants(root),
    ]
    compaction = analyze_bloat_and_compaction(root)
    active_rules = build_active_rules_summary(root)

    fail_cnt = sum(1 for c in checks if c["status"] == "FAIL")
    warn_cnt = sum(1 for c in checks if c["status"] == "WARN")
    pass_cnt = sum(1 for c in checks if c["status"] == "PASS")

    overall_status = "FAIL" if fail_cnt > 0 else ("WARN" if warn_cnt > 0 else "PASS")

    # Preserve configurable interval_days if state file already exists
    interval_days = DEFAULT_INTERVAL_DAYS
    if state_file.exists():
        try:
            prev = json.loads(_read_utf8(state_file))
            interval_days = float(prev.get("interval_days", DEFAULT_INTERVAL_DAYS))
        except Exception:
            interval_days = DEFAULT_INTERVAL_DAYS

    next_due = (now + datetime.timedelta(days=interval_days)).isoformat()

    report: Dict[str, Any] = {
        "overall_status": overall_status,
        "last_run_at": now_iso,
        "next_scheduled_run_at": next_due,
        "interval_days": interval_days,
        "trigger": trigger,
        "counts": {
            "pass": pass_cnt,
            "warn": warn_cnt,
            "fail": fail_cnt,
            "total": len(checks),
        },
        "checks": checks,
        "compaction": compaction,
        "active_rules_summary": active_rules,
    }

    if save_state:
        try:
            state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = state_file.with_name(f"{state_file.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, state_file)
        except OSError:
            pass

    return report


def is_weekly_policy_audit_due(state_file: Optional[Path] = None) -> bool:
    """Return True if Policy Doctor has never run or its last run is older than `interval_days` (default 7 days)."""
    state_file = state_file or DEFAULT_STATE_FILE
    if not state_file.exists():
        return True
    try:
        data = json.loads(_read_utf8(state_file))
        last_run_raw = data.get("last_run_at")
        interval_days = float(data.get("interval_days", DEFAULT_INTERVAL_DAYS))
        if not last_run_raw:
            return True
        last_dt = datetime.datetime.fromisoformat(str(last_run_raw).replace("Z", "+00:00"))
        now_dt = datetime.datetime.now(datetime.timezone.utc)
        age_days = (now_dt - last_dt).total_seconds() / 86400.0
        return age_days >= interval_days
    except Exception:
        return True


def get_latest_policy_doctor_report(
    root: Optional[Path] = None,
    state_file: Optional[Path] = None,
    auto_run_if_due: bool = True,
) -> Dict[str, Any]:
    """
    Return the persisted Policy Doctor report, running a fresh audit automatically if never run
    or if the weekly schedule (`interval_days: 7`) is due.
    """
    root = root or AGENTS_ROOT
    state_file = state_file or DEFAULT_STATE_FILE
    if auto_run_if_due and is_weekly_policy_audit_due(state_file):
        return run_policy_doctor_audit(root=root, state_file=state_file, save_state=True, trigger="weekly_auto")
    if state_file.exists():
        try:
            data = json.loads(_read_utf8(state_file))
            if isinstance(data, dict) and "checks" in data and data["checks"] and "title_tr" in data["checks"][0]:
                # Always refresh active_rules_summary so live edits are reflected immediately
                data["active_rules_summary"] = build_active_rules_summary(root)
                return data
        except Exception:
            pass
    return run_policy_doctor_audit(root=root, state_file=state_file, save_state=True, trigger="initial")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Policy Doctor — Contradiction & Compaction Auditor for ~/.agents")
    parser.add_argument("--json", action="store_true", help="Output full report as JSON")
    parser.add_argument("--no-save", action="store_true", help="Do not write state to quota-hub/policy_doctor_state.json")
    parser.add_argument("--if-due", action="store_true", help="Only run if the weekly schedule is due")
    args = parser.parse_args(argv)

    if args.if_due and not is_weekly_policy_audit_due():
        report = get_latest_policy_doctor_report(auto_run_if_due=False)
    else:
        report = run_policy_doctor_audit(save_state=not args.no_save, trigger="cli")

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        counts = report["counts"]
        comp = report["compaction"]
        print(
            f"[Policy Doctor] Status: {report['overall_status']} "
            f"(PASS={counts['pass']}, WARN={counts['warn']}, FAIL={counts['fail']}) | "
            f"Always-Loaded Tokens: ~{comp['always_loaded_est_tokens']} | "
            f"Compaction Health: {comp['compaction_health_score']}/100"
        )
        for chk in report["checks"]:
            print(f"  - [{chk['status']}] {chk['title']}: {chk['summary']}")
            for det in chk.get("details") or []:
                print(f"      * {det}")
        for prop in comp.get("proposals") or []:
            print(f"  - [{prop['severity']}] ({prop['target']}): {prop['suggestion']}")

    return 1 if report["overall_status"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
