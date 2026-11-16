# AI Quota Hub & Reset-Aware Multi-Provider Quota Scheduler

🌐 **Language / Dil:** **English** | [Türkçe (README.tr.md)](README.tr.md)

> **⚠️ Experimental / "As-Is" Community Release (PRs Welcome)**
>
> **AI Quota Hub** is a local multi-provider quota monitoring and orchestration tool built to track live 5-hour and weekly quota windows across **Google Antigravity (`agy`)**, **Claude Code (`claude`)**, and **OpenAI Codex (`codex`)** at zero token cost, and to mathematically route multi-agent work packages without sudden quota exhaustion.
>
> Because AI CLI tools and telemetry formats evolve rapidly, this repository is shared **"as-is"** without commercial support guarantees. Issues, forks, and Pull Requests from the community are warmly welcome!

---

## 🖥️ Single-Screen Command Center Dashboard

The Web Dashboard (`http://127.0.0.1:8765`) is built around a **Single-Screen Overview + Modal Detail Overlays** layout:
- **Bilingual UI (`🌐 EN | TR`)**: Defaults to **English (`EN`)** with one-click live switching to **Turkish (`TR`)** in the top header bar.
- **Zero Vertical Page Scroll (`100vh`)**: All 10 operational subsystems (Mode/Orchestrator, Live 5h/7d Quotas, 0–10 SWRR Provider Mix, Concurrency Limits, Live Processes & Autonomous Goal, Work Package Router, Reserves & Banked Resets, Policy Doctor, 4 Pools Math & Scorecard, Model Review & JSON API) are visible at a single glance on the main overview grid.
- **Live Telemetry Hover Popovers**: Hovering over any KPI row, provider card, SWRR row, or pool pill opens a live telemetry callout popup with sub-model breakdowns, Usable/Stranded ratios, Huber cost profiles, and Policy Doctor audit results—without clicking.
- **Centered Detail Modals (`🔍 Details` / `ESC`)**: Clicking any card's `🔍 Details` button opens its deep configuration and inspection view as a centered modal overlay without shifting or scrolling the main screen.
- **Contextual `?` Help Guides**: Every card and modal header includes a `?` button that explains its metrics, formulas, and practical usage in the active language (`EN` or `TR`).

---

## 🏗️ Architecture & Key Modules

- [collectors/agy_collector.py](collectors/agy_collector.py): Discovers the local Antigravity `language_server` process, PID, and `--csrf_token`, then queries the Connect Protocol `GetUserStatus` RPC endpoint to normalize both `Gemini` (`agy_gemini`) and `Claude/GPT` (`agy_claude_gpt`) quota pools and reset countdowns.
- [collectors/claude_collector.py](collectors/claude_collector.py): Combines `~/.claude/` session/cost logs (`cost-state`, `total_tokens_reminder`) and OAuth usage windows (`fh`: 5-hour, `sd`: 7-day) into normalized `haiku`, `sonnet`, and `opus` metrics.
- [collectors/codex_collector.py](collectors/codex_collector.py): Parses the latest `rollout-*.jsonl` files under `~/.codex/sessions/` and usage endpoints for `rate_limits` (`primary` 300m, `secondary` 10080m) and `banked_resets`.
- [allocation.py](allocation.py): Implements `0–10` provider preference sliders (`preference_weights`), real-time empirical burn-rate learning (`BurnRateLearner`), sustainable pool capacity (`C(q)`), sigma bands (`saver`, `economy`, `normal`), and Smooth Weighted Round Robin (`step_swrr`) dispatch.
- [policy_doctor.py](policy_doctor.py): Automated weekly/on-demand rule consistency & token-budget auditor (`~/.agents/`), checking cross-provider routing alignment, Anti-Spike protection, and `Always-Loaded` rule compaction score (`0–100`) (`GET /api/policy-doctor`, `POST /api/policy-doctor/run`).
- [model_catalog.json](model_catalog.json) & [model_review.py](model_review.py): Externalized canonical model catalog, zero-LLM local model discovery (`agy models`, `~/.codex/models_cache.json`, `models.dev`, `OpenRouter`), blended price ratio (`0.25·in + 0.75·out`), and 14-day probation scorecard.
- [orchestrator.py](orchestrator.py) & [scheduler.py](scheduler.py): Enforces Capability Floor (`LOW`, `MID`, `HIGH`), SWRR provider selection, Cross-Family Independent Review, dynamic Codex Continuity Reserve (`R_orch`), and `85% / 95%` quota handoff gates.
- [main.py](main.py): FastAPI server (`GET /api/quotas`, `POST /api/preferences`, `POST /api/limits`, `GET /api/policy-doctor`, `POST /api/policy-doctor/run`, `GET /api/models/review`, `POST /api/models/review/run`, `POST /api/models/decision`, `POST /api/orchestrator`, `POST /api/dispatch`, `POST /api/simulate`).
- [static/index.html](static/index.html): Single-Screen Web Dashboard (`EN` default + `TR` option).

---

## 🚀 Quick Start

```powershell
# 1. Install dependencies
pip install fastapi uvicorn httpx pydantic pytest

# 2. Start the server & Web Dashboard (http://127.0.0.1:8765)
python cli.py serve --port 8765

# 3. Query normalized live quotas from the CLI (GET /api/quotas format)
python cli.py status

# 4. Run Policy Doctor rule consistency & compaction audit
python policy_doctor.py

# 5. Route a work package to the best eligible model from the CLI
python cli.py route --task "WP-REV-01: Cross-family review" --kind review --implementer-family gemini

# 6. Run the unit & integration test suite
python -m pytest -v
```

---

## 🔧 Custom Workspace Aliases (`owner-overrides/quota-hub.json`)

The live process collector ([collectors/live_activity.py](collectors/live_activity.py)) contains zero hardcoded personal project paths. You can define your own project aliases and process patterns in `~/.agents/owner-overrides/quota-hub.json`:

```json
{
  "live_activity_pattern": "my-project|another-project",
  "workspace_aliases": {
    "my project": "~/my-project"
  }
}
```

---

## ⚖️ License: GNU AGPLv3 + Commercial Dual-License

This project is dual-licensed under the **GNU Affero General Public License v3.0 ([AGPL-3.0](LICENSE))** for open-source/community use and a **Commercial License** for proprietary enterprise/SaaS use:

1. **Open-Source / Community Use (GNU AGPLv3)**:
   - Free to use, study, modify, and contribute back.
   - If you distribute modified versions or host/offer AI Quota Hub as a network service / SaaS / internal commercial platform to users, you **must** make your complete modified source code publicly available under AGPLv3.
2. **Commercial / Proprietary License**:
   - If your organization wishes to embed AI Quota Hub into a closed-source commercial product, enterprise platform, or SaaS offering without AGPLv3 copyleft obligations, please contact the copyright holder (**Onur Yıldız** via GitHub [@yildizonur07](https://github.com/yildizonur07)) to arrange a separate commercial license.
