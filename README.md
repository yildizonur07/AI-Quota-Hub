# AI Quota Hub & Reset-Aware Multi-Provider Quota Scheduler

> **⚠️ Experimental / "As-Is" Community Release (PRs Welcome)**
>
> This project is a personal developer tool built to monitor live 5-hour and weekly quota windows across **Google Antigravity (`agy`)**, **Claude Code (`claude`)**, and **OpenAI Codex (`codex`)**, and to route multi-agent work packages without sudden quota exhaustion.
> Because AI CLI tools and telemetry formats evolve rapidly, this repository is shared **"as-is"** without commercial support guarantees. Issues, forks, and Pull Requests from the community are warmly welcome!
>
> *Bu proje; **Google Antigravity (`agy`)**, **Claude Code (`claude`)** ve **OpenAI Codex (`codex`)** üzerindeki 5 saatlik ve haftalık kota pencerelerini sıfır token maliyetiyle izlemek ve alt-ajan iş paketlerini kota patlaması yaşamadan matematiksel olarak yönlendirmek için geliştirilmiş kişisel bir mühendislik aracıdır. Sağlayıcı arayüzleri sık değiştiği için **"olduğu gibi" (as-is)** açık kaynak olarak paylaşılmıştır; topluluk katkıları (PR / Fork) memnuniyetle kabul edilir.*

---

## 🎛️ SCADA / High-Performance HMI Single-Screen Dashboard

The Web Dashboard (`http://127.0.0.1:8765`) is designed with a **SCADA / High-Performance HMI (L1 Overview + L2 Drill-Down Faceplates)** philosophy:
- **Zero Vertical Page Scroll (`100vh`)**: All critical operational telemetry is visible at a single glance on **Screen 1 (`[1] 🎛️ L1 SCADA Genel Bakış`)**:
  - **3 Provider Telemetry Units** (`AGY` dual pools `agy_gemini` + `agy_claude_gpt`, `Claude Code`, `OpenAI Codex`) with 5h/7d bars, Usable/Stranded ratios, Reset timers, and collapsible `[🔍 Modeller]` faceplates.
  - **0–10 Provider Preference Sliders & Live SWRR Allocation Table** (`Target Share`, `Actual 24h Share`, `Sigma Band`, `Max Workers`, `Effort Policy`).
  - **Live Running Processes (`Coordinators` & `Headless Workers`)** with automatic workspace resolution.
  - **4 Quota Pools Live Matrix (`Usable` · `Stranded` · `Opportunity Cost Φ`)**, **Policy Doctor & Reserve Health LEDs**, and **6 One-Click L2 Faceplate Pushbuttons** (also accessible via keyboard shortcuts `1`–`7` and `ESC` to return to `L1`).
- **Contextual `?` Help Guides on Every Panel**: Every card and table header includes a `?` button that opens an interactive explanation of its metrics, formulas, and practical usage.

---

## Mimari & Dosya Yapısı (Architecture)

- [agy_collector.py](collectors/agy_collector.py): Yerel `language_server` sürecini, PID ve `--csrf_token` bilgisini yakalar; Connect Protocol üzerinden `GetUserStatus` RPC uç noktasına bağlanarak `Gemini` (`agy_gemini`) ve `Claude/GPT` (`agy_claude_gpt`) havuzlarının kotalarını ve `resetTime` sayaçlarını ayrıştırır.
- [claude_collector.py](collectors/claude_collector.py): `~/.claude/` altındaki oturum/maliyet (`cost-state`, `total_tokens_reminder`) loglarını ve OAuth kullanım verilerini (`fh`: 5-saatlik, `sd`: 7-günlük) birleştirerek `haiku`, `sonnet` ve `opus` kotalarını normalize eder.
- [codex_collector.py](collectors/codex_collector.py): `~/.codex/sessions/` altındaki en güncel `rollout-*.jsonl` dosyalarından ve kullanım uç noktasından `rate_limits` (`primary` 300dk, `secondary` 10080dk) ile `banked_resets` verilerini ayrıştırır.
- [allocation.py](allocation.py): `0–10` sağlayıcı tercih sürgüleri (`preference_weights`), gerçek zamanlı kota erime hızı öğrenicisi (`BurnRateLearner`), sürdürülebilir havuz kapasitesi (`C(q)`), sigma bantları (`saver`, `economy`, `normal`) ve Smooth Weighted Round Robin (`step_swrr`) iş dağılım motoru.
- [policy_doctor.py](policy_doctor.py): Orkestratör ve alt-ajan kurallarını (`~/.agents/`) haftada bir otomatik tarayan, model uygunluk ve kota koruma çelişkilerini yakalayan ve her mesajda yüklenen (`Always-Loaded`) kural dosyalarının token bütçesini / kompaktlık puanını (`0–100`) denetleyen **Policy Doctor** motoru (`GET /api/policy-doctor`, `POST /api/policy-doctor/run`).
- [model_catalog.json](model_catalog.json) & [model_review.py](model_review.py): Kod dışına taşınmış kanonik model kataloğu, sıfır-LLM yerel model keşfi (`agy models`, `~/.codex/models_cache.json`, `models.dev`, `OpenRouter`), harmanlanmış fiyat oranı (`blended_price = 0.25·in + 0.75·out`) ve 14 günlük deneme (`probation`) karne özeti.
- [orchestrator.py](orchestrator.py) & [scheduler.py](scheduler.py): Capability Floor (`LOW`, `MID`, `HIGH`), SWRR sağlayıcı seçimi, Cross-Family Independent Review, dinamik Codex Koordinatör Rezervi (`R_orch`) ve `%85 / %95` kota el değiştirme (handoff) kurallarını işletir.
- [main.py](main.py): FastAPI sunucusu (`GET /api/quotas`, `POST /api/preferences`, `POST /api/limits`, `GET /api/policy-doctor`, `POST /api/policy-doctor/run`, `GET /api/models/review`, `POST /api/models/review/run`, `POST /api/models/decision`, `POST /api/orchestrator`, `POST /api/dispatch`, `POST /api/simulate`).
- [static/index.html](static/index.html): SCADA HMI Tek Ekran (L1/L2) Web Dashboard arayüzü.

---

## Hızlı Başlangıç (Quick Start)

```powershell
# 1. Bağımlılıkları yükle / Install dependencies
pip install fastapi uvicorn httpx pydantic pytest

# 2. Sunucuyu ve SCADA Web Dashboard'u başlat (http://127.0.0.1:8765)
python cli.py serve --port 8765

# 3. Komut satırından anlık normalize kotaları sorgula (GET /api/quotas formatında)
python cli.py status

# 4. Policy Doctor kural çelişki ve kompaktlık denetimini çalıştır
python policy_doctor.py

# 5. Komut satırından bir iş paketini en uygun modele yönlendir
python cli.py route --task "WP-REV-01: Cross-family review" --kind review --implementer-family gemini

# 6. Birim ve entegrasyon testlerini çalıştır
python -m pytest -v
```

---

## Kişisel Çalışma Alanı Eşlemeleri (`owner-overrides/quota-hub.json`)

Canlı süreç toplayıcısı ([collectors/live_activity.py](collectors/live_activity.py)), kişisel proje isimlerini kod içinde barındırmaz. Kendi projeleriniz için `~/.agents/owner-overrides/quota-hub.json` dosyasında `workspace_aliases` ve `live_activity_pattern` tanımlayabilirsiniz:

```json
{
  "live_activity_pattern": "my-project|another-project",
  "workspace_aliases": {
    "my project": "~/my-project"
  }
}
```

## Lisans (License)

[MIT License](LICENSE)
