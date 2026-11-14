# AI Quota Hub & Reset-Aware Multi-Provider Quota Scheduler (Türkçe)

🌐 **Dil / Language:** [English (README.md)](README.md) | **Türkçe**

> **⚠️ Deneyimsel / "Olduğu Gibi" (As-Is) Açık Kaynak Sürümü (PR ve Katkılara Açık)**
>
> **AI Quota Hub**; **Google Antigravity (`agy`)**, **Claude Code (`claude`)** ve **OpenAI Codex (`codex`)** üzerindeki 5 saatlik ve haftalık kota pencerelerini sıfır token maliyetiyle izlemek ve alt-ajan (subagent) iş paketlerini kota patlaması yaşamadan matematiksel olarak yönlendirmek için geliştirilmiş SCADA tarzı bir mühendislik aracıdır.
>
> Sağlayıcı arayüzleri ve telemetri formatları hızla değiştiği için bu proje ticari destek taahhüdü olmaksızın **"olduğu gibi" (as-is)** paylaşılmıştır. Topluluk katkıları (Issue, Fork, Pull Request) memnuniyetle kabul edilir!

---

## 🎛️ SCADA / High-Performance HMI Tek Ekran Dashboard

Web Dashboard (`http://127.0.0.1:8765`), **SCADA / High-Performance HMI (L1 Ana Genel Bakış + L2 Açılır Faceplate Pencereleri)** felsefesiyle tasarlanmıştır:
- **Çift Dilli Arayüz (`🌐 EN | TR`)**: Varsayılan olarak **İngilizce (`EN`)** açılır; üst bardaki `🌐 EN | TR` butonuyla tek tıkla **Türkçe (`TR`)** diline geçer.
- **Sıfır Dikey Sayfa Kaydırma (`100vh`)**: Tüm 10 ana alt sistemin en kritik bilgileri ve birincil komutları tek bir ekranda (**L1 Genel Bakış Izgarası**) aynı anda görünür.
- **Canlı Telemetri Mouseover (Hover) Popup'ları**: Herhangi bir KPI satırı, sağlayıcı kartı, SWRR satırı veya havuz rozeti üzerine fareyle gelindiğinde (`mouseover`), tıklamaya gerek kalmadan alt model kırılımlarını, Usable/Stranded oranlarını, Huber maliyet profillerini ve Policy Doctor denetim özetini gösteren canlı SCADA popup'ı açılır.
- **Merkezi Modal Faceplate Detayları (`🔍 Detay` / `ESC`)**: Kartlardaki `🔍 Detay` butonuna basıldığında ana ekran hiç kaymadan merkezi açılır pencere (modal faceplate) içinde derinlemesine ayarlar ve tablolar açılır.
- **Bağlamsal `?` Yardım Rehberleri**: Her kartın ve pencerenin sağ üst köşesindeki `?` butonu, seçili dilde (`EN` veya `TR`) ilgili metriklerin ve formüllerin açıklamalarını sunar.

---

## 🏗️ Mimari & Dosya Yapısı

- [collectors/agy_collector.py](collectors/agy_collector.py): Yerel `language_server` sürecini, PID ve `--csrf_token` bilgisini yakalar; Connect Protocol üzerinden `GetUserStatus` RPC uç noktasına bağlanarak `Gemini` (`agy_gemini`) ve `Claude/GPT` (`agy_claude_gpt`) havuzlarının kotalarını ve `resetTime` sayaçlarını ayrıştırır.
- [collectors/claude_collector.py](collectors/claude_collector.py): `~/.claude/` altındaki oturum/maliyet (`cost-state`, `total_tokens_reminder`) loglarını ve OAuth kullanım verilerini (`fh`: 5-saatlik, `sd`: 7-günlük) birleştirerek `haiku`, `sonnet` ve `opus` kotalarını normalize eder.
- [collectors/codex_collector.py](collectors/codex_collector.py): `~/.codex/sessions/` altındaki en güncel `rollout-*.jsonl` dosyalarından ve kullanım uç noktasından `rate_limits` (`primary` 300dk, `secondary` 10080dk) ile `banked_resets` verilerini ayrıştırır.
- [allocation.py](allocation.py): `0–10` sağlayıcı tercih sürgüleri (`preference_weights`), gerçek zamanlı kota erime hızı öğrenicisi (`BurnRateLearner`), sürdürülebilir havuz kapasitesi (`C(q)`), sigma bantları (`saver`, `economy`, `normal`) ve Smooth Weighted Round Robin (`step_swrr`) iş dağılım motoru.
- [policy_doctor.py](policy_doctor.py): Orkestratör ve alt-ajan kurallarını (`~/.agents/`) otomatik tarayan, model uygunluk ve kota koruma çelişkilerini yakalayan ve her mesajda yüklenen (`Always-Loaded`) kural dosyalarının token bütçesini / kompaktlık puanını (`0–100`) denetleyen **Policy Doctor** motoru (`GET /api/policy-doctor`, `POST /api/policy-doctor/run`).
- [model_catalog.json](model_catalog.json) & [model_review.py](model_review.py): Kod dışına taşınmış kanonik model kataloğu, sıfır-LLM yerel model keşfi (`agy models`, `~/.codex/models_cache.json`, `models.dev`, `OpenRouter`), harmanlanmış fiyat oranı (`blended_price = 0.25·in + 0.75·out`) ve 14 günlük deneme (`probation`) karne özeti.
- [orchestrator.py](orchestrator.py) & [scheduler.py](scheduler.py): Capability Floor (`LOW`, `MID`, `HIGH`), SWRR sağlayıcı seçimi, Cross-Family Independent Review, dinamik Codex Koordinatör Rezervi (`R_orch`) ve `%85 / %95` kota el değiştirme (handoff) kurallarını işletir.
- [main.py](main.py): FastAPI sunucusu (`GET /api/quotas`, `POST /api/preferences`, `POST /api/limits`, `GET /api/policy-doctor`, `POST /api/policy-doctor/run`, `GET /api/models/review`, `POST /api/models/review/run`, `POST /api/models/decision`, `POST /api/orchestrator`, `POST /api/dispatch`, `POST /api/simulate`).
- [static/index.html](static/index.html): SCADA HMI Tek Ekran (L1/L2) Web Dashboard arayüzü (`EN` varsayılan + `TR` dil seçeneği).

---

## 🚀 Hızlı Başlangıç

```powershell
# 1. Bağımlılıkları yükle
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

## 🔧 Kişisel Çalışma Alanı Eşlemeleri (`owner-overrides/quota-hub.json`)

Canlı süreç toplayıcısı ([collectors/live_activity.py](collectors/live_activity.py)), kişisel proje isimlerini kod içinde barındırmaz. Kendi projeleriniz için `~/.agents/owner-overrides/quota-hub.json` dosyasında `workspace_aliases` ve `live_activity_pattern` tanımlayabilirsiniz:

```json
{
  "live_activity_pattern": "my-project|another-project",
  "workspace_aliases": {
    "my project": "~/my-project"
  }
}
```

---

## ⚖️ Lisans: GNU AGPLv3 + Ticari Çift Lisans (Dual-License)

Bu proje, açık kaynak / topluluk kullanımı için **GNU Affero General Public License v3.0 ([AGPL-3.0](LICENSE))** ve kapalı kaynak ticari / SaaS kullanımı için **Ticari Lisans (Commercial License)** olmak üzere çift lisanslıdır:

1. **Açık Kaynak / Topluluk Kullanımı (GNU AGPLv3)**:
   - Bireysel geliştiriciler ve açık kaynak projeler için kullanımı, incelenmesi ve geliştirilmesi tamamen ücretsizdir.
   - Projeyi değiştirerek dağıtanlar veya bir ağ/bulut servisi (SaaS) olarak sunanlar, yaptıkları tüm geliştirmeleri ve kaynak kodunu AGPLv3 kapsamında açık kaynak olarak yayınlamak zorundadır (böylece topluluk geliştirmeleri size ve projeye geri döner).
2. **Kapalı Kaynak Ticari Kullanım (Commercial License)**:
   - Kendi kaynak kodunu açmadan bu yazılımı kapalı kaynak ticari bir ürüne, kurumsal platforma veya ücretli bir SaaS hizmetine entegre etmek isteyen şirketler, telif hakkı sahibiyle (**Onur Yıldız** — GitHub [@yildizonur07](https://github.com/yildizonur07)) iletişime geçerek ayrı bir ticari lisans almalıdır.
