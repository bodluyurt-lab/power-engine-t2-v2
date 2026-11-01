# The Primer v2: Dağıtım, Sürüm Yönetimi ve MLOps Kılavuzu (RELEASE.md)

Bu doküman, **The Primer v2** adaptif kurgu sisteminde eğitilen yapay zeka modelinin (`model.onnx`) üretim ortamına dağıtılması, Hugging Face Hub üzerinde barındırılması, sıfır-kesinti (*zero-downtime*) ile kademeli açılması, geri alma (*rollback*) süreçleri ve farklı çalışma zamanları (*runtime environments*) arasındaki mimari ödünleşimleri (*trade-offs*) tanımlar.

---

## 1. Hugging Face Hub Resolve URL Mantığı ve Değişmezlik (Immutability)

Tarayıcı ve mobil istemciler, ONNX modelini statik bir depolama kaynağından HTTP(S) protokolü üzerinden indirir. Hugging Face Hub, Git LFS tabanlı versiyonlama ve global CDN desteği sayesinde bu ihtiyacı ek sunucu maliyeti olmadan karşılar.

### 1.1. URL Mimarisi ve Commit Hash Zorunluluğu

Hugging Face üzerindeki bir model dosyasına iki farklı formatta erişilebilir:

```
❌ Dinamik/Değişken URL (Anti-Pattern):
https://huggingface.co/<kullanici>/<repo>/resolve/main/model.onnx

✅ Değişmez/Deterministik URL (Üretim Standardı):
https://huggingface.co/<kullanici>/<repo>/resolve/<commit_sha>/model.onnx
```

### 1.2. Neden `main` Dalı Yerine `<commit_sha>` Kullanılmalıdır?

1. **CDN ve Tarayıcı Önbelleklemesi (Edge & Browser Cache Poisoning):**
   - `main` dalı dinamiktir. Yeni bir model yüklendiğinde CDN uç noktalarında (Cloudflare/Fastly) veya kullanıcının tarayıcı önbelleğinde eski model kalabilir. Bu durum, istemcinin yeni `config.json` ile eski `model.onnx` çalıştırmasına (boyut veya özellik uyumsuzluğu) yol açar.
   - Sabit `commit_sha` içeren URL'ler değişmezdir (*immutable*). CDN seviyesinde `Cache-Control: public, max-age=31536000, immutable` ile süresiz önbelleklenebilir. Bu sayede sonraki açılışlarda 0 ms indirme süresi elde edilir.
2. **Kriptografik Bütünlük ve Güvenlik:**
   - İstemci, `config.json` içinde tanımlı `sha256` özeti ile indirilen ONNX dosyasının özetini karşılaştırır:
     $$\text{SHA256}(\text{indirilen\_model}) \stackrel{?}{=} \text{config.onnx.sha256}$$
   - Aradaki bir ağ saldırısı (*man-in-the-middle*) veya yarım indirme anında tespit edilerek bozuk modelin belleğe alınması engellenir.
3. **Atomik Sözleşme Senkronizasyonu:**
   - `config.json` ve `model.onnx` aynı Git commit'i üzerinden çekildiği için özellik sıralaması veya normalizasyon katsayılarının uyuşmama riski sıfırlanır.

---

## 2. Geri Alma (Rollback) Stratejisi

Üretim ortamında bir modelin beklenmeyen bir davranış sergilemesi (örn. her oyuncuya sürekli ipucu vermesi, aşırı bellek tüketmesi veya uç cihazda hata fırlatması) durumunda uygulanacak geri alma protokolüdür.

```
┌─────────────────┐       ┌───────────────────────┐       ┌──────────────────────┐
│  Anomali Tespiti│ ────► │  site.json / Config   │ ────► │  Anında İstemci      │
│  (Alert Trigger)│       │  Commit SHA Güncelleme│       │  Rollback (< 60 sn)  │
└─────────────────┘       └───────────────────────┘       └──────────────────────┘
```

### 2.1. Tetikleme Kriterleri (Rollback Triggers)
Aşağıdaki durumlardan herhangi biri tespit edildiğinde otomatik/manuel geri alma süreci başlar:
- **İpucu Oranı Sapması:** İstemcilerden gelen anonim telemetride `P(ipucu = 1)` oranının $\ge \%75$ veya $\le \%5$ olması (Normal dağılım: $\%35 - \%55$).
- **Inference Hata Oranı:** ONNX Runtime başlatma veya çıkarım hatası alan istemci oranının $\%0.5$'i aşması.
- **Gecikme (Latency) Artışı:** İstemcide p95 çıkarım süresinin $25\text{ ms}$ üzerine çıkması.

### 2.2. Geri Alma Mekanizması (GitOps Tabanlı Dinamik İşaretçi)
- Uygulama ikilisi (*binary*) veya web kodu yeniden derlenmez.
- `web/site.json` dosyasındaki model işaretçisi bir önceki doğrulanmış commit SHA'sına çekilir:
  ```json
  {
    "repo": "5eyyah/the-primer-v2-hint-onnx",
    "commit": "a1b2c3d4e5f67890123456789abcdef012345678",
    "fallback_to_rules": true
  }
  ```
- **Acil Durum Devre Kesici (Circuit Breaker):** Eğer model kaynağına ulaşılamazsa veya hata alınırsa istemci sessizce `fallback_to_rules: true` durumuna geçer ve klasik sabit kurallı hikaye mantığına geri döner; oyuncu hiçbir çökme veya donma yaşamaz.

---

## 3. Dağıtım Stratejileri (Deployment Strategies)

### 3.1. Gölge Dağıtım (Shadow Deployment)

Yeni eğitilen model sürümü (`v2.0.0`), canlı ortamdaki oyuncuların cihazında sessizce çalıştırılır; ancak ürettiği sonuç oyun motoruna **uygulanmaz**.

```
                           ┌───────────────────────────────┐
                           │ Oyuncu Karar Anı Telemetrisi  │
                           └──────────────┬────────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
      ┌───────────────────────┐                       ┌───────────────────────┐
      │ Aktif Sistem (v1.0)   │                       │ Gölge Model (v2.0)    │
      │ • Oyuna yön verir     │                       │ • Arka planda çalışır │
      │ • İpuçları gösterir   │                       │ • Sonuç loglanır      │
      └───────────────────────┘                       └───────────┬───────────┘
                                                                  ▼
                                                      ┌───────────────────────┐
                                                      │ Sapma & Performans    │
                                                      │ Analizi (Telemetry)   │
                                                      └───────────────────────┘
```

- **Süreç:** 
  1. Karar anında mevcut kural motoru ipucu durumunu belirler.
  2. Arka planda eşzamanlı olarak `model.onnx` çıkarım yapar.
  3. Çıkarım süresi, bellek tüketimi ve iki kararın uyumu telemetriye yazılır.
- **Kabul Kriteri:** 10.000 gölge çıkarımda sıfır kilitlenme (*freeze*) ve p99 gecikme $< 15\text{ ms}$.

---

### 3.2. Kademeli Dağıtım (Canary Deployment)

Kullanıcı kitlesinin kademeli olarak yeni modelle tanıştırılması sürecidir. Kullanıcı ayrımı, `oturum_id` veya cihaz kimliğinin deterministik hash'i ile yapılır:

$$\text{dilim} = \text{CRC32}(\text{oturum\_id}) \pmod{100}$$

```
   Faz 1: %5 Canary           Faz 2: %25 Genişletme        Faz 3: %100 Genel Dağıtım
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│  %5 Yeni Model (v2)  │    │ %25 Yeni Model (v2)  │    │                      │
│ ──────────────────── │ ──►│ ──────────────────── │ ──►│  %100 Yeni Model     │
│  %95 Eski Model (v1) │    │ %75 Eski Model (v1)  │    │                      │
└──────────────────────┘    └──────────────────────┘    └──────────────────────┘
      (24 Saat)                   (48 Saat)                    (Kararlı)
```

1. **Faz 1 (%5 Trafik - 24 Saat):**
   - Amaç: Uç cihaz çeşitliliğinde (düşük segment Android, eski Safari tarayıcılar) donanım uyumluluğu ve bellek sızıntısı kontrolü.
   - İzlenen Metrik: Crash-free session rate $\ge \%99.9$.
2. **Faz 2 (%25 Trafik - 48 Saat):**
   - Amaç: İstatistiksel A/B hipotez testi.
   - Hipotez: Yeni modelle ipucu alan oyuncuların hikayeyi tamamlama oranı (completion rate), kontrol grubuna göre en az $\%12$ daha yüksek olmalıdır ($p < 0.05$).
3. **Faz 3 (%100 Trafik - Tam Yayın):**
   - Bütün kullanıcılar için `site.json` güncellenir ve eski model arşivlenir.

---

## 4. Çıkarım Mimarileri Ödünleşim Matrisi (Trade-off Matrix)

Aşağıdaki matris, modelin çalıştırılabileceği 3 farklı mimari yaklaşımın (Web Runtime, Mobil Native ve Sunucu API) kilit mühendislik kriterlerine göre karşılaştırmasını sunar:

| Kriter | Tarayıcı Çalışma Zamanı (`onnxruntime-web`) | Mobil Cihaz İçi (`onnxruntime-react-native`) | Sunucu API (`Rails 8 / FastAPI`) |
| :--- | :--- | :--- | :--- |
| **Gecikme (Latency - p95)** | 🟢 **1 - 5 ms** (Yerel Wasm/WebGL) | 🟢 **2 - 8 ms** (Yerel NPU/CPU) | 🔴 **60 - 250 ms** (Ağ RTT + Kuyruk) |
| **KVKK / Veri Gizliliği** | 🟢 **Mükemmel** (Sıfır veri aktarımı, cihazda kalır) | 🟢 **Mükemmel** (Sıfır veri aktarımı, cihazda kalır) | 🟡 **Hassas** (Telemetrinin sunucuya iletilmesi gerekir) |
| **Tersine Mühendislik / Model Güvenliği** | 🔴 **Zayıf** (`model.onnx` ağ sekmesinden indirilebilir) | 🟡 **Orta** (APK/IPA içine gömülür, decompile edilebilir) | 🟢 **Mükemmel** (Model sunucu belleğinde gizlidir) |
| **Altyapı & Sunucu Maliyeti** | 🟢 **\$0** (Statik GitHub Pages / CDN maliyetsiz) | 🟢 **\$0** (Cihazın donanımı kullanılır) | 🔴 **Yüksek** (Trafikle artan CPU/RAM/Sunucu maliyeti) |
| **Çevrimdışı Çalışma (Offline-first)** | 🟢 **Evet** (ServiceWorker / CacheStorage ile) | 🟢 **Evet** (Tamamen bağımsız yerel çalışma) | 🔴 **Hayır** (İnternet bağlantısı şarttır) |
| **Dağıtım & Sürüm Güncelleme Çevikliği** | 🟢 **Anında** (Yeni dosya URL'si ile saniyeler içinde) | 🟡 **Orta/Yavaş** (OTA güncellemesi veya Store onayı gerekir) | 🟢 **Anında** (Sunucuda tek komutla deploy) |
| **Donanım Çeşitliliği Riski** | 🟡 **Orta** (WebGL / Wasm tarayıcı uyumsuzlukları) | 🟡 **Orta** (Android NDK / GPU sürücü farkları) | 🟢 **Sıfır** (Kontrollü sunucu donanımı) |

### 4.1. The Primer v2 İçin Mimari Karar Gerekçesi
The Primer v2 projesinde **İstemci Çıkarımı (Web / Mobil ONNX)** mimarisi tercih edilmiştir:
- Hikaye karar anında oyuncunun $150-200\text{ ms}$ sürecek bir ağ beklemesine tahammülü yoktur; akışın kesintisiz olması gerekir.
- Kullanıcının tereddüt süresi ve karar değiştirme gibi özel davranışsal verileri KVKK kapsamında hassas psikometrik profil oluşturabileceğinden, verilerin cihazda tutulması en güvenli hukuksal ve etik yaklaşımdır.
- Model kamuya açık bir oyun deneyimi için geliştirildiğinden, ONNX modelinin tersine mühendislikle incelenmesi ticari bir güvenlik açığı oluşturmamaktadır.

