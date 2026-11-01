# The Primer v2: Makine Öğrenmesi ile Dinamik Hikaye Uyarlaması Şartnamesi (spec.md)

Bu belge, "The Primer v2" etkileşimli kurgu platformuna entegre edilecek istemci-taraflı (client-side edge ML) yapay zeka modelinin fonksiyonel gereksinimlerini, veri sözleşmesini, kabul eşiklerini ve değerlendirme kapılarını (Evaluation Gate) tanımlar.

---

## 1. Proje Amacı ve Çözülen Problem

Sabit kural ağaçlarıyla ilerleyen klasik metin tabanlı etkileşimli hikayelerde oyuncular iki temel problem yaşar:
1. **Bunalma / Tıkanma (Frustration):** Oyuncunun zorlandığı, kararsız kaldığı ve oyunu terk etme (churn) riski taşıdığı anlarda yardım alamaması.
2. **Aşırı Yönlendirilme (Boredom):** Tecrübeli ve hızlı ilerleyen oyunculara gereksiz ipucu veya kolaylaştırıcı sunularak akışın (flow state) bozulması.

**Çözüm:** Oyuncunun karar anındaki davranışsal telemetrilerini (tereddüt süresi, kararsızlık, geçmiş hata sıklığı ve seçilen dalın zorluğu) istemcide anlık olarak işleyen dikey/hafif bir sınıflandırma modeli (`scikit-learn` -> `ONNX`). Model, tarayıcıda (`onnxruntime-web`) veya mobil cihazda sıfır ağ gecikmesi ve tam KVKK uyumluluğuyla çalışarak dinamik iskele (`scaffolding / ipucu`) verilip verilmeyeceğini belirler.

---

## 2. Model Girdi ve Çıktı Değişkenleri Sözleşmesi

Model, her karar kavşağında (decision point) aşağıdaki özellikleri (features) tek bir tensör vektörü halinde girdi olarak alır:

### 2.1. Girdi Özellikleri (Features - `X`)

| Özellik Adı (Python/JSON) | Veri Tipi | Birim / Format | Değer Aralığı | Açıklama |
| :--- | :--- | :--- | :--- | :--- |
| `tereddut_suresi_sn` | `float32` | Saniye | `[0.0, 120.0]` | Karar sayfasının yüklendiği an ile seçim butonuna tıklandığı an arasındaki süre. |
| `karar_degistirme_sayisi` | `int64` | Adet | `[0, 20]` | Oyuncunun nihai kararından önce seçenekler/radyo butonları arasında yaptığı değişiklik adedi. |
| `onceki_hata_sayisi` | `int64` | Adet | `[0, 10]` | Oyuncunun mevcut bölüme gelene kadar hikayede karşılaştığı çıkmaz sokak veya başarısız seçim sayısı. |
| `secilen_dal_zorlugu` | `int64` | Ordinal Derece | `1, 2, 3` | Oyuncunun seçtiği kararın zorluk derecesi (1: Kolay/Doğrusal, 2: Orta, 3: Karmaşık/Riskli). |

> **Önemli Not (Feature Sıralaması):** ONNX giriş tensöründeki sütun sırası kesinlikle `[tereddut_suresi_sn, karar_degistirme_sayisi, onceki_hata_sayisi, secilen_dal_zorlugu]` olmalıdır.

### 2.2. Çıktı Değişkeni (Target - `y`)

| Hedef Değişken | Veri Tipi | Olası Değerler | Semantik Anlamı |
| :--- | :--- | :--- | :--- |
| `ipucu_verilsin_mi` | `int64` | `0` veya `1` | `0`: İpucu Gösterilmez (Standart Akış).<br>`1`: İpucu Gösterilir (Dinamik Destek Aktif). |

- **Olasılık Çıktısı (`probabilities`):** Model aynı zamanda `[P(y=0), P(y=1)]` olasılık dağılımını üretir. Varsayılan sınıflandırma eşik değeri (threshold) $\tau = 0.50$'dir.

---

## 3. Kabul Eşikleri ve Değerlendirme Kapısı (Evaluation Gate)

Bu eşikler model eğitimi çalıştırılmadan **ÖNCE** belirlenmiştir. `train.py` eğitim betiği, test kümesi üzerinde bu eşiklerin altında kalırsa süreci **hata vererek durdurmakla (Evaluation Gate)** yükümlüdür.

1. **Doğruluk (Accuracy) Eşiği:** $\ge 0.80$ (%80)
2. **Dengeli F1-Skoru (F1-Score Macro) Eşiği:** $\ge 0.75$
3. **ROC-AUC Eşiği:** $\ge 0.85$
4. **ONNX Sayısal Doğruluk Paritesi:** $| P_{sklearn}(y=1) - P_{onnx}(y=1) | < 10^{-4}$ (Maksimum mutlak hata)

---

## 4. Somut Kabul Test Senaryoları (Acceptance Test Cases)

Modelin iş mantığına ve pedagojik beklentilere uygunluğu aşağıdaki 3 doğrulanabilir test senaryosu üzerinden test edilir:

### Senaryo 1: Kendinden Emin / Hızlı Oyuncu (Minimal İskele)
- **Girdi:** 
  - `tereddut_suresi_sn`: `3.5`
  - `karar_degistirme_sayisi`: `0`
  - `onceki_hata_sayisi`: `0`
  - `secilen_dal_zorlugu`: `1`
- **Beklenen Çıktı:** `ipucu_verilsin_mi = 0` ($P(y=1) < 0.20$)
- **Gerekçe:** Oyuncu seçeneği hemen kavramış, kararsızlık yaşamamış, geçmişte hatası yok ve kolay dalda ilerliyor. İpucu verilmesi oyun akışını baltalar.

### Senaryo 2: Zorlanan / Bocalayan Oyuncu (Dinamik İskele Şart)
- **Girdi:**
  - `tereddut_suresi_sn`: `42.0`
  - `karar_degistirme_sayisi`: `4`
  - `onceki_hata_sayisi`: `3`
  - `secilen_dal_zorlugu`: `3`
- **Beklenen Çıktı:** `ipucu_verilsin_mi = 1` ($P(y=1) \ge 0.75$)
- **Gerekçe:** Karmaşık bir dalda 40 saniyenin üzerinde düşünülmüş, seçenekler arasında 4 kez gidip gelinmiş ve daha önce 3 kez hata yapılmış. Oyuncu tıkanma eşiğindedir; ipucu desteği devreye girmelidir.

### Senaryo 3: Sınır Durum (Boundary Case - Yalnızca Süresi Uzun, Kararı Net)
- **Girdi:**
  - `tereddut_suresi_sn`: `22.0`
  - `karar_degistirme_sayisi`: `0`
  - `onceki_hata_sayisi`: `1`
  - `secilen_dal_zorlugu`: `2`
- **Beklenen Çıktı:** `ipucu_verilsin_mi = 0` ($P(y=1) < 0.50$)
- **Gerekçe:** Oyuncu metni dikkatli okuduğu için süre uzundur, ancak hiç karar değiştirmemiştir ve geçmiş hata sayısı düşüktür. Tereddüt süresi tek başına ipucu tetiklemek için yeterli olmamalıdır; model çoklu faktör etkileşimini doğru ayrıştırmalıdır.

---

## 5. Sistem Kısıtları ve KVKK Uyumluluğu

1. **Sıfır Kişisel Veri (Zero PII):** İsim, e-posta, IP adresi, konum, donanım kimliği gibi kişisel veriler kesinlikle toplanmaz ve model tensörüne dahil edilmez.
2. **Uçta Çıkarım (Edge Inference):** Çıkarım tamamen istemcinin tarayıcısında WebAssembly / WebGL donanım hızlandırmasıyla çalışır.
3. **Maksimum İzin Verilen Çıkarım Gecikmesi (Latency):** İstemcide tekil çıkarım süresi $t_{infer} \le 15\text{ ms}$ olmalıdır.

