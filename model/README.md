---
language:
- tr
- en
license: mit
tags:
- onnx
- tabular-classification
- scikit-learn
- edge-ml
- interactive-fiction
- adaptive-learning
pipeline_tag: tabular-classification
---

# The Primer v2: Adaptive Hint Classifier (v2.0.0)

Oyuncunun davranışsal telemetrilerinden bilişsel tıkanmayı sezerek dinamik iskele / ipucu sağlayan ONNX modeli.

## 🎯 Model Amacı ve Kapsamı
Bu model, etkileşimli kurgu platformu **The Primer v2** içerisinde oyuncunun kognitif yükünü ve tıkanma seviyesini tespit ederek **dinamik iskele (scaffolding / hint)** gösterilip gösterilmeyeceğine anlık olarak karar verir.

- **Hesaplama Yeri:** İstemci tarafında uçta çıkarım (Browser via `onnxruntime-web`, Mobil via `onnxruntime-react-native`).
- **Ağ İhtiyacı:** Sıfır ağ çağrısı (Offline-first, 0 ms network latency).
- **Gizlilik:** Sıfır PII (Kişisel Veri İçermez). Veriler cihaz dışına çıkmaz.

---

## 📐 Model Mimarisi & Sözleşme

- **Çatı (Framework):** `scikit-learn` Pipeline (`StandardScaler` → `LogisticRegression`)
- **Dışa Aktarım (ONNX):** `skl2onnx` (Opset 17, `zipmap=False`)
- **Girdi Tensörü (`input`):** `[N, 4]` boyutunda `float32` tensörü.
  1. `tereddut_suresi_sn` (sn): Karar anındaki bekleme süresi `[0.0, 120.0]`
  2. `karar_degistirme_sayisi` (adet): Seçenekler arası gidip gelme sayısı `[0, 20]`
  3. `onceki_hata_sayisi` (adet): Karşılaşılan başarısızlık adedi `[0, 10]`
  4. `secilen_dal_zorlugu` (derece): Dalın zorluk derecesi `[1, 2, 3]`
- **Çıktı Tensörleri:**
  - `label` (`int64` `[N]`): `0` (Standart Akış) veya `1` (İpucu Göster)
  - `probabilities` (`float32` `[N, 2]`): `[P(y=0), P(y=1)]` olasılık vektörü.

---

## 📊 Doğrulama Metrikleri (Test Seti %20 Hold-out)

| Metrik | Değer | Şartname Eşiği (spec.md) | Durum |
| :--- | :--- | :--- | :--- |
| accuracy | 0.8333 |
| f1_macro | 0.8303 |
| roc_auc | 0.8822 |

- **ONNX Sayısal Doğruluk Sapması:** `1.20e-07` (Eşik: `< 1e-4` ✅)
- **Model Dosyası:** `model.onnx` (SHA256: `6fdb00b33969dc6d3463291d377c9994d9bca40a21c00a61b8eebedfd0031161`)

---

## ⚠️ Kısıtlar ve Kullanılmaması Gereken Durumlar (Out-of-Scope)

1. **Klinik / Tıbbi Teşhis:** Bu model bir kognitif bozukluk, dikkat eksikliği veya nörolojik durum teşhis edicisi DEĞİLDİR.
2. **Akademik Değerlendirme / Notlandırma:** Model sadece kullanıcı deneyimini zenginleştirmek için dinamik oyun ipucu tetikler; başarı notlandırması veya yetenek sıralaması amacıyla kullanılamaz.
3. **Farklı Girdi Dağılımı:** Karar süresinin 120 saniyeden uzun olduğu durumlarda girdi 120'ye kırpılmalıdır; doğrudan uç değerlerle beslenmemelidir.

---

## 💻 Örnek Çıkarım (Python & ONNX)

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
# Girdi: [tereddut_sn, karar_degistirme, onceki_hata, zorluk]
ornek_girdi = np.array([[42.0, 4, 3, 3]], dtype=np.float32)

etiket, olasiliklar = session.run(["label", "probabilities"], {"input": ornek_girdi})
print(f"Karar: {etiket[0]} | İpucu Olasılığı: {olasiliklar[0][1]:.4f}")
```
