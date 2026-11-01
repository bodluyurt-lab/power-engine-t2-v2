#!/usr/bin/env python3
"""
The Primer v2: Tekrarlanabilir Model Eğitimi, Eval Kapısı & ONNX Paketleme (train.py)

Bu betik:
1. data/oturumlar.csv dosyasını yükler ve şema bütünlüğünü doğrular.
2. Sabit seed (random_state=42) ile train/test kümelerine ayırır (%80 train, %20 test, stratified).
3. scikit-learn Pipeline (StandardScaler + LogisticRegression) modelini eğitir.
4. Test kümesi üzerinde doğruluk metriklerini hesaplar.
5. Değerlendirme Kapısı (Evaluation Gate): spec.md'de tanımlı eşiklerin (Accuracy >= 0.80, F1 >= 0.75, ROC-AUC >= 0.85)
   altında kalırsa sys.exit(1) ile işlemi durdurur.
6. skl2onnx kullanarak modeli model/model.onnx (input [n, 4] -> label, probabilities [n, 2], zipmap=False) olarak aktarır.
7. ONNX çıktısı ile sklearn çıktısı arasındaki maksimum olasılık farkının < 1e-4 olduğunu doğrular (Parite Kapısı).
8. model/config.json (Sözleşme) ve model/README.md (Model Kartı) artefaktlarını üretir.
"""

import os
import sys
import json
import argparse
import hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import joblib

import onnx
import onnxruntime as ort
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Sabitler ve Sözleşmeler
SEED = 42
FEATURE_COLUMNS = [
    "tereddut_suresi_sn",
    "karar_degistirme_sayisi",
    "onceki_hata_sayisi",
    "secilen_dal_zorlugu"
]
TARGET_COLUMN = "ipucu_verilsin_mi"

# spec.md Değerlendirme Kapısı Eşikleri (Evaluation Gate Thresholds)
THRESHOLDS = {
    "min_accuracy": 0.80,
    "min_f1_macro": 0.75,
    "min_roc_auc": 0.85,
    "max_onnx_divergence": 1e-4
}


def veri_yukle_ve_dogrula(csv_yolu: str) -> pd.DataFrame:
    """Veri setini okur ve şema/bütünlük kontrollerini uygular."""
    if not os.path.exists(csv_yolu):
        raise FileNotFoundError(f"Veri seti bulunamadı: {csv_yolu}")

    df = pd.read_csv(csv_yolu)
    
    beklenen_sutunlar = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    eksik = beklenen_sutunlar - set(df.columns)
    if eksik:
        raise ValueError(f"Veri setinde eksik sütunlar mevcut: {eksik}")

    if df[list(beklenen_sutunlar)].isnull().any().any():
        raise ValueError("Veri setinde NaN / boş değerler tespit edildi!")

    # Aykırı değer kırpma (spec.md gereği)
    df["tereddut_suresi_sn"] = df["tereddut_suresi_sn"].clip(lower=0.0, upper=120.0)

    return df


def model_olustur() -> Pipeline:
    """Tekrarlanabilir ölçekleme ve lojistik regresyon pipeline'ı oluşturur."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            random_state=SEED,
            C=1.0,
            solver="lbfgs",
            max_iter=500
        ))
    ])


def onnx_aktar_ve_dogrula(model: Pipeline, X_ornek: pd.DataFrame, onnx_yolu: str) -> float:
    """
    Modeli ONNX formatına dönüştürür ve sklearn ile ONNX çıktısı arasındaki 
    maksimum olasılık farkını < 1e-4 eşiğinde doğrular.
    """
    # Girdi tanımı: 'input' adında [None, 4] boyutlu float32 tensör
    initial_types = [("input", FloatTensorType([None, X_ornek.shape[1]]))]
    
    # zipmap=False ile çıktı doğrudan [N, C] float tensör probabilities olarak döner
    onx = to_onnx(
        model,
        initial_types=initial_types,
        options={id(model): {"zipmap": False}},
        target_opset=17
    )
    
    with open(onnx_yolu, "wb") as f:
        f.write(onx.SerializeToString())
    
    # ONNX Eşdeğerlik / Tolerans Kontrolü
    session = ort.InferenceSession(onnx_yolu)
    input_data = X_ornek.to_numpy(dtype=np.float32)
    
    onnx_probs = session.run(["probabilities"], {"input": input_data})[0]
    sklearn_probs = model.predict_proba(X_ornek)
    
    maks_fark = float(np.abs(onnx_probs - sklearn_probs).max())
    print(f"🔬 ONNX <-> Scikit-Learn Sayısal Farkı (Maksimum Mutlak Hata): {maks_fark:.2e}")
    
    if maks_fark > THRESHOLDS["max_onnx_divergence"]:
        raise ValueError(
            f"❌ ONNX Parite Hatası: Fark ({maks_fark:.2e}) izin verilen toleransı "
            f"({THRESHOLDS['max_onnx_divergence']:.2e}) aşıyor!"
        )
    
    print(f"✅ ONNX Parite Kontrolü Başarılı! (Fark < {THRESHOLDS['max_onnx_divergence']:.2e})")
    return maks_fark


def dosya_sha256(dosya_yolu: str) -> str:
    """Bir dosyanın SHA256 özetini üretir."""
    hasher = hashlib.sha256()
    with open(dosya_yolu, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def model_karti_olustur(config: dict) -> str:
    """model/README.md için kurumsal Hugging Face uyumlu Model Kartı üretir."""
    metrikler_tablosu = "\n".join(
        f"| {k} | {v} |" for k, v in config["metrics"].items() if k != "confusion_matrix"
    )
    return f"""---
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

# {config["title"]} (v{config["version"]})

{config["description"]}

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
{metrikler_tablosu}

- **ONNX Sayısal Doğruluk Sapması:** `{config["onnx"]["max_probability_divergence"]:.2e}` (Eşik: `< 1e-4` ✅)
- **Model Dosyası:** `model.onnx` (SHA256: `{config["onnx"]["sha256"]}`)

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

etiket, olasiliklar = session.run(["label", "probabilities"], {{"input": ornek_girdi}})
print(f"Karar: {{etiket[0]}} | İpucu Olasılığı: {{olasiliklar[0][1]:.4f}}")
```
"""


def egit_ve_paketle(csv_yolu: str = "data/oturumlar.csv", cikis_dizini: str = "model"):
    """Tam eğitim, eval kapısı, ONNX dönüştürme ve artefakt oluşturma akışı."""
    print("=" * 65)
    print("🎯 The Primer v2: Uçtan Uca Model Eğitimi & ONNX Paketleme")
    print(f"📌 Seed: {SEED} | Veri: {csv_yolu} | Çıktı: {cikis_dizini}")
    print("=" * 65)

    # 1. Veri Yükleme
    df = veri_yukle_ve_dogrula(csv_yolu)
    X = df[FEATURE_COLUMNS].astype(np.float32)
    y = df[TARGET_COLUMN].astype(np.int64)

    print(f"📊 Toplam Oturum: {len(df)} (Sınıf 0: {(y == 0).sum()}, Sınıf 1: {(y == 1).sum()})")

    # 2. Train-Test Ayrımı (%80 Train, %20 Test, Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=SEED,
        stratify=y
    )

    # 3. Model Eğitimi
    pipeline = model_olustur()
    pipeline.fit(X_train, y_train)

    # 4. Değerlendirme (Evaluation Gate Kontrolü)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    conf_mat = confusion_matrix(y_test, y_pred).tolist()

    print("\n🔍 Test Kümesi Başarım Metrikleri:")
    print(f"   - Doğruluk (Accuracy): {acc:.4f} (Eşik: >= {THRESHOLDS['min_accuracy']})")
    print(f"   - F1-Skoru (Macro):    {f1_macro:.4f} (Eşik: >= {THRESHOLDS['min_f1_macro']})")
    print(f"   - ROC-AUC:             {roc_auc:.4f} (Eşik: >= {THRESHOLDS['min_roc_auc']})")

    hatalar = []
    if acc < THRESHOLDS["min_accuracy"]:
        hatalar.append(f"Accuracy {acc:.4f} < Eşik {THRESHOLDS['min_accuracy']}")
    if f1_macro < THRESHOLDS["min_f1_macro"]:
        hatalar.append(f"F1-Macro {f1_macro:.4f} < Eşik {THRESHOLDS['min_f1_macro']}")
    if roc_auc < THRESHOLDS["min_roc_auc"]:
        hatalar.append(f"ROC-AUC {roc_auc:.4f} < Eşik {THRESHOLDS['min_roc_auc']}")

    if hatalar:
        print("\n❌ EVALUATION GATE BAŞARISIZ!")
        for h in hatalar:
            print(f"   🔴 {h}")
        sys.exit(1)

    print("\n✅ EVALUATION GATE GEÇİLDİ!")

    # 5. Dizin Hazırlığı ve Sklearn Kaydı
    os.makedirs(cikis_dizini, exist_ok=True)
    joblib_path = os.path.join(cikis_dizini, "model.joblib")
    joblib.dump(pipeline, joblib_path)
    print(f"💾 Scikit-learn Pipeline kaydedildi: {joblib_path}")

    # 6. ONNX Dönüştürme ve Sayısal Doğruluk Kontrolü (< 1e-4)
    onnx_path = os.path.join(cikis_dizini, "model.onnx")
    onnx_fark = onnx_aktar_ve_dogrula(pipeline, X_test, onnx_path)
    onnx_sha = dosya_sha256(onnx_path)

    # 7. Model Sözleşmesi (config.json)
    referans_girdi = {
        "tereddut_suresi_sn": 42.0,
        "karar_degistirme_sayisi": 4,
        "onceki_hata_sayisi": 3,
        "secilen_dal_zorlugu": 3
    }
    ref_df = pd.DataFrame([referans_girdi])[FEATURE_COLUMNS].astype(np.float32)
    ref_tahmin = int(pipeline.predict(ref_df)[0])
    ref_olasilik = float(pipeline.predict_proba(ref_df)[0][1])

    config = {
        "title": "The Primer v2: Adaptive Hint Classifier",
        "description": "Oyuncunun davranışsal telemetrilerinden bilişsel tıkanmayı sezerek dinamik iskele / ipucu sağlayan ONNX modeli.",
        "model_name": "the-primer-v2-hint-classifier",
        "version": "2.0.0",
        "task": "binary-classification",
        "seed": SEED,
        "target": TARGET_COLUMN,
        "classes": ["0", "1"],
        "class_labels": {
            "0": "standart_akis",
            "1": "ipucu_ver"
        },
        "threshold": 0.50,
        "features": [
            {
                "name": "tereddut_suresi_sn",
                "label": "Tereddüt Süresi (sn)",
                "type": "float32",
                "example": 42.0,
                "range": [0.0, 120.0]
            },
            {
                "name": "karar_degistirme_sayisi",
                "label": "Karar Değiştirme Sayısı",
                "type": "int64",
                "example": 4.0,
                "range": [0, 20]
            },
            {
                "name": "onceki_hata_sayisi",
                "label": "Önceki Hata Sayısı",
                "type": "int64",
                "example": 3.0,
                "range": [0, 10]
            },
            {
                "name": "secilen_dal_zorlugu",
                "label": "Seçilen Dal Zorluğu",
                "type": "int64",
                "example": 3.0,
                "range": [1, 3]
            }
        ],
        "metrics": {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1_macro, 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix": {
                "tn": conf_mat[0][0], "fp": conf_mat[0][1],
                "fn": conf_mat[1][0], "tp": conf_mat[1][1]
            }
        },
        "onnx": {
            "input_name": "input",
            "input_shape": [None, 4],
            "input_type": "float32",
            "outputs": ["label", "probabilities"],
            "opset": 17,
            "max_probability_divergence": round(onnx_fark, 8),
            "sha256": onnx_sha
        },
        "eval_gate": {
            "status": "PASSED",
            "thresholds": THRESHOLDS
        },
        "reference_inference": {
            "inputs": referans_girdi,
            "expected_label": ref_tahmin,
            "expected_p1_probability": round(ref_olasilik, 4)
        },
        "environment": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "sklearn_version": sklearn.__version__,
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__
        },
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    config_path = os.path.join(cikis_dizini, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"📄 Model Sözleşmesi kaydedildi: {config_path}")

    # 8. Model Kartı (README.md)
    readme_path = os.path.join(cikis_dizini, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(model_karti_olustur(config))
    print(f"📖 Model Kartı (README.md) kaydedildi: {readme_path}")
    print("=" * 65)
    print("✨ ADIM 4 Artefakt Üretimi Başarıyla Tamamlandı!")
    print("=" * 65)

    return config


def main():
    parser = argparse.ArgumentParser(description="The Primer v2 Model Eğitimi & ONNX Dışa Aktarımı")
    parser.add_argument("--veri", type=str, default="data/oturumlar.csv", help="Veri CSV yolu")
    parser.add_argument("--cikis", type=str, default="model", help="Artefakt çıktı dizini")
    args = parser.parse_args()

    egit_ve_paketle(csv_yolu=args.veri, cikis_dizini=args.cikis)


if __name__ == "__main__":
    main()
