#!/usr/bin/env python3
"""
The Primer v2: Diferansiyel Parite Testi & Python Referans Oracle (tests/parite.py)

Bu betik:
1. model/model.joblib (Scikit-Learn) ve model/model.onnx (ONNX Runtime) modellerini yükler.
2. 5 adet belirlenmiş somut referans girdi senaryosu üzerinde eşzamanlı çıkarım yapar.
3. Her girdi için sklearn ve ONNX olasılıklarını karşılaştırır.
4. Maksimum mutlak farkın < 1e-4 tolerans sınırında olduğunu doğrular (Diferansiyel Test).
5. Tüm referans girdileri ve çıkarım sonuçlarını tests/parite.json dosyasına kaydeder.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

TOLERANS = 1e-4

FEATURE_COLUMNS = [
    "tereddut_suresi_sn",
    "karar_degistirme_sayisi",
    "onceki_hata_sayisi",
    "secilen_dal_zorlugu"
]

# 5 Adet Belirlenmiş Somut Referans Test Senaryosu
REFERANS_SENARYOLAR = [
    {
        "id": "senaryo_1_emin_oyuncu",
        "aciklama": "Kendinden emin, hızlı karar veren, sıfır hatalı oyuncu (Kolay Dal)",
        "inputs": {
            "tereddut_suresi_sn": 3.5,
            "karar_degistirme_sayisi": 0,
            "onceki_hata_sayisi": 0,
            "secilen_dal_zorlugu": 1
        },
        "beklenen_etiket": 0
    },
    {
        "id": "senaryo_2_bocalayan_oyuncu",
        "aciklama": "Aşırı tereddütlü, 4 kez fikir değiştirmiş, 3 hatalı oyuncu (Zor Dal)",
        "inputs": {
            "tereddut_suresi_sn": 42.0,
            "karar_degistirme_sayisi": 4,
            "onceki_hata_sayisi": 3,
            "secilen_dal_zorlugu": 3
        },
        "beklenen_etiket": 1
    },
    {
        "id": "senaryo_3_sinir_yavas_okuyucu",
        "aciklama": "Sınır Durum: Metni yavaş okuyan ama net kararlı oyuncu (Orta Dal)",
        "inputs": {
            "tereddut_suresi_sn": 22.0,
            "karar_degistirme_sayisi": 0,
            "onceki_hata_sayisi": 1,
            "secilen_dal_zorlugu": 2
        },
        "beklenen_etiket": 0
    },
    {
        "id": "senaryo_4_orta_tereddut_kararsiz",
        "aciklama": "Orta süre tereddütlü, seçenekler arasında kararsız kalmış oyuncu (Zor Dal)",
        "inputs": {
            "tereddut_suresi_sn": 28.5,
            "karar_degistirme_sayisi": 3,
            "onceki_hata_sayisi": 2,
            "secilen_dal_zorlugu": 3
        },
        "beklenen_etiket": 1
    },
    {
        "id": "senaryo_5_yeni_baslayan_hizli_hatali",
        "aciklama": "Hızlı ama önceki adımlarda hata yapmış oyuncu (Kolay Dal)",
        "inputs": {
            "tereddut_suresi_sn": 7.0,
            "karar_degistirme_sayisi": 1,
            "onceki_hata_sayisi": 2,
            "secilen_dal_zorlugu": 1
        },
        "beklenen_etiket": 0
    }
]


def parite_testini_calistir(
    model_joblib_yolu: str = "model/model.joblib",
    model_onnx_yolu: str = "model/model.onnx",
    cikis_json_yolu: str = "tests/parite.json"
):
    print("=" * 65)
    print("🔬 The Primer v2: Diferansiyel Parite Testi & Referans Oracle")
    print(f"📌 Tolerans: < {TOLERANS:.1e}")
    print("=" * 65)

    if not os.path.exists(model_joblib_yolu):
        raise FileNotFoundError(f"Sklearn modeli bulunamadı: {model_joblib_yolu}")
    if not os.path.exists(model_onnx_yolu):
        raise FileNotFoundError(f"ONNX modeli bulunamadı: {model_onnx_yolu}")

    # Modelleri yükle
    sklearn_model = joblib.load(model_joblib_yolu)
    onnx_session = ort.InferenceSession(model_onnx_yolu)

    sonuclar = []
    tum_farklar = []

    for s in REFERANS_SENARYOLAR:
        girdi_dict = s["inputs"]
        # DataFrame oluştur
        df_girdi = pd.DataFrame([girdi_dict])[FEATURE_COLUMNS].astype(np.float32)
        arr_girdi = df_girdi.to_numpy(dtype=np.float32)

        # 1. Scikit-learn Çıkarımı
        sklearn_pred = int(sklearn_model.predict(df_girdi)[0])
        sklearn_probs = sklearn_model.predict_proba(df_girdi)[0].tolist()

        # 2. ONNX Runtime Çıkarımı
        onnx_outputs = onnx_session.run(["label", "probabilities"], {"input": arr_girdi})
        onnx_pred = int(onnx_outputs[0][0])
        onnx_probs = onnx_outputs[1][0].tolist()

        # 3. Diferansiyel Fark Analizi
        fark_p0 = abs(sklearn_probs[0] - onnx_probs[0])
        fark_p1 = abs(sklearn_probs[1] - onnx_probs[1])
        maks_fark = max(fark_p0, fark_p1)
        tum_farklar.append(maks_fark)

        parite_gecti = maks_fark < TOLERANS
        etiket_esit = sklearn_pred == onnx_pred

        print(f"\n🧪 [{s['id']}] {s['aciklama']}")
        print(f"   Girdi: {girdi_dict}")
        print(f"   Sklearn -> Sınıf: {sklearn_pred}, Olasılıklar: [P0: {sklearn_probs[0]:.4f}, P1: {sklearn_probs[1]:.4f}]")
        print(f"   ONNX    -> Sınıf: {onnx_pred}, Olasılıklar: [P0: {onnx_probs[0]:.4f}, P1: {onnx_probs[1]:.4f}]")
        print(f"   Maksimum Fark: {maks_fark:.2e} | Durum: {'✅ GEÇTİ' if parite_gecti else '❌ BAŞARISIZ'}")

        if not parite_gecti:
            raise AssertionError(
                f"Diferansiyel test başarısız! {s['id']} için fark {maks_fark:.2e} >= {TOLERANS:.1e}"
            )
        if not etiket_esit:
            raise AssertionError(
                f"Sınıf etiketi uyumsuzluğu! Sklearn: {sklearn_pred} != ONNX: {onnx_pred}"
            )

        sonuclar.append({
            "id": s["id"],
            "aciklama": s["aciklama"],
            "inputs": girdi_dict,
            "sklearn": {
                "label": sklearn_pred,
                "probabilities": [round(p, 6) for p in sklearn_probs]
            },
            "onnx": {
                "label": onnx_pred,
                "probabilities": [round(p, 6) for p in onnx_probs]
            },
            "divergence": {
                "max_absolute_difference": float(maks_fark),
                "passed": parite_gecti
            }
        })

    genel_maks_fark = max(tum_farklar)
    print("\n" + "=" * 65)
    print(f"🎉 5 Referans Senaryonun Tümü Diferansiyel Testi Geçti!")
    print(f"   Genel Maksimum Fark: {genel_maks_fark:.2e} (Tolerans Eşiği: < {TOLERANS:.1e})")
    print("=" * 65)

    # tests/parite.json dosyasına kaydet
    os.makedirs(os.path.dirname(cikis_json_yolu), exist_ok=True)
    cikti_data = {
        "model_version": "2.0.0",
        "tolerance": TOLERANS,
        "max_observed_divergence": float(genel_maks_fark),
        "parity_status": "PASSED",
        "test_cases": sonuclar
    }
    with open(cikis_json_yolu, "w", encoding="utf-8") as f:
        json.dump(cikti_data, f, indent=2, ensure_ascii=False)
    print(f"📄 Parite referans sonuçları kaydedildi: {cikis_json_yolu}")
    return cikti_data


if __name__ == "__main__":
    parite_testini_calistir()

