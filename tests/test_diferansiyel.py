#!/usr/bin/env python3
"""
The Primer v2: Otomatik Diferansiyel Parite Birim Testi (tests/test_diferansiyel.py)
"""

import unittest
import os
import json
import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort

FEATURE_COLUMNS = [
    "tereddut_suresi_sn",
    "karar_degistirme_sayisi",
    "onceki_hata_sayisi",
    "secilen_dal_zorlugu"
]
TOLERANS = 1e-4


class TestPrimerParite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.joblib_yolu = "model/model.joblib"
        cls.onnx_yolu = "model/model.onnx"
        cls.parite_json_yolu = "tests/parite.json"

        self_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(self_dir)

        cls.joblib_yolu = os.path.join(repo_root, cls.joblib_yolu)
        cls.onnx_yolu = os.path.join(repo_root, cls.onnx_yolu)
        cls.parite_json_yolu = os.path.join(repo_root, cls.parite_json_yolu)

        if not os.path.exists(cls.joblib_yolu):
            raise FileNotFoundError(f"Model joblib bulunamadı: {cls.joblib_yolu}")
        if not os.path.exists(cls.onnx_yolu):
            raise FileNotFoundError(f"Model ONNX bulunamadı: {cls.onnx_yolu}")

        cls.sklearn_model = joblib.load(cls.joblib_yolu)
        cls.onnx_session = ort.InferenceSession(cls.onnx_yolu)

        with open(cls.parite_json_yolu, "r", encoding="utf-8") as f:
            cls.parite_verisi = json.load(f)

    def test_parite_status_passed(self):
        """tests/parite.json dosyasının genel parite durumunu doğrular."""
        self.assertEqual(self.parite_verisi.get("parity_status"), "PASSED")
        self.assertLess(self.parite_verisi.get("max_observed_divergence"), TOLERANS)

    def test_referans_senaryolar_tolerans_ici(self):
        """5 referans senaryo için Sklearn ve ONNX olasılık farkının < 1e-4 olduğunu test eder."""
        for tc in self.parite_verisi["test_cases"]:
            with self.subTest(senaryo=tc["id"]):
                inputs = tc["inputs"]
                df_in = pd.DataFrame([inputs])[FEATURE_COLUMNS].astype(np.float32)
                arr_in = df_in.to_numpy(dtype=np.float32)

                # Sklearn
                sk_prob = self.sklearn_model.predict_proba(df_in)[0]
                # ONNX
                onx_prob = self.onnx_session.run(["probabilities"], {"input": arr_in})[0][0]

                # Maksimum fark
                fark = float(np.abs(sk_prob - onx_prob).max())
                self.assertLess(
                    fark,
                    TOLERANS,
                    f"{tc['id']} için tolerans aşıldı! Fark: {fark:.2e} >= {TOLERANS:.1e}"
                )

    def test_kabul_senaryolari_mantiksal_uygunluk(self):
        """spec.md'deki 1. ve 2. kabul senaryolarının etiket ve olasılık beklentilerini doğrular."""
        # Senaryo 1: Kendinden emin oyuncu -> İpucu verilmemeli (P1 < 0.20)
        s1 = next(t for t in self.parite_verisi["test_cases"] if t["id"] == "senaryo_1_emin_oyuncu")
        self.assertEqual(s1["onnx"]["label"], 0)
        self.assertLess(s1["onnx"]["probabilities"][1], 0.20)

        # Senaryo 2: Bocalayan oyuncu -> İpucu verilmeli (P1 >= 0.75)
        s2 = next(t for t in self.parite_verisi["test_cases"] if t["id"] == "senaryo_2_bocalayan_oyuncu")
        self.assertEqual(s2["onnx"]["label"], 1)
        self.assertGreaterEqual(s2["onnx"]["probabilities"][1], 0.75)


if __name__ == "__main__":
    unittest.main()

