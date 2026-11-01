#!/usr/bin/env python3
"""
The Primer v2 - Sentetik Davranışsal Teleometri Veri Üreticisi
KVKK Uyumlu, Tasarım Yoluyla Gizlilik (Privacy by Design) Prensiplerine Uygun Sentetik Veri Simülatörü.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")



def tereddut_skoru_hesapla(
    tereddut_suresi: float,
    karar_degistirme: int,
    onceki_hata: int,
    zorluk: int,
) -> float:
    """
    Oyuncunun bilişsel tıkanma (cognitive overload / frustration) olasılığını hesaplar.
    Lojistik sigmoid fonksiyonu ile 0-1 aralığına normalize edilir.
    """
    # Ağırlık katsayıları
    w_tereddut = 0.08      # Her 10 saniye tereddüt skoru +0.8 artırır
    w_karar_degis = 0.65   # Karar değiştirme kararsızlığın güçlü göstergesidir
    w_hata = 0.55          # Önceki hatalar toleransı düşürür
    w_zorluk = 0.70        # Zor dallar daha fazla destek gerektirir
    bias = -3.8            # Temel eğilim (ipucu vermeme yönünde)

    z = (
        (w_tereddut * tereddut_suresi)
        + (w_karar_degis * karar_degistirme)
        + (w_hata * onceki_hata)
        + (w_zorluk * zorluk)
        + bias
    )
    # Sigmoid
    return 1.0 / (1.0 + np.exp(-z))


def sentetik_veri_uret(
    adet: int = 300,
    seed: int = 42,
    mevcut_csv: str = None,
) -> pd.DataFrame:
    """
    Belirtilen adette sentetik oturum verisi üretir.
    Varsa mevcut gerçek kayıtları korur ve yeni sentetik verileri ekler.
    """
    np.random.seed(seed)

    mevcut_df = None
    baslangic_id = 1
    if mevcut_csv and os.path.exists(mevcut_csv):
        mevcut_df = pd.read_csv(mevcut_csv)
        # Sadece gerçek kayıtları al
        mevcut_gercek = mevcut_df[mevcut_df["kaynak"] == "gercek"]
        if not mevcut_gercek.empty:
            baslangic_id = len(mevcut_gercek) + 1

    kayitlar = []

    for i in range(adet):
        oturum_no = baslangic_id + i
        oturum_id = f"oturum_syn_{oturum_no:04d}"

        # 3 farklı oyuncu arketipleri:
        # 1. Hızlı/Emin (%40): Düşük süre, 0 karar değiştirme, az hata
        # 2. Ortalama/Düşünceli (%35): Orta süre, 1-2 karar değiştirme, orta hata
        # 3. Bocalayan/Zorlanan (%25): Yüksek süre, çoklu karar değiştirme, yüksek hata
        arketip = np.random.choice([1, 2, 3], p=[0.40, 0.35, 0.25])

        if arketip == 1:
            tereddut = np.clip(np.random.exponential(scale=6.0) + 1.0, 1.0, 30.0)
            karar_degis = np.random.choice([0, 1], p=[0.85, 0.15])
            onceki_hata = np.random.choice([0, 1], p=[0.80, 0.20])
            zorluk = np.random.choice([1, 2, 3], p=[0.50, 0.35, 0.15])
        elif arketip == 2:
            tereddut = np.clip(np.random.normal(loc=18.0, scale=6.0), 5.0, 45.0)
            karar_degis = np.random.choice([0, 1, 2], p=[0.30, 0.50, 0.20])
            onceki_hata = np.random.choice([0, 1, 2], p=[0.40, 0.40, 0.20])
            zorluk = np.random.choice([1, 2, 3], p=[0.25, 0.50, 0.25])
        else:
            tereddut = np.clip(np.random.normal(loc=38.0, scale=12.0), 15.0, 110.0)
            karar_degis = np.random.choice([1, 2, 3, 4, 5], p=[0.10, 0.30, 0.30, 0.20, 0.10])
            onceki_hata = np.random.choice([1, 2, 3, 4], p=[0.25, 0.35, 0.25, 0.15])
            zorluk = np.random.choice([1, 2, 3], p=[0.10, 0.40, 0.50])

        tereddut = round(float(tereddut), 1)
        karar_degis = int(karar_degis)
        onceki_hata = int(onceki_hata)
        zorluk = int(zorluk)

        p_ipucu = tereddut_skoru_hesapla(tereddut, karar_degis, onceki_hata, zorluk)
        # Olasılığa göre Bernoulli denemesi (bir miktar insan davranışı gürültüsü)
        ipucu = int(np.random.binomial(1, p_ipucu))

        kayitlar.append({
            "oturum_id": oturum_id,
            "tereddut_suresi_sn": tereddut,
            "karar_degistirme_sayisi": karar_degis,
            "onceki_hata_sayisi": onceki_hata,
            "secilen_dal_zorlugu": zorluk,
            "ipucu_verilsin_mi": ipucu,
            "kaynak": "sentetik"
        })

    sentetik_df = pd.DataFrame(kayitlar)

    if mevcut_df is not None:
        # Mevcut gerçek verileri koru, sentetikleri ekle
        gercekler = mevcut_df[mevcut_df["kaynak"] == "gercek"]
        nihai_df = pd.concat([gercekler, sentetik_df], ignore_index=True)
    else:
        nihai_df = sentetik_df

    return nihai_df


def main():
    parser = argparse.ArgumentParser(description="The Primer v2 Sentetik Veri Simülatörü")
    parser.add_argument("--adet", type=int, default=280, help="Üretilecek sentetik oturum adedi (varsayılan: 280)")
    parser.add_argument("--seed", type=int, default=42, help="Rastgelelik tohumu (varsayılan: 42)")
    parser.add_argument("--cikis", type=str, default="data/oturumlar.csv", help="Hedef CSV dosya yolu")
    args = parser.parse_args()

    df = sentetik_veri_uret(adet=args.adet, seed=args.seed, mevcut_csv=args.cikis)
    os.makedirs(os.path.dirname(os.path.abspath(args.cikis)), exist_ok=True)
    df.to_csv(args.cikis, index=False)
    print(f"✅ Başarılı: {len(df)} oturum kaydedildi ({args.cikis})")
    print(f"   - Gerçek Oturumlar: {len(df[df['kaynak'] == 'gercek'])}")
    print(f"   - Sentetik Oturumlar: {len(df[df['kaynak'] == 'sentetik'])}")
    print(f"   - Sınıf Dağılımı: 0 -> {(df['ipucu_verilsin_mi'] == 0).sum()}, 1 -> {(df['ipucu_verilsin_mi'] == 1).sum()}")


if __name__ == "__main__":
    main()

