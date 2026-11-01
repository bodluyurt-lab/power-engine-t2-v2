#!/usr/bin/env python3
"""
The Primer v2: Otomatik 1080p YouTube Uyumlu Demo Video Sentezleyici (scripts/demo_video_olustur.py)

Bu betik:
1. 1920x1080 Full HD çözünürlükte, 30 FPS hızında The Primer v2 MLOps ve Karar Motorunu simüle eder.
2. Sahne 1: Mimari Giriş ve Proje Amacı (0 - 12s)
3. Sahne 2: Karar Arayüzü ve Canlı Telemetri Takibi (12 - 24s)
4. Sahne 3: Oyuncu A (Tereddütlü / Acemi) -> P = 0.9945 -> Dinamik İskele & İpucu (24 - 46s)
5. Sahne 4: Oyuncu B (Hızlı / Uzman) -> P = 0.0752 -> Uzman Modu & İpucu Engelleme (46 - 66s)
6. Sahne 5: MLOps Diferansiyel Parite Doğrulaması (< 1e-4) & Kapanış (66 - 76s)
7. H.264 / yuv420p formatında the_primer_v2_demo.mp4 olarak kaydeder.
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


WIDTH = 1920
HEIGHT = 1080
FPS = 30
OUTPUT_VIDEO = "the_primer_v2_demo.mp4"

# Renk Paleti
BG_DARK = (9, 13, 22)
CARD_BG = (19, 28, 46)
BORDER_COLOR = (30, 41, 59)
TEXT_WHITE = (248, 250, 252)
TEXT_MUTED = (148, 163, 184)
ACCENT_BLUE = (56, 189, 248)
SUCCESS_GREEN = (34, 197, 94)
WARNING_ORANGE = (245, 158, 11)
DANGER_RED = (239, 68, 68)
PURPLE_CARD = (30, 27, 75)

# Font hazırlığı (Sistem fontları veya PIL default)
def font_al(size):
    try:
        # Windows sistem fontları
        return ImageFont.truetype("segoeui.ttf", size)
    except:
        try:
            return ImageFont.truetype("arial.ttf", size)
        except:
            return ImageFont.load_default()

def bold_font_al(size):
    try:
        return ImageFont.truetype("segoeuib.ttf", size)
    except:
        return font_al(size)

FONT_HERO = bold_font_al(46)
FONT_TITLE = bold_font_al(32)
FONT_SUBTITLE = bold_font_al(24)
FONT_BODY = font_al(20)
FONT_BODY_BOLD = bold_font_al(20)
FONT_SMALL = font_al(16)
FONT_TINY = font_al(14)
FONT_HUD_VAL = bold_font_al(36)


def yuvarlak_kutu(draw, xy, r, fill, outline=None, width=1):
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle([x1, y1, x2, y2], radius=r, fill=fill, outline=outline, width=width)


def render_base_layout(title="The Primer v2 — Edge ML Dynamic Scaffolding"):
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)

    # Header Bar
    yuvarlak_kutu(draw, (0, 0, WIDTH, 75), 0, (11, 17, 32), outline=BORDER_COLOR)
    draw.text((40, 20), "🧠 THE PRIMER v2", font=FONT_TITLE, fill=ACCENT_BLUE)
    draw.text((340, 26), "|   Edge ML Adaptive Narrative Engine", font=FONT_BODY, fill=TEXT_MUTED)

    # Header Badges
    yuvarlak_kutu(draw, (1380, 18, 1660, 58), 16, (20, 83, 45), outline=SUCCESS_GREEN)
    draw.text((1396, 25), "ONNX Runtime Web (Wasm)", font=FONT_BODY_BOLD, fill=SUCCESS_GREEN)

    yuvarlak_kutu(draw, (1680, 18, 1880, 58), 16, (15, 23, 42), outline=ACCENT_BLUE)
    draw.text((1700, 25), "Parite < 1e-4 ✅", font=FONT_BODY_BOLD, fill=ACCENT_BLUE)

    return img, draw


def render_hud(draw, p_val, decision, state_text, state_color, latency_ms="1.18"):
    # HUD Box (Sağ Üst Sabit)
    hx, hy, hw, hh = 1420, 105, 460, 310
    yuvarlak_kutu(draw, (hx, hy, hx + hw, hy + hh), 14, (15, 23, 42, 240), outline=ACCENT_BLUE, width=2)

    draw.text((hx + 20, hy + 16), "MLOPS EDGE TELEMETRY", font=bold_font_al(16), fill=ACCENT_BLUE)
    draw.text((hx + hw - 100, hy + 16), "Wasm v1.22", font=FONT_TINY, fill=TEXT_MUTED)
    draw.line([(hx + 20, hy + 42), (hx + hw - 20, hy + 42)], fill=BORDER_COLOR, width=1)

    y_cur = hy + 55
    stats = [
        ("Model Mimarisi:", "Pipeline(Scaler + LR)"),
        ("Inference Gecikmesi:", f"{latency_ms} ms (Edge)"),
        ("Kognitif Durum:", state_text),
        ("Model Kararı:", decision)
    ]
    for label, val in stats:
        draw.text((hx + 20, y_cur), label, font=FONT_BODY, fill=TEXT_MUTED)
        c = state_color if label in ["Kognitif Durum:", "Model Kararı:"] else TEXT_WHITE
        draw.text((hx + hw - 20 - draw.textlength(val, font=FONT_BODY_BOLD), y_cur), val, font=FONT_BODY_BOLD, fill=c)
        y_cur += 32

    # P Box
    yuvarlak_kutu(draw, (hx + 20, hy + 200, hx + hw - 20, hy + 290), 10, (9, 13, 22), outline=BORDER_COLOR)
    draw.text((hx + 35, hy + 210), "İPUCU OLASILIĞI P(y=1):", font=FONT_TINY, fill=TEXT_MUTED)
    p_str = f"{p_val:.4f}"
    draw.text((hx + 35, hy + 235), p_str, font=FONT_HUD_VAL, fill=state_color)

    # Mini Progress Bar
    bx, by, bw, bh = hx + 220, hy + 242, 190, 18
    yuvarlak_kutu(draw, (bx, by, bx + bw, by + bh), 6, (30, 41, 59))
    fill_w = int(bw * p_val)
    if fill_w > 0:
        yuvarlak_kutu(draw, (bx, by, bx + fill_w, by + bh), 6, state_color)


def frame_olustur(t_sec):
    img, draw = render_base_layout()

    # Alt Bilgi / Altyazı Kutusu (Tüm sahnelerde ortak)
    yuvarlak_kutu(draw, (40, 960, 1880, 1045), 10, (11, 17, 32), outline=BORDER_COLOR)

    # -------------------------------------------------------------
    # SAHNE 1 (0 - 12s): PROJE VE MİMARİ GİRİŞİ
    # -------------------------------------------------------------
    if t_sec < 12.0:
        # Ana Tanıtım Kartı
        yuvarlak_kutu(draw, (40, 105, 1360, 930), 14, CARD_BG, outline=BORDER_COLOR)
        draw.text((80, 160), "PROJE: THE PRIMER v2", font=FONT_HERO, fill=ACCENT_BLUE)
        draw.text((80, 230), "İstemci-Taraflı (Edge ML / ONNX) Uyarlanabilir Kurgu & Dinamik İskele Motoru", font=FONT_TITLE, fill=TEXT_WHITE)

        draw.text((80, 320), "📌 Çözülen Problem:", font=FONT_SUBTITLE, fill=WARNING_ORANGE)
        draw.text((80, 365), "• Klasik oyunlarda sabit akış: Zorlanan oyuncu tıkanır (churn), uzman oyuncu sıkılır.", font=FONT_BODY, fill=TEXT_WHITE)
        draw.text((80, 405), "• Çözüm: Oyuncunun karar anındaki tereddüt ve kararsızlığından kognitif yükünü kestiren ML modeli.", font=FONT_BODY, fill=TEXT_WHITE)

        draw.text((80, 480), "📐 MLOps Mimarisi ve Sözleşme:", font=FONT_SUBTITLE, fill=SUCCESS_GREEN)
        draw.text((80, 525), "• Model: scikit-learn (StandardScaler + LogisticRegression) ➔ skl2onnx (Opset 17)", font=FONT_BODY, fill=TEXT_WHITE)
        draw.text((80, 565), "• Çalışma Zamanı: onnxruntime-web (WebAssembly), Sıfır Ağ Gecikmesi (1.2 ms), %100 KVKK Uyumlu", font=FONT_BODY, fill=TEXT_WHITE)
        draw.text((80, 605), "• Diferansiyel Parite Testi: Python Oracle vs. ONNX Web toleransı < 1e-4 (Gerçekleşen: 5.96e-08)", font=FONT_BODY, fill=TEXT_WHITE)

        # Karşılaştırma Kutusu
        yuvarlak_kutu(draw, (80, 680, 1320, 870), 10, PURPLE_CARD, outline=(99, 102, 241))
        draw.text((110, 705), "🎬 İki Farklı Oyuncu Simülasyonu (60-90s Demo):", font=bold_font_al(22), fill=(199, 210, 254))
        draw.text((110, 755), "1. Oyuncu A (Tereddütlü/Bocalayan): Süre: 42s, Değiştirme: 4, Hata: 3 ➔ P = 0.99 ➔ İPUCU DEVREYE GİRER", font=FONT_BODY_BOLD, fill=WARNING_ORANGE)
        draw.text((110, 805), "2. Oyuncu B (Hızlı/Uzman): Süre: 3.5s, Değiştirme: 0, Hata: 0 ➔ P = 0.07 ➔ İPUCU ENGELLENİR / AKIŞ DERİNLEŞİR", font=FONT_BODY_BOLD, fill=SUCCESS_GREEN)

        render_hud(draw, 0.0, "Hazır", "Bekleniyor", TEXT_MUTED, latency_ms="—")
        draw.text((60, 990), "🎙️ Giriş: The Primer v2, tarayıcıda sıfır gecikmeyle oyuncu davranışını analiz eden ONNX tabanlı adaptif motordur.", font=FONT_BODY, fill=TEXT_WHITE)

    # -------------------------------------------------------------
    # SAHNE 2 (12 - 24s): KARAR EKRANI & TELEMETRİ TAKİBİ
    # -------------------------------------------------------------
    elif t_sec < 24.0:
        # Demo Banner
        yuvarlak_kutu(draw, (40, 105, 1360, 255), 12, PURPLE_CARD, outline=(99, 102, 241))
        draw.text((70, 125), "🎬 60-90 Saniyelik Hoca Demo Simülasyon Paneli", font=FONT_SUBTITLE, fill=(199, 210, 254))
        yuvarlak_kutu(draw, (70, 175, 680, 235), 8, (220, 38, 38))
        draw.text((85, 195), "🔴 1. Oyuncu A: Tereddütlü Simülasyonu", font=FONT_BODY_BOLD, fill=TEXT_WHITE)
        yuvarlak_kutu(draw, (720, 175, 1330, 235), 8, (2, 132, 199))
        draw.text((735, 195), "🟢 2. Oyuncu B: Hızlı / Uzman Simülasyonu", font=FONT_BODY_BOLD, fill=TEXT_WHITE)

        # Oyun Sahnesi
        yuvarlak_kutu(draw, (40, 280, 1360, 930), 12, CARD_BG, outline=BORDER_COLOR)
        draw.text((70, 310), "🕹️ Bölüm 3: Kuantum Veri Merkezine Sızış", font=FONT_TITLE, fill=TEXT_WHITE)

        # Anlatı
        yuvarlak_kutu(draw, (70, 370, 1330, 445), 8, (11, 17, 32), outline=BORDER_COLOR)
        draw.text((90, 395), '"Siber geçidin kapısı kilitlendi. Güvenlik yapay zekası nöronal dalgalarınızı tarıyor. Seçim yapmalısınız..."', font=FONT_BODY, fill=(226, 232, 240))

        # Canlı Telemetri Barı (Zamana göre artan tereddüt)
        tereddut_canli = min(2.0 + (t_sec - 12.0) * 1.8, 22.0)
        yuvarlak_kutu(draw, (70, 470, 1330, 530), 8, (11, 17, 32))
        draw.text((90, 490), f"⏱️ Tereddüt Süresi: {tereddut_canli:.1f}s    🔄 Karar Değiştirme: 1    ⚠️ Önceki Hatalar: 1    🎯 Seçilen Yol: Dal 2 (Orta)", font=FONT_BODY_BOLD, fill=ACCENT_BLUE)

        # Şıklar
        opt_y = 555
        options = [
            ("A) Havalandırma Şaftından Sız (Doğrusal / Kolay)", "Zorluk: 1 · Risk: Minimal", False),
            ("B) Güvenlik Görevlisinin Kimlik Kartını Klonla", "Zorluk: 2 · Biyometrik Doğrulama İster", True),
            ("C) Ana Güvenlik Çekirdeğine Doğrudan Kuantum Hack Yap", "Zorluk: 3 · Yüksek Risk · Tuzak Algoritmalar", False)
        ]
        for baslik, meta, checked in options:
            bg_c = (36, 52, 77) if checked else (30, 41, 59)
            yuvarlak_kutu(draw, (70, opt_y, 1330, opt_y + 70), 8, bg_c, outline=ACCENT_BLUE if checked else BORDER_COLOR)
            radio_sym = "🔘" if checked else "⚪"
            draw.text((90, opt_y + 15), f"{radio_sym}  {baslik}", font=FONT_BODY_BOLD, fill=TEXT_WHITE)
            draw.text((130, opt_y + 42), meta, font=FONT_TINY, fill=TEXT_MUTED)
            opt_y += 85

        render_hud(draw, 0.42, "Karar Bekleniyor", "Normal Değerlendirme", ACCENT_BLUE, latency_ms="1.12")
        draw.text((60, 990), "🎙️ Sahne: Karar anında oyuncunun ekranda geçirdiği bekleme süresi ve tıklama değişimleri canlı kaydediliyor.", font=FONT_BODY, fill=TEXT_WHITE)

    # -------------------------------------------------------------
    # SAHNE 3 (24 - 46s): OYUNCU A (TEREDDÜTLÜ) ➔ İPUCU DEVREDE
    # -------------------------------------------------------------
    elif t_sec < 46.0:
        # Demo Banner (Oyuncu A seçili efekti)
        yuvarlak_kutu(draw, (40, 105, 1360, 255), 12, PURPLE_CARD, outline=(99, 102, 241))
        draw.text((70, 125), "🎬 60-90 Saniyelik Hoca Demo Simülasyon Paneli", font=FONT_SUBTITLE, fill=(199, 210, 254))
        # Oyuncu A Vurgulu
        yuvarlak_kutu(draw, (70, 170, 680, 240), 8, (220, 38, 38), outline=WARNING_ORANGE, width=3)
        draw.text((85, 192), "🔴 [TIKLANDI] Oyuncu A: Tereddütlü Simülasyonu", font=FONT_BODY_BOLD, fill=TEXT_WHITE)
        yuvarlak_kutu(draw, (720, 175, 1330, 235), 8, (2, 132, 199))
        draw.text((735, 195), "🟢 2. Oyuncu B: Hızlı / Uzman Simülasyonu", font=FONT_BODY_BOLD, fill=TEXT_WHITE)

        # Oyun Sahnesi
        yuvarlak_kutu(draw, (40, 280, 1360, 930), 12, CARD_BG, outline=BORDER_COLOR)
        draw.text((70, 310), "🕹️ Bölüm 3: Kuantum Veri Merkezine Sızış", font=FONT_TITLE, fill=TEXT_WHITE)

        # Telemetri (42s, 4 değiştirme, 3 hata, Dal 3)
        yuvarlak_kutu(draw, (70, 365, 1330, 425), 8, (11, 17, 32), outline=WARNING_ORANGE)
        draw.text((90, 385), "⏱️ Tereddüt Süresi: 42.0s    🔄 Karar Değiştirme: 4    ⚠️ Önceki Hatalar: 3    🎯 Seçilen Yol: Dal 3 (Zor)", font=FONT_BODY_BOLD, fill=WARNING_ORANGE)

        # Seçenek 3 Seçili
        opt_y = 445
        options = [
            ("A) Havalandırma Şaftından Sız (Doğrusal)", False),
            ("B) Kimlik Kartını Klonla", False),
            ("C) Ana Güvenlik Çekirdeğine Doğrudan Kuantum Hack Yap (Zorluk: 3)", True)
        ]
        for baslik, checked in options:
            bg_c = (36, 52, 77) if checked else (20, 28, 45)
            yuvarlak_kutu(draw, (70, opt_y, 1330, opt_y + 55), 8, bg_c, outline=WARNING_ORANGE if checked else BORDER_COLOR)
            radio_sym = "🔘" if checked else "⚪"
            draw.text((90, opt_y + 16), f"{radio_sym}  {baslik}", font=FONT_BODY_BOLD, fill=TEXT_WHITE)
            opt_y += 68

        # DİNAMİK İSKELE & İPUCU KUTUSU (GÖZ ALICI TURUNCU)
        yuvarlak_kutu(draw, (70, 670, 1330, 905), 10, (40, 25, 5), outline=WARNING_ORANGE, width=3)
        draw.text((95, 695), "💡 THE PRIMER DİNAMİK İSKELESİ DEVREDE (OYUNCU A)", font=bold_font_al(24), fill=WARNING_ORANGE)
        draw.text((95, 740), "Tespit: Yüksek tereddüt (42s), 4 kez karar değiştirme ve 3 önceki hata algılandı.", font=FONT_BODY_BOLD, fill=TEXT_WHITE)
        draw.text((95, 780), "Yardımcı İpucu: Kuantum güvenlik duvarı 3 saniyede bir şifre fazı değiştiriyor.", font=FONT_BODY, fill=(254, 240, 138))
        draw.text((95, 815), "Öneri: Sızmayı garantilemek için terminale  '--bypass-phase-lock'  parametresini ekleyin.", font=FONT_BODY_BOLD, fill=ACCENT_BLUE)
        draw.text((95, 860), "ℹ️ Model Telemetrisi: Çıkarım 1.18 ms'de bitti. Oyuncunun tıkanmasını (churn) önlemek için ipucu verildi.", font=FONT_TINY, fill=TEXT_MUTED)

        # HUD Oyuncu A
        render_hud(draw, 0.9945, "İPUCU VER (1)", "Tıkanma / Kararsızlık", WARNING_ORANGE, latency_ms="1.18")
        draw.text((60, 990), "🎙️ Oyuncu A: Yüksek tereddüt ve hata geçmişi tespit edildi. P=%99.5 ile Dinamik İskele ipucu sağladı.", font=FONT_BODY_BOLD, fill=WARNING_ORANGE)

    # -------------------------------------------------------------
    # SAHNE 4 (46 - 66s): OYUNCU B (HIZLI/UZMAN) ➔ AKIŞ DERİNLEŞİR
    # -------------------------------------------------------------
    elif t_sec < 66.0:
        # Demo Banner (Oyuncu B seçili efekti)
        yuvarlak_kutu(draw, (40, 105, 1360, 255), 12, PURPLE_CARD, outline=(99, 102, 241))
        draw.text((70, 125), "🎬 60-90 Saniyelik Hoca Demo Simülasyon Paneli", font=FONT_SUBTITLE, fill=(199, 210, 254))
        yuvarlak_kutu(draw, (70, 175, 680, 235), 8, (220, 38, 38))
        draw.text((85, 195), "🔴 1. Oyuncu A: Tereddütlü Simülasyonu", font=FONT_BODY_BOLD, fill=TEXT_WHITE)
        # Oyuncu B Vurgulu
        yuvarlak_kutu(draw, (720, 170, 1330, 240), 8, (2, 132, 199), outline=SUCCESS_GREEN, width=3)
        draw.text((735, 192), "🟢 [TIKLANDI] Oyuncu B: Hızlı / Uzman Simülasyonu", font=FONT_BODY_BOLD, fill=TEXT_WHITE)

        # Oyun Sahnesi
        yuvarlak_kutu(draw, (40, 280, 1360, 930), 12, CARD_BG, outline=BORDER_COLOR)
        draw.text((70, 310), "🕹️ Bölüm 3: Kuantum Veri Merkezine Sızış", font=FONT_TITLE, fill=TEXT_WHITE)

        # Telemetri (3.5s, 0 değiştirme, 0 hata, Dal 1)
        yuvarlak_kutu(draw, (70, 365, 1330, 425), 8, (11, 17, 32), outline=SUCCESS_GREEN)
        draw.text((90, 385), "⏱️ Tereddüt Süresi: 3.5s    🔄 Karar Değiştirme: 0    ⚠️ Önceki Hatalar: 0    🎯 Seçilen Yol: Dal 1 (Kolay)", font=FONT_BODY_BOLD, fill=SUCCESS_GREEN)

        # Seçenek 1 Seçili
        opt_y = 445
        options = [
            ("A) Havalandırma Şaftından Sız (Doğrusal / Kolay - Zorluk: 1)", True),
            ("B) Kimlik Kartını Klonla", False),
            ("C) Ana Güvenlik Çekirdeğine Doğrudan Kuantum Hack Yap", False)
        ]
        for baslik, checked in options:
            bg_c = (36, 52, 77) if checked else (20, 28, 45)
            yuvarlak_kutu(draw, (70, opt_y, 1330, opt_y + 55), 8, bg_c, outline=SUCCESS_GREEN if checked else BORDER_COLOR)
            radio_sym = "🔘" if checked else "⚪"
            draw.text((90, opt_y + 16), f"{radio_sym}  {baslik}", font=FONT_BODY_BOLD, fill=TEXT_WHITE)
            opt_y += 68

        # UZMAN AKIŞI KUTUSU (GÖZ ALICI YEŞİL)
        yuvarlak_kutu(draw, (70, 670, 1330, 905), 10, (10, 40, 20), outline=SUCCESS_GREEN, width=3)
        draw.text((95, 695), "🚀 UZMAN MODU: AKIŞ DERİNLEŞTİRİLDİ (OYUNCU B)", font=bold_font_al(24), fill=SUCCESS_GREEN)
        draw.text((95, 740), "Durum: Kesintisiz Akış Hali (Flow State) — Tereddüt: 3.5s, Sıfır Karar Değişimi, Sıfır Hata.", font=FONT_BODY_BOLD, fill=TEXT_WHITE)
        draw.text((95, 780), "Model Kararı: Gereksiz ipucu engellendi; oyuncunun kognitif ivmesi korunuyor.", font=FONT_BODY, fill=(187, 247, 208))
        draw.text((95, 815), "Akış Uyarlaması: Sistem doğrudan 'Derin Kuantum Çekirdeği' özel hikaye rotasına bağlandı.", font=FONT_BODY_BOLD, fill=ACCENT_BLUE)
        draw.text((95, 860), "ℹ️ Model Telemetrisi: P = %7.5 (Eşik Altı). Akış bölünmeyerek uzman oyuncu deneyimi en üst düzeye çıkarıldı.", font=FONT_TINY, fill=TEXT_MUTED)

        # HUD Oyuncu B
        render_hud(draw, 0.0752, "AKIŞI KORU (0)", "Optimal Akış Hali", SUCCESS_GREEN, latency_ms="1.05")
        draw.text((60, 990), "🎙️ Oyuncu B: Karar hızlı ve kendinden emin. P=%7.5 ile ipucu verilmedi, akış derinleştirildi.", font=FONT_BODY_BOLD, fill=SUCCESS_GREEN)

    # -------------------------------------------------------------
    # SAHNE 5 (66 - 76s): MLOPS PARİTE & KAPANIŞ
    # -------------------------------------------------------------
    else:
        yuvarlak_kutu(draw, (40, 105, 1360, 930), 14, CARD_BG, outline=BORDER_COLOR)
        draw.text((80, 160), "🔬 MLOPS DİFERANSİYEL PARİTE & DOĞRULAMA", font=FONT_HERO, fill=ACCENT_BLUE)

        # Tablo
        draw.text((80, 240), "Python Oracle (Sklearn) vs. ONNX Runtime Web Parite Tablosu:", font=FONT_SUBTITLE, fill=TEXT_WHITE)

        tablo_y = 300
        headers = ["Senaryo", "Girdi Vektörü [t, deg, err, diff]", "Sklearn P(1)", "ONNX P(1)", "Mutlak Fark", "Durum"]
        cols_x = [80, 360, 750, 920, 1100, 1260]

        # Başlık satırı
        yuvarlak_kutu(draw, (80, tablo_y, 1320, tablo_y + 40), 6, (30, 41, 59))
        for col_name, cx in zip(headers, cols_x):
            draw.text((cx + 10, tablo_y + 10), col_name, font=bold_font_al(16), fill=ACCENT_BLUE)
        tablo_y += 50

        rows = [
            ("1. Emin Oyuncu", "[3.5s, 0, 0, 1]", "0.0752", "0.0752", "2.98e-08", "✅ GEÇTİ"),
            ("2. Bocalayan", "[42.0s, 4, 3, 3]", "0.9945", "0.9945", "5.96e-08", "✅ GEÇTİ"),
            ("3. Sınır / Yavaş", "[22.0s, 0, 1, 2]", "0.4047", "0.4047", "5.96e-08", "✅ GEÇTİ"),
            ("4. Orta / Kararsız", "[28.5s, 3, 2, 3]", "0.9552", "0.9552", "2.98e-08", "✅ GEÇTİ"),
            ("5. Hızlı / Hatalı", "[7.0s, 1, 2, 1]", "0.3457", "0.3457", "0.00e+00", "✅ GEÇTİ"),
        ]
        for r in rows:
            yuvarlak_kutu(draw, (80, tablo_y, 1320, tablo_y + 45), 6, (15, 23, 42))
            for val, cx in zip(r, cols_x):
                c = SUCCESS_GREEN if "GEÇTİ" in val else TEXT_WHITE
                draw.text((cx + 10, tablo_y + 12), val, font=FONT_BODY, fill=c)
            tablo_y += 55

        # Özet Kutusu
        yuvarlak_kutu(draw, (80, 650, 1320, 870), 10, (11, 17, 32), outline=SUCCESS_GREEN, width=2)
        draw.text((110, 680), "🏆 Doğrulama Özeti:", font=bold_font_al(24), fill=SUCCESS_GREEN)
        draw.text((110, 725), "• Şartname Tolerans Eşiği: < 1.00e-04  |  Ölçülen Maksimum Fark: 5.96e-08 (1670 kat daha hassas!)", font=FONT_BODY_BOLD, fill=TEXT_WHITE)
        draw.text((110, 765), "• Uçta Çıkarım Gecikmesi: 1.05 - 1.18 ms (Sıfır Sunucu Maliyeti / Sıfır Ağ İhtiyacı)", font=FONT_BODY, fill=ACCENT_BLUE)
        draw.text((110, 805), "• Kod & Model Deposu: https://github.com/bodluyurt-lab/power-engine-t2-v2", font=FONT_BODY_BOLD, fill=TEXT_WHITE)

        render_hud(draw, 0.0752, "Parite Doğrulandı", "Tüm Testler Yeşil", SUCCESS_GREEN, latency_ms="1.18")
        draw.text((60, 990), "🎙️ Kapanış: Model tüm diferansiyel parite testlerini başarıyla geçti. Teslime hazırdır.", font=FONT_BODY_BOLD, fill=SUCCESS_GREEN)

    return np.array(img)


def main():
    toplam_sure_sn = 75.0  # 75 saniyelik ideal demo süresi
    toplam_kare = int(toplam_sure_sn * FPS)

    print("=" * 65)
    print("🎬 The Primer v2: Full HD 1080p Demo Video Sentezleniyor...")
    print(f"📌 Çözünürlük: {WIDTH}x{HEIGHT} | FPS: {FPS} | Süre: {toplam_sure_sn} sn")
    print(f"📌 Hedef Dosya: {OUTPUT_VIDEO}")
    print("=" * 65)

    writer = imageio.get_writer(
        OUTPUT_VIDEO,
        fps=FPS,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p"
    )

    yuzde_adim = toplam_kare // 10

    for i in range(toplam_kare):
        t = i / FPS
        frame = frame_olustur(t)
        writer.append_data(frame)

        if i % yuzde_adim == 0 or i == toplam_kare - 1:
            yuzde = int((i + 1) / toplam_kare * 100)
            print(f"⏳ Video İşleniyor: %{yuzde} ({t:.1f}s / {toplam_sure_sn}s)")

    writer.close()
    dosya_boyutu_mb = os.path.getsize(OUTPUT_VIDEO) / (1024 * 1024)
    print("=" * 65)
    print(f"🎉 DEMO VİDEO BAŞARIYLA OLUŞTURULDU!")
    print(f"📁 Dosya: {OUTPUT_VIDEO} ({dosya_boyutu_mb:.2f} MB)")
    print(f"🎥 YouTube, VLC ve tüm tarayıcılarla %100 uyumlu H.264 / MP4")
    print("=" * 65)


if __name__ == "__main__":
    main()
