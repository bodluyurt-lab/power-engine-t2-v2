# Sizinle Uyumlanan Sınav Tasarımı (SINAV_TASARIMI.md)

> **Düşünce Egzersizi & Klasik Sınav Sorusu Tasarımı**  
> *Kapsam: Bilgisayarlı Uyarlanabilir Testler (CAT), Kenar Yapay Zeka (Edge ML) ve Pedagojik Ölçme-Değerlendirme Mimarisi*

---

## 📌 Sınav Sorusu (Problem Bildirimi)

> **Soru:**  
> Geleneksel lineer sınavlar, her adaya aynı soru sırasını ve sabit süreyi dayatarak adayın gerçek bilgi derinliğini ölçmek yerine sınav kaygısını ve zaman baskısını ölçme eğilimindedir. **The Primer v2** projesinde geliştirdiğiniz istemci-taraflı (Edge ML / ONNX) dinamik iskele (*scaffolding*) mimarisini bir sonraki adıma taşıyarak; adayın anlık kognitif yüküne göre soru güçlüğünü ve ipucu seviyesini dinamik olarak uyarlayan **"Sizinle Uyumlanan Sınav" (Adaptive Exam Engine)** sistemini tasarlayınız.  
> 
> Tasarımınızda:
> 1. Davranışsal telemetri girdi vektörünü ve pedagojik hedefini,
> 2. Madde Tepki Kuramı (Item Response Theory - IRT) ile Makine Öğrenmesi arasındaki hibrit puanlama mekanizmasını,
> 3. Algoritmik adalet (*fairness*), açıklanabilirlik (*XAI*) ve Goodhart Yasası ("Bir ölçüt hedef haline gelirse, iyi bir ölçüt olmaktan çıkar") risklerine karşı güvenlik önlemlerini açıklayınız.

---

## 🧠 Vizyon ve Çözüm Mimarisi

### 1. Pedagojik Zemin: Yakınsak Gelişim Alanı (ZPD) ve Akış Hali
Lev Vygotsky’nin *Yakınsak Gelişim Alanı* kuramına göre ideal ölçüm, adayın bağımsız yapabildikleri ile rehberlikle yapabildikleri arasındaki eşikte gerçekleşir. Uyumlanan sınav, Mihaly Csikszentmihalyi'nin *Akış (Flow)* modelini temel alır: Soru zorluğu öğrencinin anlık yetkinlik algısına paralel yükseltilerek öğrenci ne sıkıntıya (*boredom*) ne de felç edici kaygıya (*frustration*) sürüklenir.

```
       ▲  Yüksek
       │               /  Kaygı / Tıkanma Bölgesi
  Soru │              /   (Yüksek Hata + Uzun Tereddüt)
Zorluğu│             /
       │  ───────►  [ İDEAL AKIŞ & ZPD KORİDORU ] ◄──────
       │           /   (Dinamik Uyumlanan Model)
       │          /
       │         /  Can Sıkıntısı Bölgesi (Aşırı Hızlı Karar)
       └──────────────────────────────────────────────►
         Düşük          Öğrenci Kognitif Yetkinliği        Yüksek
```

### 2. İstemci-Taraflı Telemetri Tensörü ve Karar Mekanizması
Her soru ekranında tarayıcıda (`onnxruntime-web`) çalışan hafif model şu tensörü işler:

$$X = \Big[ t_{\text{okuma}}, \; n_{\text{silme/duzeltme}}, \; \mathcal{H}_{\text{imleç}}, \; \theta_{\text{kümülatif\_hata}}, \; \delta_{\text{madde\_guclugu}} \Big]$$

- $t_{\text{okuma}}$: Soru kökünün okunma ve ilk etkileşim süresi (sn).
- $n_{\text{silme/duzeltme}}$: Metin kutusunda silinen karakter sayısı veya şık değiştirme adedi.
- $\mathcal{H}_{\text{imleç}}$: Fare / dokunmatik imleç hareketlerinin kararsızlık entropisi.
- $\theta$: Önceki sorulardan elde edilen güncel aday yetenek kestirimi.

**Modelin Çıktısı ($y$):**
1. **$y = 0$ (Yüksek Özgüven / Akış):** Bir sonraki adımda madde güçlüğü $+0.5\sigma$ artırılır.
2. **$y = 1$ (Kognitif Aşırı Yüklenme):** Soru değiştirilmez; ancak soru iki basamaklı alt adımlara bölünür (*scaffolding* açılır) veya kavramsal ipucu katmanı belirir.

### 3. Hibrit Notlandırma: IRT (Madde Tepki Kuramı) + ML
İpucu alan veya kolay soru çözen adayın haksız avantaj/dezavantaj yaşamasını önlemek için klasik doğru sayısı yerine 3-Parametreli Lojistik IRT modeli uygulanır:

$$P(\theta) = c_i + \frac{1 - c_i}{1 + e^{-a_i (\theta - b_i)}}$$

- İpucu kullanıldığı anda maddenin ayrımcılık parametresi $a_i$ ve güçlük parametresi $b_i$ dinamik olarak cezalandırılır. Böylece zor soruyu desteksiz çözen aday en yüksek yetenek puanını ($\theta$) alırken; ipucuyla çözen aday da sıfır çekmek yerine kısmi yetenek puanı kazanır.

### 4. Algoritmik Adalet ve Goodhart Yasası Savunması
- **Sistemi Hackleme Girişimi (Goodhart Yasası):** Aday bilerek yavaşlayıp ipucu tetiklemeye çalışırsa ne olur? Model, cevabın doğruluğu ile harcanan zamanın tutarlılığını analiz eder. Kolay soruda yapay gecikme tespit edilirse IRT modeli tahmin parametresini ($c_i$) yukarı çeker ve adayın yetenek kestirim katsayısını düşürür.
- **Açıklanabilirlik (XAI):** Sınav bitiminde adaya gizli bir algoritma kararı yerine; *"3. soruda formülü hatırlamakta 45 sn tereddüt ettiniz, sunulan türetim iskesiyle soruyu tamamladınız (+%65 kısmi puan)"* şeklinde şeffaf bir pedagojik bilişsel harita sunulur.

