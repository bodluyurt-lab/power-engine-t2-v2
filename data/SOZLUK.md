# The Primer v2: Veri Mimarisi ve Veri Sözlüğü (`data/SOZLUK.md`)

Bu doküman, "The Primer v2" adaptif hikaye anlatım sisteminde toplanan ve model eğitiminde kullanılan veri setinin şemasını, özellik tanımlarını, birimlerini, değer aralıklarını ve KVKK (Kişisel Verilerin Korunması Kanunu) / GDPR uyumluluk prensiplerini tanımlar.

---

## 1. Veri Gizliliği ve KVKK / GDPR İlkeleri

Bu projede uygulanan veri mimarisi "Tasarım Yoluyla Gizlilik" (Privacy by Design) prensibine dayanır:
1. **Kişisel Veri İçermeme (No PII):** Veri setinde ad, soyad, e-posta, IP adresi, coğrafi konum, cihaz parmak izi (fingerprint) veya çerez kimliği gibi kişisel veriler kesinlikle yer almaz.
2. **Sahte / Maskeli Kimliklendirme (Pseudonymization):** Her oturum `oturum_id` ile `oturum_001`, `oturum_002` şeklinde soyut bir kimlikle anılır. Bu kimliğin gerçek bir kişiyle eşleştirilmesi teknik olarak imkansızdır.
3. **Yalnızca Davranışsal Telemetri:** Toplanan veriler sadece karar anındaki eylemleri (süre, tıklama sayısı, hata frekansı) temsil eden matematiksel metriklerdir.

---

## 2. Veri Şeması ve Sütun Açıklamaları

Aşağıdaki tablo, `data/oturumlar.csv` dosyasında bulunan tüm sütunların teknik sözlüğüdür:

| Sütun Adı | Veri Tipi | Rolü | Birim | Değer Aralığı / Olası Değerler | Eksik Veri İzni | Açıklama ve Toplama Yöntemi |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `oturum_id` | `string` | Metadata (ID) | - | `oturum_[0-9]{3,}` | Hayır (Boş Olamaz) | Oturuma ait benzersiz, anonimleştirilmiş tekil kimlik. |
| `tereddut_suresi_sn` | `float` | Girdi (Feature 1) | Saniye | `[0.0, 120.0]` | Hayır | Karar ekranının render edilmesinden kullanıcının karar butonuna tıklamasına kadar geçen süre. |
| `karar_degistirme_sayisi` | `int` | Girdi (Feature 2) | Adet | `[0, 20]` | Hayır | Kullanıcının son seçimini yapmadan önce farklı şıklar/seçenekler arasında yaptığı değişim sayısı. |
| `onceki_hata_sayisi` | `int` | Girdi (Feature 3) | Adet | `[0, 10]` | Hayır | Oyuncunun mevcut bölüme kadar hikaye dallarında karşılaştığı toplam başarısızlık/çıkmaz sokak adedi. |
| `secilen_dal_zorlugu` | `int` | Girdi (Feature 4) | Derece | `1, 2, 3` | Hayır | Kararın zorluk seviyesi (1: Kolay/Doğrusal, 2: Orta Seviye, 3: Karmaşık/Riskli). |
| `ipucu_verilsin_mi` | `int` | Hedef (Target Label) | İkili | `0, 1` | Hayır | Denetimli öğrenme hedef sınıfı.<br>`0`: İpucu gerekmiyor.<br>`1`: İpucu/dinamik iskele gösterilmeli. |
| `kaynak` | `string` | Doğrulama / MLOps | Kategori | `gercek`, `sentetik` | Hayır | Verinin gerçek kullanıcı oyun testinden mi yoksa simüle edilmiş veri üretecinden mi geldiğini belirten zorunlu alan. |

---

## 3. Özellik (Feature) Mühendisliği ve Sezgisel Mantık

Modelin hedefi, oyuncunun kognitif yükünü (cognitive load) ve tıkanma düzeyini yakalamaktır:
- **Yüksek Tereddüt + Çoklu Karar Değişimi:** Oyuncunun seçenekleri tartmakta zorlandığını veya ipuçlarını kaçırdığını gösterir.
- **Yüksek Önceki Hata:** Oyuncunun özgüveninin ve toleransının düştüğünü gösterir; bir sonraki hata oyunu bırakmasına neden olabilir.
- **Yüksek Dal Zorluğu:** Karmaşık dallarda hata maliyeti daha yüksektir; bu nedenle eşik değeri daha erken tetiklenebilir.

---

## 4. Veri Doğrulama ve Bütünlük Kuralları (Data Validation)

Eğitim veya veri yükleme aşamasında şu kurallar kontrol edilir:
1. Hiçbir hücrede `NaN`, `null` veya boş değer bulunamaz.
2. `tereddut_suresi_sn` $\ge 0.0$ olmalıdır. $120.0$ saniyenin üzerindeki değerler aykırı değer (outlier) olarak $120.0$'ye kırpılır (clip).
3. `karar_degistirme_sayisi` $\ge 0$ tamsayı olmalıdır.
4. `onceki_hata_sayisi` $\ge 0$ tamsayı olmalıdır.
5. `secilen_dal_zorlugu` kümesi $\{1, 2, 3\}$ elemanı olmalıdır.
6. `ipucu_verilsin_mi` kümesi $\{0, 1\}$ elemanı olmalıdır.
7. `kaynak` yalnızca `gercek` veya `sentetik` değerlerini alabilir.

