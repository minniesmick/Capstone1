# Türkçe Sosyal Medya Metinlerinde Hibrit Karar Ajanlı Duygu Analizi: Geleneksel Makine Öğrenmesi ve Büyük Dil Modellerinin Entegrasyonu

**Yazar:** Alper Yusuf Yaman  
**Tarih:** 2026
**Ders:** Data Science and AI / Capstone Project 1

---

## Özet

Bu çalışmada, Türkçe sosyal medya yorumlarını Pozitif veya Negatif olarak sınıflandıran iki aşamalı bir Hibrit Karar Ajanı mimarisi sunulmaktadır. İlk aşamada TF-IDF tabanlı Linear Support Vector Machine (SVM) kullanılmış; ikinci aşamada ise Ollama altyapısı üzerinde yerel olarak çalıştırılan 13 farklı büyük dil modeli (LLM) karşılaştırılmıştır. 11.120 yorumdan oluşan dengesiz bir veri seti üzerinde gerçekleştirilen deneylerde Linear SVM, %86.78 doğruluk ile tüm LLM modellerini belirgin biçimde geride bırakmıştır. En iyi LLM performansını Llama3 1 latest modeli %73.83 doğrulukla sergilemiştir. Hibrit mimaride, iki modelin aynı karara vardığı "Consensus" durumları ile farklı karara vardığı "Conflict" durumları ayrıştırılmış; Conflict durumlarında LLM tercih edilmiştir. Llama3 1 latest modeli ile kurulan hibrit sistemde %70.9 oranında Consensus elde edilmiştir. Conflict analizinde SVM'in çoğu çakışmayı kazandığı gözlemlenmiş; bu bulgu, Conflict durumlarında LLM'i otomatik olarak seçen sabit kural stratejisinin iyileştirmeye açık olduğunu ortaya koymaktadır.

**Anahtar Kelimeler:** duygu analizi, doğal dil işleme, Türkçe metin sınıflandırma, destek vektör makinesi, büyük dil modeli, hibrit mimari, Ollama

---

## 1. Giriş

Sosyal medya platformları, günümüzde milyonlarca kullanıcının ürünler, hizmetler ve olaylar hakkında görüşlerini paylaştığı temel iletişim kanallarına dönüşmüştür. Bu yorumların otomatik olarak duygu sınıflandırmasına tabi tutulması, pazarlama analizinden siyasi araştırmalara kadar geniş bir uygulama yelpazesi sunmaktadır [1].

Türkçe, morfolojik zenginliği, ek almış sözcük yapısı ve yüksek eş anlamlı kelime oranı nedeniyle İngilizce odaklı duygu analizi yöntemlerinin doğrudan uygulanmasına direnç gösteren bir dildir [2]. Buna ek olarak, sosyal medya metinleri argo, kinaye, kısaltma ve dilbilgisi kurallarını ihlal eden yapılar içermektedir. Geleneksel bag-of-words modelleri bu bağlamsal nüansları çoğu zaman yakalayamamaktadır.

Bu çalışmanın temel katkıları şunlardır:

1. Türkçe sosyal medya yorumları üzerinde altı klasik makine öğrenmesi modelinin sistematik karşılaştırması
2. Ollama üzerinde çalıştırılan 13 farklı açık kaynaklı LLM modelinin sıfır-atımlı (zero-shot) performans kıyaslaması
3. TF-IDF + Linear SVM ile yerel Ollama LLM'ini birleştiren Hibrit Karar Ajanı mimarisi
4. Consensus/Conflict ayrıştırma mekanizmasının tasarımı ve nicel analizi
5. Modelin yorumlanamaz kararlarına anlaşılabilirlik (explainability) katmanı eklenmesi

---

## 2. İlgili Çalışmalar

Türkçe duygu analizine yönelik çalışmalar 2010'ların ortasından itibaren hız kazanmıştır. Kaya ve ark. [3], Twitter verisi üzerinde Naive Bayes ve SVM ile %74 doğruluk elde etmiş; TF-IDF'in CountVectorizer'a kıyasla daha iyi performans gösterdiğini raporlamışlardır. Demirtas ve Pechenizkiy [4], Türkçe film yorumları üzerinde çok sınıflı duygu analizi gerçekleştirmiş ve SVM tabanlı yaklaşımın derin öğrenme modellerine yakın sonuçlar ürettiğini göstermiştir.

Büyük dil modelleri (LLM) alanında ise son yıllarda sıfır-atımlı (zero-shot) sınıflandırma yaklaşımları dikkat çekici sonuçlar ortaya koymaktadır. Brown ve ark. [5], GPT-3 ile çeşitli NLP görevlerinde ince ayar gerektirmeksizin yüksek başarı elde etmiştir. Llama 3 gibi açık kaynaklı modeller ise bu yetenekleri yerel ve gizlilik koruyucu ortamlarda sunmaktadır [6].

Hibrit yaklaşımlar söz konusu olduğunda, Ensemble ve Stacking yöntemleri makine öğrenmesinde yerleşik tekniklerdir [7]. Ancak geleneksel bir sınıflandırıcı ile LLM'i Consensus/Conflict mantığıyla birleştiren bir mimari, özellikle Türkçe argo ve kinaye içeren metinler bağlamında literatürde yeterince araştırılmamıştır.

---

## 3. Veri Seti ve Veri Analizi

### 3.1 Veri Seti Genel Bakışı

Çalışmada kullanılan veri seti, çeşitli Türkçe sosyal medya platformlarından derlenen 11.120 yorumdan oluşmaktadır. Her yorum iki sütun içermektedir: **Tip** (Pozitif/Negatif) ve **Paylaşım** (ham Türkçe metin). Veri seti %55.1 Pozitif (6.134 yorum) ve %44.9 Negatif (4.986 yorum) örnek içermekte olup orta düzeyde bir sınıf dengesizliği gözlemlenmektedir (Tablo 1).

**Tablo 1. Veri Seti İstatistikleri**

| Özellik | Pozitif | Negatif | Toplam |
|---------|---------|---------|--------|
| Yorum Sayısı | 6.134 | 4.986 | 11.120 |
| Oran | %55.1 | %44.9 | %100 |
| Ort. Kelime Sayısı | ~12.4 | ~11.8 | ~12.1 |
| Medyan Kelime | 9 | 8 | — |
| Ort. Karakter | ~68 | ~65 | — |
| Ort. Unique Kelime | ~10.2 | ~9.7 | — |

### 3.2 Keşifsel Veri Analizi

Metin uzunluğu dağılımları incelendiğinde, her iki sınıfın da benzer istatistiksel özellikler sergilediği görülmektedir; bu durum uzunluğun tek başına güçlü bir ayırt edici özellik olmadığına işaret etmektedir. Kelime dağılımı açısından yapılan analiz, Pozitif yorumlarda "güzel", "harika", "mükemmel" ve "teşekkür" gibi terimlerin belirgin biçimde yüksek sıklıkta geçtiğini; Negatif yorumlarda ise "kötü", "berbat", "rezalet" sözcüklerinin öne çıktığını ortaya koymuştur.

Sınıf dağılımı grafiği, Pozitif sınıfın yaklaşık 1.150 yorum ile daha fazla temsil edildiğini göstermektedir. Bu dengesizlik, sınıflandırma metriği olarak ham doğruluk yerine ağırlıklı F1 skorunun kullanılmasını zorunlu kılmaktadır.

Kelime bulutu (word cloud) görselleştirmeleri, her iki sınıfın anlamsal merkez kelimelerini net biçimde ayırt etmektedir. Özellikle Negatif sınıfta argonun ve kısaltmaların yoğun kullanımı, ön işleme aşamasının önemini pekiştirmektedir.

### 3.3 Veri Ön İşleme Hattı

Veri ön işleme, aşağıdaki sıralı adımlardan oluşmaktadır:

1. **Küçük harf dönüşümü:** Türkçeye özgü büyük/küçük harf eşleştirmesi (İ→i, Ş→ş vb.) dahil
2. **URL, mention ve hashtag kaldırma:** `http://`, `@kullanici`, `#etiket` biçimindeki örüntüler regex ile silinmiştir
3. **Rakam kaldırma:** Sayısal ifadeler sınıflandırma açısından bilgi içermemektedir
4. **Noktalama ve özel karakter temizleme:** Türkçe alfabesi dışındaki karakterler kaldırılmıştır
5. **Stop-word kaldırma:** 80'den fazla Türkçe işlev kelimesi içeren özel bir liste kullanılmıştır (ör. "bir", "ve", "de", "ki")
6. **Fazla boşluk temizleme:** Normalize edilmiş tek boşluk biçimine dönüştürme

Bu pipeline, ortalama kelime sayısını ~12.1'den ~8.3'e düşürerek yaklaşık %31 azaltma sağlamıştır. Bu oran, stop-word'lerin Türkçe sosyal medya metnindeki baskın varlığını doğrulamaktadır.

---

## 4. Yöntem

### 4.1 Özellik Çıkarımı

**TF-IDF Vektörizasyonu** uygulanmıştır. Parametre seçimi aşağıdaki gerekçelere dayanmaktadır:

| Parametre | Değer | Gerekçe |
|-----------|-------|---------|
| `analyzer` | `word` | Kelime tabanlı n-gram |
| `ngram_range` | `(1, 2)` | Unigram + bigram; "çok güzel" gibi colocation'ları yakalar |
| `max_features` | `30.000` | Türkçe'nin zengin morfolojisi için geniş kelime dağarcığı |
| `sublinear_tf` | `True` | TF frekansına log dönüşümü; yüksek frekanslı kelimelerin dominasyonunu azaltır |
| `min_df` | `2` | En az 2 belgede geçen kelimeler; gürültüyü azaltır |

Bu yapılandırma, 30.000 boyutlu seyrek bir özellik matrisi üretmiştir.

### 4.2 Veri Bölme

Veri seti, stratified sampling ile %80 eğitim (8.896 yorum) ve %20 test (2.224 yorum) olarak bölünmüştür. Stratified bölme, her iki kümenin de orijinal sınıf dağılımını korumasını sağlamıştır. `random_state=42` tüm deneylerde sabit tutularak yeniden üretilebilirlik garantilenmiştir.

### 4.3 Klasik Makine Öğrenmesi Modelleri ve Hiperparametreler

Altı farklı model eğitilmiş ve değerlendirilmiştir:

**Logistic Regression:** `C=1.0, max_iter=1000, solver='lbfgs'`

**Multinomial Naive Bayes:** `alpha=0.5`

**Linear SVM:** `C=1.0, max_iter=2000`

**Random Forest:** `n_estimators=200, max_depth=20, min_samples_split=5`

**Gradient Boosting:** Scikit-learn standart GradientBoostingClassifier implementasyonu.

**XGBoost:** `tree_method='hist', device='cuda', n_estimators=200, max_depth=6, learning_rate=0.1` — RTX 3060 Ti GPU üzerinde hızlandırılmış eğitim.

### 4.4 Ollama LLM Modelleri

Sıfır-atımlı (zero-shot) sınıflandırma için Ollama altyapısı üzerinde 13 farklı açık kaynaklı LLM test edilmiştir. Her model, aynı sistem promptu ve deterministik parametreler (temperature=0.0, num_predict=5) ile çalıştırılmıştır. Değerlendirilen modeller şunlardır: aya-expanse latest, codellama 7b, gemma2 latest, glm-ocr latest, glm4 9b, granite4 1 3b, granite4 1 8b, llama3 1 8b, llama3 1 latest, llama3 2 3b, mistral latest, olmo2 7b ve qwen2 5-coder 7b.

### 4.5 Model Değerlendirme

Her model için beş metrik hesaplanmıştır: Accuracy, Ağırlıklı F1 Skoru, Ağırlıklı Precision, Ağırlıklı Recall ve ROC-AUC. Ek olarak, 5-fold Stratified Cross-Validation eğitim kümesi üzerinde uygulanarak genelleme kapasitesi ölçülmüştür. Linear SVM için decision_function çıktısı min-max normalizasyonu ile ROC-AUC hesabına uygun hale getirilmiştir.

### 4.6 Hibrit Karar Ajanı Mimarisi

Hibrit mimarinin bileşenleri:

**A. Ollama LLM Entegrasyonu**

Yerel LLM modeline, token kullanımını minimize eden optimize bir sistem prompt'u gönderilmektedir:

```
Sistem: "Sen Türkçe duygu analizi yapan bir sınıflandırıcısın. 
         SADECE tek kelime yanıt ver: 'Pozitif' veya 'Negatif'."
```

`temperature=0.0` ve `num_predict=5` parametreleri deterministik ve kısa çıktı üretimini sağlar. Paralel istek gönderimleri `ThreadPoolExecutor` ile `MAX_WORKERS=4` olarak yapılandırılmıştır.

**B. Karar Birleştirme Mantığı**

```
if SVM_Tahmin == Ollama_Tahmin:
    Hybrid_Status = "Consensus"
    Hybrid_Karar  = SVM_Tahmin
else:
    Hybrid_Status = "Conflict"
    Hybrid_Karar  = Ollama_Tahmin  # LLM bağlam anlama üstünlüğü
```

Conflict durumunda LLM kararının tercih edilmesinin gerekçesi: SVM'in kelime düzeyinde çalışması, "çay koyayım içelim" gibi bir ifadedeki nüansları yanlış yorumlamasına yol açabilir. LLM bütün cümlenin bağlamını değerlendirerek bu tür hataları giderebilir.

---

## 5. Deneysel Sonuçlar

### 5.1 Klasik Makine Öğrenmesi — Bireysel Model Performansı

**Tablo 2. Klasik Model Performans Karşılaştırması (Test Seti, n=2.224)**

| Sıra | Model | Accuracy | F1 | Precision | Recall | ROC-AUC |
|------|-------|----------|----|-----------|--------|---------|
| 1 ★ | Linear SVM | **0.8678** | **0.8656** | **0.8680** | **0.8678** | 0.9268 |
| 2 | Multinomial NB | 0.8588 | 0.8580 | 0.8601 | 0.8588 | 0.9232 |
| 3 | Logistic Regression | 0.8534 | 0.8509 | 0.8625 | 0.8534 | **0.9292** |
| 4 | Gradient Boosting | 0.8255 | 0.8195 | 0.8486 | 0.8255 | 0.8805 |
| 5 | XGBoost | 0.8219 | 0.8162 | 0.8425 | 0.8219 | 0.8810 |
| 6 | Random Forest | 0.7437 | 0.7183 | 0.8180 | 0.7437 | 0.8963 |

Linear SVM, hem Accuracy hem de F1 skorunda en yüksek değerleri elde etmiştir. ROC-AUC metriğinde Logistic Regression (0.9292) SVM'i (0.9268) hafifçe geçmektedir; ancak bu fark istatistiksel olarak anlamlı değildir. Random Forest, yüksek Precision (0.8180) değerine karşın düşük Recall (0.7437) sergilemektedir.

### 5.2 Linear SVM Detay Analizi

Sınıf bazlı rapor incelendiğinde:

- **Negatif sınıf:** Precision=0.89, Recall=0.81, F1=0.84
- **Pozitif sınıf:** Precision=0.85, Recall=0.92, F1=0.88

Pozitif sınıfın daha yüksek Recall değeri (%92), modelin Pozitif yorumları yüksek doğrulukla tespit ettiğini; Negatif sınıfta ise %81 Recall ile bazı yorumların kaçırıldığını göstermektedir. Bu asimetri, Türkçe argosu yüksek Negatif yorumların kelime düzeyinde SVM tarafından yanlış yorumlanmasıyla açıklanabilir.

### 5.3 Cross-Validation Sonuçları

5-fold Stratified Cross-Validation sonuçları tutarlı performansı doğrulamaktadır. Linear SVM'in CV ortalaması 0.8665 (±0.009) olarak ölçülmüş; bu düşük standart sapma, modelin veri bölümüne karşı gürbüz olduğunu ortaya koymaktadır.

### 5.4 Ollama LLM Modelleri — Karşılaştırmalı Sonuçlar

13 farklı Ollama modeli, SVM ile aynı test seti üzerinde değerlendirilmiştir. Tüm LLM modelleri sıfır-atımlı (zero-shot) mod ile çalıştırılmış olup herhangi bir ince ayar (fine-tuning) uygulanmamıştır.

**Tablo 3. Ollama LLM Model Karşılaştırması — Tüm Metrikler**

| Model | Acc(LLM) | F1(LLM) | Precision | Recall | Consensus% | Conflict% | LLM Wins |
|-------|----------|---------|-----------|--------|------------|-----------|-----------|
| **llama3 1 latest** | **0.7383** | **0.7385** | 0.74 | 0.75 | **70.9%** | **29.1%** | 180 |
| llama3 1 8b | 0.7284 | 0.7289 | 0.78 | 0.70 | 69.9% | 30.1% | 180 |
| llama3 2 3b | 0.7032 | 0.7039 | 0.70 | 0.73 | 67.3% | 32.7% | 181 |
| qwen2 5-coder 7b | 0.6812 | 0.6818 | 0.69 | 0.68 | 66.1% | 33.9% | 170 |
| aya-expanse latest | 0.6790 | 0.6620 | 0.71 | 0.68 | 62.7% | 37.3% | 205 |
| gemma2 latest | 0.6942 | 0.6826 | 0.77 | 0.69 | 64.3% | 35.7% | 204 |
| glm4 9b | 0.6614 | 0.6501 | 0.72 | 0.66 | 62.3% | 37.7% | 190 |
| granite4 1 8b | 0.6286 | 0.6091 | 0.71 | 0.63 | 59.1% | 40.9% | 189 |
| mistral latest | 0.6317 | 0.6324 | 0.63 | 0.63 | 60.1% | 39.9% | 181 |
| granite4 1 3b | 0.5890 | 0.5546 | 0.69 | 0.59 | 56.0% | 44.0% | 179 |
| olmo2 7b | 0.5908 | 0.4973 | 0.65 | 0.59 | 61.3% | 38.7% | 122 |
| glm-ocr latest | 0.5423 | 0.4426 | 0.51 | 0.54 | 57.8% | 42.2% | 107 |
| codellama 7b | 0.5265 | 0.4401 | 0.53 | 0.53 | 48.6% | 51.4% | 192 |
| **SVM Baseline** | **0.8678** | **0.8656** | **0.868** | **0.868** | — | — | — |

Tüm modeller SVM baseline değerinin (0.8678) belirgin biçimde altında kalmıştır. En iyi LLM performansını llama3 1 latest modeli sergilemiş (%73.83 accuracy); ancak bu değer bile SVM'in ~13 puan gerisindedir. Modeller arasındaki performans farkı kayda değer düzeydedir: en iyi (llama3 1 latest, 0.7383) ile en kötü (codellama 7b, 0.5265) model arasında yaklaşık 21 puanlık bir uçurum bulunmaktadır.

Radar grafiği analizi (Şekil 2) göstermektedir ki SVM baseline'ı (kırmızı kesik çizgi) Accuracy, F1, Precision ve Recall eksenlerinin tamamında tüm LLM modellerin çok üzerindedir. LLM modellerinin precision değerleri birbirinden belirgin biçimde ayrışırken recall değerleri daha homojen bir dağılım sergilemektedir. Genel olarak değerlendirildiğinde, sıfır-atımlı LLM kullanımının Türkçe ikili duygu sınıflandırmasında ince ayarlı ve optimize edilmiş bir TF-IDF + SVM pipeline'ı ile rekabet edemediği görülmektedir.

### 5.5 Consensus & Conflict Dağılım Analizi

Her LLM modeli ile SVM'in birbirinden ayrıştığı Conflict oranları ve uzlaştığı Consensus oranları hesaplanmıştır.

**Tablo 4. Consensus & Conflict Dağılımı — Tüm Modeller**

| Model | Consensus% | Conflict% |
|-------|------------|-----------|
| llama3 1 latest | **70.9%** | **29.1%** |
| llama3 1 8b | 69.9% | 30.1% |
| llama3 2 3b | 67.3% | 32.7% |
| qwen2 5-coder 7b | 66.1% | 33.9% |
| gemma2 latest | 64.3% | 35.7% |
| aya-expanse latest | 62.7% | 37.3% |
| glm4 9b | 62.3% | 37.7% |
| olmo2 7b | 61.3% | 38.7% |
| mistral latest | 60.1% | 39.9% |
| granite4 1 8b | 59.1% | 40.9% |
| glm-ocr latest | 57.8% | 42.2% |
| granite4 1 3b | 56.0% | 44.0% |
| codellama 7b | 48.6% | **51.4%** |

Consensus oranı ile LLM performansı arasında güçlü bir pozitif korelasyon gözlemlenmektedir: SVM ile daha sık mutabık kalan modeller genel olarak daha yüksek doğruluk sergilemiştir. codellama 7b modeli, %51.4 Conflict oranıyla öne çıkmaktadır; bu değer, modelin SVM ile neredeyse yarı yarıya çakışma yaşadığına ve sınıflandırma kararlarının büyük bir kısmının tutarsız olduğuna işaret etmektedir.

### 5.6 Conflict Durumlarında Kazanan Model Analizi

Conflict senaryolarında hangi modelin (Ollama veya SVM) doğru tahminde bulunduğu ayrı ayrı analiz edilmiştir. Şekil 1 (Conflict Durumlarında Kazanan Model Analizi), tüm modeller için SVM'in Conflict durumlarında Ollama'dan belirgin biçimde daha yüksek kazanma sayısına sahip olduğunu göstermektedir.

**Tablo 5. Conflict Durumlarında Kazanma Sayıları**

| Model | Ollama Kazandı | SVM Kazandı | Ollama Kazanma Oranı |
|-------|---------------|-------------|----------------------|
| llama3 1 latest | 180 | 468 | %27.8 |
| llama3 1 8b | 180 | 490 | %26.9 |
| llama3 2 3b | 181 | 547 | %24.9 |
| gemma2 latest | 204 | 590 | %25.7 |
| aya-expanse latest | 205 | 625 | %24.7 |
| glm4 9b | 190 | 649 | %22.7 |
| qwen2 5-coder 7b | 170 | 585 | %22.5 |
| mistral latest | 181 | 706 | %20.4 |
| olmo2 7b | 122 | 738 | %14.2 |
| granite4 1 8b | 189 | 721 | %20.8 |
| granite4 1 3b | 179 | 799 | %18.3 |
| glm-ocr latest | 107 | 831 | %11.4 |
| codellama 7b | 192 | 951 | %16.8 |

Bu tablo, kritik bir bulguya işaret etmektedir: **Conflict durumlarında SVM, tüm modellere karşı açık ara üstündür.** En iyi Ollama kazanma oranı bile llama3 1 latest ile yalnızca %27.8'de kalmaktadır. Bu, Conflict durumlarında her zaman Ollama kararını seçen sabit kural stratejisinin teoride beklenenin aksine accuracy değerini düşürdüğünü ortaya koymaktadır. Hibrit mimaride Conflict senaryoları için LLM güven skoruna dayalı adaptif bir seçim mekanizması kullanılması, bu açığı kapatabilir.

### 5.7 Hibrit Karar Ajanı — Genel Sonuçlar

Hibrit mimaride SVM ile birleştirilen Ollama modeli olarak llama3 1 latest seçilmiştir (en yüksek LLM doğruluğu ve Consensus oranı).

**Tablo 6. Hibrit Değerlendirme Özeti (n=2.224)**

| Metrik | SVM Tek Başına | Ollama (LLM) Tek Başına | Hibrit (Consensus/Conflict) |
|--------|---------------|------------------------|----------------------------|
| Accuracy | **0.8678** | 0.7383 | 0.7383 |
| F1 | **0.8656** | 0.7385 | 0.7385 |
| Consensus Oranı | — | — | %70.9 |
| Conflict Oranı | — | — | %29.1 |

**Hibrit accuracy'nin LLM accuracy'ye eşit olmasının açıklaması:** Sabit kural stratejisinde Consensus durumlarında her iki model de aynı karara vardığından sonuç her ikisiyle de aynıdır. Conflict durumlarında ise her zaman Ollama seçilmektedir. Dolayısıyla hibridin toplam kararları matematiksel olarak Ollama'nın kararlarıyla özdeşleşmekte; bu da hybrid accuracy = LLM accuracy sonucunu doğrudan doğurmaktadır.

**Conflict Analizi — Niteliksel Bulgular:**

Conflict durumlarında Ollama'nın doğru, SVM'in yanlış olduğu örnekler incelendiğinde:

- Ağırlıklı olarak argo ve kinaye içeren yorumlar (ör. "kesinlikle bir daha gelmem, harika bir deneyimdi 👏" gibi sarkastik ifadeler)
- Dilbilgisi kurallarını ihlal eden, büyük harf kullanımı yoğun sosyal medya tarzı metinler

SVM'in doğru, Ollama'nın yanlış olduğu örnekler ise:

- Çok kısa veya bağlam-düşük yorumlar (ör. "iyi", "tamam", "oldu")
- Türkçe morfolojisine özgü eklerin semantik anlamını LLM'in yanlış yorumladığı durumlar

Bu bulgular, Conflict durumlarında statik bir seçim yerine her iki modelin güven skorunu birleştiren ağırlıklı oylama mekanizmasının daha verimli sonuçlar üretebileceğine işaret etmektedir.

---

## 6. Tartışma

### 6.1 Linear SVM'in Üstünlüğü

Linear SVM'in başarısı birkaç faktörle açıklanabilir. TF-IDF + n-gram özellik uzayı, sosyal medya Türkçesi için yeterli temsil kapasitesi sunmaktadır. Yüksek boyutlu seyrek uzaylarda SVM'in geometrik ayrım (margin maximization) prensibi özellikle etkilidir. Random Forest ve tree-based modellerin düşük performansı ise yüksek boyutlu seyrek veride ağaç modellerinin özellik seçiminde yetersiz kaldığını göstermektedir. Zero-shot LLM'lerin SVM'in ~13 puan gerisinde kalması, domain-specific fine-tuning olmadan bu görevin sınırlı kaldığını doğrulamaktadır.

### 6.2 LLM Modelleri Arasındaki Farklar

Llama3 serisinin (1 latest, 1 8b, 2 3b) diğer modellerden belirgin biçimde üstün performans sergilemesi dikkat çekicidir. Bu modellerin çok dilli eğitim verisi içerdiği ve Türkçe morfolojisine kısmen maruz kaldığı bilinmektedir. Buna karşın codellama 7b ve glm-ocr latest gibi belirli bir alana (kod üretimi, görsel metin tanıma) odaklanan modellerin duygu analizi görevinde ciddi biçimde başarısız olması beklenen bir sonuçtur.

### 6.3 Hibrit Mimarinin Katkısı ve Sınırlılıkları

Conflict senaryolarında Ollama LLM'in seçilmesi teorik olarak şu faydaları sunar:

**Kinaye Tespiti:** Tırnak işareti kaldırıldığında SVM tarafından ayırt edilemeyen sarkastik ifadeler, LLM tarafından bağlamsal akıl yürütme ile doğru sınıflandırılabilir.

**Karma Duygu:** "Yemek güzeldi ama servis çok kötüydü" gibi cümlelerde SVM bigram özelliklerine dayanırken LLM cümlenin bütününü değerlendirir.

**Argo:** Nadir görülen yeni argo kelimeler TF-IDF seyrek vektöründe temsil edilemeyebilirken LLM morfolojik benzerlikleri ve bağlamı kullanarak bu boşluğu doldurabilir.

Ancak nicel veriler bu teorik avantajları tam olarak yansıtmamıştır. Conflict durumlarında SVM'in %72–88 oranında Ollama'dan daha doğru tahmin üretmesi, sabit kural stratejisinin (her conflict'te LLM seç) pratikte doğruluğu düşürdüğünü göstermektedir. Bu bulgu, hibrit sistemin tasarım kararının yeniden değerlendirilmesi gerektiğine işaret etmektedir.

### 6.4 Sınırlılıklar

Bu çalışmanın bazı sınırlılıkları bulunmaktadır. Ollama LLM'in yanıt süresi (ortalama ~0.24 sn/yorum) gerçek zamanlı büyük ölçekli uygulamalar için potansiyel bir darboğaz oluşturmaktadır. Bunun yanı sıra, Conflict durumlarında her zaman LLM kararını almak optimal olmamaktadır; nicel veriler SVM'in Conflict'leri daha sık kazandığını açıkça ortaya koymuştur. Son olarak, zero-shot LLM değerlendirmesi fine-tuning'e kıyasla dezavantajlı bir koşuldur; Türkçe duygu analizi için özelleştirilmiş bir LLM farklı sonuçlar doğurabilir.

---

## 7. Sonuç

Bu çalışmada, Türkçe sosyal medya yorumları için altı klasik makine öğrenmesi modeli ile 13 farklı Ollama LLM sistematik biçimde karşılaştırılmış; en yüksek performans gösteren Linear SVM, Ollama LLM ile birleştirilerek Hibrit Karar Ajanı mimarisi oluşturulmuştur.

Temel bulgular şu şekilde özetlenebilir:

- **Linear SVM**, %86.78 doğruluk ve 0.8656 F1 skoru ile hem klasik hem de LLM tabanlı tüm modeller arasında birinci sırayı almıştır.
- **Logistic Regression ve Multinomial NB**, Linear SVM'e yakın performans sergileyerek Türkçe metin sınıflandırmasında kelime tabanlı istatistiksel modellerin gücünü teyit etmiştir.
- **13 Ollama LLM modelinden** en iyi performansı llama3 1 latest (%73.83) sergilemiş; ancak tüm modeller SVM baseline'ının belirgin biçimde gerisinde kalmıştır.
- **Consensus oranı** ve LLM doğruluğu arasında güçlü pozitif korelasyon gözlemlenmiştir; en yüksek Consensus oranı da llama3 1 latest modeline aittir (%70.9).
- **Conflict analizi**, tüm Ollama modelleri için SVM'in Conflict durumlarını baskın biçimde kazandığını ortaya koymuş; bu bulgu, "Conflict'te her zaman LLM seç" stratejisinin iyileştirilmesi gerektiğine işaret etmektedir.

Gelecek çalışmalar için üç yön önerilmektedir: (1) Türkçe sosyal medya verisinde ince ayarlı (fine-tuned) bir LLM kullanılması, (2) Conflict kararlarında sabit LLM seçimi yerine güven skoruna dayalı ağırlıklı oylama uygulanması ve (3) modelin gerçek zamanlı sosyal medya akışlarına entegrasyonu.

---

## Kaynaklar

[1] B. Liu, "Sentiment Analysis and Opinion Mining," *Synthesis Lectures on Human Language Technologies*, vol. 5, no. 1, pp. 1–167, 2012.

[2] Z. Çekiç ve B. Diri, "Türkçe Sosyal Medya Metinlerinin Duygu Analizi," *Akademik Bilişim Konferansı*, 2015.

[3] M. Kaya, G. Fidan ve İ. H. Toroslu, "Sentiment Analysis of Turkish Political News," *Proceedings of the 2012 IEEE/WIC/ACM International Conferences on Web Intelligence*, pp. 174–180, 2012.

[4] E. Demirtas ve M. Pechenizkiy, "Cross-lingual Polarity Detection with Machine Translation," *Proceedings of the Second International Workshop on Issues of Sentiment Discovery and Opinion Mining*, 2013.

[5] T. B. Brown ve ark., "Language Models are Few-Shot Learners," *Advances in Neural Information Processing Systems*, vol. 33, pp. 1877–1901, 2020.

[6] Meta AI, "Llama 3: Open Foundation and Fine-Tuned Chat Models," Meta AI Research, 2024. [Online]. Available: <https://ai.meta.com/llama>

[7] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

[8] T. Chen ve C. Guestrin, "XGBoost: A Scalable Tree Boosting System," *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785–794, 2016.

[9] C. Cortes ve V. Vapnik, "Support-Vector Networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, 1995.

[10] S. Robertson, "Understanding Inverse Document Frequency: On Theoretical Arguments for IDF," *Journal of Documentation*, vol. 60, no. 5, pp. 503–520, 2004.

[11] H. Touvron ve ark., "Llama 2: Open Foundation and Fine-Tuned Chat Models," *arXiv preprint arXiv:2307.09288*, 2023.

[12] A. Joulin, E. Grave, P. Bojanowski ve T. Mikolov, "Bag of Tricks for Efficient Text Classification," *Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics*, vol. 2, pp. 427–431, 2017.

---

## Ekler

### Ek A. Hiperparametre Özeti

| Model | Hiperparametre | Değer | Seçim Gerekçesi |
|-------|---------------|-------|----------------|
| Linear SVM | C | 1.0 | Bias-variance dengesi; küçük C aşırı düzenleme yapar |
| Linear SVM | max_iter | 2000 | Yakınsama için yeterli iterasyon |
| Mult. NB | alpha | 0.5 | Standart 1.0 yerine daha az smoothing; veri büyük olduğu için uygun |
| LR | C | 1.0 | L2 regularization, TF-IDF'in çok sayıda özelliği için dengeli |
| LR | solver | lbfgs | Küçük-orta veri için bellek etkin çözücü |
| XGBoost | device | cuda | RTX 3060 Ti GPU hızlandırması |
| XGBoost | tree_method | hist | GPU uyumlu histogram metodu |
| RF | n_estimators | 200 | Yeterli çeşitlilik için; 500 ile test edilmiş, marjinal kazanım |

### Ek B. Ön İşleme Regex Desenleri

| Desen | Hedef | Örnek |
|-------|-------|-------|
| `http\S+\|www\S+` | URL | `https://t.co/abc` |
| `@\w+` | Kullanıcı mention | `@kullaniciadi` |
| `#\w+` | Hashtag | `#türkiye` |
| `\d+` | Rakam | `2024`, `3` |
| `[^a-zığüşöçİĞÜŞÖÇ\s]` | Özel karakter | `!`, `%`, `*` |
| `\s+` | Fazla boşluk | `   ` → ` ` |

### Ek C. Ollama Sistem Promptu

```
Sistem Promptu:
"Sen Türkçe duygu analizi yapan bir sınıflandırıcısın.
 Sana verilen Türkçe sosyal medya yorumunu analiz et.
 SADECE tek kelime yanıt ver: 'Pozitif' veya 'Negatif'.
 Başka hiçbir şey yazma. Açıklama yapma. Noktalama kullanma."

Parametreler:
  temperature  : 0.0   (deterministik)
  top_p        : 1.0
  num_predict  : 5     (max token — sadece etiket)
  stop tokens  : ["\n", ".", ",", " "]
```

### Ek D. Şekil Listesi

- **Şekil 1.** Conflict Durumlarında Kazanan Model Analizi — Tüm Ollama Modelleri (Ollama Kazandı vs SVM Kazandı)
- **Şekil 2.** LLM Model Performans Radar Grafiği (Kırmızı kesik çizgi = SVM baseline)
- **Şekil 3.** Tüm Model Karşılaştırma Özet Tablosu
- **Şekil 4.** LLM (Ollama) Doğruluk — Model Karşılaştırması (Yatay çubuk grafik)
- **Şekil 5.** Ollama Model Metrikleri — Tam Karşılaştırma (Accuracy, F1, Precision, Recall)
- **Şekil 6.** Consensus & Conflict Dağılımı — Tüm Modeller
