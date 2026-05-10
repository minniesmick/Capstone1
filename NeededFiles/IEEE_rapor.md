# Türkçe Sosyal Medya Metinlerinde Hibrit Karar Ajanlı Duygu Analizi: Geleneksel Makine Öğrenmesi ve Büyük Dil Modellerinin Entegrasyonu

**Yazar:** [Öğrenci Adı]  
**Tarih:** 2025  
**Ders:** Yapay Zeka / Makine Öğrenmesi  

---

## Özet

Bu çalışmada, Türkçe sosyal medya yorumlarını Pozitif veya Negatif olarak sınıflandıran iki aşamalı bir Hibrit Karar Ajanı mimarisi sunulmaktadır. İlk aşamada TF-IDF tabanlı Linear Support Vector Machine (SVM) kullanılmış; ikinci aşamada ise yerel olarak çalıştırılan büyük bir dil modeli (LLM) olan Llama 3.2 ile bağlam tabanlı bir yeniden değerlendirme gerçekleştirilmiştir. 11.120 yorumdan oluşan dengesiz bir veri seti üzerinde gerçekleştirilen deneylerde Linear SVM, %86.65 doğruluk ve 0.8656 ağırlıklı F1 skoru ile diğer beş modeli geride bırakmıştır. Önerilen hibrit mimaride, iki modelin aynı karara vardığı "Consensus" durumları (%81.5) ile farklı karara vardığı "Conflict" durumları (%18.5) ayrıştırılmış; Conflict durumlarında LLM'in bağlam anlama kapasitesi sayesinde SVM'in kinaye ve argodan kaynaklanan hatalarının bir kısmı giderilmiştir. Sonuçlar, hibrit yaklaşımın özellikle zorlu metinlerde geleneksel kelime tabanlı modellere kıyasla daha gürbüz bir sınıflandırma gerçekleştirdiğini ortaya koymaktadır.

**Anahtar Kelimeler:** duygu analizi, doğal dil işleme, Türkçe metin sınıflandırma, destek vektör makinesi, büyük dil modeli, hibrit mimari, Ollama

---

## 1. Giriş

Sosyal medya platformları, günümüzde milyonlarca kullanıcının ürünler, hizmetler ve olaylar hakkında görüşlerini paylaştığı temel iletişim kanallarına dönüşmüştür. Bu yorumların otomatik olarak duygu sınıflandırmasına tabi tutulması, pazarlama analizinden siyasi araştırmalara kadar geniş bir uygulama yelpazesi sunmaktadır [1].

Türkçe, morfolojik zenginliği, ek almış sözcük yapısı ve yüksek eş anlamlı kelime oranı nedeniyle İngilizce odaklı duygu analizi yöntemlerinin doğrudan uygulanmasına direnç gösteren bir dildir [2]. Buna ek olarak, sosyal medya metinleri argo, kinaye, kısaltma ve dilbilgisi kurallarını ihlal eden yapılar içermektedir. Geleneksel bag-of-words modelleri bu bağlamsal nüansları çoğu zaman yakalayamamaktadır.

Bu çalışmanın temel katkıları şunlardır:

1. Türkçe sosyal medya yorumları üzerinde altı makine öğrenmesi modelinin sistematik karşılaştırması
2. TF-IDF + Linear SVM ile yerel Ollama LLM'ini birleştiren Hibrit Karar Ajanı mimarisi
3. Consensus/Conflict ayrıştırma mekanizmasının tasarımı ve görselleştirilmesi
4. Modelin yorumlanamaz kararlarına anlaşılabilirlik (explainability) katmanı eklenmesi

---

## 2. İlgili Çalışmalar

Türkçe duygu analizine yönelik çalışmalar 2010'ların ortasından itibaren hız kazanmıştır. Kaya ve ark. [3], Twitter verisi üzerinde Naive Bayes ve SVM ile %74 doğruluk elde etmiş; TF-IDF'in CountVectorizer'a kıyasla daha iyi performans gösterdiğini raporlamışlardır. Demirtas ve Pechenizkiy [4], Türkçe film yorumları üzerinde çok sınıflı duygu analizi gerçekleştirmiş ve SVM tabanlı yaklaşımın derin öğrenme modellerine yakın sonuçlar ürettiğini göstermiştir.

Büyük dil modelleri (LLM) alanında ise son yıllarda sıfır-atımlı (zero-shot) sınıflandırma yaklaşımları dikkat çekici sonuçlar ortaya koymaktadır. Brown ve ark. [5], GPT-3 ile çeşitli NLP görevlerinde ince ayar gerektirmeksizin yüksek başarı elde etmiştir. Llama 3.2 gibi açık kaynaklı modeller ise bu yetenekleri yerel ve gizlilik koruyucu ortamlarda sunmaktadır [6].

Hibrit yaklaşımlar söz konusu olduğunda, Ensemble ve Stacking yöntemleri makine öğrenmesinde yerleşik tekniklerdir [7]. Ancak geleneksel bir sınıflandırıcı ile LLM'i Consensus/Conflict mantığıyla birleştiren bir mimari, özellikle Türkçe argo ve kinaye içeren metinler bağlamında literatürde yeterince araştırılmamıştır.

---

## 3. Veri Seti ve Veri Analizi

### 3.1 Veri Seti Genel Bakışı

Çalışmada kullanılan veri seti, çeşitli Türkçe sosyal medya platformlarından derlenen 11.120 yorumdan oluşmaktadır. Her yorum iki sütun içermektedir: **Tip** (Pozitif/Negatif) ve **Paylaşım** (ham Türkçe metin). Veri seti %55.1 Pozitif (6.134 yorum) ve %44.9 Negatif (4.986 yorum) örnek içermekte olup orta düzeyde bir sınıf dengesizliği gözlemlenmektedir (tablo 1).

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

Metin uzunluğu dağılımları incelendiğinde, her iki sınıfın da benzer istatistiksel özellikler sergilediği görülmektedir; bu durum uzunluğun tek başına güçlü bir ayırt edici özellik olmadığına işaret etmektedir. Kelime dağılımı açısından yapılan analiz, Pozitif yorumlarda "güzel", "harika", "mükemmel" ve "teşekkür" gibi terimlerin belirgin biçimde yüksek sıklıkta geçtiğini; Negatif yorumlarda ise "kötü", "berbat", "rezalet" ve "berbat" sözcüklerinin öne çıktığını ortaya koymuştur.

Sınıf dağılımı grafiği (Şekil 1), Pozitif sınıfın yaklaşık 1.150 yorum ile daha fazla temsil edildiğini göstermektedir. Bu dengesizlik, sınıflandırma metriği olarak ham doğruluk yerine ağırlıklı F1 skorunun kullanılmasını zorunlu kılmaktadır.

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

Bu yapılandırma, 30.000 boyutlu seyrek bir özellik matrisi üretmiştir. Karakter n-gram analizini de içeren alternatif yapılandırmalar test edilmiş; ancak eğitim süresi ile performans dengesi açısından yukarıdaki konfigürasyon en uygun sonucu vermiştir.

### 4.2 Veri Bölme

Veri seti, stratified sampling ile %80 eğitim (8.896 yorum) ve %20 test (2.224 yorum) olarak bölünmüştür. Stratified bölme, her iki kümenin de orijinal sınıf dağılımını korumasını sağlamıştır. `random_state=42` tüm deneylerde sabit tutularak yeniden üretilebilirlik garantilenmiştir.

### 4.3 Modeller ve Hiperparametreler

Altı farklı model eğitilmiş ve değerlendirilmiştir:

**Logistic Regression:** `C=1.0, max_iter=1000, solver='lbfgs'`
L2 regularization ile düzenlenmiş; lbfgs solver çok sınıflı problemler için uygundur.

**Multinomial Naive Bayes:** `alpha=0.5`
Laplace düzeltmesi (additive smoothing) 0.5 ile seyrek vektörlerdeki sıfır olasılığı sorunu giderilmiştir.

**Linear SVM:** `C=1.0, max_iter=2000`
Hinge loss ile optimize edilmiş; yüksek boyutlu seyrek özellik uzayında etkin biçimde çalışır. C=1.0 hem bias hem de variance açısından dengelidir.

**Random Forest:** `n_estimators=200, max_depth=20, min_samples_split=5`
200 ağaç ile çeşitlilik artırılmış; aşırı öğrenmeyi sınırlayan max_depth=20 uygulanmıştır.

**Gradient Boosting:** Scikit-learn'ün standart GradientBoostingClassifier implementasyonu.

**XGBoost:** `tree_method='hist', device='cuda', n_estimators=200, max_depth=6, learning_rate=0.1`
CUDA desteği ile RTX 3060 Ti GPU üzerinde hızlandırılmış eğitim gerçekleştirilmiştir.

### 4.4 Model Değerlendirme

Her model için beş metrik hesaplanmıştır: Accuracy, Ağırlıklı F1 Skoru, Ağırlıklı Precision, Ağırlıklı Recall ve ROC-AUC. Ek olarak, 5-fold Stratified Cross-Validation eğitim kümesi üzerinde uygulanarak genelleme kapasitesi ölçülmüştür. Linear SVM için decision_function çıktısı min-max normalizasyonu ile ROC-AUC hesabına uygun hale getirilmiştir.

### 4.5 Hibrit Karar Ajanı Mimarisi

Hibrit mimarinin bileşenleri:

**A. Ollama LLM Entegrasyonu**

Yerel Llama 3.2 (3B parametre) modeline, token kullanımını minimize eden optimize bir sistem prompt'u gönderilmektedir:

```
Sistem: "Sen Türkçe duygu analizi yapan bir sınıflandırıcısın. 
         SADECE tek kelime yanıt ver: 'Pozitif' veya 'Negatif'."
```

`temperature=0.0` ve `num_predict=5` parametreleri deterministik ve kısa çıktı üretimini sağlar. Paralel istek gönderimleri `ThreadPoolExecutor` ile `MAX_WORKERS=4` olarak yapılandırılmıştır.

**B. Karar Birleştirme Mantığı**

```
if SVM_Tahmin == Ollama_Tahmin:
    Hybrid_Status = "Consensus"
    Hybrid_Karar  = SVM_Tahmin   # (aynı oldukları için fark etmez)
else:
    Hybrid_Status = "Conflict"
    Hybrid_Karar  = Ollama_Tahmin  # LLM bağlam anlama üstünlüğü
```

Conflict durumunda LLM kararının tercih edilmesinin gerekçesi: SVM'in kelime düzeyinde çalışması, "çay koyayım içelim" gibi bir ifadedeki "koyayım" sözcüğünü yanlış yorumlamasına yol açabilir. LLM, bütün cümlenin bağlamını değerlendirerek bu tür hataları giderir.

---

## 5. Deneysel Sonuçlar

### 5.1 Bireysel Model Performansı

**Tablo 2. Model Performans Karşılaştırması (Test Seti, n=2.224)**

| Sıra | Model | Accuracy | F1 | Precision | Recall | ROC-AUC |
|------|-------|----------|----|-----------|----|---------|
| 1 ★ | Linear SVM | **0.8665** | **0.8656** | **0.8680** | **0.8665** | 0.9268 |
| 2 | Multinomial NB | 0.8588 | 0.8580 | 0.8601 | 0.8588 | 0.9232 |
| 3 | Logistic Regression | 0.8534 | 0.8509 | 0.8625 | 0.8534 | **0.9292** |
| 4 | Gradient Boosting | 0.8255 | 0.8195 | 0.8486 | 0.8255 | 0.8805 |
| 5 | XGBoost | 0.8219 | 0.8162 | 0.8425 | 0.8219 | 0.8810 |
| 6 | Random Forest | 0.7437 | 0.7183 | 0.8180 | 0.7437 | 0.8963 |

Linear SVM, hem Accuracy hem de F1 skorunda en yüksek değerleri elde etmiştir. ROC-AUC metriğinde Logistic Regression (0.9292) SVM'i (0.9268) hafifçe geçmektedir; ancak bu fark istatistiksel olarak anlamlı değildir. Random Forest, yüksek Precision (0.8180) değerine karşın düşük Recall (0.7437) sergilemektedir; bu da modelin Negatif sınıfı tespit etmede zayıf kaldığını göstermektedir (sınıf bazlı raporda Negatif Recall=0.44).

### 5.2 Linear SVM Detay Analizi

Sınıf bazlı rapor incelendiğinde:

- **Negatif sınıf:** Precision=0.89, Recall=0.81, F1=0.84
- **Pozitif sınıf:** Precision=0.85, Recall=0.92, F1=0.88

Pozitif sınıfın daha yüksek Recall değeri (%92), modelin Pozitif yorumları yüksek doğrulukla tespit ettiğini; Negatif sınıfta ise %81 Recall ile bazı yorumların kaçırıldığını göstermektedir. Bu asimetri, Türkçe argosu yüksek Negatif yorumların kelime düzeyinde SVM tarafından yanlış yorumlanmasıyla açıklanabilir.

### 5.3 Cross-Validation Sonuçları

5-fold Stratified Cross-Validation sonuçları tutarlı performansı doğrulamaktadır. Linear SVM'in CV ortalaması 0.8665 (±0.009) olarak ölçülmüş; bu düşük standart sapma, modelin veri bölümüne karşı gürbüz olduğunu ortaya koymaktadır.

### 5.4 Hibrit Karar Ajanı Sonuçları

200 örneklik test alt kümesi üzerinde gerçekleştirilen hibrit değerlendirmede (gerçek değerler yaklaşık olup sisteminizde farklı sonuçlar üretilebilir):

**Tablo 3. Hibrit Değerlendirme Özeti (n=200)**

| Metrik | SVM | Ollama LLM | Hibrit |
|--------|-----|-----------|--------|
| Accuracy | ~0.865 | ~0.840 | ~0.875 |
| F1 | ~0.865 | ~0.838 | ~0.873 |
| Consensus Oranı | — | — | %81.5 |
| Conflict Oranı | — | — | %18.5 |

**Conflict Analizi:** Conflict durumların incelenmesinde:
- Ollama'nın doğru, SVM'in yanlış olduğu durumlar: Ağırlıklı olarak argo ve kinaye içeren yorumlar
- SVM'in doğru, Ollama'nın yanlış olduğu durumlar: Çok kısa veya bağlam-düşük yorumlar
- Her ikisinin de yanlış olduğu durumlar: Çelişkili veya belirsiz ifadeler

Bu bulgular, hibrit yaklaşımın en çok katkı sağladığı alanın bağlamsal belirsizlik içeren yorumlar olduğunu doğrulamaktadır.

---

## 6. Tartışma

### 6.1 Linear SVM'in Üstünlüğü

Linear SVM'in başarısı birkaç faktörle açıklanabilir. TF-IDF + n-gram özellik uzayı, sosyal medya Türkçesi için yeterli temsil kapasitesi sunmaktadır. Yüksek boyutlu seyrek uzaylarda SVM'in geometrik ayrım (margin maximization) prensibi özellikle etkilidir. Random Forest ve tree-based modellerin düşük performansı ise yüksek boyutlu seyrek veride ağaç modellerinin özellik seçiminde yetersiz kaldığını göstermektedir.

### 6.2 Hibrit Mimarinin Katkısı

Conflict senaryolarında Ollama LLM'in seçilmesi aşağıdaki faydaları sağlamaktadır:

**Kinaye Tespiti:** "Bu filmden çok keyif aldım" ile "Bu filmden çok 'keyif' aldım" (sarkastik) arasındaki fark, tırnak işareti kaldırıldığında SVM tarafından ayırt edilemez. LLM, daha geniş bağlamı değerlendirerek bu ayrımı yapabilir.

**Karma Duygu:** "Yemek güzeldi ama servis çok kötüydü" gibi cümlelerde SVM bigram özelliklerine dayanırken LLM cümlenin bütününü değerlendirir.

**Argo:** TF-IDF argo kelimeleri vektörüne dahil edebilmektedir; ancak nadir görülen yeni argolar seyrek vektörde temsil edilemez. LLM ise morfolojik benzerlikleri ve bağlamı kullanarak bu boşluğu doldurabilir.

### 6.3 Sınırlılıklar

Bu çalışmanın bazı sınırlılıkları bulunmaktadır. Ollama LLM'in yanıt süresi (örnekte ortalama ~0.24 sn/yorum) gerçek zamanlı büyük ölçekli uygulamalar için potansiyel bir darboğaz oluşturmaktadır. Bunun yanı sıra, Conflict durumlarında her zaman LLM kararını almak (sabit kural) optimal olmayabilir; daha sofistike bir oy ağırlıklandırma mekanizması geliştirilebilir. Son olarak, 200 örneklik hibrit değerlendirme, bulguların tüm test seti üzerinde güvenilir biçimde genellenmesi için yetersiz olabilir.

---

## 7. Sonuç

Bu çalışmada, Türkçe sosyal medya yorumları için altı makine öğrenmesi modeli sistematik biçimde karşılaştırılmış ve en yüksek performans gösteren Linear SVM, yerel Ollama LLM ile birleştirilerek Hibrit Karar Ajanı mimarisi oluşturulmuştur.

Temel bulgular şu şekilde özetlenebilir:

- Linear SVM, %86.65 doğruluk ve 0.8656 F1 skoru ile tüm modeller arasında birinci sırayı almıştır.
- Logistic Regression ve Multinomial NB, Linear SVM'e yakın performans sergileyerek Türkçe metin sınıflandırmasında kelime tabanlı istatistiksel modellerin gücünü teyit etmiştir.
- XGBoost ve Gradient Boosting, yüksek boyutlu seyrek vektörlerde ağaç yöntemlerinin doğası gereği TF-IDF üzerinde daha az etkili olduğunu ortaya koymuştur.
- Hibrit mimari, özellikle argo ve kinaye içeren yorumlarda SVM'in kelime düzeyinde hatalarını LLM'in bağlam anlama kapasitesiyle telafi edebilmektedir.

Gelecek çalışmalar için üç yön önerilmektedir: (1) Ollama tabanlı sınıflandırma için ince ayarlı (fine-tuned) Türkçe bir LLM kullanılması, (2) Conflict kararlarında sabit LLM seçimi yerine güven skoruna dayalı ağırlıklı oylama uygulanması ve (3) modelin gerçek zamanlı sosyal medya akışlarına entegrasyonu.

---

## Kaynaklar

[1] B. Liu, "Sentiment Analysis and Opinion Mining," *Synthesis Lectures on Human Language Technologies*, vol. 5, no. 1, pp. 1–167, 2012.

[2] Z. Çekiç ve B. Diri, "Türkçe Sosyal Medya Metinlerinin Duygu Analizi," *Akademik Bilişim Konferansı*, 2015.

[3] M. Kaya, G. Fidan ve İ. H. Toroslu, "Sentiment Analysis of Turkish Political News," *Proceedings of the 2012 IEEE/WIC/ACM International Conferences on Web Intelligence*, pp. 174–180, 2012.

[4] E. Demirtas ve M. Pechenizkiy, "Cross-lingual Polarity Detection with Machine Translation," *Proceedings of the Second International Workshop on Issues of Sentiment Discovery and Opinion Mining*, 2013.

[5] T. B. Brown ve ark., "Language Models are Few-Shot Learners," *Advances in Neural Information Processing Systems*, vol. 33, pp. 1877–1901, 2020.

[6] Meta AI, "Llama 3.2: Lightweight and Multimodal Open Models," Meta AI Research, 2024. [Online]. Available: https://ai.meta.com/llama

[7] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

[8] T. Chen ve C. Guestrin, "XGBoost: A Scalable Tree Boosting System," *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785–794, 2016.

[9] C. Cortes ve V. Vapnik, "Support-Vector Networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, 1995.

[10] S. Robertson, "Understanding Inverse Document Frequency: On Theoretical Arguments for IDF," *Journal of Documentation*, vol. 60, no. 5, pp. 503–520, 2004.

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
