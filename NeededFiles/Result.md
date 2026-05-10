# Türkçe Sosyal Medya Duygu Analizinde Çoklu LLM Değerlendirmesi: 13 Büyük Dil Modelinin Hibrit Karar Ajanı Mimarisiyle Sistematik Karşılaştırması

**Yazar:** [Öğrenci Adı]  
**Tarih:** 2025  
**Ders:** Yapay Zeka / Makine Öğrenmesi  

---

## Özet

Bu çalışmada, Türkçe sosyal medya yorumları üzerinde daha önce tasarlanan SVM + Ollama Hibrit Karar Ajanı mimarisi, 13 farklı büyük dil modeli (LLM) ile sistematik biçimde değerlendirilmiştir. Tüm deneyler aynı 2.224 örnekli test seti ve sabit bir TF-IDF + Linear SVM baseline (Accuracy = 0.8678, F1 = 0.8670) üzerinde gerçekleştirilmiştir. Elde edilen sonuçlar, LLM'lerin sıfır-atımlı (zero-shot) Türkçe duygu sınıflandırmasında SVM'in belirgin biçimde gerisinde kaldığını ortaya koymaktadır. En yüksek LLM doğruluğu **llama3.1:latest** modeliyle 0.7383 olarak elde edilmiş; en düşük performans ise 0.5265 ile **codellama:7b** modelinde gözlemlenmiştir. Consensus oranı %48.6 ile %70.9 arasında değişirken, Conflict durumlarında SVM'in tüm modellere karşı baskın kazananı olduğu saptanmıştır. Bu bulgular, mevcut hibrit mimarinin yeniden değerlendirilmesini ve çatışma çözüm mekanizmasının iyileştirilmesini önermektedir.

**Anahtar Kelimeler:** duygu analizi, büyük dil modeli, LLM karşılaştırması, hibrit karar ajanı, Türkçe NLP, zero-shot sınıflandırma, Ollama, SVM

---

## 1. Giriş

Önceki çalışmada [önceki rapor] TF-IDF + Linear SVM ile yerel bir LLM'i (Llama 3.2:3b) birleştiren iki aşamalı bir Hibrit Karar Ajanı mimarisi tasarlanmış ve Türkçe sosyal medya metinleri üzerinde değerlendirilmiştir. Söz konusu çalışma, tek bir LLM üzerinde yürütülmüş ve Consensus/Conflict mekanizmasının sistematik analizini kapsamamıştır.

Bu çalışmanın temel motivasyonu, **hangi LLM'in hibrit mimariye en uygun ortak olduğunu** veri odaklı biçimde belirlemektir. Bunun için:

1. 13 farklı açık kaynaklı LLM, özdeş koşullarda ve aynı test seti üzerinde değerlendirilmiştir.
2. Her model için SVM, Ollama ve Hybrid olmak üzere üç performans katmanı raporlanmıştır.
3. Consensus/Conflict dağılımı, Conflict kazanan analizi ve model boyutu–performans ilişkisi incelenmiştir.

Bu çalışmanın katkıları şunlardır:

- Türkçe duygu analizi bağlamında 13 LLM'in kapsamlı sıfır-atımlı değerlendirmesi
- Conflict durumlarında hangi modelin (SVM veya LLM) daha güvenilir olduğunun nicel analizi
- LLM seçiminin hibrit mimari performansına etkisine ilişkin pratik yönlendirici bulgular

---

## 2. İlgili Çalışmalar

Türkçe duygu analizi üzerine çalışmalar 2010'ların ortasından itibaren hız kazanmış; ancak büyük ölçüde geleneksel ML yöntemleriyle sınırlı kalmıştır [1]. Derin öğrenme tabanlı yaklaşımlar ve çok dilli ön eğitimli modeller (mBERT, XLM-R) son yıllarda öne çıksa da Türkçeye özgü sıfır-atımlı değerlendirmeler hâlâ kısıtlıdır [2].

Büyük dil modeli tabanlı sınıflandırma çalışmaları ağırlıklı olarak İngilizce üzerine yoğunlaşmaktadır [3, 4]. Türkçe için GPT-4 ve Claude gibi ticari modellerin belgelenmesi mevcut olmakla birlikte, açık kaynaklı yerel LLM'lerin karşılaştırmalı değerlendirmesi literatürde bir boşluk oluşturmaktadır.

Sıfır-atımlı sınıflandırmada istem mühendisliği (prompt engineering) doğrultusunda yapılan çalışmalar [5], talimata uyum davranışının model mimarisi ve eğitim verisi çeşitliliğiyle yakından ilişkili olduğunu ortaya koymaktadır. Bu çalışma, bu gözlemi Türkçe duygu analizi bağlamında 13 model üzerinde doğrulamaktadır.

---

## 3. Deney Tasarımı

### 3.1 Sabit Koşullar

Tüm deneylerde tutarlılık sağlamak amacıyla aşağıdaki koşullar sabit tutulmuştur:

| Parametre | Değer |
|-----------|-------|
| Test seti boyutu | 2.224 örnek |
| SVM mimarisi | Linear SVC (C=1.0, TF-IDF ngram(1,2)) |
| Sistem istemi | Türkçe zero-shot, tek kelime yanıt: "Pozitif" / "Negatif" |
| Sıcaklık (temperature) | 0.1 |
| Top-p | 0.9 |
| Max token (yanıt) | 25 |
| Veri bölme | %80 eğitim / %20 test (stratified, random_state=42) |
| Sınıf etiketleri | Pozitif, Negatif |

### 3.2 Değerlendirilen Modeller

| # | Model | Parametre Büyüklüğü | Kategori |
|---|-------|---------------------|----------|
| 1 | aya-expanse:latest | ~8B | Çok dilli genel |
| 2 | codellama:7b | 7B | Kod odaklı |
| 3 | gemma2:latest | ~9B | Genel amaçlı |
| 4 | glm-ocr:latest | ~9B | Görsel + metin |
| 5 | glm4:9b | 9B | Çok dilli |
| 6 | granite4.1:3b | 3B | Hafif genel |
| 7 | granite4.1:8b | 8B | Genel amaçlı |
| 8 | llama3.1:8b | 8B | Genel amaçlı |
| 9 | llama3.1:latest | 8B | Genel amaçlı |
| 10 | llama3.2:3b | 3B | Hafif genel |
| 11 | mistral:latest | 7B | Genel amaçlı |
| 12 | olmo2:7b | 7B | Araştırma |
| 13 | qwen2.5-coder:7b | 7B | Kod odaklı |

### 3.3 Değerlendirme Metrikleri

- **Accuracy:** Doğru sınıflandırılan örnek oranı
- **Weighted F1:** Sınıf dengesizliğini göz önünde bulunduran harmonik ortalama
- **Precision / Recall:** Sınıf ağırlıklı
- **Consensus oranı (%):** İki modelin aynı karara ulaştığı örneklerin payı
- **Conflict oranı (%):** İki modelin farklı karara ulaştığı örneklerin payı
- **Conflict kazanan dağılımı:** Conflict durumlarında SVM / Ollama / Her ikisi yanlış

---

## 4. Sonuçlar

### 4.1 Ollama LLM Doğruluk Karşılaştırması

**Tablo 1. Tüm Modellerin Performans Özeti (n=2224, SVM Baseline: Acc=0.8678, F1=0.8670)**

| Sıra | Model | Acc (LLM) | F1 (LLM) | Precision | Recall | Consensus % | Conflict % |
|------|-------|-----------|----------|-----------|--------|-------------|------------|
| 1 | llama3.1:latest | **0.7383** | **0.7385** | 0.7516 | 0.7383 | **70.9%** | 29.1% |
| 2 | llama3.1:8b | 0.7284 | 0.7289 | 0.7385 | 0.7284 | 69.9% | 30.1% |
| 3 | llama3.2:3b | 0.7032 | 0.7039 | 0.7106 | 0.7032 | 67.3% | 32.7% |
| 4 | gemma2:latest | 0.6942 | 0.6826 | 0.7696 | 0.6942 | 64.3% | 35.7% |
| 5 | qwen2.5-coder:7b | 0.6812 | 0.6818 | 0.6901 | 0.6812 | 66.1% | 33.9% |
| 6 | aya-expanse:latest | 0.6790 | 0.6620 | 0.7741 | 0.6790 | 62.7% | 37.3% |
| 7 | glm4:9b | 0.6614 | 0.6501 | 0.7227 | 0.6614 | 62.3% | 37.7% |
| 8 | mistral:latest | 0.6317 | 0.6324 | 0.6338 | 0.6317 | 60.1% | 39.9% |
| 9 | granite4.1:8b | 0.6286 | 0.6091 | 0.7065 | 0.6286 | 59.1% | 40.9% |
| 10 | granite4.1:3b | 0.5890 | 0.5546 | 0.6893 | 0.5890 | 56.0% | 44.0% |
| 11 | olmo2:7b | 0.5908 | 0.4973 | 0.6631 | 0.5908 | 61.3% | 38.7% |
| 12 | glm-ocr:latest | 0.5423 | 0.4426 | 0.5072 | 0.5423 | 57.8% | 42.2% |
| 13 | codellama:7b | 0.5265 | 0.4401 | 0.7106 | 0.5265 | **48.6%** | **51.4%** |
| — | **SVM Baseline** | **0.8678** | **0.8670** | **0.8694** | **0.8678** | — | — |

> **Not:** Mevcut hibrit mimaride Hybrid sonucu = Ollama sonucu ile özdeşleşmiştir; bu durum 4.3 bölümünde analiz edilmektedir.

### 4.2 Performans Aralığı ve Kümeleme

Modeller doğruluk performanslarına göre üç kümeye ayrılmaktadır:

**Küme A — Yüksek Performans (Acc ≥ 0.70):**  
llama3.1:latest (0.7383), llama3.1:8b (0.7284), llama3.2:3b (0.7032)  
Bu grup, Llama 3.1 ailesinin üstün sıfır-atımlı Türkçe talimat uyumuna işaret etmektedir. Llama 3.1 8B ve latest sürümleri arasındaki fark ihmal düzeyindedir (Δ=0.0099), bu da aynı mimarinin farklı kuantizasyon/sürüm varyantlarından kaynaklandığını düşündürmektedir.

**Küme B — Orta Performans (0.60 ≤ Acc < 0.70):**  
gemma2:latest, qwen2.5-coder:7b, aya-expanse:latest, glm4:9b, mistral:latest, granite4.1:8b  
Bu grupta parametre büyüklüğü ile doğruluk arasında tutarsız bir ilişki gözlemlenmektedir; 9B parametreli glm4'ün 7B parametreli qwen2.5-coder'ın altında kalması dikkat çekicidir.

**Küme C — Düşük Performans (Acc < 0.60):**  
granite4.1:3b, olmo2:7b, glm-ocr:latest, codellama:7b  
Bu grupta iki farklı başarısızlık modu gözlemlenmektedir: talimat uyumsuzluğu (codellama) ve Türkçe dil kapsamı yetersizliği (glm-ocr, olmo2).

### 4.3 Hibrit Mimari Analizi

Mevcut mimaride Conflict durumlarında karar Ollama tahminlerine devredilmektedir. Bu yaklaşım, SVM'nin Conflict durumlarındaki üstünlüğü göz önüne alındığında sistematik bir hata kaynağı oluşturmaktadır.

**Tablo 2. Conflict Durumlarında Kazanan Analizi**

| Model | Conflict (n) | Ollama Kazandı | SVM Kazandı | SVM Üstünlük Oranı |
|-------|-------------|----------------|-------------|---------------------|
| llama3.1:latest | 648 | 180 | 468 | **72.2%** |
| llama3.1:8b | 670 | 180 | 490 | 73.1% |
| llama3.2:3b | 728 | 181 | 547 | 75.1% |
| gemma2:latest | 794 | 204 | 590 | 74.3% |
| qwen2.5-coder:7b | 755 | 170 | 585 | 77.5% |
| aya-expanse:latest | 830 | 205 | 625 | 75.3% |
| glm4:9b | 839 | 190 | 649 | 77.4% |
| mistral:latest | 887 | 181 | 706 | 79.6% |
| granite4.1:8b | 910 | 189 | 721 | 79.2% |
| granite4.1:3b | 978 | 179 | 799 | 81.7% |
| olmo2:7b | 860 | 122 | 738 | **85.8%** |
| glm-ocr:latest | 938 | 107 | 831 | **88.6%** |
| codellama:7b | 1143 | 192 | 951 | 83.2% |

**Temel gözlem:** Conflict durumlarında SVM, tüm modellere karşı %72–89 oranında kazanmaktadır. Bu, mevcut hibrit mimarisinin Conflict çözüm stratejisinin yeniden tasarlanmasını zorunlu kılmaktadır.

### 4.4 Consensus Oranı ile Doğruluk İlişkisi

Consensus oranı, LLM'in SVM ile ne sıklıkta mutabık kaldığını ölçen dolaylı bir güven göstergesidir. Tablo 1 incelendiğinde, Küme A modellerin (%67–71 Consensus) Küme C modellerinden (%49–58) belirgin biçimde daha yüksek uyum sergilediği görülmektedir.

Bu ilişki, **Consensus oranının LLM kalitesinin pratik bir proxy'si** olarak kullanılabileceğini öne sürmektedir: model değerlendirmesi olmaksızın yalnızca Consensus oranına bakarak ön eleme yapılabilir.

Codellama:7b'nin %48.6 Consensus oranıyla neredeyse rastgele uyum sergilemesi, modelin Türkçe talimatları anlayamadığına ve pratikte faydasız olduğuna işaret etmektedir.

### 4.5 Precision–Recall Dengesi

Bazı modeller yüksek Precision ama düşük Recall ile dikkat çekmektedir:

- **aya-expanse:latest:** Precision=0.7741, Recall=0.6790 (Δ=0.0951) — aşırı tutucu sınıflandırma
- **gemma2:latest:** Precision=0.7696, Recall=0.6942 (Δ=0.0754) — yüksek kesinlik, düşük kapsam
- **granite4.1:3b:** Precision=0.6893, Recall=0.5890 (Δ=0.1003) — ciddi dengesizlik

Bu modellerin Negatif sınıfı kaçırma eğiliminde olduğu değerlendirilmektedir; bu da sosyal medya izleme gibi hassasiyet gerektiren uygulamalarda risklidir.

Buna karşın **llama3.1:latest** (Precision=0.7516, Recall=0.7383, Δ=0.0133) ve **mistral:latest** (Precision=0.6338, Recall=0.6317, Δ=0.0021) dengeli bir sınıflandırma profili sergilemektedir.

---

## 5. Tartışma

### 5.1 Llama 3.1 Ailesinin Üstünlüğü

Llama 3.1 ailesinin diğer 7–9B modellere kıyasla belirgin üstünlüğü birkaç hipotezi desteklemektedir: (1) Meta AI'ın eğitim veri karışımında Türkçe temsilinin görece yüksek olması, (2) Llama 3.1'in talimat ince ayarı kalitesinin özellikle çok dilli görevlere uygun olması, (3) "Pozitif" / "Negatif" gibi kısa tek kelime çıktılarına uyumu kolaylaştıran RLHF sürecinin etkisi.

### 5.2 Kod Odaklı Modellerin Yetersizliği

codellama:7b ve qwen2.5-coder:7b duygu analizi görevinde beklentilerin altında kalmıştır. Kod odaklı eğitim, modelin doğal dil duygu nüanslarından uzaklaşmasına neden olmaktadır. Ancak qwen2.5-coder'ın codellama'ya göre çok daha iyi performans sergilemesi (0.6812 vs 0.5265), Qwen2.5 taban mimarisinin güçlü genel dil anlama kapasitesini koruduğunu ortaya koymaktadır.

### 5.3 Hibrit Mimarinin Yeniden Değerlendirilmesi

Mevcut Hibrit = SVM (Consensus) + Ollama (Conflict) stratejisi, Conflict durumlarında SVM'in baskın kazananı olduğu göz önüne alındığında alt optimaldır. Önerilen alternatif stratejiler:

1. **Conflict → SVM Stratejisi:** Conflict durumlarında her zaman SVM kararını benimse. Bu strateji, SVM Accuracy'i (0.8678) muhafaza ederken Ollama'nın ek değer katacağı Consensus bölgesinde güven sağlar.

2. **Güven Eşiği Stratejisi:** LLM'den güven skoru (log-olasılık) alınarak yalnızca yüksek güvenli LLM kararları benimsenir; düşük güvenli durumlarda SVM tercih edilir.

3. **Çoklu LLM Oylaması:** Birden fazla LLM'in oylama yöntemiyle karar üretmesi; özellikle Küme A modellerinin kombinasyonu değerlendirilebilir.

### 5.4 Parametre Büyüklüğü–Performans İlişkisi

Bu deneyde parametre büyüklüğü ile doğruluk arasında doğrusal bir ilişki gözlemlenmemiştir. 3B parametreli llama3.2 (0.7032), 9B parametreli glm4 (0.6614) ve glm-ocr'ı (0.5423) geride bırakmıştır. Bu bulgu, Türkçe NLP için **model büyüklüğünün değil, eğitim veri çeşitliliğinin ve talimat uyumunun** birincil belirleyici olduğunu öne sürmektedir.

---

## 6. Sınırlamalar

Bu çalışmanın birkaç önemli sınırlaması bulunmaktadır:

1. **Istem bağımlılığı:** Tek bir istem şablonu kullanılmıştır. Farklı istem formülasyonları performansı anlamlı biçimde etkileyebilir.
2. **Yanıt hataları:** "ERROR" olarak dönen LLM yanıtları SVM ile doldurulmuştur. Hata oranının yüksek olduğu modellerde (codellama) bu durum gerçek performansı maskeleyebilir.
3. **İkili sınıflandırma sınırlılığı:** Yalnızca Pozitif/Negatif etiketleri incelenmiş; nötr veya karma duygu sınıfları kapsam dışı bırakılmıştır.
4. **Donanım farklılıkları:** Modellerin çalışma süreleri raporlanmamıştır; pratikte çıkarım hızı da önemli bir seçim kriteridir.

---

## 7. Sonuç

Bu çalışmada 13 farklı açık kaynaklı LLM, Türkçe duygu analizi bağlamında aynı hibrit mimari çerçevesinde sistematik biçimde karşılaştırılmıştır. Temel bulgular şu şekilde özetlenebilir:

1. **En iyi LLM:** llama3.1:latest (Acc=0.7383, F1=0.7385), yakın takipçisi llama3.1:8b (Acc=0.7284) ile birlikte Küme A'yı oluşturmaktadır.
2. **SVM baseline üstünlüğü:** Tüm LLM'ler SVM'in (Acc=0.8678) belirgin gerisinde kalmaktadır; en iyi LLM ile SVM arasındaki fark 13 puanlık bir boşluğa karşılık gelmektedir.
3. **Conflict çözümü kritik tasarım noktasıdır:** Conflict durumlarında SVM %72–89 oranında kazanmakta; bu nedenle mevcut "Conflict → Ollama" stratejisi yerine "Conflict → SVM" stratejisinin benimsenmesi önerilmektedir.
4. **Consensus oranı:** LLM kalitesinin pratik proxy'si olarak kullanılabilmektedir; yüksek Consensus (%65+) genellikle daha iyi LLM doğruluğuyla korelasyon göstermektedir.
5. **Kod odaklı modeller** duygu analizi görevleri için uygun değildir; bu modellerin hibrit mimaride kullanımından kaçınılmalıdır.

Gelecek çalışmalar, Llama 3.1 ile çoklu LLM oylama stratejileri ve güven eşiği tabanlı karar mekanizmalarını araştırmayı hedeflemelidir.

---

## Referanslar

[1] M. Kaya, Ş. Fidan ve İ. F. Toroslu, "Sentiment analysis of turkish political news," in Proc. IEEE/WIC/ACM Int. Conf. Web Intell., 2012.

[2] E. Demirtas ve M. Pechenizkiy, "Cross-lingual polarity detection with machine translation," in Proc. 2nd Workshop on Sentiment Anal., 2013.

[3] T. Brown ve ark., "Language models are few-shot learners," Advances in Neural Information Processing Systems, cilt 33, ss. 1877–1901, 2020.

[4] H. Touvron ve ark., "Llama 2: Open foundation and fine-tuned chat models," arXiv:2307.09288, 2023.

[5] T. Wei ve ark., "Chain-of-thought prompting elicits reasoning in large language models," Advances in Neural Information Processing Systems, cilt 35, 2022.

[6] Meta AI, "Llama 3 model card," Meta AI Research, 2024. [Çevrimiçi]. Mevcut: https://ai.meta.com/blog/meta-llama-3

[7] L. Breiman, "Bagging predictors," Machine Learning, cilt 24, no. 2, ss. 123–140, 1996.

[8] Google DeepMind, "Gemma 2: Improving open language models at a practical size," arXiv:2408.00118, 2024.

[9] A. Jiang ve ark., "Mistral 7B," arXiv:2310.06825, 2023.

[10] Qwen Team, "Qwen2.5: A party of foundation models," Alibaba Cloud, 2024. [Çevrimiçi]. Mevcut: https://qwenlm.github.io/blog/qwen2.5

---

## Ekler

### Ek A — Sistem İstemi (Prompt)

```
Sen Türkçe duygu analizi yapan bir sınıflandırıcısın.
Sana verilen Türkçe sosyal medya yorumunu analiz et.
SADECE tek kelime yanıt ver: 'Pozitif' veya 'Negatif'.
Başka hiçbir şey yazma. Açıklama yapma. Noktalama kullanma.

Yorum: {metin}
Etiket:
```

### Ek B — Model Parametreleri

| Parametre | Değer |
|-----------|-------|
| temperature | 0.1 |
| top_p | 0.9 |
| num_predict | 25 |
| max_workers | 1 |

### Ek C — Tam Metrik Tablosu (LLM Doğrulukları)

| Model | Accuracy | F1 | Precision | Recall |
|-------|----------|-----|-----------|--------|
| llama3.1:latest | 0.7383 | 0.7385 | 0.7516 | 0.7383 |
| llama3.1:8b | 0.7284 | 0.7289 | 0.7385 | 0.7284 |
| llama3.2:3b | 0.7032 | 0.7039 | 0.7106 | 0.7032 |
| gemma2:latest | 0.6942 | 0.6826 | 0.7696 | 0.6942 |
| qwen2.5-coder:7b | 0.6812 | 0.6818 | 0.6901 | 0.6812 |
| aya-expanse:latest | 0.6790 | 0.6620 | 0.7741 | 0.6790 |
| glm4:9b | 0.6614 | 0.6501 | 0.7227 | 0.6614 |
| mistral:latest | 0.6317 | 0.6324 | 0.6338 | 0.6317 |
| granite4.1:8b | 0.6286 | 0.6091 | 0.7065 | 0.6286 |
| olmo2:7b | 0.5908 | 0.4973 | 0.6631 | 0.5908 |
| granite4.1:3b | 0.5890 | 0.5546 | 0.6893 | 0.5890 |
| glm-ocr:latest | 0.5423 | 0.4426 | 0.5072 | 0.5423 |
| codellama:7b | 0.5265 | 0.4401 | 0.7106 | 0.5265 |
| **SVM Baseline** | **0.8678** | **0.8670** | **0.8694** | **0.8678** |
