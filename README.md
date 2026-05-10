# Hibrit Karar Ajani — Türkçe Sosyal Medya Duygu Analizi

**SVM + Ollama LLM Tabanlı Hibrit Duygu Sınıflandırıcısı**

---

## İçindekiler

1. [Proje Genel Bakışı](#proje-genel-bakışı)
2. [Mimari](#mimari)
3. [Önerilen Ollama Modeli](#önerilen-ollama-modeli)
4. [Kurulum](#kurulum)
5. [Dosya Yapısı](#dosya-yapısı)
6. [Kullanım](#kullanım)
7. [Çıktılar](#çıktılar)
8. [Konfigürasyon](#konfigürasyon)
9. [Sistem Gereksinimleri](#sistem-gereksinimleri)
10. [Sorun Giderme](#sorun-giderme)

---

## Proje Genel Bakışı

Bu proje, Türkçe sosyal medya yorumlarını **Pozitif / Negatif** olarak sınıflandıran iki katmanlı hibrit bir karar mekanizması sunar:

| Katman | Model | Güçlü Yönü |
|--------|-------|------------|
| 1. Katman | Linear SVM (TF-IDF) | Hız, kelime tabanlı özellikler |
| 2. Katman | Ollama LLM (llama3.2) | Bağlam anlama, argo, kinaye |
| Hibrit | Consensus / Conflict | Her iki modelin kararlarını birleştirir |

**Temel Fikir:** SVM, "koyayım" gibi kelimeleri bağlamından koparabilir; LLM bağlamı anlayarak bu hataları düzeltir. İki model aynı karardaysa güven yüksektir (Consensus); farklı karardaysa LLM'in kararı ağır basar (Conflict).

---

## Mimari

```
Türkçe Sosyal Medya Yorumu
        │
        ├──────────────────────────────────────┐
        │                                      │
        ▼                                      ▼
 ┌─────────────┐                    ┌─────────────────────┐
 │ TF-IDF      │                    │ Ollama LLM          │
 │ Vektörizasyon│                   │ (llama3.2 / mistral)│
 │ Linear SVM  │                    │ System Prompt ile   │
 │ Tahmini     │                    │ minimize edilmiş    │
 └──────┬──────┘                    └──────────┬──────────┘
        │                                      │
        │   SVM_Prediction                     │  Ollama_Prediction
        │   (Pozitif/Negatif)                  │  (Pozitif/Negatif)
        └──────────────────┬───────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Karar Motoru   │
                  ├─────────────────┤
                  │ Aynı mı?        │
                  │  ✓ → CONSENSUS  │
                  │  ✗ → CONFLICT   │
                  │   (LLM seçilir) │
                  └────────┬────────┘
                           │
                           ▼
                  Hibrit Karar (Pozitif/Negatif)
```

---

## Önerilen Ollama Modeli

### 🏆 En İyi Seçim: `llama3.2` (3B)

| Kriter | Neden Llama 3.2? |
|--------|-----------------|
| **Dil Desteği** | Türkçe dahil çok dil desteği, argo ve kinayeyi iyi anlar |
| **Hız** | 3B parametre → RTX 3060 Ti'de ~2–5 token/sn, batchlerde hızlı |
| **Bağlam** | 128K token bağlam penceresi |
| **Açık Kaynak** | Llama 3.2 lisansı — ticari kullanım dahil özgür |
| **Boyut** | ~2 GB disk (kuantize 4-bit Q4_K_M) |

```bash
# Önce bu komutu çalıştırın (bir kere yeterli):
ollama pull llama3.2
```

### Alternatif Modeller (Sıralama)

| Model | Boyut | Türkçe Performansı | Hız |
|-------|-------|-------------------|-----|
| `llama3.2` | 3B | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| `mistral` | 7B | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| `llama3.1:8b` | 8B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| `gemma3:4b` | 4B | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| `qwen2.5:7b` | 7B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

> **Not:** RTX 3060 Ti (8GB VRAM) ile `llama3.2` veya `mistral` idealdir. 8B modeller VRAM'e sığar ancak biraz daha yavaş olur. Bu proje için **`llama3.2`** önerilir — hız/kalite dengesi en iyisidir.

---

## Kurulum

### 1. Ollama Kurulumu (Windows)

```bash
# Ollama'yı indirin: https://ollama.com/download/windows
# Kurulumdan sonra:
ollama serve       # Arka planda çalıştır (veya servis olarak kurun)
ollama pull llama3.2  # Modeli indirin
```

### 2. Python Bağımlılıkları

```bash
pip install ollama pandas matplotlib seaborn scikit-learn xgboost wordcloud
```

**Tam liste (requirements_hybrid.txt):**
```
numpy
pandas
matplotlib
seaborn
wordcloud
scikit-learn
xgboost
ollama
```

### 3. Ortam Değişkeni (Opsiyonel, Model Dizini İçin)

Modelleriniz `D:\OllamaModels` dizinindeyse, Ollama'yı başlatmadan önce:
```bash
set OLLAMA_MODELS=D:\OllamaModels
ollama serve
```

---

## Dosya Yapısı

```
proje_klasoru/
├── social_media_analysis.py   # Mevcut ana dosya (SVM eğitimi)
├── hybrid_agent.py            # YENİ: Hibrit Karar Ajani
├── social_media_comments.csv  # Veri seti
├── requirements.txt           # Mevcut bağımlılıklar
├── requirements_hybrid.txt    # Hibrit bağımlılıklar
├── outputs/                   # Mevcut çıktılar (social_media_analysis.py)
│   ├── 01_EDA_genel_bakis.png
│   ├── ...
│   └── 11_final_siralama.png
└── outputs_hybrid/            # YENİ: Hibrit çıktılar
    ├── H1_venn_karsilastirma.png
    ├── H2_conflict_heatmap.png
    ├── H3_model_karsilastirma.png
    ├── H4_conflict_ornekleri.png
    ├── H5_consensus_conflict_dagilim.png
    ├── hybrid_results.csv
    └── hybrid_summary.txt
```

---

## Kullanım

### Adım 1: Ollama'yı başlatın
```bash
ollama serve
```

### Adım 2: (Opsiyonel) Mevcut projeyi çalıştırın
```bash
python social_media_analysis.py
```

### Adım 3: Hibrit ajanı çalıştırın
```bash
python hybrid_agent.py
```

### Beklenen Çıktı
```
=================================================================
  HİBRİT KARAR AJANI — SVM + Ollama (LLM)
=================================================================

[A] Veri yükleniyor ve SVM eğitiliyor...
  ✓ Linear SVM eğitildi  →  Acc=0.8665 | F1=0.8656

[B] Ollama bağlantısı test ediliyor...
  ✓ Ollama bağlantısı başarılı → Test yanıt: Pozitif

[C] Ollama LLM tahminleri alınıyor...
    Model: llama3.2 | Paralel: 4 worker
    20/200 tamamlandı (hata: 0)
    ...
    ✓ Tamamlandı — 200 tahmin, 47.3 sn

[D] Hibrit Analiz Özeti:
  Consensus     : 163 (81.5%)
  Conflict      : 37  (18.5%)
  SVM Doğruluk  : 0.8650
  Ollama Doğruluk: 0.8400
  Hibrit Doğruluk: 0.8750

[E] Görseller oluşturuluyor...
  ✓ H1_venn_karsilastirma.png
  ...
```

---

## Çıktılar

| Dosya | Açıklama |
|-------|----------|
| `H1_venn_karsilastirma.png` | SVM ve Ollama'nın Pozitif/Negatif tahminlerinin kesişim analizi |
| `H2_conflict_heatmap.png` | SVM vs Ollama tahmin matrisi + Conflict kazanan analizi |
| `H3_model_karsilastirma.png` | SVM, Ollama ve Hibrit karşılaştırmalı bar + Consensus/Conflict pasta |
| `H4_conflict_ornekleri.png` | Conflict durumlarının örnek tablo gösterimi |
| `H5_consensus_conflict_dagilim.png` | Consensus/Conflict gruplarındaki gerçek etiket dağılımı |
| `hybrid_results.csv` | Tüm tahminleri içeren tam sonuç tablosu |
| `hybrid_summary.txt` | Özet metrik raporu |

---

## Konfigürasyon

`hybrid_agent.py` içindeki değişkenler:

```python
OLLAMA_MODEL  = "llama3.2"   # Ollama model adı
OLLAMA_TIMEOUT = 30           # Saniye cinsinden zaman aşımı
MAX_WORKERS   = 4             # Paralel istek sayısı (GPU'ya göre ayarlayın)
SAMPLE_SIZE   = 200           # Test seti örnek boyutu (None = tümü)
```

> **GPU İpucu:** RTX 3060 Ti ile `MAX_WORKERS=4` idealdir. Daha güçlü GPU'larda 8'e çıkarabilirsiniz. Daha zayıf sistemlerde 1–2 yapın.

---

## Sistem Gereksinimleri

| Bileşen | Minimum | Önerilen |
|---------|---------|---------|
| GPU | GTX 1060 6GB | RTX 3060 Ti 8GB ✓ |
| RAM | 8 GB | 16 GB |
| Disk | 5 GB (model) | 10 GB |
| Python | 3.9+ | 3.11 ✓ |
| Ollama | 0.1.x | Son sürüm |

---

## Sorun Giderme

### Ollama bağlanamıyor
```bash
# Ollama çalışıyor mu?
ollama list
# Çalışmıyorsa:
ollama serve
```

### VRAM hatası (model sığmıyor)
```bash
# Daha küçük model deneyin:
ollama pull llama3.2:1b
# hybrid_agent.py içinde:
OLLAMA_MODEL = "llama3.2:1b"
```

### Yavaş tahmin
- `SAMPLE_SIZE` değerini düşürün (örn: `100`)
- `MAX_WORKERS` değerini düşürün (örn: `2`)
- Daha küçük model kullanın (`llama3.2:1b`)

### Model yanıt vermiyor / ERROR
- Sistem prompt'u optimize edilmiştir; `temperature=0.0` ile deterministik çalışır
- `ERROR` yanıtları otomatik olarak SVM fallback ile doldurulur
