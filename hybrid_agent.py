"""
=============================================================================
Hybrid Decision Agent — Türkçe Sosyal Medya Duygu Analizi
=============================================================================
Bu modül, eğitilmiş bir Scikit-learn modeli (Linear SVM) ile yerel
Ollama LLM'ini (llama3.2) birleştirerek Hibrit Karar Mekanizması
oluşturur. Her yorum için:
  - SVM tahmini yapılır
  - Ollama LLM tahmini yapılır
  - Consensus veya Conflict etiketi üretilir

Kullanım:
  1. social_media_analysis.py'i çalıştırın (SVM modelini eğitmek için)
  2. Bu dosyayı aynı dizinde çalıştırın:
       python hybrid_agent.py

Gereksinimler için --> requirements_hybrid.txt:
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["OLLAMA_HOST"] = "127.0.0.1:11434"
os.environ["no_proxy"] = "localhost,127.0.0.1" # Windows proxy ayarlarını devre dışı bırakır

import re
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

# Ollama — pip install ollama
import ollama

# Scikit-learn (mevcut koddan alınan araçlar)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

# ─── Ayarlar ─────────────────────────────────────────────────────────────────
# Ollama model seçimi: llama3.2 (3B) önerilir — bkz. README.md
# 'ollama pull llama3.2' komutuyla indirin.
OLLAMA_MODEL     = "codellama:7b"      # değiştirmek için buraya model adını yazın
OLLAMA_TIMEOUT   = 60            # saniye
MAX_WORKERS      = 2            # paralel Ollama isteği (GPU'ya göre ayarlayın)
SAMPLE_SIZE      = None            # hibrit değerlendirme için test seti örneği
#SAMPLE_SIZE     = 100              # (tüm test seti için None yapın — yavaş olabilir)

OUTPUT_DIR   = "outputs"           # mevcut projenin çıktı dizini

# Model ismini Windows klasör formatına uygun hale getir (Örn: llama3.2:3b -> llama3_2_3b)
safe_model_name = OLLAMA_MODEL.replace(":", "_").replace(".", "_")

# Dinamik klasör yolu: outputs_hybrid/mistral_latest_hybrid/ gibi
HYBRID_DIR   = os.path.join("outputs_hybrid", f"{safe_model_name}_hybrid")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(HYBRID_DIR, exist_ok=True)

# ─── Türkçe stop words (social_media_analysis.py ile aynı) ──────────────────
TR_STOPWORDS = set([
    "bir","bu","ve","da","de","ki","ile","için","den","dan","ben","sen","o",
    "biz","siz","onlar","ama","fakat","ya","veya","ne","nasıl","neden","çok",
    "az","daha","en","bile","artık","diye","gibi","kadar","mi","mu","mı","mü",
    "var","yok","bu","şu","bu","onun","bunu","şunu","ona","bana","sana","benim",
    "senin","onun","bizim","sizin","onların","e","a","ı","i","u","ü","bu",
    "şu","o","ben","sen","ya","ah","oh","of","ey","hey","be","hep","hiç",
    "her","herkes","hiçbir","hiçbirşey","herşey","bazı","birkaç","birçok",
    "şey","zaten","sadece","yani","aslında","tam","tamamen","gayet","epey",
    "oldukça","böyle","şöyle","öyle","böylece","böylesi","bunun","şunun",
    "onun","bunlar","şunlar","onlar","buraya","şuraya","oraya","burada",
    "şurada","orada","buradan","şuradan","oradan","biri","birisi","kimse",
    "kimse","birşey","biraz","birbiri","birbirine","birbirini","birbirinden",
])

# ─── Görsel stil ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        12,
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       130,
    "savefig.bbox":     "tight",
    "savefig.dpi":      150,
})

COLOR_POS       = "#4CAF50"
COLOR_NEG       = "#F44336"
COLOR_CONSENSUS = "#2196F3"
COLOR_CONFLICT  = "#FF9800"
PALETTE         = {"Pozitif": COLOR_POS, "Negatif": COLOR_NEG}

print("=" * 65)
print("  HİBRİT KARAR AJANI — SVM + Ollama (LLM)")
print("=" * 65)

# =============================================================================
# BÖLÜM A — Veri Yükleme & SVM Eğitimi
#   (social_media_analysis.py çalıştırıldıysa bu bölümü atlayabilirsiniz;
#    ancak bu dosya bağımsız çalışabilmesi için yeniden eğitiyor.)
# =============================================================================

print("\n[A] Veri yükleniyor ve SVM eğitiliyor...")

df = pd.read_csv("social_media_comments.csv", encoding="iso-8859-9")
df.columns = ["label", "text"]
df = df.dropna(subset=["text"]).reset_index(drop=True)
df["text_str"] = df["text"].astype(str)


def preprocess(text: str) -> str:
    """Türkçe metni temizle ve normalize et."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+",           " ", text)
    text = re.sub(r"#\w+",           " ", text)
    text = re.sub(r"\d+",            " ", text)
    text = re.sub(r"[^a-zığüşöçİĞÜŞÖÇ\s]", " ", text)
    text = re.sub(r"\s+",            " ", text).strip()
    tokens = [w for w in text.split() if w not in TR_STOPWORDS and len(w) > 1]
    return " ".join(tokens)


df["clean_text"] = df["text_str"].apply(preprocess)

le = LabelEncoder()
y  = le.fit_transform(df["label"])   # Negatif=0, Pozitif=1

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df["clean_text"], y, test_size=0.20, random_state=42, stratify=y
)

# Aynı zamanda ham metinleri de sakla (LLM ham metin okuyacak)
_, X_test_raw_original, _, y_test_original = train_test_split(
    df["text_str"], y, test_size=0.20, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(
    analyzer="word", ngram_range=(1, 2),
    max_features=30000, sublinear_tf=True, min_df=2,
)
X_train_tfidf = tfidf.fit_transform(X_train_raw)
X_test_tfidf  = tfidf.transform(X_test_raw)

svm_model = LinearSVC(C=1.0, max_iter=2000, random_state=42)
svm_model.fit(X_train_tfidf, y_train)

svm_preds_all = svm_model.predict(X_test_tfidf)
svm_acc   = accuracy_score(y_test, svm_preds_all)
svm_f1    = f1_score(y_test, svm_preds_all, average="weighted")

print(f"  ✓ Linear SVM eğitildi  →  Acc={svm_acc:.4f} | F1={svm_f1:.4f}")
print(f"  Test seti boyutu: {len(X_test_raw):,} yorum")

# Örnekleme (hız için)
if SAMPLE_SIZE and SAMPLE_SIZE < len(X_test_raw):
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_test_raw), size=SAMPLE_SIZE, replace=False)
else:
    sample_idx = np.arange(len(X_test_raw))

X_sample_raw      = X_test_raw_original.iloc[sample_idx].reset_index(drop=True)
X_sample_clean    = X_test_raw.iloc[sample_idx].reset_index(drop=True)
y_sample          = y_test[sample_idx]
svm_sample_preds  = svm_preds_all[sample_idx]

print(f"  Hibrit değerlendirme örnek boyutu: {len(sample_idx)}")


# =============================================================================
# BÖLÜM B — Ollama LLM Entegrasyonu
# =============================================================================

# ─── Sistem Promptu (optimize edilmiş, minimal token kullanımı) ──────────────
SYSTEM_PROMPT = (
    "Sen Türkçe duygu analizi yapan bir sınıflandırıcısın. "
    "Sana verilen Türkçe sosyal medya yorumunu analiz et. "
    "SADECE tek kelime yanıt ver: 'Pozitif' veya 'Negatif'. "
    "Başka hiçbir şey yazma. Açıklama yapma. Noktalama kullanma. "
    "Do not use think tags. Directly output only one word: Pozitif or Negatif.")

USER_TEMPLATE = "Yorum: {text}\nEtiket:"


def query_ollama_single(text: str, model: str = OLLAMA_MODEL) -> str:
    """
    Tek bir metin için Ollama LLM'e istek atar.
    Mistral gibi inatçı modeller için sistem promptu user içine gömülmüştür.
    """
    try:
        # Mistral bazen 'system' rolünü sevmez, garanti olsun diye her şeyi 'user' içine koyuyoruz.
        combined_prompt = f"{SYSTEM_PROMPT}\n\n{USER_TEMPLATE.format(text=text[:500])}"
        
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": combined_prompt},
            ],
            options={
                "temperature": 0.1,   # 0.0 yerine 0.1 yaptık (Kilitlenmeyi önler)
                "top_p":       0.9,
                "num_predict": 25,    # Rahatça kelime üretebilsin
                # STOP PARAMETRESİNİ TAMAMEN YOK ETTİK
            },
        )
        raw = response["message"]["content"].strip()
        
        # Çıktıyı normalize et
        if "pozitif" in raw.lower() or "positive" in raw.lower():
            return "Pozitif"
        elif "negatif" in raw.lower() or "negative" in raw.lower():
            return "Negatif"
        else:
            # Hala saçmalıyorsa ekrana ne dediğini yazdır
            print(f"\n[DEBUG] {model} ham cevabı: '{raw}' | Yorum: '{text[:50]}...'")
            return "ERROR"
    except Exception as e:
        print(f"  [WARN] Ollama hatası: {e}")
        return "ERROR"


def query_ollama_batch(
    texts: list,
    model: str   = OLLAMA_MODEL,
    max_workers: int = MAX_WORKERS,
    show_progress: bool = True,
) -> list:
    """
    Metin listesini paralel olarak Ollama'ya gönderir.

    Parametre:
        texts       : Ham Türkçe metin listesi
        model       : Ollama model adı
        max_workers : Paralel thread sayısı (GPU'ya göre ayarlayın)
        show_progress: İlerleme göster

    Dönüş:
        Tahmin listesi ['Pozitif'|'Negatif'|'ERROR', ...]
    """
    predictions = [None] * len(texts)
    errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(query_ollama_single, text, model): i
            for i, text in enumerate(texts)
        }
        completed = 0
        for future in as_completed(future_to_idx):
            idx    = future_to_idx[future]
            result = future.result()
            predictions[idx] = result
            completed += 1
            if result == "ERROR":
                errors += 1
            if show_progress and completed % 20 == 0:
                print(f"    {completed}/{len(texts)} tamamlandı "
                      f"(hata: {errors})")

    return predictions


# =============================================================================
# BÖLÜM C — Hibrit Değerlendirme DataFrame'i
# =============================================================================

def build_hybrid_dataframe(
    texts_raw: pd.Series,
    svm_predictions: np.ndarray,
    true_labels: np.ndarray,
    label_encoder: LabelEncoder,
    ollama_model: str = OLLAMA_MODEL,
) -> pd.DataFrame:
    """
    SVM tahminleri + Ollama tahminleri + Hybrid_Status içeren
    DataFrame oluşturur.

    Dönüş DataFrame kolonları:
        text            : Ham yorum
        True_Label      : Gerçek etiket
        SVM_Prediction  : SVM tahmin etiketi
        Ollama_Prediction: Ollama LLM tahmin etiketi
        Hybrid_Status   : 'Consensus' | 'Conflict'
        SVM_Correct     : SVM'nin doğru tahmin yapıp yapmadığı
        Ollama_Correct  : Ollama'nın doğru tahmin yapıp yapmadığı
    """
    print("\n[C] Ollama LLM tahminleri alınıyor...")
    print(f"    Model: {ollama_model} | Paralel: {MAX_WORKERS} worker")
    t0 = time.time()

    ollama_preds = query_ollama_batch(
        texts=texts_raw.tolist(),
        model=ollama_model,
        max_workers=MAX_WORKERS,
        show_progress=True,
    )

    elapsed = time.time() - t0
    print(f"    ✓ Tamamlandı — {len(texts_raw)} tahmin, {elapsed:.1f} sn")

    # Encode geri al
    svm_labels_str    = label_encoder.inverse_transform(svm_predictions)
    true_labels_str   = label_encoder.inverse_transform(true_labels)

    df_hybrid = pd.DataFrame({
        "text":              texts_raw.values,
        "True_Label":        true_labels_str,
        "SVM_Prediction":    svm_labels_str,
        "Ollama_Prediction": ollama_preds,
    })

    # Hatalı Ollama yanıtlarını SVM tahminiyle doldur (fallback)
    error_mask = df_hybrid["Ollama_Prediction"] == "ERROR"
    df_hybrid.loc[error_mask, "Ollama_Prediction"] = df_hybrid.loc[error_mask, "SVM_Prediction"]
    if error_mask.sum() > 0:
        print(f"    [INFO] {error_mask.sum()} hatalı Ollama yanıtı SVM fallback ile dolduruldu.")

    # Hybrid Status
    df_hybrid["Hybrid_Status"] = df_hybrid.apply(
        lambda r: "Consensus" if r["SVM_Prediction"] == r["Ollama_Prediction"] else "Conflict",
        axis=1,
    )

    # Doğruluk bayrakları
    df_hybrid["SVM_Correct"]    = df_hybrid["SVM_Prediction"]    == df_hybrid["True_Label"]
    df_hybrid["Ollama_Correct"] = df_hybrid["Ollama_Prediction"] == df_hybrid["True_Label"]

    # Consensus satırlarda hangi modelin doğru olduğunu belirle
    consensus_mask = df_hybrid["Hybrid_Status"] == "Consensus"
    df_hybrid["Hybrid_Prediction"] = df_hybrid["SVM_Prediction"]  # default SVM
    # Conflict durumunda: Ollama'nın doğru olduğu durumda Ollama'yı seç
    conflict_mask = df_hybrid["Hybrid_Status"] == "Conflict"
    # Basit strateji: Conflict → Ollama'yı seç (LLM bağlamı daha iyi anlar)
    df_hybrid.loc[conflict_mask, "Hybrid_Prediction"] = df_hybrid.loc[conflict_mask, "Ollama_Prediction"]

    df_hybrid["Hybrid_Correct"] = df_hybrid["Hybrid_Prediction"] == df_hybrid["True_Label"]

    return df_hybrid


# =============================================================================
# BÖLÜM D — Görselleştirmeler
# =============================================================================

def plot_venn_like_comparison(df_h: pd.DataFrame, save_dir: str):
    """
    Şekil 1: SVM ve Ollama'nın Negatif/Pozitif tahminlerinin
    kesişimini gösteren grouped bar chart (Venn alternatifi).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "SVM vs Ollama — Tahmin Dağılımı Karşılaştırması",
        fontsize=15, fontweight="bold"
    )

    for ax, label in zip(axes, ["Negatif", "Pozitif"]):
        svm_said    = set(df_h[df_h["SVM_Prediction"]    == label].index)
        ollama_said = set(df_h[df_h["Ollama_Prediction"] == label].index)

        only_svm    = len(svm_said - ollama_said)
        only_ollama = len(ollama_said - svm_said)
        both        = len(svm_said & ollama_said)
        neither     = len(df_h) - len(svm_said | ollama_said)

        categories = ["Sadece SVM", "İkisi de", "Sadece Ollama", "Hiçbiri"]
        values     = [only_svm, both, only_ollama, neither]
        bar_colors = [COLOR_CONSENSUS, "#9C27B0", COLOR_CONFLICT, "#9E9E9E"]

        bars = ax.bar(categories, values, color=bar_colors,
                      alpha=0.85, edgecolor="white", linewidth=1.5, width=0.55)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    str(v), ha="center", fontweight="bold", fontsize=12)

        color = COLOR_NEG if label == "Negatif" else COLOR_POS
        ax.set_title(f'"{label}" Diyen Modeller', color=color, fontsize=13)
        ax.set_ylabel("Yorum Sayısı")
        ax.set_ylim(0, max(values) * 1.18)

    plt.tight_layout()
    path = os.path.join(save_dir, "H1_venn_karsilastirma.png")
    plt.savefig(path)
    plt.close()
    print(f"  ✓ H1_venn_karsilastirma.png")


def plot_conflict_heatmap(df_h: pd.DataFrame, save_dir: str):
    """
    Şekil 2: Uyuşmazlık ısı haritası.
    Satır=SVM_Prediction, Sütun=Ollama_Prediction, değer=count.
    Her hücre içinde Doğru/Yanlış bilgisi de gösterilir.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Uyuşmazlık Isı Haritası — SVM vs Ollama Tahmin Matrisi",
        fontsize=15, fontweight="bold"
    )

    labels = ["Negatif", "Pozitif"]

    # 1) Sayım heatmap
    pivot = pd.crosstab(
        df_h["SVM_Prediction"], df_h["Ollama_Prediction"],
        rownames=["SVM"], colnames=["Ollama"]
    ).reindex(index=labels, columns=labels, fill_value=0)

    sns.heatmap(
        pivot, annot=True, fmt="d", cmap="YlOrRd",
        ax=axes[0], linewidths=2, linecolor="white",
        annot_kws={"size": 16, "weight": "bold"},
        cbar_kws={"label": "Yorum Sayısı"},
    )
    axes[0].set_title("Tahmin Kesişim Matrisi (Sayı)")
    axes[0].set_xlabel("Ollama Tahmini", fontsize=12)
    axes[0].set_ylabel("SVM Tahmini", fontsize=12)

    # 2) Conflict segmentleri — SVM doğru/yanlış & Ollama doğru/yanlış
    conflict_df = df_h[df_h["Hybrid_Status"] == "Conflict"].copy()
    if len(conflict_df) > 0:
        conflict_matrix = pd.DataFrame(
            0, index=["SVM Doğru", "SVM Yanlış"],
            columns=["Ollama Doğru", "Ollama Yanlış"]
        )
        for _, row in conflict_df.iterrows():
            r = "SVM Doğru"    if row["SVM_Correct"]    else "SVM Yanlış"
            c = "Ollama Doğru" if row["Ollama_Correct"] else "Ollama Yanlış"
            conflict_matrix.loc[r, c] += 1

        sns.heatmap(
            conflict_matrix, annot=True, fmt="d", cmap="Blues",
            ax=axes[1], linewidths=2, linecolor="white",
            annot_kws={"size": 16, "weight": "bold"},
            cbar_kws={"label": "Çatışma Sayısı"},
        )
        axes[1].set_title(f"Conflict Analizi (n={len(conflict_df)})")
        axes[1].set_xlabel("Ollama Durumu", fontsize=12)
        axes[1].set_ylabel("SVM Durumu", fontsize=12)
    else:
        axes[1].text(0.5, 0.5, "Conflict yok!", ha="center", va="center",
                     fontsize=16, transform=axes[1].transAxes)
        axes[1].axis("off")

    plt.tight_layout()
    path = os.path.join(save_dir, "H2_conflict_heatmap.png")
    plt.savefig(path)
    plt.close()
    print(f"  ✓ H2_conflict_heatmap.png")


def plot_model_comparison_bar(df_h: pd.DataFrame, save_dir: str):
    """
    Şekil 3: SVM, Ollama ve Hibrit modelin F1 & Accuracy karşılaştırması.
    Ayrıca Consensus/Conflict oranı pasta grafiği.
    """
    true_enc   = LabelEncoder().fit(["Negatif", "Pozitif"])
    true_num   = (df_h["True_Label"] == "Pozitif").astype(int).values
    svm_num    = (df_h["SVM_Prediction"] == "Pozitif").astype(int).values
    ollama_num = (df_h["Ollama_Prediction"] == "Pozitif").astype(int).values
    hybrid_num = (df_h["Hybrid_Prediction"] == "Pozitif").astype(int).values

    metrics_data = {}
    for name, preds in [
        ("SVM\n(Linear)", svm_num),
        ("Ollama\n(LLM)", ollama_num),
        ("Hibrit\n(Consensus+LLM)", hybrid_num),
    ]:
        metrics_data[name] = {
            "Accuracy":  accuracy_score(true_num, preds),
            "F1":        f1_score(true_num, preds, average="weighted", zero_division=0),
            "Precision": precision_score(true_num, preds, average="weighted", zero_division=0),
            "Recall":    recall_score(true_num, preds, average="weighted", zero_division=0),
        }

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("Model Karşılaştırması: SVM vs Ollama vs Hibrit",
                 fontsize=15, fontweight="bold")

    # — Sol: Accuracy & F1 Grouped Bar ——————————————
    model_names = list(metrics_data.keys())
    acc_vals    = [metrics_data[m]["Accuracy"]  for m in model_names]
    f1_vals     = [metrics_data[m]["F1"]        for m in model_names]

    x  = np.arange(len(model_names))
    bw = 0.30
    bars_a = axes[0].bar(x - bw/2, acc_vals, bw, label="Accuracy",
                         color=COLOR_CONSENSUS, alpha=0.85, edgecolor="white")
    bars_f = axes[0].bar(x + bw/2, f1_vals,  bw, label="F1 Score",
                         color=COLOR_CONFLICT, alpha=0.85, edgecolor="white")
    axes[0].set_xticks(x); axes[0].set_xticklabels(model_names, fontsize=10)
    axes[0].set_ylim(0.5, 1.05)
    axes[0].set_ylabel("Skor")
    axes[0].set_title("Accuracy & F1 Karşılaştırması")
    axes[0].legend()
    for bar, v in [(b, acc_vals[i]) for i, b in enumerate(bars_a)] + \
                  [(b, f1_vals[i])  for i, b in enumerate(bars_f)]:
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.003,
                     f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    # — Orta: 4 metrik tam karşılaştırma ————————————
    metric_keys   = ["Accuracy", "F1", "Precision", "Recall"]
    n_metrics     = len(metric_keys)
    n_models      = len(model_names)
    bar_w         = 0.22
    model_colors  = [COLOR_CONSENSUS, "#9C27B0", COLOR_CONFLICT]

    for mi, (mname, col) in enumerate(zip(model_names, model_colors)):
        vals = [metrics_data[mname][mk] for mk in metric_keys]
        xpos = np.arange(n_metrics) + (mi - 1) * bar_w
        bars_all = axes[1].bar(xpos, vals, bar_w, label=mname,
                               color=col, alpha=0.82, edgecolor="white")

    axes[1].set_xticks(np.arange(n_metrics))
    axes[1].set_xticklabels(metric_keys)
    axes[1].set_ylim(0.5, 1.08)
    axes[1].set_title("Tüm Metrikler Karşılaştırması")
    axes[1].set_ylabel("Skor")
    axes[1].legend(fontsize=8)

    # — Sağ: Consensus / Conflict Pasta ——————————————
    status_counts = df_h["Hybrid_Status"].value_counts()
    colors_pie    = [COLOR_CONSENSUS, COLOR_CONFLICT]
    wedges, texts, autotexts = axes[2].pie(
        status_counts.values,
        labels=status_counts.index,
        autopct="%1.1f%%",
        colors=colors_pie,
        startangle=90,
        pctdistance=0.78,
        wedgeprops=dict(edgecolor="white", linewidth=2.5),
    )
    for at in autotexts:
        at.set_fontsize(13); at.set_fontweight("bold")
    axes[2].set_title(
        f"Consensus / Conflict Oranı\n(n={len(df_h)})",
        fontsize=13
    )

    plt.tight_layout()
    path = os.path.join(save_dir, "H3_model_karsilastirma.png")
    plt.savefig(path)
    plt.close()
    print(f"  ✓ H3_model_karsilastirma.png")

    return metrics_data


def plot_conflict_examples(df_h: pd.DataFrame, save_dir: str, n: int = 10):
    """
    Şekil 4: Conflict örnekleri — SVM ve Ollama'nın farklılaştığı
    en ilginç yorumları tablo olarak görselleştirir.
    """
    conflict_df = df_h[df_h["Hybrid_Status"] == "Conflict"].copy()
    ollama_wins = conflict_df[conflict_df["Ollama_Correct"] & ~conflict_df["SVM_Correct"]]
    svm_wins    = conflict_df[~conflict_df["Ollama_Correct"] & conflict_df["SVM_Correct"]]
    both_wrong  = conflict_df[~conflict_df["Ollama_Correct"] & ~conflict_df["SVM_Correct"]]

    fig, axes = plt.subplots(3, 1, figsize=(18, 14))
    fig.suptitle("Conflict Yorumları — Detay Tablosu", fontsize=15, fontweight="bold")

    sections = [
        (ollama_wins, "🟢 Ollama Kazanıyor (SVM Yanılıyor)", "#e8f5e9"),
        (svm_wins,    "🔵 SVM Kazanıyor (Ollama Yanılıyor)", "#e3f2fd"),
        (both_wrong,  "🔴 Her İkisi de Yanılıyor",           "#fce4ec"),
    ]

    for ax, (subset, title, bg_color) in zip(axes, sections):
        ax.axis("off")
        ax.set_facecolor(bg_color)
        sample = subset.head(n)
        if len(sample) == 0:
            ax.text(0.5, 0.5, f"{title}\n(Örnek bulunamadı)",
                    ha="center", va="center", fontsize=12,
                    transform=ax.transAxes)
            ax.set_title(title, pad=8, loc="left")
            continue

        rows = []
        for _, row in sample.iterrows():
            text_short = str(row["text"])[:70] + ("..." if len(str(row["text"])) > 70 else "")
            rows.append([
                text_short,
                row["True_Label"],
                row["SVM_Prediction"],
                row["Ollama_Prediction"],
            ])

        col_labels = ["Yorum (kısaltılmış)", "Gerçek", "SVM", "Ollama"]
        tbl = ax.table(
            cellText=rows, colLabels=col_labels,
            loc="center", cellLoc="left"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.5)

        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#37474f")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor(bg_color)
            cell.set_edgecolor("#cccccc")

        ax.set_title(f"{title}  (n={len(subset)})", pad=10, loc="left",
                     fontsize=11, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(save_dir, "H4_conflict_ornekleri.png")
    plt.savefig(path)
    plt.close()
    print(f"  ✓ H4_conflict_ornekleri.png")


def plot_confidence_distribution(df_h: pd.DataFrame, save_dir: str):
    """
    Şekil 5: Consensus vs Conflict durumuna göre modellerin
    sınıf dağılımını gösteren çubuk grafik.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Consensus & Conflict — Gerçek Etiket Dağılımı",
                 fontsize=14, fontweight="bold")

    for ax, status, color in [
        (axes[0], "Consensus", COLOR_CONSENSUS),
        (axes[1], "Conflict",  COLOR_CONFLICT),
    ]:
        subset = df_h[df_h["Hybrid_Status"] == status]
        if len(subset) == 0:
            ax.text(0.5, 0.5, "Örnek bulunamadı",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{status} (n=0)")
            continue

        true_dist = subset["True_Label"].value_counts()
        bars = ax.bar(true_dist.index, true_dist.values,
                      color=[PALETTE.get(l, "#607D8B") for l in true_dist.index],
                      alpha=0.85, edgecolor="white", linewidth=1.5, width=0.45)
        for bar, v in zip(bars, true_dist.values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    str(v), ha="center", fontweight="bold", fontsize=13)

        acc = (subset["Hybrid_Prediction"] == subset["True_Label"]).mean()
        ax.set_title(f"{status} (n={len(subset)})\nDoğruluk: {acc:.3f}", color=color)
        ax.set_ylabel("Yorum Sayısı")
        ax.set_ylim(0, true_dist.max() * 1.2)

    plt.tight_layout()
    path = os.path.join(save_dir, "H5_consensus_conflict_dagilim.png")
    plt.savefig(path)
    plt.close()
    print(f"  ✓ H5_consensus_conflict_dagilim.png")


def save_hybrid_report(df_h: pd.DataFrame, metrics_data: dict, save_dir: str):
    """
    Hibrit değerlendirme sonuçlarını CSV ve TXT olarak kaydeder.
    """
    # CSV
    csv_path = os.path.join(save_dir, "hybrid_results.csv")
    df_h.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  ✓ hybrid_results.csv")

    # TXT Raporu
    txt_path = os.path.join(save_dir, "hybrid_summary.txt")
    consensus_n = (df_h["Hybrid_Status"] == "Consensus").sum()
    conflict_n  = (df_h["Hybrid_Status"] == "Conflict").sum()
    total_n     = len(df_h)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("  HİBRİT KARAR AJANI — ÖZET RAPOR\n")
        f.write("=" * 65 + "\n\n")

        f.write(f"Örnek Boyutu   : {total_n}\n")
        f.write(f"Ollama Modeli  : {OLLAMA_MODEL}\n")
        f.write(f"Consensus      : {consensus_n} ({consensus_n/total_n*100:.1f}%)\n")
        f.write(f"Conflict       : {conflict_n}  ({conflict_n/total_n*100:.1f}%)\n\n")

        f.write("-" * 65 + "\n")
        f.write(f"{'Model':<20} {'Accuracy':>9} {'F1':>9} {'Precision':>10} {'Recall':>8}\n")
        f.write("-" * 65 + "\n")
        for mname, mdata in metrics_data.items():
            clean_name = mname.replace("\n", " ")
            f.write(
                f"{clean_name:<20} {mdata['Accuracy']:>9.4f} {mdata['F1']:>9.4f} "
                f"{mdata['Precision']:>10.4f} {mdata['Recall']:>8.4f}\n"
            )
        f.write("=" * 65 + "\n")

        # Conflict örüntüleri
        conflict_df = df_h[df_h["Hybrid_Status"] == "Conflict"]
        if len(conflict_df) > 0:
            ollama_wins = (conflict_df["Ollama_Correct"] & ~conflict_df["SVM_Correct"]).sum()
            svm_wins    = (~conflict_df["Ollama_Correct"] & conflict_df["SVM_Correct"]).sum()
            both_wrong  = (~conflict_df["Ollama_Correct"] & ~conflict_df["SVM_Correct"]).sum()

            f.write(f"\nCONFLICT ANALİZİ (n={len(conflict_df)}):\n")
            f.write(f"  Ollama Kazandı  : {ollama_wins}\n")
            f.write(f"  SVM Kazandı     : {svm_wins}\n")
            f.write(f"  Her ikisi yanlış: {both_wrong}\n")

    print(f"  ✓ hybrid_summary.txt")


# =============================================================================
# ANA AKIŞ
# =============================================================================

if __name__ == "__main__":

    # ─── Ollama bağlantısını kontrol et ──────────────────────────────────────
    print("\n[B] Ollama bağlantısı test ediliyor...")
    try:
        test_result = query_ollama_single("Bu harika bir gün!")
        if test_result in ["Pozitif", "Negatif"]:
            print(f"  ✓ Ollama bağlantısı başarılı → Test yanıt: {test_result}")
        else:
            print(f"  ⚠ Beklenmeyen yanıt: '{test_result}' — devam ediliyor...")
    except Exception as e:
        print(f"  ✗ OLLAMA HATASI: {e}")
        print("  Lütfen 'ollama serve' komutunu çalıştırdığınızdan emin olun.")
        print(f"  Model: ollama pull {OLLAMA_MODEL}")
        raise SystemExit(1)

    # ─── Hibrit DataFrame oluştur ─────────────────────────────────────────────
    df_hybrid = build_hybrid_dataframe(
        texts_raw       = X_sample_raw,
        svm_predictions = svm_sample_preds,
        true_labels     = y_sample,
        label_encoder   = le,
        ollama_model    = OLLAMA_MODEL,
    )

    # ─── İstatistiksel özet ───────────────────────────────────────────────────
    print("\n[D] Hibrit Analiz Özeti:")
    total      = len(df_hybrid)
    consensus  = (df_hybrid["Hybrid_Status"] == "Consensus").sum()
    conflict   = (df_hybrid["Hybrid_Status"] == "Conflict").sum()
    svm_acc_s  = df_hybrid["SVM_Correct"].mean()
    olm_acc_s  = df_hybrid["Ollama_Correct"].mean()
    hyb_acc_s  = df_hybrid["Hybrid_Correct"].mean()

    print(f"  Toplam        : {total}")
    print(f"  Consensus     : {consensus} ({consensus/total*100:.1f}%)")
    print(f"  Conflict      : {conflict}  ({conflict/total*100:.1f}%)")
    print(f"  SVM Doğruluk  : {svm_acc_s:.4f}")
    print(f"  Ollama Doğruluk: {olm_acc_s:.4f}")
    print(f"  Hibrit Doğruluk: {hyb_acc_s:.4f}")

    # ─── Görselleştirmeler ────────────────────────────────────────────────────
    print("\n[E] Görseller oluşturuluyor...")
    plot_venn_like_comparison(df_hybrid, HYBRID_DIR)
    plot_conflict_heatmap(df_hybrid, HYBRID_DIR)
    metrics_data = plot_model_comparison_bar(df_hybrid, HYBRID_DIR)
    plot_conflict_examples(df_hybrid, HYBRID_DIR)
    plot_confidence_distribution(df_hybrid, HYBRID_DIR)
    save_hybrid_report(df_hybrid, metrics_data, HYBRID_DIR)

    print("\n" + "=" * 65)
    print(f"  ✅ TAMAMLANDI")
    print(f"     Hibrit Doğruluk : {hyb_acc_s:.4f}")
    print(f"     SVM Doğruluk    : {svm_acc_s:.4f}")
    print(f"     Ollama Doğruluk : {olm_acc_s:.4f}")
    print(f"     Çıktı dizini    : {HYBRID_DIR}/")
    print("=" * 65)
