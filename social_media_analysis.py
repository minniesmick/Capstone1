"""
=============================================================================
Social Media Comment Sentiment Analysis
Turkish Text Classification: Pozitif vs Negatif
=============================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS & SETUP
# ──────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns

from collections import Counter
from wordcloud import WordCloud

# Scikit-learn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_curve, auc,
    ConfusionMatrixDisplay, roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

import xgboost as xgb

# ─── Output directory ────────────────────────────────────────────────────────
# Burayı senin yerel klasörüne uygun hale getiriyoruz
OUTPUT_DIR = "outputs" 
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Style ───────────────────────────────────────────────────────────────────
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

PALETTE   = {"Pozitif": "#4CAF50", "Negatif": "#F44336"}
COLOR_POS = "#4CAF50"
COLOR_NEG = "#F44336"
COLOR_A   = "#2196F3"
COLOR_B   = "#FF9800"

print("=" * 65)
print("  SOCIAL MEDIA SENTIMENT ANALYSIS — TÜRKÇE")
print("=" * 65)

# ──────────────────────────────────────────────────────────────────────────────
# 1. VERİ YÜKLEME
# ──────────────────────────────────────────────────────────────────────────────
print("\n[1/7] Veri yükleniyor...")

df = pd.read_csv("social_media_comments.csv", encoding="iso-8859-9")
df.columns = ["label", "text"]
df = df.dropna(subset=["text"]).reset_index(drop=True)

print(f"  Toplam kayıt : {len(df):,}")
print(f"  Pozitif      : {(df['label']=='Pozitif').sum():,}")
print(f"  Negatif      : {(df['label']=='Negatif').sum():,}")

# ──────────────────────────────────────────────────────────────────────────────
# 2. KEŞİFSEL VERİ ANALİZİ (EDA) + VİZÜELLER
# ──────────────────────────────────────────────────────────────────────────────
print("\n[2/7] Keşifsel Veri Analizi (EDA)...")

# ── Metin istatistikleri hesapla ──────────────────────────────────────────────
df["text_str"]      = df["text"].astype(str)
df["char_count"]    = df["text_str"].apply(len)
df["word_count"]    = df["text_str"].apply(lambda x: len(x.split()))
df["unique_words"]  = df["text_str"].apply(lambda x: len(set(x.lower().split())))
df["avg_word_len"]  = df["text_str"].apply(
    lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
)

# ──────────────────────────────────────────────────────────────────────────────
# FİGÜR 1 — Genel Bakış (2×3)
# ──────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)
fig.suptitle("Sosyal Medya Yorumları — Keşifsel Veri Analizi", fontsize=17,
             fontweight="bold", y=1.01)

# — 1a) Sınıf Dağılımı (Pie) ——————————————————————————————
ax1 = fig.add_subplot(gs[0, 0])
counts  = df["label"].value_counts()
colors  = [PALETTE[c] for c in counts.index]
wedges, texts, autotexts = ax1.pie(
    counts.values, labels=counts.index,
    autopct="%1.1f%%", colors=colors,
    startangle=140, pctdistance=0.78,
    wedgeprops=dict(edgecolor="white", linewidth=2)
)
for t in autotexts: t.set_fontsize(12); t.set_fontweight("bold")
ax1.set_title("Sınıf Dağılımı")

# — 1b) Sınıf Dağılımı (Bar) ——————————————————————————————
ax2 = fig.add_subplot(gs[0, 1])
bars = ax2.bar(counts.index, counts.values,
               color=[PALETTE[c] for c in counts.index],
               width=0.5, edgecolor="white", linewidth=1.5)
for bar, v in zip(bars, counts.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f"{v:,}", ha="center", va="bottom", fontweight="bold", fontsize=13)
ax2.set_title("Sınıf Frekansları")
ax2.set_ylabel("Yorum Sayısı")
ax2.set_ylim(0, counts.max() * 1.18)

# — 1c) Kelime Sayısı Dağılımı ————————————————————————————
ax3 = fig.add_subplot(gs[0, 2])
for lbl, col in PALETTE.items():
    subset = df[df["label"] == lbl]["word_count"]
    subset_clipped = subset.clip(upper=60)
    ax3.hist(subset_clipped, bins=35, alpha=0.65, color=col,
             label=lbl, edgecolor="white", linewidth=0.5)
ax3.set_title("Kelime Sayısı Dağılımı")
ax3.set_xlabel("Kelime Sayısı (≤60)")
ax3.set_ylabel("Frekans")
ax3.legend()

# — 1d) Karakter Sayısı Dağılımı ——————————————————————————
ax4 = fig.add_subplot(gs[1, 0])
for lbl, col in PALETTE.items():
    subset = df[df["label"] == lbl]["char_count"].clip(upper=300)
    ax4.hist(subset, bins=35, alpha=0.65, color=col,
             label=lbl, edgecolor="white", linewidth=0.5)
ax4.set_title("Karakter Sayısı Dağılımı")
ax4.set_xlabel("Karakter Sayısı (≤300)")
ax4.set_ylabel("Frekans")
ax4.legend()

# — 1e) Boxplot: Kelime Sayısı vs Sınıf ——————————————————
ax5 = fig.add_subplot(gs[1, 1])
data_box = [df[df["label"] == l]["word_count"].clip(upper=60).values
            for l in ["Pozitif", "Negatif"]]
bp = ax5.boxplot(data_box, labels=["Pozitif", "Negatif"],
                 patch_artist=True, notch=True, widths=0.45,
                 medianprops=dict(color="white", linewidth=2.5))
for patch, col in zip(bp["boxes"], [COLOR_POS, COLOR_NEG]):
    patch.set_facecolor(col); patch.set_alpha(0.75)
ax5.set_title("Kelime Sayısı (Sınıfa Göre)")
ax5.set_ylabel("Kelime Sayısı")

# — 1f) İstatistik Özet Tablosu ———————————————————————————
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
summary = df.groupby("label").agg(
    Adet=("text", "count"),
    Ort_Kelime=("word_count", lambda x: f"{x.mean():.1f}"),
    Med_Kelime=("word_count", lambda x: f"{x.median():.0f}"),
    Ort_Karakter=("char_count", lambda x: f"{x.mean():.0f}"),
    Ort_UniqueK=("unique_words", lambda x: f"{x.mean():.1f}"),
).reset_index()

col_labels = ["Sınıf", "Adet", "Ort.\nKelime", "Med.\nKelime", "Ort.\nKarakter", "Ort.\nUnique"]
table_data = summary.values.tolist()
table = ax6.table(cellText=table_data, colLabels=col_labels,
                  loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.1, 1.8)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
    elif row == 1:
        cell.set_facecolor("#d5f5e3")
    else:
        cell.set_facecolor("#fde8e8")
    cell.set_edgecolor("#cccccc")
ax6.set_title("İstatistik Özeti", pad=10)

plt.savefig(f"{OUTPUT_DIR}/01_EDA_genel_bakis.png")
plt.close()
print("  ✓ 01_EDA_genel_bakis.png")

# ──────────────────────────────────────────────────────────────────────────────
# FİGÜR 2 — WordCloud (Pozitif & Negatif)
# ──────────────────────────────────────────────────────────────────────────────
# Türkçe stop words
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

def make_wordcloud(text_series, title, colormap, bg="white"):
    text = " ".join(text_series.astype(str).str.lower())
    wc = WordCloud(
        width=900, height=500, background_color=bg,
        colormap=colormap, max_words=150,
        stopwords=TR_STOPWORDS, collocations=False,
        min_word_length=3
    ).generate(text)
    return wc

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("En Sık Kullanılan Kelimeler — WordCloud", fontsize=16, fontweight="bold")

wc_pos = make_wordcloud(df[df["label"]=="Pozitif"]["text_str"], "Pozitif", "Greens")
axes[0].imshow(wc_pos, interpolation="bilinear")
axes[0].set_title("Pozitif Yorumlar", fontsize=14, color=COLOR_POS, fontweight="bold")
axes[0].axis("off")

wc_neg = make_wordcloud(df[df["label"]=="Negatif"]["text_str"], "Negatif", "Reds")
axes[1].imshow(wc_neg, interpolation="bilinear")
axes[1].set_title("Negatif Yorumlar", fontsize=14, color=COLOR_NEG, fontweight="bold")
axes[1].axis("off")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_wordcloud.png")
plt.close()
print("  ✓ 02_wordcloud.png")

# ──────────────────────────────────────────────────────────────────────────────
# FİGÜR 3 — Top-N Kelimeler (Bar chart)
# ──────────────────────────────────────────────────────────────────────────────
def get_top_words(series, n=20):
    tokens = []
    for t in series.astype(str):
        for w in re.sub(r"[^a-zA-ZığüşöçİĞÜŞÖÇa-z]", " ", t.lower()).split():
            if len(w) > 2 and w not in TR_STOPWORDS:
                tokens.append(w)
    return Counter(tokens).most_common(n)

top_pos = get_top_words(df[df["label"]=="Pozitif"]["text_str"])
top_neg = get_top_words(df[df["label"]=="Negatif"]["text_str"])

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("En Sık Kullanılan 20 Kelime (Stop-words hariç)", fontsize=15, fontweight="bold")

for ax, top_words, title, color in [
    (axes[0], top_pos, "Pozitif Yorumlar", COLOR_POS),
    (axes[1], top_neg, "Negatif Yorumlar", COLOR_NEG)
]:
    words, freqs = zip(*top_words)
    y_pos = range(len(words))
    bars = ax.barh(y_pos, freqs, color=color, alpha=0.82, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Frekans")
    ax.set_title(title, color=color, fontweight="bold")
    for bar, freq in zip(bars, freqs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                str(freq), va="center", fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_top_kelimeler.png")
plt.close()
print("  ✓ 03_top_kelimeler.png")

# ──────────────────────────────────────────────────────────────────────────────
# 3. VERİ ÖN İŞLEME
# ──────────────────────────────────────────────────────────────────────────────
print("\n[3/7] Veri Ön İşleme...")

def preprocess(text):
    text = str(text).lower()
    # URL, mention, hashtag, sayı temizle
    text = re.sub(r"http\S+|www\S+",        " ", text)
    text = re.sub(r"@\w+",                  " ", text)
    text = re.sub(r"#\w+",                  " ", text)
    text = re.sub(r"\d+",                   " ", text)
    # Türkçe olmayan karakterleri kaldır (Türkçe harfler korunur)
    text = re.sub(r"[^a-zığüşöçİĞÜŞÖÇ\s]", " ", text)
    # Tekrarlanan boşlukları temizle
    text = re.sub(r"\s+",                   " ", text).strip()
    # Stop words kaldır
    tokens = [w for w in text.split() if w not in TR_STOPWORDS and len(w) > 1]
    return " ".join(tokens)

df["clean_text"] = df["text_str"].apply(preprocess)

# Ön işleme istatistikleri
df["clean_word_count"] = df["clean_text"].apply(lambda x: len(x.split()))
print(f"  Ham ortalama kelime  : {df['word_count'].mean():.1f}")
print(f"  Temiz ortalama kelime: {df['clean_word_count'].mean():.1f}")
print(f"  Azaltma oranı        : {(1 - df['clean_word_count'].mean()/df['word_count'].mean())*100:.1f}%")

# ──────────────────────────────────────────────────────────────────────────────
# FİGÜR 4 — Ön İşleme Karşılaştırması
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Ön İşleme Öncesi vs Sonrası Karşılaştırması", fontsize=15, fontweight="bold")

# Ham vs Temiz — Kelime Sayısı Dağılımı
for col, label, ls in [("word_count","Ham","--"), ("clean_word_count","Temiz","-")]:
    clipped = df[col].clip(upper=50)
    axes[0].hist(clipped, bins=30, alpha=0.60, label=label,
                 edgecolor="white", linewidth=0.4)
axes[0].set_title("Kelime Sayısı Dağılımı")
axes[0].set_xlabel("Kelime Sayısı (≤50)")
axes[0].set_ylabel("Frekans")
axes[0].legend()

# Ortalama kelime sayıları (grouped bar)
labels_g = ["Pozitif", "Negatif"]
raw_means  = [df[df["label"]==l]["word_count"].mean()      for l in labels_g]
clean_means= [df[df["label"]==l]["clean_word_count"].mean() for l in labels_g]
x = np.arange(len(labels_g)); w = 0.35
axes[1].bar(x - w/2, raw_means,   w, label="Ham",   color=COLOR_A, alpha=0.8, edgecolor="white")
axes[1].bar(x + w/2, clean_means, w, label="Temiz", color=COLOR_B, alpha=0.8, edgecolor="white")
axes[1].set_title("Ortalama Kelime (Sınıfa Göre)")
axes[1].set_xticks(x); axes[1].set_xticklabels(labels_g)
axes[1].set_ylabel("Ortalama Kelime Sayısı")
axes[1].legend()
for xi, (r, c) in enumerate(zip(raw_means, clean_means)):
    axes[1].text(xi - w/2, r + 0.2, f"{r:.1f}", ha="center", fontsize=9)
    axes[1].text(xi + w/2, c + 0.2, f"{c:.1f}", ha="center", fontsize=9)

# Ön işleme adımları tablosu
axes[2].axis("off")
steps = [
    ["Adım", "İşlem"],
    ["1", "Küçük harfe dönüştürme"],
    ["2", "URL, mention, hashtag kaldırma"],
    ["3", "Rakam kaldırma"],
    ["4", "Noktalama / özel karakter kaldırma"],
    ["5", "Stop-word kaldırma"],
    ["6", "Gereksiz boşluk temizleme"],
    ["7", "TF-IDF Vektörizasyon"],
]
tbl = axes[2].table(cellText=steps[1:], colLabels=steps[0],
                    loc="center", cellLoc="left")
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.2, 1.6)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#eaf6fb")
    cell.set_edgecolor("#cccccc")
axes[2].set_title("Ön İşleme Adımları", pad=12)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_onisleme.png")
plt.close()
print("  ✓ 04_onisleme.png")

# ──────────────────────────────────────────────────────────────────────────────
# 4. TRAIN/TEST SPLIT & FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────
print("\n[4/7] Feature Extraction & Split...")

le = LabelEncoder()
y  = le.fit_transform(df["label"])   # Negatif=0, Pozitif=1

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df["clean_text"], y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Train: {len(X_train_raw):,} | Test: {len(X_test_raw):,}")

# TF-IDF (karakter n-gram + kelime n-gram birleşik)
tfidf = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=30000,
    sublinear_tf=True,
    min_df=2,
)
X_train = tfidf.fit_transform(X_train_raw)
X_test  = tfidf.transform(X_test_raw)

print(f"  TF-IDF features: {X_train.shape[1]:,}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. MODEL EĞİTİMİ & DEĞERLENDİRME
# ──────────────────────────────────────────────────────────────────────────────
print("\n[5/7] Model Eğitimi...")

MODELS = {
    "Logistic Regression": LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", random_state=42
    ),
    "Multinomial NB": MultinomialNB(
        alpha=0.5
    ),
    "Linear SVM": LinearSVC(
        C=1.0, max_iter=2000, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=5,
        n_jobs=-1, random_state=42
    ),
    "XGBoost": xgb.XGBClassifier(
        tree_method='hist',     # GPU hızlandırmalı histogram metodunu kullanır
        device='cuda',          # RTX 3060 Ti'ı (CUDA) aktif eder
        n_estimators=200, 
        max_depth=6, 
        learning_rate=0.1,
        subsample=0.8, 
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42, 
        verbosity=0
    ),
}

# Cross-validation (5-fold, train üzerinde)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}
for name, model in MODELS.items():
    print(f"  Eğitiliyor: {name}...", end=" ", flush=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ROC için olasılık/decision function
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X_test)
        y_prob = (raw - raw.min()) / (raw.max() - raw.min())
    else:
        y_prob = y_pred.astype(float)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted")
    prec = precision_score(y_test, y_pred, average="weighted")
    rec  = recall_score(y_test, y_pred, average="weighted")
    roc  = roc_auc_score(y_test, y_prob)

    # CV skoru
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                scoring="accuracy", n_jobs=-1)

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "roc_auc": roc,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "report": classification_report(y_test, y_pred,
                                        target_names=le.classes_)
    }
    print(f"Acc={acc:.4f} | F1={f1:.4f} | ROC-AUC={roc:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# FİGÜR 5 — Model Karşılaştırma (Grouped Bar + CV)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[6/7] Görseller oluşturuluyor...")

model_names = list(results.keys())
metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
metric_labels = ["Accuracy", "F1 Score", "Precision", "Recall", "ROC-AUC"]

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Model Karşılaştırması — Test Seti Sonuçları", fontsize=17, fontweight="bold")
axes = axes.flatten()

colors_models = plt.cm.tab10(np.linspace(0, 0.9, len(model_names)))

# 5 metrik için bar grafikleri
for i, (met, mlabel) in enumerate(zip(metrics, metric_labels)):
    vals = [results[n][met] for n in model_names]
    bars = axes[i].bar(range(len(model_names)), vals,
                       color=colors_models, alpha=0.85, edgecolor="white", linewidth=1.2)
    axes[i].set_xticks(range(len(model_names)))
    axes[i].set_xticklabels([n.replace(" ", "\n") for n in model_names], fontsize=9)
    axes[i].set_ylim(min(vals)*0.96, 1.01)
    axes[i].set_ylabel(mlabel)
    axes[i].set_title(f"{mlabel} Karşılaştırması")
    for bar, v in zip(bars, vals):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    best_idx = int(np.argmax(vals))
    axes[i].get_children()[best_idx].set_edgecolor("gold")
    axes[i].get_children()[best_idx].set_linewidth(3)

# Cross-validation sonuçları (son panel)
ax_cv = axes[5]
cv_means = [results[n]["cv_mean"] for n in model_names]
cv_stds  = [results[n]["cv_std"]  for n in model_names]
bars_cv = ax_cv.bar(range(len(model_names)), cv_means,
                    yerr=cv_stds, capsize=6,
                    color=colors_models, alpha=0.85,
                    edgecolor="white", linewidth=1.2,
                    error_kw=dict(ecolor="black", elinewidth=1.5))
ax_cv.set_xticks(range(len(model_names)))
ax_cv.set_xticklabels([n.replace(" ", "\n") for n in model_names], fontsize=9)
ax_cv.set_ylim(min(cv_means)*0.96, 1.01)
ax_cv.set_title("5-Fold Cross Validation (Train)")
ax_cv.set_ylabel("CV Accuracy (±std)")
for bar, m, s in zip(bars_cv, cv_means, cv_stds):
    ax_cv.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.002,
               f"{m:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_model_karsilastirma.png")
plt.close()
print("  ✓ 05_model_karsilastirma.png")

# ──────────────────────────────────────────────────────────────────────────────
# FİGÜR 6 — Confusion Matrices (tüm modeller, 2×3)
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Confusion Matrix — Tüm Modeller (Test Seti)", fontsize=16, fontweight="bold")
axes = axes.flatten()

cmaps = ["Greens", "Blues", "Purples", "Oranges", "Reds", "YlOrBr"]
for i, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=axes[i], colorbar=False, cmap=cmaps[i], values_format="d")
    axes[i].set_title(f"{name}\nAcc={res['accuracy']:.3f} | F1={res['f1']:.3f}",
                      fontsize=11)
    axes[i].set_xlabel("Tahmin")
    axes[i].set_ylabel("Gerçek")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_confusion_matrices.png")
plt.close()
print("  ✓ 06_confusion_matrices.png")

# ──────────────────────────────────────────────────────────────────────────────
# FİGÜR 7 — ROC Eğrileri (tüm modeller üst üste)
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot([0,1],[0,1], "k--", alpha=0.4, label="Random (AUC=0.50)")

for (name, res), col in zip(results.items(), colors_models):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    roc_val = res["roc_auc"]
    ax.plot(fpr, tpr, linewidth=2.2, color=col,
            label=f"{name} (AUC={roc_val:.3f})")

ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate", fontsize=13)
ax.set_title("ROC Eğrileri — Tüm Modeller", fontsize=15, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.25)
ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_roc_curves.png")
plt.close()
print("  ✓ 07_roc_curves.png")

# ──────────────────────────────────────────────────────────────────────────────
# FİGÜR 8 — Özet Radar / Heatmap Tablosu
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle("Model Performans Özeti", fontsize=15, fontweight="bold")

# Heatmap
metric_keys = ["accuracy","f1","precision","recall","roc_auc","cv_mean"]
metric_names = ["Accuracy","F1","Precision","Recall","ROC-AUC","CV-Acc"]
heatmap_data = pd.DataFrame(
    {name: [results[name][m] for m in metric_keys] for name in model_names},
    index=metric_names
)
sns.heatmap(heatmap_data, ax=axes[0], annot=True, fmt=".3f",
            cmap="RdYlGn", vmin=0.8, vmax=1.0,
            linewidths=0.5, linecolor="white",
            annot_kws={"size": 11, "weight": "bold"})
axes[0].set_title("Metrik Heatmap")
axes[0].set_xticklabels([n.replace(" ", "\n") for n in model_names], fontsize=9)
axes[0].tick_params(axis="y", labelsize=10)

# Radar chart
from matplotlib.patches import FancyArrowPatch
num_metrics = len(metric_names)
angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
angles += angles[:1]

ax_r = axes[1]
ax_r.remove()
ax_r = fig.add_subplot(1, 2, 2, polar=True)

for (name, res), col in zip(results.items(), colors_models):
    vals = [res[m] for m in metric_keys]
    vals += vals[:1]
    ax_r.plot(angles, vals, "o-", linewidth=2, color=col, label=name, markersize=4)
    ax_r.fill(angles, vals, alpha=0.05, color=col)

ax_r.set_xticks(angles[:-1])
ax_r.set_xticklabels(metric_names, size=10)
ax_r.set_ylim(0.7, 1.0)
ax_r.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
ax_r.set_yticklabels(["0.75","0.80","0.85","0.90","0.95","1.00"], size=8)
ax_r.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
ax_r.set_title("Radar Chart — Metrik Karşılaştırması", pad=15, fontsize=12, fontweight="bold")
ax_r.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_ozet_heatmap_radar.png")
plt.close()
print("  ✓ 08_ozet_heatmap_radar.png")

# ──────────────────────────────────────────────────────────────────────────────
# FİGÜR 9 — En iyi modelin TF-IDF feature importance
# ──────────────────────────────────────────────────────────────────────────────
best_name = max(results, key=lambda x: results[x]["f1"])
best_model = results[best_name]["model"]

# Logistic Regression veya Linear SVM için coef kullan
coef_model_name = None
for candidate in ["Linear SVM", "Logistic Regression"]:
    if candidate in results:
        coef_model_name = candidate
        break

if coef_model_name:
    coef_model = results[coef_model_name]["model"]
    coef = coef_model.coef_[0] if hasattr(coef_model, "coef_") else None
    if coef is not None:
        feature_names = np.array(tfidf.get_feature_names_out())
        top_n = 20
        top_pos_idx = coef.argsort()[-top_n:][::-1]
        top_neg_idx = coef.argsort()[:top_n]

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f"En Etkili Kelimeler — {coef_model_name} (TF-IDF Katsayıları)",
                     fontsize=15, fontweight="bold")

        for ax, idx, title, color in [
            (axes[0], top_pos_idx, "Pozitif → En Yüksek Katkı", COLOR_POS),
            (axes[1], top_neg_idx, "Negatif → En Yüksek Katkı", COLOR_NEG),
        ]:
            words = feature_names[idx]
            vals  = coef[idx]
            y_pos = range(len(words))
            bars  = ax.barh(y_pos, vals, color=color, alpha=0.80, edgecolor="white")
            ax.set_yticks(y_pos); ax.set_yticklabels(words, fontsize=11)
            ax.invert_yaxis()
            ax.set_xlabel("TF-IDF Katsayısı")
            ax.set_title(title, color=color, fontweight="bold")
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/09_feature_importance.png")
        plt.close()
        print("  ✓ 09_feature_importance.png")

# ──────────────────────────────────────────────────────────────────────────────
# 7. SONUÇ RAPORU (TXT)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[7/7] Sonuç raporu yazılıyor...")

best_by_f1 = max(results, key=lambda x: results[x]["f1"])

with open(f"{OUTPUT_DIR}/10_model_sonuclari.txt", "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("  SOCIAL MEDIA SENTIMENT ANALİZİ — MODEL SONUÇLARI\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Veri Seti:\n")
    f.write(f"  Toplam Kayıt  : {len(df):,}\n")
    f.write(f"  Pozitif       : {(df['label']=='Pozitif').sum():,} ({(df['label']=='Pozitif').mean()*100:.1f}%)\n")
    f.write(f"  Negatif       : {(df['label']=='Negatif').sum():,} ({(df['label']=='Negatif').mean()*100:.1f}%)\n")
    f.write(f"  Train / Test  : {len(X_train_raw):,} / {len(X_test_raw):,}\n")
    f.write(f"  TF-IDF Özellik: {X_train.shape[1]:,}\n\n")

    f.write("-" * 70 + "\n")
    f.write(f"{'Model':<25} {'Accuracy':>9} {'F1':>9} {'Precision':>10} {'Recall':>8} {'ROC-AUC':>9} {'CV±std':>12}\n")
    f.write("-" * 70 + "\n")
    for name, res in results.items():
        marker = " ★" if name == best_by_f1 else ""
        f.write(
            f"{name+marker:<25} {res['accuracy']:>9.4f} {res['f1']:>9.4f} "
            f"{res['precision']:>10.4f} {res['recall']:>8.4f} {res['roc_auc']:>9.4f} "
            f"{res['cv_mean']:.4f}±{res['cv_std']:.4f}\n"
        )
    f.write("-" * 70 + "\n\n")

    f.write(f"\n{'='*70}\nDETAYLI SINIFLANDIRMA RAPORLARI\n{'='*70}\n\n")
    for name, res in results.items():
        f.write(f"\n{'─'*50}\n{name}\n{'─'*50}\n")
        f.write(res["report"])

    f.write(f"\n{'='*70}\nEN İYİ MODEL: {best_by_f1} (F1={results[best_by_f1]['f1']:.4f})\n{'='*70}\n")

# ──────────────────────────────────────────────────────────────────────────────
# FİGÜR 10 — Özet Sıralama Tablosu
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
ax.axis("off")

table_data = []
sorted_results = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)
for rank, (name, res) in enumerate(sorted_results, 1):
    table_data.append([
        f"#{rank}",
        name,
        f"{res['accuracy']:.4f}",
        f"{res['f1']:.4f}",
        f"{res['precision']:.4f}",
        f"{res['recall']:.4f}",
        f"{res['roc_auc']:.4f}",
        f"{res['cv_mean']:.4f}±{res['cv_std']:.4f}",
    ])

cols = ["Rank", "Model", "Accuracy", "F1", "Precision", "Recall", "ROC-AUC", "CV (5-fold)"]
tbl = ax.table(cellText=table_data, colLabels=cols,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.1, 2.2)

header_color = "#2c3e50"
rank_colors  = ["#ffd700", "#c0c0c0", "#cd7f32"]  # gold, silver, bronze

for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")
    elif row <= 3:
        cell.set_facecolor(rank_colors[row - 1] if row <= 3 else "#f5f5f5")
        cell.set_alpha(0.55)
    else:
        cell.set_facecolor("#f5f5f5")
    cell.set_edgecolor("#dddddd")

ax.set_title("Model Performans Sıralaması (F1 Skoruna Göre)", fontsize=14,
             fontweight="bold", pad=15)
fig.suptitle("★ = En İyi Model", fontsize=11, color="#888", y=0.02)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/11_final_siralama.png")
plt.close()
print("  ✓ 11_final_siralama.png")

# ──────────────────────────────────────────────────────────────────────────────
# TAMAMLANDI
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"  ✅ TAMAMLANDI — En İyi Model: {best_by_f1}")
print(f"     F1={results[best_by_f1]['f1']:.4f} | Acc={results[best_by_f1]['accuracy']:.4f} | ROC-AUC={results[best_by_f1]['roc_auc']:.4f}")
print(f"\n  Çıktı dizini: {OUTPUT_DIR}/")
print("=" * 65)
