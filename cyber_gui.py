"""
╔══════════════════════════════════════════════════════════════════╗
║        HYBRID DECISION AGENT — CYBERPUNK GUI                     ║
║        SVM + Ollama LLM  |  PySide6 + QSS Dark Neon             ║
║        v2.1 — Multi-Model Comparator Module Added                ║
╚══════════════════════════════════════════════════════════════════╝
Kullanım:
    pip install -r requirements_gui.txt
    python cyber_gui.py
"""

import sys
import os
import re
import json
import time
import warnings
import subprocess
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QSlider, QSpinBox,
    QComboBox, QProgressBar, QTabWidget, QTextEdit, QScrollArea,
    QFrame, QSplitter, QGroupBox, QGridLayout, QSizePolicy,
    QMessageBox, QDoubleSpinBox, QDialog, QCheckBox, QListWidget,
    QListWidgetItem, QDialogButtonBox, QAbstractItemView
)
from PySide6.QtCore import (
    Qt, QThread, Signal, QTimer, QPropertyAnimation,
    QEasingCurve, QRect, QSize
)
from PySide6.QtGui import (
    QFont, QPixmap, QPainter, QColor, QLinearGradient,
    QDragEnterEvent, QDropEvent, QPalette, QIcon
)

# ─── Cyberpunk Renk Paleti ────────────────────────────────────────────────────
CYBER_BG        = "#0a0a0f"       # Derin siyah
CYBER_BG2       = "#0f0f1a"       # Panel arka planı
CYBER_BG3       = "#13131f"       # Kart arka planı
CYBER_BLUE      = "#00d4ff"       # Neon mavi
CYBER_PURPLE    = "#bf5fff"       # Neon mor
CYBER_RED       = "#ff2a6d"       # Neon kırmızı/pembe
CYBER_CYAN      = "#05ffa1"       # Neon yeşil-cyan (aksan)
CYBER_ORANGE    = "#ff9f1c"       # Uyarı / karşılaştırma rengi
CYBER_YELLOW    = "#f5e642"       # Neon sarı (karşılaştırma aksan)
CYBER_TEXT      = "#e0e0ff"       # Ana metin
CYBER_TEXT_DIM  = "#6060a0"       # Soluk metin
CYBER_BORDER    = "#1a1a3a"       # Kenarlık

# ─── Global QSS Stylesheet ───────────────────────────────────────────────────
CYBERPUNK_QSS = f"""
/* ═══════════════════ GLOBAL ═══════════════════ */
QMainWindow, QWidget {{
    background-color: {CYBER_BG};
    color: {CYBER_TEXT};
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
}}

/* ═══════════════════ GROUP BOX ═══════════════════ */
QGroupBox {{
    background-color: {CYBER_BG3};
    border: 1px solid {CYBER_BORDER};
    border-top: 2px solid {CYBER_BLUE};
    border-radius: 6px;
    margin-top: 14px;
    padding: 12px 8px 8px 8px;
    font-size: 11px;
    font-weight: bold;
    color: {CYBER_BLUE};
    letter-spacing: 1px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    left: 10px;
    top: -2px;
    color: {CYBER_BLUE};
    background-color: {CYBER_BG};
}}

/* ═══════════════════ BUTTONS ═══════════════════ */
QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #0a0a1f, stop:1 #0d0d25);
    color: {CYBER_BLUE};
    border: 1px solid {CYBER_BLUE};
    border-radius: 4px;
    padding: 7px 18px;
    font-family: "Consolas", monospace;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 1px;
    min-height: 28px;
}}
QPushButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #001a33, stop:1 #00264d);
    border-color: {CYBER_CYAN};
    color: {CYBER_CYAN};
}}
QPushButton:pressed {{
    background: {CYBER_BLUE};
    color: {CYBER_BG};
}}
QPushButton:disabled {{
    background: {CYBER_BG3};
    color: {CYBER_TEXT_DIM};
    border-color: {CYBER_BORDER};
}}

QPushButton#btn_run {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1a0033, stop:1 #33004d);
    color: {CYBER_PURPLE};
    border: 1px solid {CYBER_PURPLE};
    font-size: 13px;
    min-height: 42px;
    border-radius: 6px;
    letter-spacing: 2px;
}}
QPushButton#btn_run:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #260040, stop:1 #400060);
    border-color: {CYBER_CYAN};
    color: {CYBER_CYAN};
}}
QPushButton#btn_run:pressed {{
    background: {CYBER_PURPLE};
    color: {CYBER_BG};
}}

QPushButton#btn_stop {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #330011, stop:1 #4d0022);
    color: {CYBER_RED};
    border: 1px solid {CYBER_RED};
    font-size: 12px;
    min-height: 36px;
    border-radius: 5px;
}}
QPushButton#btn_stop:hover {{
    background: {CYBER_RED};
    color: {CYBER_BG};
}}

QPushButton#btn_compare {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1a0d00, stop:1 #2d1800);
    color: {CYBER_ORANGE};
    border: 1px solid {CYBER_ORANGE};
    font-size: 12px;
    min-height: 36px;
    border-radius: 5px;
    letter-spacing: 2px;
}}
QPushButton#btn_compare:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #2d1800, stop:1 #3d2000);
    border-color: {CYBER_YELLOW};
    color: {CYBER_YELLOW};
}}
QPushButton#btn_compare:pressed {{
    background: {CYBER_ORANGE};
    color: {CYBER_BG};
}}

QPushButton#btn_compare_run {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1a0d00, stop:1 #2d1800);
    color: {CYBER_ORANGE};
    border: 1px solid {CYBER_ORANGE};
    font-size: 13px;
    min-height: 42px;
    border-radius: 6px;
    letter-spacing: 2px;
}}
QPushButton#btn_compare_run:hover {{
    background: {CYBER_ORANGE};
    color: {CYBER_BG};
}}

/* ═══════════════════ LINE EDIT ═══════════════════ */
QLineEdit {{
    background-color: {CYBER_BG2};
    color: {CYBER_CYAN};
    border: 1px solid {CYBER_BORDER};
    border-radius: 4px;
    padding: 5px 8px;
    font-family: "Consolas", monospace;
    font-size: 11px;
    selection-background-color: {CYBER_PURPLE};
}}
QLineEdit:focus {{
    border-color: {CYBER_BLUE};
    background-color: #0c0c1e;
}}

/* ═══════════════════ COMBO BOX ═══════════════════ */
QComboBox {{
    background-color: {CYBER_BG2};
    color: {CYBER_TEXT};
    border: 1px solid {CYBER_BORDER};
    border-radius: 4px;
    padding: 5px 8px;
    font-family: "Consolas", monospace;
    font-size: 11px;
    min-height: 24px;
}}
QComboBox:hover, QComboBox:focus {{
    border-color: {CYBER_BLUE};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox::down-arrow {{
    width: 10px;
    height: 10px;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {CYBER_BLUE};
}}
QComboBox QAbstractItemView {{
    background-color: {CYBER_BG2};
    color: {CYBER_TEXT};
    border: 1px solid {CYBER_BLUE};
    selection-background-color: #1a0040;
    selection-color: {CYBER_PURPLE};
    outline: none;
}}

/* ═══════════════════ SLIDER ═══════════════════ */
QSlider::groove:horizontal {{
    border: 1px solid {CYBER_BORDER};
    height: 4px;
    background: {CYBER_BG2};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {CYBER_BLUE};
    border: 1px solid {CYBER_CYAN};
    width: 14px;
    height: 14px;
    border-radius: 7px;
    margin: -6px 0;
}}
QSlider::handle:horizontal:hover {{
    background: {CYBER_CYAN};
    border-color: {CYBER_BLUE};
}}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {CYBER_BLUE}, stop:1 {CYBER_PURPLE});
    border-radius: 2px;
    height: 4px;
}}

/* ═══════════════════ PROGRESS BAR ═══════════════════ */
QProgressBar {{
    background-color: {CYBER_BG2};
    border: 1px solid {CYBER_BORDER};
    border-radius: 6px;
    height: 22px;
    text-align: center;
    color: {CYBER_TEXT};
    font-family: "Consolas", monospace;
    font-size: 11px;
    font-weight: bold;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {CYBER_BLUE},
        stop:0.5 {CYBER_PURPLE},
        stop:1 {CYBER_RED});
    border-radius: 5px;
}}

/* ═══════════════════ TAB WIDGET ═══════════════════ */
QTabWidget::pane {{
    border: 1px solid {CYBER_BORDER};
    border-top: 2px solid {CYBER_PURPLE};
    background-color: {CYBER_BG2};
    border-radius: 0 4px 4px 4px;
}}
QTabBar::tab {{
    background: {CYBER_BG3};
    color: {CYBER_TEXT_DIM};
    border: 1px solid {CYBER_BORDER};
    border-bottom: none;
    padding: 7px 18px;
    font-family: "Consolas", monospace;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 1px;
    margin-right: 2px;
    border-radius: 4px 4px 0 0;
}}
QTabBar::tab:selected {{
    color: {CYBER_PURPLE};
    border-top: 2px solid {CYBER_PURPLE};
    background: {CYBER_BG2};
}}
QTabBar::tab:hover:!selected {{
    color: {CYBER_BLUE};
    background: #0c0c20;
}}

/* ═══════════════════ TEXT EDIT (LOG) ═══════════════════ */
QTextEdit {{
    background-color: #050510;
    color: {CYBER_CYAN};
    border: 1px solid {CYBER_BORDER};
    border-radius: 4px;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 11px;
    padding: 4px;
    selection-background-color: {CYBER_PURPLE};
}}

/* ═══════════════════ SPINBOX ═══════════════════ */
QSpinBox, QDoubleSpinBox {{
    background-color: {CYBER_BG2};
    color: {CYBER_CYAN};
    border: 1px solid {CYBER_BORDER};
    border-radius: 4px;
    padding: 4px 6px;
    font-family: "Consolas", monospace;
    font-size: 11px;
    min-height: 24px;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {CYBER_BLUE};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background: {CYBER_BG3};
    border: none;
    width: 16px;
}}

/* ═══════════════════ SCROLL BAR ═══════════════════ */
QScrollBar:vertical {{
    background: {CYBER_BG};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {CYBER_PURPLE};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: {CYBER_BLUE};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

/* ═══════════════════ FRAME (separator) ═══════════════════ */
QFrame[frameShape="4"] {{
    color: {CYBER_BORDER};
}}

/* ═══════════════════ LABEL ═══════════════════ */
QLabel#lbl_title {{
    color: {CYBER_PURPLE};
    font-size: 20px;
    font-weight: bold;
    letter-spacing: 3px;
}}
QLabel#lbl_subtitle {{
    color: {CYBER_BLUE};
    font-size: 11px;
    letter-spacing: 2px;
}}
QLabel#lbl_status {{
    color: {CYBER_CYAN};
    font-size: 11px;
    font-weight: bold;
}}
QLabel#lbl_metric {{
    color: {CYBER_CYAN};
    font-size: 18px;
    font-weight: bold;
}}
QLabel#lbl_metric_title {{
    color: {CYBER_TEXT_DIM};
    font-size: 10px;
    letter-spacing: 1px;
}}

/* ═══════════════════ SCROLL AREA ═══════════════════ */
QScrollArea {{
    border: none;
    background-color: transparent;
}}

/* ═══════════════════ CHECKBOX ═══════════════════ */
QCheckBox {{
    color: {CYBER_TEXT};
    font-family: "Consolas", monospace;
    font-size: 11px;
    spacing: 8px;
}}
QCheckBox:hover {{
    color: {CYBER_ORANGE};
}}
QCheckBox::indicator {{
    width: 15px;
    height: 15px;
    border: 1px solid {CYBER_BORDER};
    border-radius: 3px;
    background: {CYBER_BG2};
}}
QCheckBox::indicator:unchecked:hover {{
    border-color: {CYBER_ORANGE};
}}
QCheckBox::indicator:checked {{
    background: {CYBER_ORANGE};
    border-color: {CYBER_ORANGE};
    image: none;
}}

/* ═══════════════════ LIST WIDGET ═══════════════════ */
QListWidget {{
    background-color: {CYBER_BG2};
    color: {CYBER_TEXT};
    border: 1px solid {CYBER_BORDER};
    border-radius: 4px;
    font-family: "Consolas", monospace;
    font-size: 11px;
    outline: none;
}}
QListWidget::item {{
    padding: 5px 8px;
    border-bottom: 1px solid {CYBER_BORDER};
}}
QListWidget::item:selected {{
    background-color: #1a0d00;
    color: {CYBER_ORANGE};
    border-left: 3px solid {CYBER_ORANGE};
}}
QListWidget::item:hover {{
    background-color: #13100a;
    color: {CYBER_YELLOW};
}}

/* ═══════════════════ DIALOG ═══════════════════ */
QDialog {{
    background-color: {CYBER_BG};
    color: {CYBER_TEXT};
    font-family: "Consolas", monospace;
}}
"""


# ══════════════════════════════════════════════════════════════════════════════
# DROP ZONE — CSV sürükle-bırak alanı
# ══════════════════════════════════════════════════════════════════════════════
class DropZoneEdit(QLineEdit):
    """Dosya sürükle-bırak destekli metin kutusu."""
    file_dropped = Signal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)
        self.setPlaceholderText("📁  CSV dosyası sürükle-bırak veya yol girin...")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith(".csv"):
                self.setText(path)
                self.file_dropped.emit(path)
            else:
                QMessageBox.warning(self, "Uyarı", "Sadece .csv dosyaları desteklenir!")
        event.acceptProposedAction()


# ══════════════════════════════════════════════════════════════════════════════
# NEON METRIC CARD
# ══════════════════════════════════════════════════════════════════════════════
class MetricCard(QFrame):
    """Glowing neon metrik kartı."""
    def __init__(self, title: str, value: str = "—", color: str = CYBER_BLUE, parent=None):
        super().__init__(parent)
        self.color = color
        self.setFixedHeight(80)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {CYBER_BG3};
                border: 1px solid {color};
                border-radius: 6px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        self.lbl_title = QLabel(title.upper())
        self.lbl_title.setObjectName("lbl_metric_title")
        self.lbl_title.setStyleSheet(f"color: {color}; font-size: 9px; letter-spacing: 1px;")
        self.lbl_title.setAlignment(Qt.AlignLeft)

        self.lbl_value = QLabel(value)
        self.lbl_value.setObjectName("lbl_metric")
        self.lbl_value.setStyleSheet(f"color: {color}; font-size: 22px; font-weight: bold;")
        self.lbl_value.setAlignment(Qt.AlignLeft)

        layout.addWidget(self.lbl_title)
        layout.addWidget(self.lbl_value)

    def set_value(self, v: str):
        self.lbl_value.setText(v)


# ══════════════════════════════════════════════════════════════════════════════
# WORKER THREAD — Analizi arka planda çalıştır
# ══════════════════════════════════════════════════════════════════════════════
class AnalysisWorker(QThread):
    progress      = Signal(int, str)      # (yüzde, mesaj)
    log_message   = Signal(str)           # terminal log satırı
    finished      = Signal(dict)          # sonuç dict
    error         = Signal(str)           # hata mesajı

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config  = config
        self._stop   = False

    def stop(self):
        self._stop = True

    def _log(self, msg: str):
        self.log_message.emit(msg)

    def _prog(self, pct: int, msg: str):
        self.progress.emit(pct, msg)

    def run(self):
        try:
            cfg = self.config
            self._run_analysis(cfg)
        except Exception as e:
            import traceback
            self.error.emit(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")

    def _run_analysis(self, cfg: dict):
        import warnings
        warnings.filterwarnings("ignore")

        import re
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import LabelEncoder
        from sklearn.svm import LinearSVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        import seaborn as sns
        import ollama

        TR_STOPWORDS = set([
            "bir","bu","ve","da","de","ki","ile","için","den","dan","ben","sen","o",
            "biz","siz","onlar","ama","fakat","ya","veya","ne","nasıl","neden","çok",
            "az","daha","en","bile","artık","diye","gibi","kadar","mi","mu","mı","mü",
            "var","yok","şu","onun","bunu","şunu","ona","bana","sana","benim",
            "senin","bizim","sizin","onların","e","a","ı","i","u","ü",
            "ah","oh","of","ey","hey","be","hep","hiç","her","herkes","hiçbir",
            "şey","zaten","sadece","yani","aslında","tam","tamamen","böyle","şöyle",
            "bunun","bunlar","şunlar","biri","birisi","kimse","biraz",
        ])

        def preprocess(text: str) -> str:
            text = str(text).lower()
            text = re.sub(r"http\S+|www\S+", " ", text)
            text = re.sub(r"@\w+|#\w+|\d+", " ", text)
            text = re.sub(r"[^a-zığüşöçİĞÜŞÖÇ\s]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            tokens = [w for w in text.split() if w not in TR_STOPWORDS and len(w) > 1]
            return " ".join(tokens)

        self._prog(5, "Veri yükleniyor...")
        self._log(f"[1/6] CSV yükleniyor: {cfg['csv_path']}")
        if self._stop: return

        try:
            df = pd.read_csv(cfg["csv_path"], encoding="iso-8859-9")
        except Exception:
            df = pd.read_csv(cfg["csv_path"], encoding="utf-8")

        df.columns = ["label", "text"] if len(df.columns) >= 2 else df.columns
        df = df.dropna(subset=["text"]).reset_index(drop=True)
        df["text_str"]   = df["text"].astype(str)
        df["clean_text"] = df["text_str"].apply(preprocess)
        self._log(f"    ✓ {len(df):,} yorum yüklendi.")

        self._prog(12, "Model eğitiliyor...")
        le = LabelEncoder()
        y  = le.fit_transform(df["label"])

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            df["clean_text"], y, test_size=0.20, random_state=42, stratify=y
        )
        _, X_test_orig, _, _ = train_test_split(
            df["text_str"], y, test_size=0.20, random_state=42, stratify=y
        )

        tfidf = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            max_features=30000, sublinear_tf=True, min_df=2,
        )
        X_train_tfidf = tfidf.fit_transform(X_train_raw)
        X_test_tfidf  = tfidf.transform(X_test_raw)

        self._prog(20, f"Klasik model ({cfg['classic_model']}) eğitiliyor...")
        self._log(f"[2/6] Klasik model: {cfg['classic_model']}")

        MODEL_MAP = {
            "Linear SVM":           LinearSVC(C=1.0, max_iter=2000, random_state=42),
            "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
            "Naive Bayes":          MultinomialNB(),
            "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        }
        classic_clf = MODEL_MAP.get(cfg["classic_model"],
                                     LinearSVC(C=1.0, max_iter=2000, random_state=42))
        classic_clf.fit(X_train_tfidf, y_train)
        all_preds = classic_clf.predict(X_test_tfidf)
        svm_acc   = accuracy_score(y_test, all_preds)
        svm_f1    = f1_score(y_test, all_preds, average="weighted")
        self._log(f"    ✓ {cfg['classic_model']} → Acc={svm_acc:.4f} | F1={svm_f1:.4f}")

        sample_size = cfg["sample_size"]
        if sample_size and sample_size < len(X_test_raw):
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(len(X_test_raw), size=sample_size, replace=False)
        else:
            sample_idx = np.arange(len(X_test_raw))

        X_sample_orig  = X_test_orig.iloc[sample_idx].reset_index(drop=True)
        X_sample_clean = X_test_raw.iloc[sample_idx].reset_index(drop=True)
        y_sample       = y_test[sample_idx]
        svm_sample     = all_preds[sample_idx]

        self._log(f"    Hibrit örnek boyutu: {len(sample_idx)}")
        if self._stop: return

        self._prog(30, "Ollama bağlantısı test ediliyor...")
        self._log(f"[3/6] Ollama modeli: {cfg['ollama_model']}")

        SYSTEM_PROMPT = (
            "Sen Türkçe duygu analizi yapan bir sınıflandırıcısın. "
            "Sana verilen Türkçe sosyal medya yorumunu analiz et. "
            "SADECE tek kelime yanıt ver: 'Pozitif' veya 'Negatif'. "
            "Başka hiçbir şey yazma. Açıklama yapma. Noktalama kullanma."
        )

        def query_single(text: str) -> str:
            try:
                prompt = f"{SYSTEM_PROMPT}\n\nYorum: {text[:500]}\nEtiket:"
                resp = ollama.chat(
                    model=cfg["ollama_model"],
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.1, "top_p": 0.9, "num_predict": 25},
                )
                raw = resp["message"]["content"].strip().lower()
                if "pozitif" in raw or "positive" in raw:
                    return "Pozitif"
                elif "negatif" in raw or "negative" in raw:
                    return "Negatif"
                return "ERROR"
            except Exception as exc:
                return "ERROR"

        total_texts = X_sample_orig.tolist()
        ollama_preds = [None] * len(total_texts)
        errors = 0
        max_w  = max(1, cfg["max_workers"])

        with ThreadPoolExecutor(max_workers=max_w) as executor:
            future_to_idx = {
                executor.submit(query_single, text): i
                for i, text in enumerate(total_texts)
            }
            completed = 0
            for future in as_completed(future_to_idx):
                if self._stop:
                    executor.shutdown(wait=False, cancel_futures=True)
                    self._log("  ⚠ Kullanıcı tarafından durduruldu.")
                    return
                idx    = future_to_idx[future]
                result = future.result()
                ollama_preds[idx] = result
                completed += 1
                if result == "ERROR":
                    errors += 1

                pct = 30 + int(completed / len(total_texts) * 50)
                msg = f"Ollama tahmini: {completed}/{len(total_texts)}  (hata: {errors})"
                self._prog(pct, msg)
                if completed % 10 == 0:
                    self._log(f"    {completed}/{len(total_texts)} tamamlandı (hata: {errors})")

        self._prog(82, "Hibrit DataFrame oluşturuluyor...")
        self._log("[4/6] Hibrit karar tablosu hazırlanıyor...")

        svm_labels_str  = le.inverse_transform(svm_sample)
        true_labels_str = le.inverse_transform(y_sample)

        df_h = pd.DataFrame({
            "text":               X_sample_orig.values,
            "True_Label":         true_labels_str,
            "SVM_Prediction":     svm_labels_str,
            "Ollama_Prediction":  ollama_preds,
        })

        err_mask = df_h["Ollama_Prediction"] == "ERROR"
        df_h.loc[err_mask, "Ollama_Prediction"] = df_h.loc[err_mask, "SVM_Prediction"]
        if err_mask.sum() > 0:
            self._log(f"    [INFO] {err_mask.sum()} hatalı Ollama yanıtı SVM ile dolduruldu.")

        df_h["Hybrid_Status"] = df_h.apply(
            lambda r: "Consensus" if r["SVM_Prediction"] == r["Ollama_Prediction"] else "Conflict",
            axis=1,
        )
        df_h["SVM_Correct"]    = df_h["SVM_Prediction"]    == df_h["True_Label"]
        df_h["Ollama_Correct"] = df_h["Ollama_Prediction"] == df_h["True_Label"]

        conflict_mask = df_h["Hybrid_Status"] == "Conflict"
        df_h["Hybrid_Prediction"] = df_h["SVM_Prediction"]
        df_h.loc[conflict_mask, "Hybrid_Prediction"] = df_h.loc[conflict_mask, "Ollama_Prediction"]
        df_h["Hybrid_Correct"] = df_h["Hybrid_Prediction"] == df_h["True_Label"]

        def metrics(preds, labels):
            return {
                "Accuracy":  accuracy_score(labels, preds),
                "F1":        f1_score(labels, preds, average="weighted", zero_division=0),
                "Precision": precision_score(labels, preds, average="weighted", zero_division=0),
                "Recall":    recall_score(labels, preds, average="weighted", zero_division=0),
            }

        true_enc  = le.transform(df_h["True_Label"])
        svm_enc   = le.transform(df_h["SVM_Prediction"])
        ollama_enc = np.array([
            (le.transform([x])[0] if x in le.classes_ else 0)
            for x in df_h["Ollama_Prediction"]
        ])
        hybrid_enc = le.transform(df_h["Hybrid_Prediction"])

        metrics_data = {
            cfg["classic_model"]: metrics(svm_enc, true_enc),
            f"Ollama ({cfg['ollama_model'].split(':')[0]})": metrics(ollama_enc, true_enc),
            "Hybrid": metrics(hybrid_enc, true_enc),
        }

        self._prog(88, "Grafikler oluşturuluyor...")
        self._log("[5/6] Görseller oluşturuluyor...")
        safe_name = cfg["ollama_model"].replace(":", "_").replace(".", "_")
        out_dir   = os.path.join("outputs_hybrid", f"{safe_name}_hybrid")
        os.makedirs(out_dir, exist_ok=True)

        COLOR_POS       = "#4CAF50"
        COLOR_NEG       = "#F44336"
        COLOR_CONSENSUS = "#2196F3"
        COLOR_CONFLICT  = "#FF9800"
        PALETTE         = {"Pozitif": COLOR_POS, "Negatif": COLOR_NEG}

        plt.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 12,
            "axes.titlesize": 14, "axes.titleweight": "bold",
            "axes.spines.top": False, "axes.spines.right": False,
            "figure.dpi": 130, "savefig.bbox": "tight", "savefig.dpi": 150,
        })

        plot_paths = {}

        # H1
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle("SVM vs Ollama — Tahmin Dağılımı", fontsize=15, fontweight="bold")
        for ax, label in zip(axes, ["Negatif", "Pozitif"]):
            svm_s    = set(df_h[df_h["SVM_Prediction"]    == label].index)
            oll_s    = set(df_h[df_h["Ollama_Prediction"] == label].index)
            vals     = [len(svm_s - oll_s), len(svm_s & oll_s), len(oll_s - svm_s), len(df_h) - len(svm_s | oll_s)]
            cats     = ["Sadece SVM", "İkisi de", "Sadece Ollama", "Hiçbiri"]
            colors   = [COLOR_CONSENSUS, "#9C27B0", COLOR_CONFLICT, "#9E9E9E"]
            bars = ax.bar(cats, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5, width=0.55)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1, str(v), ha="center", fontweight="bold", fontsize=12)
            color = COLOR_NEG if label == "Negatif" else COLOR_POS
            ax.set_title(f'"{label}" Diyen Modeller', color=color, fontsize=13)
            ax.set_ylabel("Yorum Sayısı")
        plt.tight_layout()
        p = os.path.join(out_dir, "H1_venn_karsilastirma.png")
        plt.savefig(p); plt.close()
        plot_paths["H1 — Tahmin Dağılımı"] = p

        # H2
        import seaborn as sns
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle("Uyuşmazlık Isı Haritası", fontsize=15, fontweight="bold")
        labels_list = ["Negatif", "Pozitif"]
        pivot = pd.crosstab(df_h["SVM_Prediction"], df_h["Ollama_Prediction"],
                            rownames=["SVM"], colnames=["Ollama"]
                           ).reindex(index=labels_list, columns=labels_list, fill_value=0)
        sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=axes[0],
                    linewidths=2, linecolor="white", annot_kws={"size": 16, "weight": "bold"})
        axes[0].set_title("Tahmin Kesişim Matrisi")
        conflict_df = df_h[df_h["Hybrid_Status"] == "Conflict"].copy()
        if len(conflict_df) > 0:
            cm = pd.DataFrame(0, index=["SVM Doğru","SVM Yanlış"], columns=["Ollama Doğru","Ollama Yanlış"])
            for _, row in conflict_df.iterrows():
                r = "SVM Doğru" if row["SVM_Correct"] else "SVM Yanlış"
                c = "Ollama Doğru" if row["Ollama_Correct"] else "Ollama Yanlış"
                cm.loc[r, c] += 1
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                        linewidths=2, linecolor="white", annot_kws={"size": 16, "weight": "bold"})
            axes[1].set_title(f"Conflict Analizi (n={len(conflict_df)})")
        plt.tight_layout()
        p = os.path.join(out_dir, "H2_conflict_heatmap.png")
        plt.savefig(p); plt.close()
        plot_paths["H2 — Conflict Heatmap"] = p

        # H3
        fig, axes = plt.subplots(1, 4, figsize=(18, 7))
        fig.suptitle("Model Performans Karşılaştırması", fontsize=15, fontweight="bold")
        metric_names = ["Accuracy", "F1", "Precision", "Recall"]
        bar_colors = [CYBER_BLUE, CYBER_PURPLE, CYBER_ORANGE]
        for ax, metric in zip(axes, metric_names):
            model_names = list(metrics_data.keys())
            vals = [metrics_data[m][metric] for m in model_names]
            bars = ax.bar(range(len(model_names)), vals, color=bar_colors[:len(model_names)],
                          alpha=0.85, edgecolor="white", linewidth=1.5, width=0.55)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f"{v:.3f}", ha="center", fontweight="bold", fontsize=11)
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels([m.replace(" ", "\n") for m in model_names], fontsize=9)
            ax.set_title(metric)
            ax.set_ylim(0, 1.15)
        plt.tight_layout()
        p = os.path.join(out_dir, "H3_model_karsilastirma.png")
        plt.savefig(p); plt.close()
        plot_paths["H3 — Model Karşılaştırma"] = p

        # H4
        sections = [
            (df_h[(df_h["Hybrid_Status"]=="Conflict") & df_h["Ollama_Correct"] & ~df_h["SVM_Correct"]],
             "Ollama Kazandı", "#e8f5e9"),
            (df_h[(df_h["Hybrid_Status"]=="Conflict") & ~df_h["Ollama_Correct"] & df_h["SVM_Correct"]],
             "SVM Kazandı", "#e3f2fd"),
            (df_h[(df_h["Hybrid_Status"]=="Conflict") & ~df_h["Ollama_Correct"] & ~df_h["SVM_Correct"]],
             "Her İkisi de Yanlış", "#fff3e0"),
        ]
        fig, axes = plt.subplots(3, 1, figsize=(18, 14))
        fig.suptitle("Conflict Örnekleri", fontsize=14, fontweight="bold")
        for ax, (subset, title, bg) in zip(axes, sections):
            ax.axis("off"); ax.set_facecolor(bg)
            sample = subset.head(5)
            if len(sample) == 0:
                ax.text(0.5, 0.5, f"{title} — Örnek yok", ha="center", va="center",
                        transform=ax.transAxes); ax.set_title(title); continue
            rows = [[str(r["text"])[:70]+"...", r["True_Label"], r["SVM_Prediction"], r["Ollama_Prediction"]]
                    for _, r in sample.iterrows()]
            tbl = ax.table(cellText=rows, colLabels=["Yorum", "Gerçek", "SVM", "Ollama"],
                           loc="center", cellLoc="left")
            tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.5)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_facecolor("#37474f" if r == 0 else bg)
                if r == 0: cell.set_text_props(color="white", fontweight="bold")
                cell.set_edgecolor("#cccccc")
            ax.set_title(f"{title}  (n={len(subset)})", loc="left", fontsize=11, fontweight="bold")
        plt.tight_layout()
        p = os.path.join(out_dir, "H4_conflict_ornekleri.png")
        plt.savefig(p); plt.close()
        plot_paths["H4 — Conflict Örnekleri"] = p

        # H5
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Consensus & Conflict — Gerçek Etiket Dağılımı", fontsize=14, fontweight="bold")
        for ax, (status, color) in zip(axes, [("Consensus", COLOR_CONSENSUS), ("Conflict", COLOR_CONFLICT)]):
            subset = df_h[df_h["Hybrid_Status"] == status]
            if len(subset) == 0:
                ax.text(0.5, 0.5, "Örnek yok", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{status} (n=0)"); continue
            dist = subset["True_Label"].value_counts()
            bars = ax.bar(dist.index, dist.values,
                          color=[PALETTE.get(l, "#607D8B") for l in dist.index],
                          alpha=0.85, edgecolor="white", linewidth=1.5, width=0.45)
            for bar, v in zip(bars, dist.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.5, str(v),
                        ha="center", fontweight="bold", fontsize=13)
            acc = (subset["Hybrid_Prediction"] == subset["True_Label"]).mean()
            ax.set_title(f"{status} (n={len(subset)})\nDoğruluk: {acc:.3f}", color=color)
            ax.set_ylabel("Yorum Sayısı")
        plt.tight_layout()
        p = os.path.join(out_dir, "H5_consensus_conflict_dagilim.png")
        plt.savefig(p); plt.close()
        plot_paths["H5 — Consensus/Conflict Dağılımı"] = p

        df_h.to_csv(os.path.join(out_dir, "hybrid_results.csv"), index=False, encoding="utf-8-sig")

        self._prog(100, "✅  Analiz tamamlandı!")
        self._log(f"[6/6] ✅ Tüm çıktılar → {out_dir}/")

        summary = {
            "svm_acc":     metrics_data[cfg["classic_model"]]["Accuracy"],
            "ollama_acc":  list(metrics_data.values())[1]["Accuracy"],
            "hybrid_acc":  metrics_data["Hybrid"]["Accuracy"],
            "consensus_n": int((df_h["Hybrid_Status"] == "Consensus").sum()),
            "conflict_n":  int((df_h["Hybrid_Status"] == "Conflict").sum()),
            "total_n":     len(df_h),
            "plot_paths":  plot_paths,
            "out_dir":     out_dir,
            "metrics":     metrics_data,
        }
        self.finished.emit(summary)


# ══════════════════════════════════════════════════════════════════════════════
# COMPARE WORKER — Çoklu model karşılaştırması arka planda
# ══════════════════════════════════════════════════════════════════════════════
class CompareWorker(QThread):
    """outputs_hybrid altındaki model klasörlerini okuyup karşılaştırma grafikleri üretir."""
    progress    = Signal(int, str)
    log_message = Signal(str)
    finished    = Signal(dict)
    error       = Signal(str)

    def __init__(self, model_entries: list, parent=None):
        """
        model_entries: [(display_name, folder_path), ...]
        """
        super().__init__(parent)
        self.model_entries = model_entries

    def _log(self, msg):
        self.log_message.emit(msg)

    def _prog(self, pct, msg):
        self.progress.emit(pct, msg)

    def run(self):
        try:
            self._compare()
        except Exception as e:
            import traceback
            self.error.emit(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")

    def _compare(self):
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        import seaborn as sns

        results = {}
        n = len(self.model_entries)

        self._log("=" * 55)
        self._log("  KARŞILAŞTIRMA MOTORU BAŞLADI")
        self._log("=" * 55)

        for i, (name, folder) in enumerate(self.model_entries):
            pct = int((i / n) * 70)
            self._prog(pct, f"Okunuyor: {name}")
            self._log(f"[{i+1}/{n}] {name}")

            csv_path = os.path.join(folder, "hybrid_results.csv")
            if not os.path.exists(csv_path):
                self._log(f"    ⚠ hybrid_results.csv bulunamadı: {csv_path}")
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                self._log(f"    ✗ CSV okuma hatası: {e}")
                continue

            required_cols = {"True_Label", "SVM_Prediction", "Ollama_Prediction",
                             "Hybrid_Prediction", "Hybrid_Status"}
            if not required_cols.issubset(df.columns):
                self._log(f"    ✗ Eksik sütunlar: {required_cols - set(df.columns)}")
                continue

            true      = df["True_Label"]
            svm_pred  = df["SVM_Prediction"]
            oll_pred  = df["Ollama_Prediction"]
            hyb_pred  = df["Hybrid_Prediction"]

            def _m(pred, ref):
                return {
                    "Accuracy":  round(accuracy_score(ref, pred), 4),
                    "F1":        round(f1_score(ref, pred, average="weighted", zero_division=0), 4),
                    "Precision": round(precision_score(ref, pred, average="weighted", zero_division=0), 4),
                    "Recall":    round(recall_score(ref, pred, average="weighted", zero_division=0), 4),
                }

            consensus_n = int((df["Hybrid_Status"] == "Consensus").sum())
            conflict_n  = int((df["Hybrid_Status"] == "Conflict").sum())
            total       = len(df)

            # Conflict kazananları
            if "SVM_Correct" in df.columns and "Ollama_Correct" in df.columns:
                conf_df = df[df["Hybrid_Status"] == "Conflict"]
                ollama_won = int((conf_df["Ollama_Correct"] & ~conf_df["SVM_Correct"]).sum())
                svm_won    = int((~conf_df["Ollama_Correct"] & conf_df["SVM_Correct"]).sum())
                both_wrong = int((~conf_df["Ollama_Correct"] & ~conf_df["SVM_Correct"]).sum())
            else:
                ollama_won = svm_won = both_wrong = 0

            results[name] = {
                "SVM":        _m(svm_pred, true),
                "Ollama":     _m(oll_pred, true),
                "Hybrid":     _m(hyb_pred, true),
                "Consensus_N": consensus_n,
                "Conflict_N":  conflict_n,
                "Total":       total,
                "Consensus_Pct": round(consensus_n / total * 100, 1) if total else 0,
                "Conflict_Pct":  round(conflict_n  / total * 100, 1) if total else 0,
                "Ollama_Won":  ollama_won,
                "SVM_Won":     svm_won,
                "Both_Wrong":  both_wrong,
            }

            self._log(f"    ✓ Acc(Ollama)={results[name]['Ollama']['Accuracy']:.4f} "
                      f"| Acc(Hybrid)={results[name]['Hybrid']['Accuracy']:.4f} "
                      f"| Consensus={consensus_n} ({results[name]['Consensus_Pct']}%)")

        if not results:
            self.error.emit("Karşılaştırılabilir model verisi bulunamadı.\n"
                            "hybrid_results.csv dosyalarının mevcut olduğundan emin olun.")
            return

        # ── Grafik üretimi ────────────────────────────────────────────────────
        self._prog(72, "Karşılaştırma grafikleri oluşturuluyor...")
        self._log("\n  Grafikler oluşturuluyor...")

        out_dir = os.path.join("outputs_hybrid", "_compare_results")
        os.makedirs(out_dir, exist_ok=True)

        plt.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 11,
            "axes.titlesize": 13, "axes.titleweight": "bold",
            "axes.spines.top": False, "axes.spines.right": False,
            "figure.dpi": 120, "savefig.bbox": "tight", "savefig.dpi": 150,
            "axes.facecolor": "#0d0d1a", "figure.facecolor": "#0a0a0f",
            "axes.labelcolor": "#c0c0e0", "xtick.color": "#c0c0e0",
            "ytick.color": "#c0c0e0", "text.color": "#e0e0ff",
            "grid.color": "#1a1a3a", "grid.alpha": 0.5,
        })

        model_names = list(results.keys())
        n_models = len(model_names)

        # Renk paleti — her model için farklı neon renk
        NEON_PALETTE = [
            "#00d4ff", "#bf5fff", "#05ffa1", "#ff9f1c",
            "#ff2a6d", "#f5e642", "#7b61ff", "#00ff99",
            "#ff6b35", "#4fc3f7", "#ce93d8", "#80cbc4",
            "#ffb74d", "#ef9a9a",
        ]
        model_colors = {m: NEON_PALETTE[i % len(NEON_PALETTE)] for i, m in enumerate(model_names)}

        plot_paths = {}

        # ── C1: Ollama Accuracy Karşılaştırması (yatay bar) ──────────────────
        fig, ax = plt.subplots(figsize=(14, max(5, n_models * 0.65)))
        fig.patch.set_facecolor("#0a0a0f")
        ax.set_facecolor("#0d0d1a")

        acc_vals  = [results[m]["Ollama"]["Accuracy"]  for m in model_names]
        svm_acc   = results[model_names[0]]["SVM"]["Accuracy"]  # SVM tüm modeller için aynı
        colors    = [model_colors[m] for m in model_names]

        bars = ax.barh(model_names, acc_vals, color=colors, alpha=0.85,
                       edgecolor="#1a1a3a", linewidth=1.2, height=0.6)
        ax.axvline(svm_acc, color="#ff2a6d", linewidth=2, linestyle="--",
                   label=f"SVM Baseline ({svm_acc:.4f})", alpha=0.9)

        for bar, v, c in zip(bars, acc_vals, colors):
            ax.text(v + 0.003, bar.get_y() + bar.get_height()/2,
                    f"{v:.4f}", va="center", fontsize=10, fontweight="bold", color=c)

        ax.set_xlabel("Accuracy", color="#c0c0e0")
        ax.set_title("LLM (Ollama) Doğruluk — Model Karşılaştırması", color="#00d4ff", pad=14)
        ax.set_xlim(0, min(1.0, max(acc_vals) * 1.18))
        ax.legend(loc="lower right", facecolor="#13131f", edgecolor="#1a1a3a",
                  labelcolor="#ff2a6d", fontsize=10)
        ax.tick_params(colors="#c0c0e0")
        ax.grid(axis="x", alpha=0.3, color="#1a1a3a")
        plt.tight_layout()
        p = os.path.join(out_dir, "C1_ollama_accuracy.png")
        plt.savefig(p, facecolor="#0a0a0f"); plt.close()
        plot_paths["C1 — Ollama Doğruluk"] = p

        # ── C2: Tüm Metrikler — Grouped Bar ──────────────────────────────────
        metrics_list = ["Accuracy", "F1", "Precision", "Recall"]
        x = np.arange(len(metrics_list))
        width = 0.8 / n_models

        fig, ax = plt.subplots(figsize=(16, 7))
        fig.patch.set_facecolor("#0a0a0f")
        ax.set_facecolor("#0d0d1a")

        for i, m in enumerate(model_names):
            vals = [results[m]["Ollama"][k] for k in metrics_list]
            offset = (i - n_models/2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width=width * 0.92,
                          color=model_colors[m], alpha=0.85, label=m,
                          edgecolor="#0a0a0f", linewidth=0.8)

        # SVM referans çizgileri
        for j, metric in enumerate(metrics_list):
            ref_val = results[model_names[0]]["SVM"][metric]
            ax.hlines(ref_val, j - 0.45, j + 0.45,
                      colors="#ff2a6d", linewidth=1.5, linestyle="--", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(metrics_list, fontsize=12, color="#c0c0e0")
        ax.set_ylim(0, 1.1)
        ax.set_title("Ollama Model Metrikleri — Tam Karşılaştırma\n(Kesik çizgi = SVM baseline)",
                     color="#00d4ff", pad=14)
        ax.tick_params(colors="#c0c0e0")
        ax.grid(axis="y", alpha=0.3, color="#1a1a3a")

        legend = ax.legend(loc="upper right", facecolor="#13131f", edgecolor="#1a1a3a",
                           fontsize=9, ncol=max(1, n_models // 5))
        for text in legend.get_texts():
            text.set_color("#e0e0ff")

        plt.tight_layout()
        p = os.path.join(out_dir, "C2_metrics_grouped.png")
        plt.savefig(p, facecolor="#0a0a0f"); plt.close()
        plot_paths["C2 — Tüm Metrikler"] = p

        # ── C3: Consensus / Conflict Oranları ────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(16, max(5, n_models * 0.65)))
        fig.patch.set_facecolor("#0a0a0f")

        cons_pcts = [results[m]["Consensus_Pct"] for m in model_names]
        conf_pcts = [results[m]["Conflict_Pct"]  for m in model_names]
        colors    = [model_colors[m] for m in model_names]

        for ax, vals, title, accent in zip(
            axes,
            [cons_pcts, conf_pcts],
            ["Consensus Oranı (%)", "Conflict Oranı (%)"],
            ["#2196F3", "#FF9800"]
        ):
            ax.set_facecolor("#0d0d1a")
            bars = ax.barh(model_names, vals, color=colors, alpha=0.85,
                           edgecolor="#1a1a3a", linewidth=1, height=0.6)
            for bar, v, c in zip(bars, vals, colors):
                ax.text(v + 0.5, bar.get_y() + bar.get_height()/2,
                        f"{v:.1f}%", va="center", fontsize=10, fontweight="bold", color=c)
            ax.set_title(title, color=accent, pad=12)
            ax.set_xlabel("%", color="#c0c0e0")
            ax.tick_params(colors="#c0c0e0")
            ax.grid(axis="x", alpha=0.3, color="#1a1a3a")

        fig.suptitle("Consensus & Conflict Dağılımı — Tüm Modeller",
                     color="#e0e0ff", fontsize=14, fontweight="bold")
        plt.tight_layout()
        p = os.path.join(out_dir, "C3_consensus_conflict.png")
        plt.savefig(p, facecolor="#0a0a0f"); plt.close()
        plot_paths["C3 — Consensus/Conflict"] = p

        # ── C4: Conflict Kazan/Kaybet Analizi ────────────────────────────────
        fig, ax = plt.subplots(figsize=(16, max(5, n_models * 0.7)))
        fig.patch.set_facecolor("#0a0a0f")
        ax.set_facecolor("#0d0d1a")

        ollama_won_vals = [results[m]["Ollama_Won"] for m in model_names]
        svm_won_vals    = [results[m]["SVM_Won"]    for m in model_names]
        both_wrong_vals = [results[m]["Both_Wrong"] for m in model_names]

        y_pos = np.arange(n_models)
        bar_h = 0.25

        b1 = ax.barh(y_pos + bar_h, ollama_won_vals, height=bar_h * 0.9,
                     color="#05ffa1", alpha=0.85, label="Ollama Kazandı", edgecolor="#0a0a0f")
        b2 = ax.barh(y_pos,          svm_won_vals,    height=bar_h * 0.9,
                     color="#00d4ff", alpha=0.85, label="SVM Kazandı",    edgecolor="#0a0a0f")
        b3 = ax.barh(y_pos - bar_h, both_wrong_vals, height=bar_h * 0.9,
                     color="#ff2a6d", alpha=0.85, label="Her İkisi Yanlış", edgecolor="#0a0a0f")

        for bars_group, vals in [(b1, ollama_won_vals), (b2, svm_won_vals), (b3, both_wrong_vals)]:
            for bar, v in zip(bars_group, vals):
                if v > 0:
                    ax.text(v + 1, bar.get_y() + bar.get_height()/2,
                            str(v), va="center", fontsize=9, fontweight="bold",
                            color=bar.get_facecolor())

        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names, fontsize=10, color="#c0c0e0")
        ax.set_xlabel("Conflict Sayısı", color="#c0c0e0")
        ax.set_title("Conflict Durumlarında Kazanan Model Analizi",
                     color="#ff9f1c", pad=14)
        ax.tick_params(colors="#c0c0e0")
        ax.grid(axis="x", alpha=0.3, color="#1a1a3a")

        legend = ax.legend(loc="lower right", facecolor="#13131f",
                           edgecolor="#1a1a3a", fontsize=10)
        for text in legend.get_texts():
            text.set_color("#e0e0ff")

        plt.tight_layout()
        p = os.path.join(out_dir, "C4_conflict_winner.png")
        plt.savefig(p, facecolor="#0a0a0f"); plt.close()
        plot_paths["C4 — Conflict Kazan/Kaybet"] = p

        # ── C5: Radar / Spider Grafiği ────────────────────────────────────────
        from matplotlib.patches import FancyArrowPatch
        metrics_radar = ["Accuracy", "F1", "Precision", "Recall"]
        N = len(metrics_radar)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # kapat

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("#0a0a0f")
        ax.set_facecolor("#0d0d1a")
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_radar, color="#c0c0e0", fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],
                           color="#6060a0", fontsize=8)
        ax.grid(color="#1a1a3a", alpha=0.6)
        ax.spines["polar"].set_color("#1a1a3a")

        for m in model_names:
            vals = [results[m]["Ollama"][k] for k in metrics_radar]
            vals += vals[:1]
            c = model_colors[m]
            ax.plot(angles, vals, "o-", linewidth=2, color=c, alpha=0.9,
                    markersize=5, label=m)
            ax.fill(angles, vals, alpha=0.08, color=c)

        # SVM referans
        svm_vals = [results[model_names[0]]["SVM"][k] for k in metrics_radar]
        svm_vals += svm_vals[:1]
        ax.plot(angles, svm_vals, "--", linewidth=2.5, color="#ff2a6d",
                alpha=0.9, label="SVM Baseline")

        legend = ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
                           facecolor="#13131f", edgecolor="#1a1a3a", fontsize=9)
        for text in legend.get_texts():
            text.set_color("#e0e0ff")

        ax.set_title("LLM Model Performans Radar\n(Kırmızı kesik = SVM baseline)",
                     color="#00d4ff", pad=28, fontsize=13, fontweight="bold")

        plt.tight_layout()
        p = os.path.join(out_dir, "C5_radar.png")
        plt.savefig(p, facecolor="#0a0a0f", bbox_inches="tight"); plt.close()
        plot_paths["C5 — Radar Grafiği"] = p

        # ── C6: Özet Tablo Görseli ───────────────────────────────────────────
        table_data = []
        col_labels = ["Model", "Acc(SVM)", "Acc(LLM)", "Acc(Hybrid)",
                      "F1(LLM)", "Consensus%", "Conflict%", "LLM Wins"]
        for m in model_names:
            r = results[m]
            table_data.append([
                m[:28],
                f"{r['SVM']['Accuracy']:.4f}",
                f"{r['Ollama']['Accuracy']:.4f}",
                f"{r['Hybrid']['Accuracy']:.4f}",
                f"{r['Ollama']['F1']:.4f}",
                f"{r['Consensus_Pct']:.1f}%",
                f"{r['Conflict_Pct']:.1f}%",
                str(r["Ollama_Won"]),
            ])

        fig, ax = plt.subplots(figsize=(20, max(4, n_models * 0.55 + 2)))
        fig.patch.set_facecolor("#0a0a0f")
        ax.set_facecolor("#0a0a0f")
        ax.axis("off")

        tbl = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.8)

        header_color = "#0d001a"
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor("#1a1a3a")
            if row == 0:
                cell.set_facecolor(header_color)
                cell.set_text_props(color=CYBER_PURPLE, fontweight="bold")
            else:
                m_name = model_names[row - 1]
                cell.set_facecolor("#0d0d1a" if row % 2 == 0 else "#0f0f1f")
                # Renk: LLM accuracy sütunu
                if col == 2:
                    cell.set_text_props(color=model_colors.get(m_name, CYBER_CYAN),
                                        fontweight="bold")
                elif col == 1:
                    cell.set_text_props(color="#ff2a6d", fontweight="bold")
                else:
                    cell.set_text_props(color="#c0c0e0")

        ax.set_title("Tüm Model Karşılaştırma Özet Tablosu",
                     color="#00d4ff", fontsize=13, fontweight="bold", pad=16,
                     loc="center")

        plt.tight_layout()
        p = os.path.join(out_dir, "C6_ozet_tablo.png")
        plt.savefig(p, facecolor="#0a0a0f"); plt.close()
        plot_paths["C6 — Özet Tablo"] = p

        self._prog(100, "✅  Karşılaştırma tamamlandı!")
        self._log(f"\n  ✅ Karşılaştırma grafikleri → {out_dir}/")
        self._log(f"  Karşılaştırılan model sayısı: {len(results)}")
        self._log("=" * 55)

        self.finished.emit({
            "results":    results,
            "plot_paths": plot_paths,
            "out_dir":    out_dir,
        })


# ══════════════════════════════════════════════════════════════════════════════
# SONUÇ GÖRÜNTÜLEME SEKMESİ  (ortak kullanım)
# ══════════════════════════════════════════════════════════════════════════════
class ResultsTab(QWidget):
    """H1–H5 / C1–C6 grafikleri sekmeli gösterir."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

    def load_plots(self, plot_paths: dict):
        self.tab_widget.clear()
        for tab_name, img_path in plot_paths.items():
            if not os.path.exists(img_path):
                continue
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            container = QWidget()
            vbox = QVBoxLayout(container)
            vbox.setAlignment(Qt.AlignCenter)
            lbl = QLabel()
            pix = QPixmap(img_path)
            if pix.width() > 1200:
                pix = pix.scaledToWidth(1200, Qt.SmoothTransformation)
            lbl.setPixmap(pix)
            lbl.setAlignment(Qt.AlignCenter)
            vbox.addWidget(lbl)
            scroll.setWidget(container)
            self.tab_widget.addTab(scroll, tab_name)


# ══════════════════════════════════════════════════════════════════════════════
# KARŞILAŞTIRMA SONUÇ PENCERESİ
# ══════════════════════════════════════════════════════════════════════════════
class CompareResultWindow(QDialog):
    """Karşılaştırma grafik sonuçlarını gösteren tam ekran dialog."""
    def __init__(self, plot_paths: dict, results: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚖  MODEL KARŞILAŞTIRMA SONUÇLARI")
        self.setMinimumSize(1300, 820)
        self.resize(1440, 900)
        self.setWindowModality(Qt.NonModal)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Başlık
        hdr = QFrame()
        hdr.setFixedHeight(54)
        hdr.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0d0800, stop:0.5 #1a0d00, stop:1 #0d0800);
                border-bottom: 1px solid {CYBER_ORANGE};
                border-radius: 4px;
            }}
        """)
        hbox = QHBoxLayout(hdr)
        hbox.setContentsMargins(16, 0, 16, 0)

        icon_lbl = QLabel("⚖")
        icon_lbl.setStyleSheet(f"color: {CYBER_ORANGE}; font-size: 24px;")

        title_lbl = QLabel("MODEL KARŞILAŞTIRMA SONUÇLARI")
        title_lbl.setStyleSheet(f"color: {CYBER_ORANGE}; font-size: 17px; font-weight: bold; letter-spacing: 3px;")

        n_models = len(results)
        info_lbl = QLabel(f"{n_models} MODEL  //  {list(results.values())[0]['Total']} ÖRNEK")
        info_lbl.setStyleSheet(f"color: {CYBER_TEXT_DIM}; font-size: 10px; letter-spacing: 1px;")

        hbox.addWidget(icon_lbl)
        hbox.addSpacing(10)
        hbox.addWidget(title_lbl)
        hbox.addStretch()
        hbox.addWidget(info_lbl)
        layout.addWidget(hdr)

        # Splitter: grafik + özet bilgi
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {CYBER_BORDER}; }}")

        # Sol: Grafikler
        self.results_tab = ResultsTab()
        self.results_tab.load_plots(plot_paths)
        splitter.addWidget(self.results_tab)

        # Sağ: Mini özet metrikleri
        right_panel = self._build_summary_panel(results)
        splitter.addWidget(right_panel)
        splitter.setSizes([1050, 380])

        layout.addWidget(splitter, 1)

        # Kapat butonu
        btn_close = QPushButton("✕  KAPAT")
        btn_close.setFixedHeight(36)
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close)

    def _build_summary_panel(self, results: dict) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(340)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        grp = QGroupBox("[ MODEL ÖZETİ ]")
        grp.setStyleSheet(f"""
            QGroupBox {{
                border-top: 2px solid {CYBER_ORANGE};
                color: {CYBER_ORANGE};
            }}
            QGroupBox::title {{
                color: {CYBER_ORANGE};
            }}
        """)
        grp_lay = QVBoxLayout(grp)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        container = QWidget()
        c_lay = QVBoxLayout(container)
        c_lay.setSpacing(6)

        # Modelleri Ollama accuracy'e göre sırala
        sorted_models = sorted(results.items(),
                               key=lambda x: x[1]["Ollama"]["Accuracy"],
                               reverse=True)

        NEON_PALETTE = [
            "#00d4ff", "#bf5fff", "#05ffa1", "#ff9f1c",
            "#ff2a6d", "#f5e642", "#7b61ff", "#00ff99",
        ]

        svm_ref = list(results.values())[0]["SVM"]["Accuracy"]

        for rank, (m_name, r) in enumerate(sorted_models):
            color = NEON_PALETTE[rank % len(NEON_PALETTE)]
            card = QFrame()
            card.setStyleSheet(f"""
                QFrame {{
                    background-color: {CYBER_BG3};
                    border: 1px solid {color};
                    border-left: 4px solid {color};
                    border-radius: 5px;
                    padding: 2px;
                }}
            """)
            card_lay = QVBoxLayout(card)
            card_lay.setContentsMargins(8, 6, 8, 6)
            card_lay.setSpacing(2)

            medal = ["🥇", "🥈", "🥉"][rank] if rank < 3 else f"#{rank+1}"
            name_lbl = QLabel(f"{medal}  {m_name[:30]}")
            name_lbl.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 10px;")
            name_lbl.setWordWrap(True)

            acc_ollama = r["Ollama"]["Accuracy"]
            delta = acc_ollama - svm_ref
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            delta_color = CYBER_CYAN if delta >= 0 else CYBER_RED

            metrics_lbl = QLabel(
                f"  LLM Acc : {acc_ollama:.4f}  "
                f"<span style='color:{delta_color}'>[{delta_str} vs SVM]</span>\n"
                f"  F1      : {r['Ollama']['F1']:.4f}\n"
                f"  Consensus: {r['Consensus_Pct']:.1f}%  |  Conflict: {r['Conflict_Pct']:.1f}%"
            )
            metrics_lbl.setTextFormat(Qt.RichText)
            metrics_lbl.setStyleSheet(f"color: {CYBER_TEXT}; font-size: 10px; font-family: Consolas;")

            card_lay.addWidget(name_lbl)
            card_lay.addWidget(metrics_lbl)
            c_lay.addWidget(card)

        # SVM baseline bilgisi
        svm_card = QFrame()
        svm_card.setStyleSheet(f"""
            QFrame {{
                background-color: {CYBER_BG3};
                border: 1px solid {CYBER_RED};
                border-left: 4px solid {CYBER_RED};
                border-radius: 5px;
            }}
        """)
        svm_lay = QVBoxLayout(svm_card)
        svm_lay.setContentsMargins(8, 6, 8, 6)
        svm_lbl = QLabel(f"📌  SVM BASELINE\n  Accuracy : {svm_ref:.4f}")
        svm_lbl.setStyleSheet(f"color: {CYBER_RED}; font-size: 10px; font-weight: bold; font-family: Consolas;")
        svm_lay.addWidget(svm_lbl)
        c_lay.addWidget(svm_card)

        c_lay.addStretch()
        scroll.setWidget(container)
        grp_lay.addWidget(scroll)
        layout.addWidget(grp, 1)
        return panel


# ══════════════════════════════════════════════════════════════════════════════
# KARŞILAŞTIRMA DİALOGU — Model seçimi + çalıştırma
# ══════════════════════════════════════════════════════════════════════════════
class CompareDialog(QDialog):
    """outputs_hybrid altındaki modelleri listeler; kullanıcı seçer, karşılaştırır."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚖  MODEL KARŞILAŞTIRICI — Seçim Ekranı")
        self.setMinimumSize(680, 560)
        self.resize(720, 620)
        self.compare_worker = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # Başlık banner
        banner = QLabel("  ⚖  KARŞILAŞTIRMA MODELLERİNİ SEÇİN")
        banner.setStyleSheet(f"""
            color: {CYBER_ORANGE};
            background: #1a0d00;
            border: 1px solid {CYBER_ORANGE};
            border-radius: 4px;
            font-size: 14px;
            font-weight: bold;
            letter-spacing: 2px;
            padding: 8px 14px;
        """)
        layout.addWidget(banner)

        # Dizin seçimi
        dir_grp = QGroupBox("[ OUTPUTS DİZİNİ ]")
        dir_grp.setStyleSheet(f"QGroupBox {{ border-top: 2px solid {CYBER_ORANGE}; color: {CYBER_ORANGE}; }}"
                              f"QGroupBox::title {{ color: {CYBER_ORANGE}; }}")
        dir_lay = QHBoxLayout(dir_grp)

        self.edit_dir = QLineEdit("outputs_hybrid")
        self.edit_dir.setPlaceholderText("outputs_hybrid klasör yolu...")
        btn_browse_dir = QPushButton("📂")
        btn_browse_dir.setFixedWidth(38)
        btn_browse_dir.setToolTip("Klasör seç")
        btn_browse_dir.clicked.connect(self._browse_dir)

        btn_scan = QPushButton("🔍  TARA")
        btn_scan.setFixedWidth(90)
        btn_scan.clicked.connect(self._scan_models)

        dir_lay.addWidget(self.edit_dir)
        dir_lay.addWidget(btn_browse_dir)
        dir_lay.addWidget(btn_scan)
        layout.addWidget(dir_grp)

        # Model listesi
        model_grp = QGroupBox("[ BULUNAN MODELLER ]")
        model_grp.setStyleSheet(f"QGroupBox {{ border-top: 2px solid {CYBER_ORANGE}; color: {CYBER_ORANGE}; }}"
                                f"QGroupBox::title {{ color: {CYBER_ORANGE}; }}")
        model_lay = QVBoxLayout(model_grp)

        # Araç çubuğu
        tools_row = QHBoxLayout()
        btn_sel_all  = QPushButton("☑  Tümünü Seç")
        btn_sel_none = QPushButton("☐  Temizle")
        self.lbl_found = QLabel("—")
        self.lbl_found.setStyleSheet(f"color: {CYBER_ORANGE}; font-size: 11px; font-weight: bold;")
        btn_sel_all.setFixedHeight(28)
        btn_sel_none.setFixedHeight(28)
        btn_sel_all.clicked.connect(self._select_all)
        btn_sel_none.clicked.connect(self._select_none)
        tools_row.addWidget(btn_sel_all)
        tools_row.addWidget(btn_sel_none)
        tools_row.addStretch()
        tools_row.addWidget(self.lbl_found)
        model_lay.addLayout(tools_row)

        # Liste
        self.list_widget = QListWidget()
        self.list_widget.setMinimumHeight(200)
        self.list_widget.setSelectionMode(QAbstractItemView.NoSelection)
        model_lay.addWidget(self.list_widget)
        layout.addWidget(model_grp, 1)

        # İlerleme
        prog_grp = QGroupBox("[ KARŞILAŞTIRMA DURUMU ]")
        prog_grp.setStyleSheet(f"QGroupBox {{ border-top: 2px solid {CYBER_ORANGE}; color: {CYBER_ORANGE}; }}"
                               f"QGroupBox::title {{ color: {CYBER_ORANGE}; }}")
        prog_lay = QVBoxLayout(prog_grp)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(24)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {CYBER_ORANGE}, stop:1 {CYBER_YELLOW});
                border-radius: 5px;
            }}
        """)

        self.lbl_status = QLabel("Modelleri taramak için TARA butonuna basın.")
        self.lbl_status.setStyleSheet(f"color: {CYBER_TEXT_DIM}; font-size: 10px;")
        self.lbl_status.setWordWrap(True)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(80)
        self.log_box.setStyleSheet(f"background: #050510; color: {CYBER_ORANGE}; font-size: 10px;")

        prog_lay.addWidget(self.progress_bar)
        prog_lay.addWidget(self.lbl_status)
        prog_lay.addWidget(self.log_box)
        layout.addWidget(prog_grp)

        # Butonlar
        btn_row = QHBoxLayout()
        self.btn_compare = QPushButton("⚖  KARŞILAŞTIRMAYI BAŞLAT")
        self.btn_compare.setObjectName("btn_compare_run")
        self.btn_compare.setEnabled(False)
        self.btn_compare.clicked.connect(self._start_compare)

        btn_cancel = QPushButton("✕  KAPAT")
        btn_cancel.setFixedWidth(110)
        btn_cancel.clicked.connect(self.close)

        btn_row.addWidget(self.btn_compare)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        # İlk tarama
        self._scan_models()

    # ── Yardımcılar ──────────────────────────────────────────────────────────
    def _browse_dir(self):
        path = QFileDialog.getExistingDirectory(self, "outputs_hybrid Klasörü Seç", "")
        if path:
            self.edit_dir.setText(path)
            self._scan_models()

    def _scan_models(self):
        """outputs_hybrid altında *_hybrid klasörlerini tarar."""
        base = self.edit_dir.text().strip() or "outputs_hybrid"
        self.list_widget.clear()
        found = []

        if os.path.isdir(base):
            for entry in sorted(os.listdir(base)):
                full = os.path.join(base, entry)
                if os.path.isdir(full) and entry.endswith("_hybrid") and not entry.startswith("_"):
                    csv_ok = os.path.exists(os.path.join(full, "hybrid_results.csv"))
                    found.append((entry, full, csv_ok))
        else:
            self.lbl_status.setText(f"⚠  Dizin bulunamadı: {base}")

        for name, folder, csv_ok in found:
            item = QListWidgetItem()
            # Modelin görünen adı: klasör adından _hybrid çıkar
            display = name.replace("_hybrid", "").replace("_", " ").replace("  ", ":")
            item.setText(f"  {'✓' if csv_ok else '✗'}  {display}")
            item.setData(Qt.UserRole, (display, folder, csv_ok))
            item.setCheckState(Qt.Checked if csv_ok else Qt.Unchecked)
            if not csv_ok:
                item.setForeground(QColor(CYBER_RED))
                item.setToolTip("hybrid_results.csv bulunamadı — karşılaştırma için çalıştırın.")
            else:
                item.setForeground(QColor(CYBER_ORANGE))
            self.list_widget.addItem(item)

        n_ok = sum(1 for _, _, ok in found if ok)
        self.lbl_found.setText(f"{len(found)} klasör  //  {n_ok} hazır")
        self.btn_compare.setEnabled(n_ok >= 2)
        if n_ok < 2:
            self.lbl_status.setText(
                "⚠  Karşılaştırma için en az 2 modelin çalıştırılmış olması gerekir."
            )
        else:
            self.lbl_status.setText(
                f"✓  {n_ok} model hazır. Seçimlerinizi yapıp KARŞILAŞTIRMAYI BAŞLAT'a basın."
            )

    def _select_all(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            _, _, csv_ok = item.data(Qt.UserRole)
            if csv_ok:
                item.setCheckState(Qt.Checked)

    def _select_none(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Unchecked)

    def _get_selected(self):
        selected = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            display, folder, csv_ok = item.data(Qt.UserRole)
            if item.checkState() == Qt.Checked and csv_ok:
                selected.append((display, folder))
        return selected

    def _start_compare(self):
        selected = self._get_selected()
        if len(selected) < 2:
            QMessageBox.warning(self, "Eksik Seçim",
                                "Karşılaştırma için en az 2 model seçmelisiniz!")
            return

        self.btn_compare.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_box.clear()
        self.lbl_status.setText(f"⚡  {len(selected)} model karşılaştırılıyor...")

        self.compare_worker = CompareWorker(selected)
        self.compare_worker.progress.connect(self._on_progress)
        self.compare_worker.log_message.connect(self._on_log)
        self.compare_worker.finished.connect(self._on_finished)
        self.compare_worker.error.connect(self._on_error)
        self.compare_worker.start()

    def _on_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.lbl_status.setText(msg)

    def _on_log(self, msg: str):
        self.log_box.append(msg)
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum()
        )

    def _on_finished(self, result: dict):
        self.btn_compare.setEnabled(True)
        self.lbl_status.setText(f"✅  Tamamlandı! {len(result['results'])} model karşılaştırıldı.")

        win = CompareResultWindow(
            result["plot_paths"],
            result["results"],
            parent=self.parent()
        )
        win.show()
        # Store reference so it doesn't get GC'd
        if not hasattr(self, "_result_windows"):
            self._result_windows = []
        self._result_windows.append(win)

    def _on_error(self, msg: str):
        self.btn_compare.setEnabled(True)
        self.lbl_status.setText("✗  HATA!")
        QMessageBox.critical(self, "Karşılaştırma Hatası",
                             f"Hata oluştu:\n\n{msg[:600]}")


# ══════════════════════════════════════════════════════════════════════════════
# ANA PENCERE
# ══════════════════════════════════════════════════════════════════════════════
class CyberHybridGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker      = None
        self.ollama_models = []
        self._compare_dialog = None
        self.setWindowTitle("⬡  HYBRID DECISION AGENT  |  SVM + Ollama  |  CYBERPUNK EDITION")
        self.setMinimumSize(1280, 820)
        self.resize(1440, 900)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 8, 12, 8)
        root_layout.setSpacing(6)

        self._build_header(root_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {CYBER_BORDER}; }}")

        left_panel  = self._build_left_panel()
        right_panel = self._build_right_panel()

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([420, 1020])

        root_layout.addWidget(splitter, 1)
        self._build_status_bar(root_layout)
        self._refresh_ollama_models()

    # ── Başlık ───────────────────────────────────────────────────────────────
    def _build_header(self, parent_layout):
        hdr = QFrame()
        hdr.setFixedHeight(64)
        hdr.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0a0014, stop:0.5 #0d001a, stop:1 #14000a);
                border-bottom: 1px solid {CYBER_PURPLE};
            }}
        """)
        hbox = QHBoxLayout(hdr)
        hbox.setContentsMargins(16, 0, 16, 0)

        lbl_icon = QLabel("⬡")
        lbl_icon.setStyleSheet(f"color: {CYBER_RED}; font-size: 28px;")

        title_box = QVBoxLayout()
        lbl_title    = QLabel("HYBRID DECISION AGENT")
        lbl_title.setObjectName("lbl_title")
        lbl_subtitle = QLabel("SVM  ×  OLLAMA LLM  //  TÜRKÇE DUYGU ANALİZİ")
        lbl_subtitle.setObjectName("lbl_subtitle")
        title_box.addWidget(lbl_title)
        title_box.addWidget(lbl_subtitle)

        hbox.addWidget(lbl_icon)
        hbox.addSpacing(10)
        hbox.addLayout(title_box)
        hbox.addStretch()

        self.lbl_ollama_status = QLabel("● OLLAMA: kontrol ediliyor...")
        self.lbl_ollama_status.setStyleSheet(
            f"color: {CYBER_ORANGE}; font-size: 11px; font-weight: bold;"
        )
        hbox.addWidget(self.lbl_ollama_status)

        parent_layout.addWidget(hdr)

    # ── Sol Panel ─────────────────────────────────────────────────────────────
    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(380)
        panel.setMaximumWidth(460)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(8)

        # Veri Girişi
        grp_data = QGroupBox("[ VERİ GİRİŞİ ]")
        gd_lay = QVBoxLayout(grp_data)

        lbl_csv = QLabel("CSV Dosyası:")
        lbl_csv.setStyleSheet(f"color: {CYBER_TEXT_DIM}; font-size: 10px;")
        self.edit_csv = DropZoneEdit("social_media_comments.csv")
        self.edit_csv.file_dropped.connect(self._on_csv_dropped)

        btn_browse = QPushButton("📂  DOSYA SEÇ")
        btn_browse.clicked.connect(self._browse_csv)

        gd_lay.addWidget(lbl_csv)
        gd_lay.addWidget(self.edit_csv)
        gd_lay.addWidget(btn_browse)
        layout.addWidget(grp_data)

        # Model Seçimi
        grp_model = QGroupBox("[ MODEL SEÇİMİ ]")
        gm_lay = QGridLayout(grp_model)
        gm_lay.setSpacing(8)

        lbl_classic = QLabel("Klasik Model:")
        lbl_classic.setStyleSheet(f"color: {CYBER_TEXT_DIM}; font-size: 10px;")
        self.combo_classic = QComboBox()
        self.combo_classic.addItems([
            "Linear SVM",
            "Logistic Regression",
            "Naive Bayes",
            "Random Forest",
        ])

        lbl_ollama_model = QLabel("Ollama Modeli:")
        lbl_ollama_model.setStyleSheet(f"color: {CYBER_TEXT_DIM}; font-size: 10px;")
        self.combo_ollama = QComboBox()
        self.combo_ollama.setEditable(True)

        btn_refresh = QPushButton("↺  YENİLE")
        btn_refresh.setFixedWidth(90)
        btn_refresh.clicked.connect(self._refresh_ollama_models)

        gm_lay.addWidget(lbl_classic, 0, 0)
        gm_lay.addWidget(self.combo_classic, 0, 1, 1, 2)
        gm_lay.addWidget(lbl_ollama_model, 1, 0)
        gm_lay.addWidget(self.combo_ollama, 1, 1)
        gm_lay.addWidget(btn_refresh, 1, 2)
        layout.addWidget(grp_model)

        # Dinamik Ayarlar
        grp_settings = QGroupBox("[ DİNAMİK AYARLAR ]")
        gs_lay = QGridLayout(grp_settings)
        gs_lay.setSpacing(10)

        lbl_sample = QLabel("SAMPLE_SIZE")
        lbl_sample.setStyleSheet(f"color: {CYBER_TEXT_DIM}; font-size: 10px;")
        self.spin_sample = QSpinBox()
        self.spin_sample.setRange(0, 50000)
        self.spin_sample.setValue(0)
        self.spin_sample.setSpecialValueText("Tümü (None)")
        self.spin_sample.setSingleStep(50)
        lbl_sample_hint = QLabel("0 = tüm test seti")
        lbl_sample_hint.setStyleSheet(f"color: {CYBER_TEXT_DIM}; font-size: 9px;")

        lbl_workers = QLabel("MAX_WORKERS")
        lbl_workers.setStyleSheet(f"color: {CYBER_TEXT_DIM}; font-size: 10px;")
        workers_row = QHBoxLayout()
        self.slider_workers = QSlider(Qt.Horizontal)
        self.slider_workers.setRange(1, 16)
        self.slider_workers.setValue(1)
        self.slider_workers.setTickInterval(1)
        self.slider_workers.setTickPosition(QSlider.TicksBelow)
        self.lbl_workers_val = QLabel("1")
        self.lbl_workers_val.setStyleSheet(
            f"color: {CYBER_CYAN}; font-weight: bold; min-width: 22px;"
        )
        self.slider_workers.valueChanged.connect(
            lambda v: self.lbl_workers_val.setText(str(v))
        )
        workers_row.addWidget(self.slider_workers)
        workers_row.addWidget(self.lbl_workers_val)

        gs_lay.addWidget(lbl_sample, 0, 0)
        gs_lay.addWidget(self.spin_sample, 0, 1)
        gs_lay.addWidget(lbl_sample_hint, 0, 2)
        gs_lay.addWidget(lbl_workers, 1, 0)
        gs_lay.addLayout(workers_row, 1, 1, 1, 2)

        layout.addWidget(grp_settings)

        # İlerleme
        grp_prog = QGroupBox("[ İLERLEME ]")
        gp_lay = QVBoxLayout(grp_prog)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setFixedHeight(26)

        self.lbl_progress_msg = QLabel("Bekleniyor...")
        self.lbl_progress_msg.setObjectName("lbl_status")
        self.lbl_progress_msg.setWordWrap(True)

        gp_lay.addWidget(self.progress_bar)
        gp_lay.addWidget(self.lbl_progress_msg)
        layout.addWidget(grp_prog)

        # ── Kontrol butonları ─────────────────────────────────────────────────
        self.btn_run = QPushButton("▶  ANALİZİ BAŞLAT")
        self.btn_run.setObjectName("btn_run")
        self.btn_run.clicked.connect(self._start_analysis)

        self.btn_stop = QPushButton("■  DURDUR")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_analysis)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_stop)
        layout.addLayout(btn_row)

        # ── Karşılaştırma butonu ──────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {CYBER_BORDER};")
        layout.addWidget(sep)

        self.btn_compare = QPushButton("⚖  MODEL KARŞILAŞTIR")
        self.btn_compare.setObjectName("btn_compare")
        self.btn_compare.setToolTip(
            "outputs_hybrid klasöründeki tamamlanmış model sonuçlarını karşılaştırır"
        )
        self.btn_compare.clicked.connect(self._open_compare)
        layout.addWidget(self.btn_compare)

        layout.addStretch()
        return panel

    # ── Sağ Panel ─────────────────────────────────────────────────────────────
    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        metrics_row = QHBoxLayout()
        self.card_svm    = MetricCard("SVM Doğruluk",    "—", CYBER_BLUE)
        self.card_ollama = MetricCard("Ollama Doğruluk", "—", CYBER_PURPLE)
        self.card_hybrid = MetricCard("Hibrit Doğruluk", "—", CYBER_CYAN)
        self.card_con    = MetricCard("Consensus",       "—", CYBER_ORANGE)
        self.card_conf   = MetricCard("Conflict",        "—", CYBER_RED)
        for card in [self.card_svm, self.card_ollama, self.card_hybrid,
                     self.card_con, self.card_conf]:
            metrics_row.addWidget(card)
        layout.addLayout(metrics_row)

        self.main_tabs = QTabWidget()

        self.results_tab = ResultsTab()
        self.main_tabs.addTab(self.results_tab, "📊  GRAFİKLER")

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setPlaceholderText("Terminal çıktısı burada görünecek...")
        self.main_tabs.addTab(self.log_edit, "🖥  TERMINAL LOG")

        layout.addWidget(self.main_tabs, 1)
        return panel

    # ── Durum Çubuğu ─────────────────────────────────────────────────────────
    def _build_status_bar(self, parent_layout):
        bar = QFrame()
        bar.setFixedHeight(26)
        bar.setStyleSheet(f"""
            QFrame {{
                background-color: {CYBER_BG3};
                border-top: 1px solid {CYBER_BORDER};
            }}
        """)
        hbox = QHBoxLayout(bar)
        hbox.setContentsMargins(12, 2, 12, 2)

        self.lbl_status = QLabel("■  HAZIR")
        self.lbl_status.setObjectName("lbl_status")
        ver_lbl = QLabel("HYBRID AGENT v2.1  |  PySide6  |  SVM + Ollama  |  Multi-Model Comparator")
        ver_lbl.setStyleSheet(f"color: {CYBER_TEXT_DIM}; font-size: 10px;")

        hbox.addWidget(self.lbl_status)
        hbox.addStretch()
        hbox.addWidget(ver_lbl)
        parent_layout.addWidget(bar)

    # ══════════════════════════════════════════════════════════════════════════
    # OLLAMA MODEL LİSTESİ
    # ══════════════════════════════════════════════════════════════════════════
    def _refresh_ollama_models(self):
        models = []

        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True, timeout=8
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n")[1:]:
                    parts = line.split()
                    if parts:
                        models.append(parts[0])
                self.lbl_ollama_status.setText(f"● OLLAMA: {len(models)} model bulundu")
                self.lbl_ollama_status.setStyleSheet(
                    f"color: {CYBER_CYAN}; font-size: 11px; font-weight: bold;"
                )
        except Exception:
            self.lbl_ollama_status.setText("● OLLAMA: bağlanamadı")
            self.lbl_ollama_status.setStyleSheet(
                f"color: {CYBER_RED}; font-size: 11px; font-weight: bold;"
            )

        custom_model_dir = r"D:\OllamaModels"
        if os.path.isdir(custom_model_dir):
            for root, dirs, files in os.walk(custom_model_dir):
                for f in files:
                    if "latest" in f or f.endswith(".json"):
                        rel = os.path.relpath(root, custom_model_dir)
                        candidate = rel.replace(os.sep, "/") + ":" + f.replace(".json", "")
                        if len(candidate) < 60 and candidate not in models:
                            models.append(candidate)

        if not models:
            models = ["llama3.2:latest", "aya-expanse:latest", "mistral:latest"]

        self.combo_ollama.clear()
        self.combo_ollama.addItems(models)

        for i, m in enumerate(models):
            if "aya-expanse" in m:
                self.combo_ollama.setCurrentIndex(i)
                break

    # ══════════════════════════════════════════════════════════════════════════
    # CSV SEÇİM
    # ══════════════════════════════════════════════════════════════════════════
    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "CSV Dosyası Seç", "", "CSV Dosyaları (*.csv);;Tüm Dosyalar (*)"
        )
        if path:
            self.edit_csv.setText(path)

    def _on_csv_dropped(self, path: str):
        self._log(f"  ✓ CSV sürükle-bırak: {path}")

    # ══════════════════════════════════════════════════════════════════════════
    # ANALİZ BAŞLAT / DURDUR
    # ══════════════════════════════════════════════════════════════════════════
    def _start_analysis(self):
        csv_path = self.edit_csv.text().strip()
        if not csv_path:
            QMessageBox.warning(self, "Eksik Parametre", "Lütfen bir CSV dosyası seçin!")
            return
        if not os.path.exists(csv_path):
            QMessageBox.warning(self, "Dosya Bulunamadı", f"Dosya mevcut değil:\n{csv_path}")
            return

        ollama_model = self.combo_ollama.currentText().strip()
        if not ollama_model:
            QMessageBox.warning(self, "Eksik Parametre", "Ollama modeli seçin!")
            return

        sample_val = self.spin_sample.value()
        config = {
            "csv_path":      csv_path,
            "classic_model": self.combo_classic.currentText(),
            "ollama_model":  ollama_model,
            "sample_size":   sample_val if sample_val > 0 else None,
            "max_workers":   self.slider_workers.value(),
        }

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_compare.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_edit.clear()
        self._reset_metrics()
        self.lbl_status.setText("⚡  ANALİZ ÇALIŞIYOR...")
        self.main_tabs.setCurrentIndex(1)

        self._log("=" * 60)
        self._log("  HİBRİT KARAR AJANI — ANALİZ BAŞLADI")
        self._log("=" * 60)
        self._log(f"  CSV          : {csv_path}")
        self._log(f"  Klasik Model : {config['classic_model']}")
        self._log(f"  Ollama Model : {ollama_model}")
        self._log(f"  Örnek Boyutu : {config['sample_size'] or 'Tümü'}")
        self._log(f"  Max Workers  : {config['max_workers']}")
        self._log("=" * 60)

        self.worker = AnalysisWorker(config)
        self.worker.progress.connect(self._on_progress)
        self.worker.log_message.connect(self._log)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _stop_analysis(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.lbl_status.setText("⚠  DURDURULDU")
            self.btn_stop.setEnabled(False)
            self.btn_run.setEnabled(True)
            self.btn_compare.setEnabled(True)

    # ══════════════════════════════════════════════════════════════════════════
    # KARŞILAŞTIRMA AÇ
    # ══════════════════════════════════════════════════════════════════════════
    def _open_compare(self):
        """Model karşılaştırma diyalogunu açar."""
        if self._compare_dialog is None or not self._compare_dialog.isVisible():
            self._compare_dialog = CompareDialog(parent=self)
        self._compare_dialog.show()
        self._compare_dialog.raise_()
        self._compare_dialog.activateWindow()

    # ══════════════════════════════════════════════════════════════════════════
    # SIGNAL HANDLERS
    # ══════════════════════════════════════════════════════════════════════════
    def _on_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.lbl_progress_msg.setText(msg)

    def _log(self, msg: str):
        self.log_edit.append(msg)
        sb = self.log_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_finished(self, result: dict):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_compare.setEnabled(True)
        self.lbl_status.setText("✅  ANALİZ TAMAMLANDI")

        self.card_svm.set_value(f"{result['svm_acc']:.3f}")
        self.card_ollama.set_value(f"{result['ollama_acc']:.3f}")
        self.card_hybrid.set_value(f"{result['hybrid_acc']:.3f}")
        self.card_con.set_value(str(result["consensus_n"]))
        self.card_conf.set_value(str(result["conflict_n"]))

        self.results_tab.load_plots(result["plot_paths"])
        self.main_tabs.setCurrentIndex(0)

        self._log("\n" + "=" * 60)
        self._log(f"  ✅  TAMAMLANDI  |  Toplam: {result['total_n']}")
        self._log(f"  SVM Doğruluk    : {result['svm_acc']:.4f}")
        self._log(f"  Ollama Doğruluk : {result['ollama_acc']:.4f}")
        self._log(f"  Hibrit Doğruluk : {result['hybrid_acc']:.4f}")
        self._log(f"  Consensus       : {result['consensus_n']}")
        self._log(f"  Conflict        : {result['conflict_n']}")
        self._log(f"  Çıktı dizini    : {result['out_dir']}/")
        self._log("=" * 60)

        QMessageBox.information(
            self, "✅  ANALİZ TAMAMLANDI",
            f"Hibrit Doğruluk  : {result['hybrid_acc']:.4f}\n"
            f"SVM Doğruluk     : {result['svm_acc']:.4f}\n"
            f"Ollama Doğruluk  : {result['ollama_acc']:.4f}\n\n"
            f"Consensus : {result['consensus_n']}\n"
            f"Conflict  : {result['conflict_n']}\n\n"
            f"Çıktılar → {result['out_dir']}/"
        )

    def _on_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_compare.setEnabled(True)
        self.lbl_status.setText("✗  HATA")
        self._log(f"\n[HATA]\n{msg}")
        QMessageBox.critical(self, "Hata", f"Analiz sırasında hata oluştu:\n\n{msg[:500]}")

    def _reset_metrics(self):
        for card in [self.card_svm, self.card_ollama, self.card_hybrid,
                     self.card_con, self.card_conf]:
            card.set_value("—")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Hybrid Decision Agent — Cyberpunk GUI")
    app.setStyleSheet(CYBERPUNK_QSS)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    font = QFont("Consolas", 10)
    app.setFont(font)

    win = CyberHybridGUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()