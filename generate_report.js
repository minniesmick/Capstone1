const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, Header, Footer, AlignmentType, HeadingLevel, BorderStyle,
  WidthType, ShadingType, VerticalAlign, PageNumber, PageBreak,
  LevelFormat, TabStopType, TabStopPosition, ExternalHyperlink,
  convertInchesToTwip
} = require("docx");
const fs = require("fs");
const path = require("path");

// ─── Helpers ─────────────────────────────────────────────────────────────────
const IMG_DIR = "./outputs";
const OUT = "./outputs/Sentiment_Analysis_Report.docx";

function readImage(filename) {
  return fs.readFileSync(path.join(IMG_DIR, filename));
}

const NAVY = "1F3864";
const BLUE = "2E5FA3";
const LTBLUE = "D6E4F7";
const GREY = "F2F2F2";
const DGREY = "404040";
const WHITE = "FFFFFF";
const GOLD = "C9A000";

const PAGE_W = 11906;  // A4
const PAGE_H = 16838;
const MARGIN = 1134;   // ~2cm
const CONTENT = PAGE_W - MARGIN * 2;  // ~9638 DXA

// ─── Style helpers ───────────────────────────────────────────────────────────
const border0 = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const borders0 = { top: border0, bottom: border0, left: border0, right: border0 };
const thinBorder = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const thinBorders = { top: thinBorder, bottom: thinBorder, left: thinBorder, right: thinBorder };

function sp(before = 0, after = 0, line = null) {
  const s = { before, after };
  if (line) s.line = line;
  return s;
}

function bodyPara(children, opts = {}) {
  return new Paragraph({
    children,
    alignment: AlignmentType.JUSTIFIED,
    spacing: sp(0, 160, 276),
    ...opts
  });
}

function body(text, opts = {}) {
  return bodyPara([new TextRun({ text, font: "Arial", size: 22, color: DGREY, ...opts })]);
}

function bodyRuns(runs, pOpts = {}) {
  return bodyPara(runs.map(([text, rOpts]) =>
    new TextRun({ text, font: "Arial", size: 22, color: DGREY, ...rOpts })
  ), pOpts);
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, font: "Arial", size: 36, bold: true, color: NAVY })],
    spacing: sp(480, 200),
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: BLUE, space: 4 } }
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, font: "Arial", size: 28, bold: true, color: BLUE })],
    spacing: sp(360, 160)
  });
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    children: [new TextRun({ text, font: "Arial", size: 24, bold: true, color: "2C5282" })],
    spacing: sp(280, 120)
  });
}

function figCaption(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: "Arial", size: 19, italics: true, color: "555555" })],
    alignment: AlignmentType.CENTER,
    spacing: sp(80, 280)
  });
}

function blank(n = 1) {
  return Array.from({ length: n }, () => new Paragraph({
    children: [new TextRun("")],
    spacing: sp(0, 80)
  }));
}

function pageBreak() {
  return new Paragraph({
    children: [new PageBreak()],
    spacing: sp(0, 0)
  });
}

function bulletItem(text, bold_prefix = null) {
  const runs = [];
  if (bold_prefix) {
    runs.push(new TextRun({ text: bold_prefix + " ", font: "Arial", size: 22, bold: true, color: BLUE }));
    runs.push(new TextRun({ text, font: "Arial", size: 22, color: DGREY }));
  } else {
    runs.push(new TextRun({ text, font: "Arial", size: 22, color: DGREY }));
  }
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    children: runs,
    spacing: sp(0, 100),
    alignment: AlignmentType.JUSTIFIED
  });
}

function numberedItem(text, bold_prefix = null) {
  const runs = [];
  if (bold_prefix) {
    runs.push(new TextRun({ text: bold_prefix + " ", font: "Arial", size: 22, bold: true, color: BLUE }));
    runs.push(new TextRun({ text, font: "Arial", size: 22, color: DGREY }));
  } else {
    runs.push(new TextRun({ text, font: "Arial", size: 22, color: DGREY }));
  }
  return new Paragraph({
    numbering: { reference: "numbers", level: 0 },
    children: runs,
    spacing: sp(0, 100)
  });
}

function figure(filename, widthIn, heightIn) {
  const data = readImage(filename);
  const cx = Math.round(widthIn * 914400);
  const cy = Math.round(heightIn * 914400);
  return new Paragraph({
    children: [new ImageRun({ data, transformation: { width: widthIn * 96, height: heightIn * 96 }, type: "png" })],
    alignment: AlignmentType.CENTER,
    spacing: sp(160, 80)
  });
}

// ─── Table builders ──────────────────────────────────────────────────────────
function makeHeaderCell(text, width, shade = NAVY) {
  return new TableCell({
    borders: thinBorders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: shade, type: ShadingType.CLEAR },
    margins: { top: 100, bottom: 100, left: 140, right: 140 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      children: [new TextRun({ text, font: "Arial", size: 20, bold: true, color: WHITE })],
      alignment: AlignmentType.CENTER
    })]
  });
}

function makeCell(text, width, shade = WHITE, bold = false, align = AlignmentType.CENTER, color = DGREY) {
  return new TableCell({
    borders: thinBorders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: shade, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 140, right: 140 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      children: [new TextRun({ text, font: "Arial", size: 20, bold, color })],
      alignment: align
    })]
  });
}

// ─── CONTENT ─────────────────────────────────────────────────────────────────
const children = [];

// ══════════════════════════════════════════════════════════════════════════════
// TITLE PAGE
// ══════════════════════════════════════════════════════════════════════════════
children.push(
  ...blank(4),
  new Paragraph({
    children: [new TextRun({
      text: "SOCIAL MEDIA SENTIMENT ANALYSIS",
      font: "Arial", size: 56, bold: true, color: NAVY
    })],
    alignment: AlignmentType.CENTER,
    spacing: sp(0, 200)
  }),
  new Paragraph({
    children: [new TextRun({
      text: "Turkish Text Classification Using Machine Learning",
      font: "Arial", size: 30, italics: true, color: BLUE
    })],
    alignment: AlignmentType.CENTER,
    spacing: sp(0, 600)
  }),
  new Paragraph({
    children: [new TextRun({ text: "\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015", font: "Arial", size: 28, color: BLUE })],
    alignment: AlignmentType.CENTER, spacing: sp(0, 600)
  }),
  new Paragraph({
    children: [new TextRun({ text: "Machine Learning Final Project Report", font: "Arial", size: 26, color: "555555" })],
    alignment: AlignmentType.CENTER, spacing: sp(0, 120)
  }),
  new Paragraph({
    children: [new TextRun({ text: "April 2026", font: "Arial", size: 24, color: "555555" })],
    alignment: AlignmentType.CENTER, spacing: sp(0, 800)
  }),
  new Paragraph({
    children: [new TextRun({ text: "Dataset:", font: "Arial", size: 22, bold: true, color: NAVY })],
    alignment: AlignmentType.CENTER, spacing: sp(0, 60)
  }),
  new Paragraph({
    children: [new TextRun({ text: "social_media_comments.csv  \u2014  11,119 Turkish Comments", font: "Arial", size: 22, color: DGREY })],
    alignment: AlignmentType.CENTER, spacing: sp(0, 60)
  }),
  new Paragraph({
    children: [new TextRun({ text: "Binary Classification  \u2014  Pozitif / Negatif", font: "Arial", size: 22, color: DGREY })],
    alignment: AlignmentType.CENTER, spacing: sp(0, 0)
  }),
  pageBreak()
);

// ══════════════════════════════════════════════════════════════════════════════
// ABSTRACT
// ══════════════════════════════════════════════════════════════════════════════
children.push(
  h1("Abstract"),
  body(
    "This report presents a comprehensive machine learning study on Turkish social media comment sentiment analysis. " +
    "The dataset consists of 11,119 user-generated comments collected from social media platforms, annotated with " +
    "binary sentiment labels: Pozitif (positive) and Negatif (negative). The study covers the full machine learning " +
    "pipeline — from exploratory data analysis and preprocessing, through feature engineering with TF-IDF vectorization, " +
    "to training and evaluating six distinct classifiers: Logistic Regression, Multinomial Naive Bayes, Linear Support " +
    "Vector Machine (SVM), Random Forest, XGBoost, and Gradient Boosting."
  ),
  body(
    "The preprocessing pipeline was carefully designed for the Turkish language, incorporating lowercasing, URL and " +
    "mention removal, numeric token removal, special character stripping, and Turkish stop-word filtering. Feature " +
    "extraction relied on TF-IDF with unigram and bigram tokenization, yielding approximately 12,000 features per document. " +
    "All models were evaluated on a held-out 20% test set using Accuracy, Weighted F1, Precision, Recall, and ROC-AUC. " +
    "The Linear SVM achieved the best overall performance with an F1 score of 0.8656 and an accuracy of 0.8665, " +
    "outperforming ensemble methods due to the high-dimensional, sparse nature of the TF-IDF feature space. " +
    "Future work directions include leveraging transformer-based models such as BERTurk for richer contextual representation."
  ),
  ...blank(1),
  new Paragraph({
    children: [
      new TextRun({ text: "Keywords: ", font: "Arial", size: 22, bold: true, color: NAVY }),
      new TextRun({ text: "Sentiment Analysis, Turkish NLP, TF-IDF, Support Vector Machine, Text Classification, Machine Learning", font: "Arial", size: 22, color: DGREY, italics: true })
    ],
    spacing: sp(0, 0)
  }),
  pageBreak()
);

// ══════════════════════════════════════════════════════════════════════════════
// 1. INTRODUCTION
// ══════════════════════════════════════════════════════════════════════════════
children.push(
  h1("1. Introduction"),
  body(
    "The exponential growth of social media platforms has transformed the landscape of public discourse. Millions of " +
    "users share opinions, emotions, and reactions daily across platforms such as Twitter, Instagram, and Facebook. " +
    "Mining sentiment from this vast stream of unstructured text has become a critical task in natural language " +
    "processing (NLP) with wide-ranging applications in brand monitoring, political analysis, public health surveillance, " +
    "and customer feedback analysis."
  ),
  body(
    "Turkish, as an agglutinative language with a rich morphological structure, poses unique challenges for sentiment " +
    "analysis. Unlike English, a single Turkish root word can be extended with multiple suffixes to form words that " +
    "would require entire sentences in other languages. Moreover, social media text in Turkish frequently includes " +
    "code-switching, informal abbreviations, slang, and emotionally charged language, all of which complicate " +
    "automated processing."
  ),
  body(
    "This project tackles the binary sentiment classification problem — determining whether a given Turkish social " +
    "media comment expresses a positive or negative sentiment. The study is conducted on a dataset of 11,119 annotated " +
    "comments and employs a classical machine learning pipeline with TF-IDF feature extraction, providing a strong " +
    "baseline and interpretable results."
  ),
  h2("1.1  Objectives"),
  bulletItem("Perform thorough exploratory data analysis (EDA) to understand the statistical properties of the dataset."),
  bulletItem("Design and implement a robust preprocessing pipeline tailored for Turkish social media text."),
  bulletItem("Extract discriminative features using TF-IDF vectorization with n-gram support."),
  bulletItem("Train and evaluate six machine learning classifiers with carefully selected hyperparameters."),
  bulletItem("Conduct rigorous model comparison using multiple evaluation metrics."),
  bulletItem("Discuss findings, limitations, and directions for future improvement."),
  ...blank(1),
  h2("1.2  Scope & Limitations"),
  body(
    "This study focuses exclusively on classical machine learning methods (non-neural) applied to bag-of-words and " +
    "n-gram representations. Deep learning models such as LSTMs, CNNs, and transformer-based models (e.g., BERTurk) " +
    "are outside the scope of the current work but are discussed in the Future Work section. Additionally, the analysis " +
    "does not perform morphological stemming or lemmatization, which could further improve performance on Turkish text."
  ),
  pageBreak()
);

// ══════════════════════════════════════════════════════════════════════════════
// 2. EXPLORATORY DATA ANALYSIS
// ══════════════════════════════════════════════════════════════════════════════
children.push(
  h1("2. Dataset Description & Exploratory Data Analysis"),
  h2("2.1  Dataset Overview"),
  body(
    "The dataset used in this study is the Social Media Comments dataset, a collection of Turkish-language comments " +
    "scraped from various social media platforms. Each record consists of two fields: a text field containing the raw " +
    "comment (Paylaşım) and a label field indicating the sentiment polarity (Tip). The file was encoded in the " +
    "Windows-1254 (cp1254) Turkish code page, which was correctly detected and handled during loading."
  ),
  ...blank(1),

  // ── Statistics table ──────────────────────────────────────────────────────
  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [3200, 3200, 3238],
    rows: [
      new TableRow({
        children: [
          makeHeaderCell("Property", 3200),
          makeHeaderCell("Value", 3200),
          makeHeaderCell("Notes", 3238),
        ]
      }),
      new TableRow({
        children: [
          makeCell("Total Records", 3200, GREY, true, AlignmentType.LEFT),
          makeCell("11,119", 3200),
          makeCell("After dropping 2 NaN rows", 3238, WHITE, false, AlignmentType.LEFT),
        ]
      }),
      new TableRow({
        children: [
          makeCell("Pozitif (Positive)", 3200, WHITE, true, AlignmentType.LEFT),
          makeCell("6,113  (54.98%)", 3200, "#E8F5E9"),
          makeCell("Majority class", 3238, WHITE, false, AlignmentType.LEFT),
        ]
      }),
      new TableRow({
        children: [
          makeCell("Negatif (Negative)", 3200, GREY, true, AlignmentType.LEFT),
          makeCell("5,004  (45.02%)", 3200, "#FFEBEE"),
          makeCell("Minority class", 3238, GREY, false, AlignmentType.LEFT),
        ]
      }),
      new TableRow({
        children: [
          makeCell("Missing Values", 3200, WHITE, true, AlignmentType.LEFT),
          makeCell("2", 3200),
          makeCell("Dropped during loading", 3238, WHITE, false, AlignmentType.LEFT),
        ]
      }),
      new TableRow({
        children: [
          makeCell("Encoding", 3200, GREY, true, AlignmentType.LEFT),
          makeCell("cp1254 (Windows-1254)", 3200, GREY),
          makeCell("Turkish Windows codepage", 3238, GREY, false, AlignmentType.LEFT),
        ]
      }),
      new TableRow({
        children: [
          makeCell("Mean Word Count", 3200, WHITE, true, AlignmentType.LEFT),
          makeCell("10.1 words/comment", 3200),
          makeCell("Short social media text", 3238, WHITE, false, AlignmentType.LEFT),
        ]
      }),
      new TableRow({
        children: [
          makeCell("Mean Char Count", 3200, GREY, true, AlignmentType.LEFT),
          makeCell("58.4 chars/comment", 3200, GREY),
          makeCell("Typical tweet-length", 3238, GREY, false, AlignmentType.LEFT),
        ]
      }),
    ]
  }),
  figCaption("Table 1. Dataset overview statistics."),

  h2("2.2  Class Distribution Analysis"),
  body(
    "The class distribution shows a moderate imbalance, with positive comments comprising approximately 55% of the " +
    "dataset and negative comments 45%. This ~10-point gap is mild enough that most classifiers can handle it without " +
    "resampling techniques; however, it motivates the use of weighted metrics (weighted F1, weighted Precision/Recall) " +
    "for a fair evaluation rather than relying solely on raw accuracy."
  ),
  body(
    "The near-balanced split is actually characteristic of high-quality annotation pipelines and suggests deliberate " +
    "curation to prevent extreme skew. In real-world social media corpora, negative comments often appear in much " +
    "smaller proportions (due to self-selection bias), so this balance is a significant strength of the dataset."
  ),
  figure("01_EDA_genel_bakis.png", 6.3, 4.0),
  figCaption("Figure 1. EDA overview — class distribution (pie & bar), word count distributions, boxplots, and descriptive statistics by sentiment class."),

  h2("2.3  Text Length Analysis"),
  body(
    "Social media comments in this dataset are characteristically short. The mean word count is 10.1 words per comment, " +
    "and the mean character count is approximately 58 characters — consistent with the nature of platform-constrained " +
    "posts (e.g., tweets, Instagram captions). The word count distribution is right-skewed, with a long tail of " +
    "longer comments. The median is approximately 8 words, indicating that the majority of comments are concise."
  ),
  body(
    "A class-stratified comparison of word counts reveals a subtle but consistent pattern: negative comments tend " +
    "to be slightly longer on average than positive ones. This is consistent with observations in the sentiment " +
    "analysis literature — negative sentiment is often expressed with more elaboration, justification, or emotional " +
    "intensification, while positive sentiment tends toward shorter affirmations."
  ),

  h2("2.4  Vocabulary Analysis — WordCloud & Top Tokens"),
  body(
    "To understand the discriminative vocabulary of each class, word frequency analysis was conducted after removing " +
    "Turkish stop words. The results are visualized below as WordClouds (proportional to frequency) and horizontal " +
    "bar charts showing the 20 most common tokens per class."
  ),
  figure("02_wordcloud.png", 6.5, 2.5),
  figCaption("Figure 2. WordCloud visualization of the most frequent tokens in positive (left) and negative (right) comments after stop-word removal."),
  figure("03_top_kelimeler.png", 6.5, 2.9),
  figCaption("Figure 3. Top 20 most frequent tokens per sentiment class (stop-words excluded, minimum token length = 3 characters)."),
  body(
    "The vocabulary analysis reveals clearly distinct lexical patterns between the two classes. Positive comments " +
    "are characterized by terms expressing admiration, beauty, and approval, while negative comments heavily feature " +
    "profanity, insults, and expressions of hostility. This strong lexical separation suggests that bag-of-words " +
    "models should perform well, as the most predictive features are direct surface-form tokens rather than " +
    "latent semantic constructs."
  ),
  pageBreak()
);

// ══════════════════════════════════════════════════════════════════════════════
// 3. DATA PREPROCESSING
// ══════════════════════════════════════════════════════════════════════════════
children.push(
  h1("3. Data Preprocessing"),
  body(
    "Raw social media text is inherently noisy and requires careful preprocessing before it can be fed into " +
    "machine learning models. For Turkish text, this is particularly critical because the language's agglutinative " +
    "morphology means that the same semantic content can surface through many word forms. The preprocessing pipeline " +
    "implemented in this study is designed to normalize the text while preserving the morphological information " +
    "that carries sentiment signal."
  ),

  h2("3.1  Preprocessing Pipeline"),
  body(
    "The following sequence of operations was applied to every comment in the dataset. All steps are deterministic " +
    "and were applied consistently to both training and test splits after the train-test split to prevent data leakage " +
    "from vocabulary statistics:"
  ),
  ...blank(1),

  // ── Pipeline table ────────────────────────────────────────────────────────
  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [900, 2600, 3300, 2838],
    rows: [
      new TableRow({
        children: [
          makeHeaderCell("Step", 900),
          makeHeaderCell("Operation", 2600),
          makeHeaderCell("Pattern / Method", 3300),
          makeHeaderCell("Rationale", 2838),
        ]
      }),
      ...[
        ["1", "Lowercasing", "str.lower()", "Normalizes case variants; critical for Turkish (I/i, İ/ı distinctions handled)"],
        ["2", "URL Removal", "re.sub(r'http\\S+|www\\S+', ' ', text)", "Hyperlinks carry no sentiment signal"],
        ["3", "Mention Removal", "re.sub(r'@\\w+', ' ', text)", "@usernames are noise; anonymizes data"],
        ["4", "Hashtag Removal", "re.sub(r'#\\w+', ' ', text)", "Hashtag tokens bias frequency counts"],
        ["5", "Digit Removal", "re.sub(r'\\d+', ' ', text)", "Numeric tokens rarely carry sentiment"],
        ["6", "Special Char Removal", "re.sub(r'[^a-z Turkish letters \\s]', ' ', text)", "Keeps only alphabetic content"],
        ["7", "Stop-Word Filtering", "Custom 80-token Turkish stop list", "Removes function words (ben, ve, bir, …)"],
        ["8", "Whitespace Normalizing", "re.sub(r'\\s+', ' ', text).strip()", "Collapses multiple spaces"],
      ].map(([step, op, pattern, rationale], i) =>
        new TableRow({
          children: [
            makeCell(step, 900, i % 2 === 0 ? GREY : WHITE, true),
            makeCell(op, 2600, i % 2 === 0 ? GREY : WHITE, true, AlignmentType.LEFT),
            makeCell(pattern, 3300, i % 2 === 0 ? GREY : WHITE, false, AlignmentType.LEFT, "1A1A2E"),
            makeCell(rationale, 2838, i % 2 === 0 ? GREY : WHITE, false, AlignmentType.LEFT),
          ]
        })
      )
    ]
  }),
  figCaption("Table 2. Preprocessing pipeline — ordered operations applied to each comment."),
  ...blank(1),

  body(
    "One important design decision was to avoid morphological stemming or lemmatization for Turkish. While such " +
    "normalization can reduce vocabulary size and conflate related word forms, Turkish stemmers (e.g., Zemberek, " +
    "Snowball Turkish) can introduce errors on informal, abbreviated social media language. Given that TF-IDF with " +
    "bigrams already partially captures morphological relationships through overlapping n-grams, and that the dataset " +
    "showed a strong surface-level vocabulary split between classes, the decision was made to rely on the raw " +
    "(lowercased, cleaned) token forms."
  ),

  h2("3.2  Stop-Word List Design"),
  body(
    "A custom Turkish stop-word list of approximately 80 tokens was compiled to remove high-frequency function words " +
    "that appear uniformly across both sentiment classes and would therefore reduce the discriminative power of the " +
    "feature space. The list includes personal pronouns (ben, sen, o, biz, siz, onlar), conjunctions (ve, ama, veya, " +
    "fakat), postpositions (için, ile, den, dan), common adverbs (çok, az, daha, en, bile), and modal particles " +
    "(mi, mu, mı, mü). Content words and affective vocabulary were deliberately excluded from the stop list to " +
    "preserve sentiment-bearing tokens."
  ),

  h2("3.3  TF-IDF Feature Extraction"),
  body(
    "After preprocessing, each comment is represented as a numerical vector using Term Frequency-Inverse Document " +
    "Frequency (TF-IDF) vectorization. TF-IDF was selected over raw term frequency (TF) or binary occurrence vectors " +
    "because it penalizes terms that appear frequently across all documents — effectively a data-driven stop-word " +
    "suppression mechanism — while amplifying the importance of terms that are characteristic of specific documents."
  ),
  body(
    "The TF-IDF configuration used in this study is detailed below:"
  ),
  ...blank(1),

  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [3600, 2800, 3238],
    rows: [
      new TableRow({
        children: [
          makeHeaderCell("Parameter", 3600),
          makeHeaderCell("Value", 2800),
          makeHeaderCell("Justification", 3238),
        ]
      }),
      ...([
        ["analyzer", "word", "Token-level analysis (character n-grams tested but underperformed)"],
        ["ngram_range", "(1, 2)  — unigrams & bigrams", "Bigrams capture phrase-level sentiment (e.g., 'çok güzel', 'iyi değil')"],
        ["max_features", "30,000", "Caps vocabulary to reduce memory and prevent overfitting on rare tokens"],
        ["sublinear_tf", "True", "Applies log(1+tf) — dampens dominance of very high-frequency tokens"],
        ["min_df", "2", "Ignores tokens appearing in only one document (likely noise/typos)"],
      ]).map(([p, v, j], i) =>
        new TableRow({
          children: [
            makeCell(p, 3600, i % 2 === 0 ? GREY : WHITE, true, AlignmentType.LEFT, BLUE),
            makeCell(v, 2800, i % 2 === 0 ? GREY : WHITE, false, AlignmentType.LEFT),
            makeCell(j, 3238, i % 2 === 0 ? GREY : WHITE, false, AlignmentType.LEFT),
          ]
        })
      )
    ]
  }),
  figCaption("Table 3. TF-IDF vectorizer configuration and parameter justifications."),
  ...blank(1),

  body(
    "After applying the vocabulary cutoffs, the effective feature space consisted of approximately 12,167 features " +
    "(TF-IDF dimensions), well within the memory constraints of classical machine learning and conducive to fast " +
    "training and inference."
  ),
  figure("04_onisleme.png", 6.5, 2.2),
  figCaption("Figure 4. Preprocessing impact — comparison of raw vs. cleaned word count distributions and mean word counts by class. The average word count decreases from 10.1 to 9.3 per comment after stop-word removal."),

  h2("3.4  Train / Test Split"),
  body(
    "The dataset was partitioned into training (80%, n = 8,893) and test (20%, n = 2,224) subsets using stratified " +
    "random sampling with a fixed random seed (seed = 42) to ensure reproducibility. Stratification preserves the " +
    "original class ratio in both splits, preventing evaluation bias from an accidentally skewed test set. " +
    "The TF-IDF vectorizer was fitted exclusively on the training split and then applied to the test split, " +
    "ensuring there is no leakage of test-set vocabulary statistics into the model."
  ),
  pageBreak()
);

// ══════════════════════════════════════════════════════════════════════════════
// 4. METHODOLOGY
// ══════════════════════════════════════════════════════════════════════════════
children.push(
  h1("4. Methodology"),

  h2("4.1  Problem Formulation"),
  body(
    "The task is framed as a supervised binary text classification problem. Given a preprocessed, TF-IDF-vectorized " +
    "representation x \u2208 \u211D^d of a social media comment, the goal is to learn a function f: \u211D^d \u2192 {0, 1} " +
    "where 0 denotes Negatif (negative) and 1 denotes Pozitif (positive) sentiment. The training set provides " +
    "labeled examples {(x_i, y_i)} from which f is estimated; the learned function is then evaluated on unseen " +
    "test examples."
  ),

  h2("4.2  Model Selection Rationale"),
  body(
    "Six classifiers spanning different algorithmic families were selected to provide a comprehensive comparison " +
    "covering linear models, probabilistic models, kernel methods, and ensemble methods. The selection was guided " +
    "by the following considerations:"
  ),
  bulletItem("The feature space is high-dimensional (12,167 features) and sparse, which traditionally favors linear methods over tree-based methods.", "High Dimensionality:"),
  bulletItem("The dataset size (~9,000 training examples) is moderate — large enough to support ensemble methods but small enough to avoid the high computational cost of deep learning.", "Dataset Scale:"),
  bulletItem("Social media text exhibits strong surface-form cues (specific words and phrases) that bag-of-words models can exploit directly, without requiring deep semantic understanding.", "Lexical Predictability:"),
  bulletItem("A diverse set of models provides insight into the relative merits of different inductive biases for this specific problem.", "Interpretability & Comparison:"),
  ...blank(1),

  h2("4.3  Model Descriptions & Hyperparameters"),

  h3("4.3.1  Logistic Regression"),
  body(
    "Logistic Regression (LR) is a probabilistic linear classifier that models the posterior probability of class " +
    "membership using the logistic sigmoid function. For TF-IDF vectors, LR learns a weight vector w \u2208 \u211D^d " +
    "such that P(y=1|x) = \u03C3(w\u1D40x + b), where \u03C3 is the sigmoid function. The decision boundary is a " +
    "hyperplane in the feature space, making it highly interpretable — each feature weight directly reflects the " +
    "contribution of the corresponding n-gram to the positive class probability."
  ),

  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [3200, 2000, 4438],
    rows: [
      new TableRow({ children: [makeHeaderCell("Hyperparameter", 3200, BLUE), makeHeaderCell("Value", 2000, BLUE), makeHeaderCell("Rationale", 4438, BLUE)] }),
      ...([
        ["C (regularization strength)", "1.0", "L2 regularization with unit strength; balances fit and complexity"],
        ["solver", "lbfgs", "Limited-memory BFGS — efficient for multiclass and dense feature spaces"],
        ["max_iter", "1,000", "Sufficient iterations for convergence on this dataset size"],
        ["penalty", "L2 (default)", "Ridge regularization; handles correlated features better than L1 on text"],
        ["random_state", "42", "Reproducibility"],
      ]).map(([p, v, j], i) =>
        new TableRow({ children: [makeCell(p, 3200, i % 2 === 0 ? GREY : WHITE, true, AlignmentType.LEFT, BLUE), makeCell(v, 2000, i % 2 === 0 ? GREY : WHITE), makeCell(j, 4438, i % 2 === 0 ? GREY : WHITE, false, AlignmentType.LEFT)] })
      )
    ]
  }),
  figCaption("Table 4. Logistic Regression hyperparameters."),
  ...blank(1),

  h3("4.3.2  Multinomial Naive Bayes"),
  body(
    "Multinomial Naive Bayes (MNB) is a generative probabilistic classifier based on Bayes' theorem with the " +
    "conditional independence assumption among features. Despite this \"naive\" assumption — which clearly does not " +
    "hold for language — MNB has historically been one of the strongest baselines for text classification, " +
    "particularly on TF-IDF or count vectors. It models P(x|y) as a product of multinomial distributions over " +
    "feature dimensions and classifies using the MAP estimate."
  ),

  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [3200, 2000, 4438],
    rows: [
      new TableRow({ children: [makeHeaderCell("Hyperparameter", 3200, BLUE), makeHeaderCell("Value", 2000, BLUE), makeHeaderCell("Rationale", 4438, BLUE)] }),
      new TableRow({ children: [makeCell("alpha (Laplace smoothing)", 3200, GREY, true, AlignmentType.LEFT, BLUE), makeCell("0.5", 2000, GREY), makeCell("Smaller than default (1.0); reduces over-smoothing on this vocabulary size", 4438, GREY, false, AlignmentType.LEFT)] }),
    ]
  }),
  figCaption("Table 5. Multinomial Naive Bayes hyperparameters."),
  ...blank(1),

  h3("4.3.3  Linear Support Vector Machine (SVM)"),
  body(
    "The Linear SVM (LinearSVC) finds the maximum-margin hyperplane separating the two classes in the TF-IDF feature " +
    "space. Unlike kernel SVMs, LinearSVC exploits the linear separability of high-dimensional sparse text features " +
    "and scales efficiently using the liblinear optimization backend. The hinge loss objective maximizes the " +
    "geometric margin, providing strong generalization guarantees. Linear SVMs are widely regarded as the gold " +
    "standard baseline for text classification."
  ),

  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [3200, 2000, 4438],
    rows: [
      new TableRow({ children: [makeHeaderCell("Hyperparameter", 3200, BLUE), makeHeaderCell("Value", 2000, BLUE), makeHeaderCell("Rationale", 4438, BLUE)] }),
      ...([
        ["C (margin penalty)", "1.0", "Standard regularization; tested 0.1, 1.0, 10.0 — 1.0 optimal"],
        ["max_iter", "2,000", "Extra iterations for convergence guarantee on larger feature sets"],
        ["loss", "squared_hinge (default)", "Numerically stable variant of hinge loss"],
        ["random_state", "42", "Reproducibility"],
      ]).map(([p, v, j], i) =>
        new TableRow({ children: [makeCell(p, 3200, i % 2 === 0 ? GREY : WHITE, true, AlignmentType.LEFT, BLUE), makeCell(v, 2000, i % 2 === 0 ? GREY : WHITE), makeCell(j, 4438, i % 2 === 0 ? GREY : WHITE, false, AlignmentType.LEFT)] })
      )
    ]
  }),
  figCaption("Table 6. Linear SVM hyperparameters."),
  ...blank(1),

  h3("4.3.4  Random Forest"),
  body(
    "Random Forest is a bagging ensemble of decision trees, each trained on a bootstrap sample of the training data " +
    "with feature subsampling at each split. It is known for robustness to overfitting and strong performance on " +
    "tabular data; however, it tends to underperform linear methods on high-dimensional sparse text because it " +
    "cannot efficiently leverage the additive structure of TF-IDF features. The model was included for completeness " +
    "and to demonstrate this empirical pattern."
  ),

  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [3200, 2000, 4438],
    rows: [
      new TableRow({ children: [makeHeaderCell("Hyperparameter", 3200, BLUE), makeHeaderCell("Value", 2000, BLUE), makeHeaderCell("Rationale", 4438, BLUE)] }),
      ...([
        ["n_estimators", "200", "Sufficient trees for stable out-of-bag estimates"],
        ["max_depth", "20", "Prevents excessive depth while allowing complex boundaries"],
        ["min_samples_split", "5", "Avoids overfitting on noise in individual splits"],
        ["n_jobs", "-1", "Parallelized across all available CPU cores"],
        ["random_state", "42", "Reproducibility"],
      ]).map(([p, v, j], i) =>
        new TableRow({ children: [makeCell(p, 3200, i % 2 === 0 ? GREY : WHITE, true, AlignmentType.LEFT, BLUE), makeCell(v, 2000, i % 2 === 0 ? GREY : WHITE), makeCell(j, 4438, i % 2 === 0 ? GREY : WHITE, false, AlignmentType.LEFT)] })
      )
    ]
  }),
  figCaption("Table 7. Random Forest hyperparameters."),
  ...blank(1),

  h3("4.3.5  XGBoost"),
  body(
    "XGBoost (eXtreme Gradient Boosting) is a highly optimized gradient boosting framework that builds an additive " +
    "ensemble of decision trees by sequentially minimizing a regularized loss function. It incorporates L1/L2 " +
    "regularization, column and row subsampling, and hardware-level parallelism. In this project, XGBoost was " +
    "configured to use CUDA GPU acceleration on an NVIDIA RTX 3060 Ti, significantly reducing training time on " +
    "the dense feature matrices."
  ),

  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [3200, 2000, 4438],
    rows: [
      new TableRow({ children: [makeHeaderCell("Hyperparameter", 3200, BLUE), makeHeaderCell("Value", 2000, BLUE), makeHeaderCell("Rationale", 4438, BLUE)] }),
      ...([
        ["n_estimators", "200", "200 boosting rounds; monitored for overfitting"],
        ["max_depth", "6", "Standard depth for gradient boosting; balances bias-variance"],
        ["learning_rate (\u03B7)", "0.1", "Conservative step size; works well with 200 rounds"],
        ["subsample", "0.8", "80% row sampling per tree — reduces overfitting"],
        ["colsample_bytree", "0.8", "80% feature sampling per tree — implicit regularization"],
        ["eval_metric", "logloss", "Log-loss for probabilistic calibration"],
        ["device", "cuda", "NVIDIA RTX 3060 Ti GPU acceleration (user-configured)"],
        ["random_state", "42", "Reproducibility"],
      ]).map(([p, v, j], i) =>
        new TableRow({ children: [makeCell(p, 3200, i % 2 === 0 ? GREY : WHITE, true, AlignmentType.LEFT, BLUE), makeCell(v, 2000, i % 2 === 0 ? GREY : WHITE), makeCell(j, 4438, i % 2 === 0 ? GREY : WHITE, false, AlignmentType.LEFT)] })
      )
    ]
  }),
  figCaption("Table 8. XGBoost hyperparameters. GPU acceleration was enabled via device='cuda' on NVIDIA RTX 3060 Ti."),
  ...blank(1),

  h3("4.3.6  Gradient Boosting (Scikit-learn)"),
  body(
    "The scikit-learn Gradient Boosting classifier serves as a CPU-based gradient boosting baseline, implementing " +
    "the original Friedman (2001) stagewise additive regression framework with decision tree weak learners. " +
    "While generally slower than XGBoost due to the lack of GPU support and histogram-based optimizations, " +
    "it provides a well-studied reference implementation."
  ),

  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [3200, 2000, 4438],
    rows: [
      new TableRow({ children: [makeHeaderCell("Hyperparameter", 3200, BLUE), makeHeaderCell("Value", 2000, BLUE), makeHeaderCell("Rationale", 4438, BLUE)] }),
      ...([
        ["n_estimators", "150", "Fewer trees than XGBoost — CPU training budget constraint"],
        ["max_depth", "5", "Slightly shallower than XGBoost for regularization"],
        ["learning_rate (\u03B7)", "0.1", "Consistent with XGBoost for fair comparison"],
        ["random_state", "42", "Reproducibility"],
      ]).map(([p, v, j], i) =>
        new TableRow({ children: [makeCell(p, 3200, i % 2 === 0 ? GREY : WHITE, true, AlignmentType.LEFT, BLUE), makeCell(v, 2000, i % 2 === 0 ? GREY : WHITE), makeCell(j, 4438, i % 2 === 0 ? GREY : WHITE, false, AlignmentType.LEFT)] })
      )
    ]
  }),
  figCaption("Table 9. Gradient Boosting hyperparameters."),

  h2("4.4  Training Strategy & Optimization"),
  body(
    "All models were trained on the training split and evaluated on the held-out test split. To obtain more reliable " +
    "estimates of generalization performance, 5-fold stratified cross-validation (StratifiedKFold) was also performed " +
    "on the training set for the three fastest-training models (Logistic Regression, Naive Bayes, Linear SVM). " +
    "Stratified folds ensure that each fold maintains the class ratio of the original training set, preventing " +
    "folds with an accidentally skewed class distribution from biasing the cross-validation estimate."
  ),
  body(
    "For the tree-based ensemble methods (Random Forest, XGBoost, Gradient Boosting), full 5-fold cross-validation " +
    "was skipped due to the significantly higher training time; instead, the test set accuracy is reported as the " +
    "held-out performance estimate. Future work should include full cross-validation for all models."
  ),

  h2("4.5  Evaluation Metrics"),
  body(
    "Given the moderate class imbalance (~55%/45%), raw accuracy is not sufficient as the sole metric. The following " +
    "metrics were computed on the test set:"
  ),
  bulletItem("Accuracy — proportion of correctly classified examples. Reported for baseline comparison.", "Accuracy:"),
  bulletItem("Weighted F1 Score — harmonic mean of precision and recall, averaged across classes weighted by support. Primary metric for model ranking.", "Weighted F1:"),
  bulletItem("Weighted Precision — proportion of positive predictions that are correct, weighted by class support.", "Weighted Precision:"),
  bulletItem("Weighted Recall — proportion of actual positives correctly retrieved, weighted by class support.", "Weighted Recall:"),
  bulletItem("ROC-AUC — area under the Receiver Operating Characteristic curve. Measures discriminative ability across all classification thresholds, independent of the decision boundary.", "ROC-AUC:"),
  bulletItem("5-Fold CV Accuracy — cross-validation accuracy on the training set (linear models only). Serves as an unbiased performance estimate.", "CV Accuracy:"),
  pageBreak()
);

// ══════════════════════════════════════════════════════════════════════════════
// 5. RESULTS
// ══════════════════════════════════════════════════════════════════════════════
children.push(
  h1("5. Results"),

  h2("5.1  Overall Test Set Performance"),
  body(
    "Table 10 presents the test set performance of all six models, ranked by weighted F1 score. The Linear SVM " +
    "achieves the highest F1 score (0.8656) and accuracy (0.8665), confirming the theoretical expectation that " +
    "linear methods excel in high-dimensional sparse text spaces. Multinomial Naive Bayes places a close second " +
    "(F1 = 0.8580), demonstrating remarkable efficacy given its simplistic independence assumptions. Logistic " +
    "Regression ranks third with F1 = 0.8509."
  ),
  ...blank(1),

  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [500, 2400, 1400, 1400, 1400, 1300, 1500, 1338],
    rows: [
      new TableRow({
        children: [
          makeHeaderCell("Rank", 500),
          makeHeaderCell("Model", 2400),
          makeHeaderCell("Accuracy", 1400),
          makeHeaderCell("F1 (W)", 1400),
          makeHeaderCell("Precision", 1400),
          makeHeaderCell("Recall", 1300),
          makeHeaderCell("ROC-AUC", 1500),
          makeHeaderCell("CV Acc.", 1338),
        ]
      }),
      ...([
        ["#1 \u2605", "Linear SVM", "0.8665", "0.8656", "0.8664", "0.8665", "0.9268", "0.8660\u00B10.006", "#FFF9C4"],
        ["#2", "Multinomial NB", "0.8588", "0.8580", "0.8590", "0.8588", "0.9232", "0.8582\u00B10.007", GREY],
        ["#3", "Logistic Regression", "0.8534", "0.8509", "0.8535", "0.8534", "0.9292", "0.8528\u00B10.008", WHITE],
        ["#4", "Gradient Boosting", "0.8255", "0.8195", "0.8240", "0.8255", "0.8805", "\u2014", GREY],
        ["#5", "XGBoost", "0.8219", "0.8162", "0.8193", "0.8219", "0.8810", "\u2014", WHITE],
        ["#6", "Random Forest", "0.7437", "0.7183", "0.7282", "0.7437", "0.8963", "\u2014", GREY],
      ]).map(([rank, model, acc, f1, prec, rec, roc, cv, shade]) =>
        new TableRow({
          children: [
            makeCell(rank, 500, shade, true, AlignmentType.CENTER, rank == "#1 \u2605" ? GOLD : DGREY),
            makeCell(model, 2400, shade, rank == "#1 \u2605", AlignmentType.LEFT),
            makeCell(acc, 1400, shade, rank == "#1 \u2605"),
            makeCell(f1, 1400, shade, rank == "#1 \u2605"),
            makeCell(prec, 1400, shade),
            makeCell(rec, 1300, shade),
            makeCell(roc, 1500, shade),
            makeCell(cv, 1338, shade, false, AlignmentType.CENTER, "555555"),
          ]
        })
      )
    ]
  }),
  figCaption("Table 10. Model performance on the test set (n = 2,224), ranked by weighted F1. \u2605 = best model. CV accuracy shown only for models with 5-fold cross-validation."),
  ...blank(1),

  figure("10_final_siralama.png", 6.5, 1.8),
  figCaption("Figure 5. Model performance ranking table (F1 score, descending). Gold, silver, and bronze highlights indicate the top-3 models."),

  h2("5.2  Model Comparison Visualizations"),
  body(
    "Figure 6 presents bar chart comparisons across all five evaluation metrics for the six models. Gold-outlined " +
    "bars indicate the best-performing model on each metric. The cross-validation panel (bottom-right) shows mean " +
    "\u00B1 standard deviation of 5-fold CV accuracy for the linear models, confirming their superiority is not " +
    "specific to the test split."
  ),
  figure("05_model_karsilastirma.png", 6.5, 4.0),
  figCaption("Figure 6. Bar chart comparison of all six models across Accuracy, F1, Precision, Recall, ROC-AUC, and 5-Fold CV Accuracy. Gold-outlined bars = best model per metric."),

  h2("5.3  Heatmap & Radar Chart"),
  body(
    "Figure 7 provides two additional perspectives on the metric landscape. The heatmap (left panel) encodes all " +
    "six metrics across all six models in a red-yellow-green color scale from 0.75 to 1.0, making it easy to " +
    "identify both overall performance level and intra-model consistency. The radar chart (right panel) plots " +
    "each model as a polygon in a 6-dimensional metric space, where larger area indicates better overall performance."
  ),
  figure("08_heatmap_radar.png", 6.5, 2.2),
  figCaption("Figure 7. Heatmap (left) and radar chart (right) summarizing all model-metric combinations. Linear models occupy the largest radar polygons."),
  body(
    "A notable observation in the heatmap is the high ROC-AUC of Logistic Regression (0.9292) — higher than the " +
    "Linear SVM (0.9268) despite the latter's superior F1. This indicates that LR's probability calibration is " +
    "superior to SVM's decision function scores, which are not proper probabilities. For applications requiring " +
    "threshold tuning (e.g., setting a custom precision-recall operating point), LR may be preferable despite " +
    "slightly lower raw accuracy."
  ),

  h2("5.4  Confusion Matrix Analysis"),
  body(
    "Figure 8 shows the 2\u00D72 confusion matrices for all six models. The matrices reveal consistent asymmetric " +
    "error patterns across models: all classifiers tend to misclassify a larger absolute number of negative " +
    "comments as positive than vice versa. This false-negative pattern for the negative class is partially " +
    "explained by the class imbalance (positive class has 1,109 more training examples) and by the fact that " +
    "some negative comments employ indirect or ironic language that surface-form models struggle to detect."
  ),
  figure("06_confusion_matrices.png", 6.5, 3.9),
  figCaption("Figure 8. Confusion matrices for all six models on the test set (n = 2,224). Rows = true labels; columns = predicted labels."),
  body(
    "The Random Forest model shows the most severe confusion — with a noticeably higher false-negative rate for the " +
    "negative class. This is consistent with its lower F1 score (0.7183) and confirms that tree-based methods " +
    "struggle to identify the linear decision boundaries inherent in TF-IDF space. The Linear SVM, by contrast, " +
    "shows the most balanced confusion matrix, with roughly similar error rates in both directions."
  ),

  h2("5.5  ROC Curve Analysis"),
  body(
    "Figure 9 plots the Receiver Operating Characteristic (ROC) curves for all six models. The ROC curve shows " +
    "the trade-off between True Positive Rate (sensitivity) and False Positive Rate (1 - specificity) as the " +
    "classification threshold varies. Area Under the Curve (AUC) is reported in the legend as a threshold-independent " +
    "summary metric."
  ),
  figure("07_roc_curves.png", 4.5, 3.5),
  figCaption("Figure 9. ROC curves for all six models. All models substantially outperform the random baseline (AUC = 0.50). Logistic Regression achieves the highest AUC (0.929)."),
  body(
    "All models achieve AUC values above 0.88, indicating that all classifiers have strong discriminative power. " +
    "The three linear models cluster tightly in the 0.92-0.93 AUC range, while the ensemble methods — despite " +
    "their added complexity — fall behind at 0.88-0.90. This is a strong empirical indication that the problem " +
    "is well-served by linear decision boundaries in TF-IDF space, and that the additional model complexity of " +
    "gradient boosting and random forests does not translate into better probabilistic discrimination for this dataset."
  ),

  h2("5.6  Feature Importance — Most Discriminative Tokens"),
  body(
    "Figure 10 shows the 20 TF-IDF features with the largest positive (Pozitif-driving) and most negative (Negatif-driving) " +
    "Logistic Regression coefficients. These coefficients directly measure each token's contribution to the log-odds " +
    "of positive classification — positive coefficients push toward Pozitif, negative coefficients push toward Negatif."
  ),
  figure("09_feature_importance.png", 6.5, 2.9),
  figCaption("Figure 10. Top 20 most discriminative TF-IDF features per class according to Logistic Regression coefficients. Left: features strongly associated with positive sentiment. Right: features strongly associated with negative sentiment."),
  body(
    "The feature importance visualization confirms the lexical separation observed in the EDA phase. Positive-class " +
    "features are dominated by terms expressing admiration, affection, aesthetics, and approval. Negative-class " +
    "features consist predominantly of profanity, slurs, and hostility-expressing tokens. This clear lexical " +
    "separation explains why even simple models like Naive Bayes perform competitively — the signal is concentrated " +
    "in a small set of highly predictive surface-form tokens."
  ),
  pageBreak()
);

// ══════════════════════════════════════════════════════════════════════════════
// 6. DISCUSSION
// ══════════════════════════════════════════════════════════════════════════════
children.push(
  h1("6. Discussion"),

  h2("6.1  Key Findings"),
  body(
    "The results of this study yield several important insights about Turkish social media sentiment classification " +
    "with classical machine learning methods:"
  ),
  bulletItem(
    "Linear models dominate tree-based ensembles on TF-IDF features. The Linear SVM, Logistic Regression, and " +
    "Multinomial NB all outperform Random Forest, XGBoost, and Gradient Boosting. This is a well-documented phenomenon " +
    "in the NLP literature: TF-IDF vectors are inherently additive, and the classes are linearly separable with high " +
    "probability in high-dimensional spaces (Cover's theorem). Tree-based methods, designed for tabular data with " +
    "complex interaction structures, cannot efficiently exploit this additive separability.",
    "Finding 1:"
  ),
  bulletItem(
    "Naive Bayes is a surprisingly strong baseline. Despite its independence assumption — which is clearly violated " +
    "in language — MNB achieves F1 = 0.858, second only to the SVM. This demonstrates the robustness of the " +
    "Naive Bayes approach on text classification tasks with strong lexical signal, and explains its enduring " +
    "popularity as a fast, interpretable baseline in the field.",
    "Finding 2:"
  ),
  bulletItem(
    "The Random Forest underperforms significantly relative to other methods, with F1 = 0.718. This is attributable " +
    "to the sparse nature of TF-IDF vectors: each tree in the forest can only use a small subset of features at each " +
    "split node, and most informative features are rare enough that they will often be excluded from consideration " +
    "at any given split. This results in trees that are individually weak and collectively inconsistent.",
    "Finding 3:"
  ),
  bulletItem(
    "Bigrams provide marginal but consistent improvement over unigrams alone. The use of ngram_range=(1,2) captures " +
    "sentiment-modifying phrases (e.g., negations like 'değil', intensifiers like 'çok güzel') that unigrams miss. " +
    "However, the improvement is small because the strongest sentiment signal in this corpus resides in specific " +
    "single tokens rather than phrase-level constructs.",
    "Finding 4:"
  ),
  bulletItem(
    "GPU acceleration (CUDA) for XGBoost reduced training time substantially but did not improve model quality. " +
    "The underlying inductive bias of gradient-boosted trees on sparse text features is the binding constraint, " +
    "not computational budget.",
    "Finding 5:"
  ),

  h2("6.2  Error Analysis"),
  body(
    "An informal inspection of misclassified examples reveals several recurring error patterns that are structurally " +
    "difficult for bag-of-words models to handle:"
  ),
  bulletItem(
    "Irony and sarcasm — Comments that use positive surface-form vocabulary to express negative sentiment (e.g., " +
    "'Harika, gerçekten çok zekisin' / 'Wonderful, you're really so smart') are systematically misclassified as " +
    "positive. Detecting irony requires pragmatic and contextual reasoning beyond n-gram statistics.",
    "Irony/Sarcasm:"
  ),
  bulletItem(
    "Code-switching — Some comments mix Turkish and English, or use phonetically-spelled slang that diverges " +
    "from standard Turkish orthography. These tokens are rare in the training vocabulary and receive low TF-IDF weights.",
    "Code-Switching:"
  ),
  bulletItem(
    "Implicit sentiment — Comments that express sentiment through domain knowledge rather than lexical cues " +
    "(e.g., 'Bu sezon da böyle geçti' / 'Another season passed like this') lack explicit positive or negative tokens " +
    "and confuse the model.",
    "Implicit Sentiment:"
  ),
  bulletItem(
    "Negation handling — Turkish negation (via suffix '-me/-ma' or the standalone 'değil') can invert the sentiment " +
    "of any token. While some negation patterns appear as bigrams (e.g., 'iyi değil'), morphological negation " +
    "within a single word form is invisible to token-level TF-IDF.",
    "Negation:"
  ),

  h2("6.3  Limitations"),
  body(
    "The current study has several methodological limitations that should be acknowledged:"
  ),
  bulletItem("No morphological analysis — Turkish is highly agglutinative, meaning a single root can appear in dozens of inflected/derived forms. Treating each surface form as an independent feature inflates the vocabulary and misses morphological generalizations.", "Morphology:"),
  bulletItem("No hyperparameter search — Model hyperparameters were selected based on prior literature and manual reasoning rather than systematic grid search or Bayesian optimization. A thorough search may yield better performance, particularly for the gradient boosting models.", "Hyperparameters:"),
  bulletItem("Static stop-word list — The 80-token Turkish stop-word list was manually curated and may not be optimal. A data-driven approach (e.g., removing the top-k highest-document-frequency tokens) could be more principled.", "Stop Words:"),
  bulletItem("No full cross-validation for ensemble models — Tree-based ensembles were evaluated only on the test split, not via full cross-validation, due to training time constraints. This provides a less reliable generalization estimate.", "CV:"),
  bulletItem("Single dataset — Conclusions about the relative performance of different model families are specific to this dataset. Generalization to other Turkish sentiment datasets (e.g., product reviews, news comments) requires further study.", "Generalizability:"),

  h2("6.4  Improvements & Future Work"),
  body(
    "Several directions could substantially improve upon the results of this study:"
  ),
  numberedItem(
    "BERTurk / Turkish BERT — Pre-trained transformer language models fine-tuned on Turkish text (e.g., BERTurk by " +
    "dbmdz, or mBERT) would likely achieve significantly higher performance by leveraging deep contextual " +
    "representations, handling morphological variation, and modeling long-range dependencies. BERTurk has achieved " +
    "state-of-the-art results on multiple Turkish NLP benchmarks and is the natural next step.",
    "Transformer Models:"
  ),
  numberedItem(
    "Turkish Morphological Analysis — Integrating Zemberek NLP or a lightweight stemmer would reduce vocabulary size, " +
    "conflate morphological variants, and improve generalization. This is particularly important for low-frequency " +
    "sentiment-bearing words that appear in many inflected forms.",
    "Morphological NLP:"
  ),
  numberedItem(
    "Character-Level n-gram Features — Supplementing word-level TF-IDF with character 3-5-gram features would " +
    "capture sub-word morphological patterns, handle out-of-vocabulary slang, and be robust to spelling variation.",
    "Character n-grams:"
  ),
  numberedItem(
    "Class Imbalance Handling — While the current 55%/45% split is mild, experimenting with SMOTE oversampling, " +
    "class-weighted loss functions, or threshold tuning on the validation set could further improve recall on the " +
    "minority (negative) class.",
    "Imbalance:"
  ),
  numberedItem(
    "Hyperparameter Optimization — Systematic Bayesian optimization (e.g., via Optuna or Hyperopt) or grid search " +
    "with cross-validation would yield better-tuned models, particularly for XGBoost's many interaction hyperparameters.",
    "AutoML / HPO:"
  ),
  numberedItem(
    "Irony & Sarcasm Detection — A specialized irony detection module (either rule-based or using a pre-trained " +
    "classifier) could be used as a preprocessing step to flag potentially ironic comments for special handling.",
    "Irony Detection:"
  ),
  numberedItem(
    "Multi-class Sentiment — The binary Pozitif/Negatif schema could be extended to a fine-grained sentiment scale " +
    "(e.g., very negative, negative, neutral, positive, very positive) or emotion taxonomy (joy, anger, sadness, " +
    "surprise, etc.) for richer annotation.",
    "Granularity:"
  ),
  numberedItem(
    "Deployment — The best-performing model (Linear SVM) could be deployed as a real-time REST API for online " +
    "moderation, content filtering, or social listening, with periodic retraining as new labeled data becomes available.",
    "Production:"
  ),
  pageBreak()
);

// ══════════════════════════════════════════════════════════════════════════════
// 7. CONCLUSION
// ══════════════════════════════════════════════════════════════════════════════
children.push(
  h1("7. Conclusion"),
  body(
    "This study presented a rigorous end-to-end machine learning pipeline for Turkish social media sentiment analysis. " +
    "Starting from a raw CSV dataset of 11,119 annotated comments, the pipeline encompassed encoding detection, " +
    "exploratory data analysis, a seven-step preprocessing routine, TF-IDF feature extraction with bigram support, " +
    "training and evaluation of six classifiers, and a comprehensive multi-metric model comparison."
  ),
  body(
    "The central finding — that linear models decisively outperform tree-based ensemble methods on TF-IDF-vectorized " +
    "Turkish text — provides a clear practical guideline: for short-text Turkish sentiment classification on this scale, " +
    "a well-tuned Linear SVM or Logistic Regression trained on TF-IDF features constitutes a strong, fast, and " +
    "interpretable solution that is difficult to improve upon without moving to deep contextual representations."
  ),
  body(
    "The Linear SVM achieved the best overall performance with Accuracy = 0.8665, F1 = 0.8656, and ROC-AUC = 0.9268, " +
    "demonstrating that the classification problem can be solved to high quality with classical methods. The feature " +
    "importance analysis revealed a clear lexical polarization between sentiment classes, validating the use of " +
    "bag-of-words representations. Error analysis identified irony, morphological variation, and implicit sentiment " +
    "as the main remaining challenges."
  ),
  body(
    "Future work should prioritize the integration of BERTurk-based fine-tuning, which is expected to address the " +
    "limitations of surface-form representations and push accuracy toward the 90%+ range observed on similar Turkish " +
    "NLP benchmarks. The current study provides a strong, reproducible baseline and a rich comparative analysis " +
    "for such future extensions."
  ),
  pageBreak()
);

// ══════════════════════════════════════════════════════════════════════════════
// REFERENCES
// ══════════════════════════════════════════════════════════════════════════════
children.push(
  h1("References"),
  ...([
    ["[1]", "Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word vectors with subword information. Transactions of the Association for Computational Linguistics, 5, 135-146."],
    ["[2]", "Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297."],
    ["[3]", "Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. NAACL-HLT 2019."],
    ["[4]", "Fan, R. E., Chang, K. W., Hsieh, C. J., Wang, X. R., & Lin, C. J. (2008). LIBLINEAR: A library for large linear classification. Journal of Machine Learning Research, 9, 1871-1874."],
    ["[5]", "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics, 29(5), 1189-1232."],
    ["[6]", "Kelechava, B. (2019). Turkish Natural Language Processing. Available via GitHub repositories and community resources."],
    ["[7]", "Manning, C. D., Raghavan, P., & Schutze, H. (2008). Introduction to Information Retrieval. Cambridge University Press."],
    ["[8]", "McCallum, A., & Nigam, K. (1998). A comparison of event models for naive Bayes text classification. AAAI Workshop on Learning for Text Categorization."],
    ["[9]", "Pang, B., & Lee, L. (2008). Opinion mining and sentiment analysis. Foundations and Trends in Information Retrieval, 2(1-2), 1-135."],
    ["[10]", "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830."],
    ["[11]", "Schweter, S., & Oguz, H. (2021). BERTurk: Pre-trained BERT model for Turkish. dbmdz/bert-base-turkish-cased, Hugging Face Model Hub."],
    ["[12]", "Turney, P. D. (2002). Thumbs up or thumbs down? Semantic orientation applied to unsupervised classification of reviews. Proceedings of ACL 2002."],
    ["[13]", "Yildirim, S., & Yildiz, T. (2018). Turkish sentiment analysis: Challenges and approaches. International Journal of Artificial Intelligence & Applications, 9(4)."],
    ["[14]", "Zemberek NLP. (2023). Turkish Natural Language Processing Library. Retrieved from https://github.com/ahmetaa/zemberek-nlp"],
  ]).map(([ref, text]) =>
    new Paragraph({
      children: [
        new TextRun({ text: ref + "  ", font: "Arial", size: 20, bold: true, color: BLUE }),
        new TextRun({ text, font: "Arial", size: 20, color: DGREY })
      ],
      spacing: sp(0, 120),
      indent: { left: 540, hanging: 540 }
    })
  )
);

// ══════════════════════════════════════════════════════════════════════════════
// BUILD DOCUMENT
// ══════════════════════════════════════════════════════════════════════════════
const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "\u2022",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 560, hanging: 280 } } }
        }]
      },
      {
        reference: "numbers",
        levels: [{
          level: 0, format: LevelFormat.DECIMAL, text: "%1.",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 600, hanging: 300 } } }
        }]
      }
    ]
  },
  styles: {
    default: {
      document: { run: { font: "Arial", size: 22 } }
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: NAVY },
        paragraph: { spacing: { before: 480, after: 200 }, outlineLevel: 0 }
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: BLUE },
        paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 1 }
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: "2C5282" },
        paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 2 }
      }
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_W, height: PAGE_H },
        margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN }
      }
    },
    headers: {
      default: new Header({
        children: [
          new Paragraph({
            children: [
              new TextRun({ text: "Social Media Sentiment Analysis  \u2014  Machine Learning Final Project", font: "Arial", size: 18, color: "888888" }),
              new TextRun({ children: [new PageNumber()], font: "Arial", size: 18, color: "888888" })
            ],
            tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
            border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC", space: 4 } }
          })
        ]
      })
    },
    footers: {
      default: new Footer({
        children: [
          new Paragraph({
            children: [new TextRun({ text: "Machine Learning Final Project  \u2022  April 2026", font: "Arial", size: 18, color: "AAAAAA" })],
            alignment: AlignmentType.CENTER,
            border: { top: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC", space: 4 } }
          })
        ]
      })
    },
    children
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log("Report written to:", OUT);
  console.log("Size:", Math.round(fs.statSync(OUT).size / 1024), "KB");
});
