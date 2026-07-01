# 🎬 NLP on IMDB Movie Reviews
### Text Classification · Clustering · Sentiment Analysis

> A complete Natural Language Processing pipeline applied to 50,000 IMDB movie reviews — covering supervised classification, unsupervised clustering, and three-way sentiment analysis using lexicon, machine learning, and deep learning approaches.

---

##  Problem Statement

Online platforms accumulate thousands of user reviews every day, making it impractical to read or categorise them manually. This project explores how NLP techniques can automatically:

- **Classify** whether a movie review expresses a positive or negative opinion
- **Cluster** reviews into natural groups based purely on word usage, without using any labels
- **Analyse sentiment** using three fundamentally different approaches and compare their strengths and weaknesses

---

##  Dataset

| Property | Detail |
|---|---|
| **Name** | IMDB Dataset of 50K Movie Reviews |
| **Source** | [Kaggle — lakshmi25npathi/imdb-dataset-of-50k-movie-reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |
| **Size** | 50,000 reviews (49,582 after removing 418 duplicates) |
| **Label** | Binary — `positive` / `negative` |
| **Class balance** | 24,884 positive / 24,698 negative (near-balanced) |
| **Review length** | Mean: 231 words · Median: 173 words · Max: 2,470 words |

The dataset is loaded directly via `kagglehub` in the notebook — no manual download required.

---

##  Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| Google Colab (T4 GPU) | Runtime environment |
| NLTK | Tokenization, stopword removal, lemmatization, VADER |
| scikit-learn | TF-IDF, classification models, clustering, evaluation |
| TensorFlow / Keras | LSTM deep learning model |
| Matplotlib / Seaborn | Visualizations |
| WordCloud | Word frequency visualization |
| kagglehub | Dataset download |

---

##  Pipeline Overview

```
Raw Reviews
    │
    ▼
[EDA] Shape · Duplicates · Class balance · Review length · Word frequency
    │
    ▼
[Preprocessing]
    ├─ HTML tag removal (<br /><br />)
    ├─ Punctuation removal
    ├─ Tokenization + Lowercasing
    ├─ Stopword removal (NLTK)
    └─ Lemmatization (WordNetLemmatizer)
    │
    ▼
[Train-Test Split] 80/20 · stratified · random_state=42
    │
    ▼
[TF-IDF Vectorization] fit on train only → (39,665 × 136,532)
    │
    ├──────────────────┬──────────────────┐
    ▼                  ▼                  ▼
[Q3: Classification] [Q4: Clustering]  [Q5: Sentiment Analysis]
```

---

##  Tasks & Results

### Q3 — Text Classification

Three classical classifiers trained on TF-IDF features:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Naive Bayes | 0.8666 | 0.8668 | 0.8666 | 0.8666 |
| Logistic Regression | 0.8925 | 0.8929 | 0.8925 | 0.8925 |
| **Linear SVM** ✅ | **0.8954** | **0.8955** | **0.8954** | **0.8954** |

**Best model: Linear SVM** — its margin-based approach handles high-dimensional sparse TF-IDF data most effectively, producing more balanced results across both classes than Naive Bayes' independence assumption allows.

---

### Q4 — Text Clustering

TF-IDF reduced to 2 dimensions via **TruncatedSVD** (used instead of PCA since TF-IDF is sparse — PCA would require densifying 136k features, exceeding memory limits). Optimal k=2 selected using combined elbow method + silhouette score analysis.

| Algorithm | Cluster Sizes | Silhouette Score |
|---|---|---|
| **KMeans** ✅ | 28,242 / 11,423 | **0.4359** |
| GMM | 28,481 / 11,184 | 0.4348 |
| Agglomerative | 2,118 / 882 (subsample) | 0.3873 |

**Note:** Despite good silhouette scores, clusters did not align with sentiment labels — they appear to capture text structure (review length, vocabulary diversity) rather than sentiment polarity, highlighting that sentiment is a weak signal for unsupervised grouping since positive and negative reviews share most of the same vocabulary.

---

### Q5 — Sentiment Analysis

Three fundamentally different approaches compared on the same 9,917 test reviews:

| Approach | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| VADER (Lexicon) | 0.6934 | 0.7151 | 0.6934 | 0.6851 |
| **Logistic Regression (ML)** ✅ | **0.8925** | **0.8929** | **0.8925** | **0.8925** |
| LSTM (Deep Learning) | 0.8764 | 0.8770 | 0.8764 | 0.8763 |

| Approach | Pros | Cons |
|---|---|---|
| VADER | No training needed, instant, interpretable | Ignores context and sarcasm, generic word list |
| Logistic Regression | High accuracy, fast, interpretable weights | Ignores word order |
| LSTM | Captures word order and context | Needs more setup, prone to overfitting without regularisation |

---

##  Project Structure

```
 project/
├── NLP_and_Classification.ipynb       # Main Google Colab notebook (all code + outputs)
├── Report.pdf  # Full written report
└── README.md                # This file
```

---

##  Key Implementation Decisions

- **Split before vectorizing** — train-test split is performed before TF-IDF fitting to prevent data leakage (IDF statistics from the test set must not influence the training vocabulary)
- **TruncatedSVD over PCA** — TruncatedSVD operates directly on sparse matrices; PCA requires a dense matrix which at 136,532 features would exceed available memory
- **Agglomerative on a subsample** — pairwise distance computation for 40k reviews is prohibitively expensive; a 3,000-review subsample is used for fair comparison
- **LSTM with early stopping** — without early stopping, the LSTM overfit to 98% training accuracy while validation accuracy stagnated around 85%; early stopping with `restore_best_weights=True` raised test accuracy from 82.5% to 87.64%
- **GPU determinism** — cuDNN GPU operations are non-deterministic by default; all random seeds (Python, NumPy, TensorFlow) are explicitly fixed and `tf.config.experimental.enable_op_determinism()` is enabled to guarantee reproducible results

---

##  Running the Notebook

1. Open the notebook in **Google Colab**
2. Set runtime to **T4 GPU** (Runtime → Change runtime type → T4 GPU)
   - Required for the LSTM training step (Q5); all other cells work on CPU
3. Run all cells from top to bottom (Runtime → Run all)
   - The dataset is downloaded automatically via `kagglehub` — no manual upload needed

---

##  Summary of Findings

| Task | Best Model | Score |
|---|---|---|
| Text Classification | Linear SVM | Accuracy = 89.54% |
| Clustering | KMeans | Silhouette = 0.4359 |
| Sentiment Analysis | Logistic Regression | Accuracy = 89.25% |

The clearest finding across all three tasks is that **classical linear models (Logistic Regression and Linear SVM) on full TF-IDF features are highly competitive with deep learning models**, and can outperform them when the deep model is constrained to a smaller vocabulary and a fixed-length input window. Deep learning (LSTM) adds meaningful value primarily where word order and context are critical, given sufficient data and compute.

---

