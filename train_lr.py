import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import re
import os
import joblib

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
try:
    stop_words_id = set(stopwords.words('indonesian'))
except Exception:
    stop_words_id = set()

print(f"Stopwords ID: {len(stop_words_id)} kata")

factory = StemmerFactory()
indonesian_stemmer = factory.create_stemmer()
english_stemmer = SnowballStemmer('english')

# =============================================================================
# LOAD DATA
# =============================================================================
FILE_PATH = 'Data Merge - Sheet2.csv'
df_raw = pd.read_csv(FILE_PATH)
print(f"Loaded: {df_raw.shape}")

df = df_raw.copy()
df['text_merged'] = df['title'].fillna('') + ' ' + df['snippet'].fillna('')

# =============================================================================
# PREPROCESSING (sama persis dengan 9_eksperimen_lr)
# =============================================================================

def lowercase(text):
    return text.lower()

def remove_noise(text):
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

finance_terms = [
    'gold', 'silver', 'xauusd', 'xagusd', 'gld', 'sgol', 'comex',
    'futures', 'spot', 'etf', 'stock', 'market', 'bullion', 'ounce',
    'inflation', 'interest rate', 'fed', 'central bank', 'treasury',
    'harga emas', 'emas'
]

def protect_entities(text):
    for term in sorted(finance_terms, key=len, reverse=True):
        placeholder = term.replace(' ', '_')
        text = re.sub(
            r'\b' + re.escape(term) + r'\b',
            placeholder,
            text,
            flags=re.IGNORECASE
        )
    return text

def tokenize_text(text):
    return word_tokenize(text)

financial_keep = {
    'up', 'down', 'high', 'low', 'above', 'below', 'over', 'under',
    'out', 'only', 'no', 'not', 'but', 'against', 'between', 'through',
    'during', 'before', 'after', 'more', 'most', 'less', 'few',
    'very', 'too', 'so', 'such', 'just', 'then', 'now',
    'off', 'on', 'in', 'into', 'than', 'also', 'well',
    'top', 'bottom', 'break', 'rise', 'fall', 'gain', 'loss',
    'bull', 'bear', 'rally', 'crash', 'surge', 'slump', 'soar',
    'plunge', 'climb', 'drop', 'jump', 'slide', 'rebound',
    'support', 'resistance', 'breakout', 'breakdown',
    'overbought', 'oversold', 'volatile', 'volatility',
    'best', 'worst', 'big', 'small', 'major', 'minor',
    'new', 'old', 'record', 'all_time'
}

custom_stopwords = {
    '.', ',', '(', ')', '--', '-', '``', "''", ':', ';', "'s", '...',
    'the', 'a', 'an', 'and', 'or', 'of', 'for', 'in', 'to', 'is',
    'it', 'at', 'by', 'as', 'be', 'are', 'was', 'were', 'been',
    'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would',
    'can', 'could', 'shall', 'should', 'may', 'might', 'must',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
    'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'yang', 'di', 'ke', 'dan', 'dari', 'dengan', 'untuk',
    'dalam', 'pada', 'ini', 'itu', 'adalah', 'telah', 'akan',
    'tidak', 'ada', 'juga', 'saya', 'ia', 'kami', 'kita',
    'mereka', 'oleh', 'sebagai', 'tentang', 'karena',
    'dapat', 'bisa', 'sudah', 'belum', 'atau',
    'satu', 'dua', 'tiga', 'lebih', 'sangat',
    'lain', 'setelah', 'seperti', 'antara', 'serta',
    'namun', 'sedangkan', 'meski', 'walaupun', 'sejak',
    'tetapi', 'jika', 'bila', 'apakah', 'bagaimana',
    'kapan', 'dimana', 'siapa', 'mengapa', 'bahwa'
}

stopwords_final = custom_stopwords - financial_keep

def remove_stopwords(tokens):
    return [t for t in tokens if t.lower() not in stopwords_final]

def lemmatize_tokens(tokens):
    results = []
    for token in tokens:
        if '_' in token:
            results.append(token.replace('_', ' '))
        elif token.isascii():
            stemmed = english_stemmer.stem(token)
            results.append(stemmed if len(stemmed) > 1 else token)
        else:
            stemmed = indonesian_stemmer.stem(token)
            results.append(stemmed if len(stemmed) > 1 else token)
    return results

def preprocess(text):
    text = lowercase(text)
    text = remove_noise(text)
    text = protect_entities(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return ' '.join(tokens)

# Preprocessing seluruh dataset
print("Preprocessing...")
df['text_clean'] = df['text_merged'].apply(preprocess)
print("Preprocessing selesai.")

# =============================================================================
# SAVE PREPROCESSING CONFIG
# =============================================================================
preproc_config = {
    'stopwords_final': stopwords_final,
    'finance_terms': finance_terms,
}
os.makedirs('models', exist_ok=True)
joblib.dump(preproc_config, 'models/preprocessing_config.pkl')
print("Saved preprocessing_config.pkl")

# =============================================================================
# TF-IDF VECTORIZATION (fit on ALL data for production)
# =============================================================================
X_all = df['text_clean']
y_all = df['sentiment_label']

vec = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True
)
X_tfidf = vec.fit_transform(X_all)
joblib.dump(vec, 'models/tfidf_vectorizer.pkl')
print(f"TF-IDF shape: {X_tfidf.shape}")
print(f"Vocabulary size: {len(vec.get_feature_names_out())}")

# =============================================================================
# TRAIN TWO-STAGE LOGISTIC REGRESSION
# =============================================================================

# Stage 1: Positive vs Rest
y1 = (y_all == 'positive').astype(int).values
lr_stage1 = LogisticRegression(
    C=100, solver='saga', penalty='l1', max_iter=1000, random_state=42
)
lr_stage1.fit(X_tfidf, y1)
print(f"\nStage 1 (Positive vs Rest) trained: {lr_stage1.coef_.shape}")

# Stage 2: Negative vs Neutral (subset non-positive)
mask_nonpos = (y_all != 'positive').values
y2 = (y_all.values[mask_nonpos] == 'negative').astype(int)
lr_stage2 = LogisticRegression(
    C=100, solver='saga', penalty='l1', max_iter=1000, random_state=42
)
lr_stage2.fit(X_tfidf[mask_nonpos], y2)
print(f"Stage 2 (Negative vs Neutral) trained on {mask_nonpos.sum()} samples")

# Save models
joblib.dump(lr_stage1, 'models/lr_stage1.pkl')
joblib.dump(lr_stage2, 'models/lr_stage2.pkl')
print("\nSaved lr_stage1.pkl, lr_stage2.pkl")

# =============================================================================
# EVALUASI: 5-FOLD CV
# =============================================================================
print("\n" + "=" * 70)
print(" 5-FOLD CV — TWO-STAGE LOGISTIC REGRESSION")
print("=" * 70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accs, f1s_macro, f1s_neg, f1s_neu, f1s_pos = [], [], [], [], []

fold = 1
for train_idx, test_idx in skf.split(X_all, y_all):
    X_tr = X_all.iloc[train_idx]
    y_tr = y_all.iloc[train_idx]
    X_te = X_all.iloc[test_idx]
    y_te = y_all.iloc[test_idx]

    vec_cv = TfidfVectorizer(
        max_features=3000, ngram_range=(1, 2), min_df=3,
        max_df=0.9, sublinear_tf=True
    )
    X_tr_tfidf = vec_cv.fit_transform(X_tr)
    X_te_tfidf = vec_cv.transform(X_te)

    y1_tr = (y_tr == 'positive').astype(int).values
    lr1_cv = LogisticRegression(
        C=100, solver='saga', penalty='l1', max_iter=1000, random_state=42
    )
    lr1_cv.fit(X_tr_tfidf, y1_tr)
    p1_cv = lr1_cv.predict(X_te_tfidf)

    mask_tr = (y_tr != 'positive').values
    mask_te = (y_te != 'positive').values
    y2_tr = (y_tr.values[mask_tr] == 'negative').astype(int)
    lr2_cv = LogisticRegression(
        C=100, solver='saga', penalty='l1', max_iter=1000, random_state=42
    )
    lr2_cv.fit(X_tr_tfidf[mask_tr], y2_tr)
    p2_cv = lr2_cv.predict(X_te_tfidf[mask_te])

    final_cv = y_te.copy()
    final_cv[:] = 'positive'
    final_cv[mask_te] = np.where(p2_cv == 1, 'negative', 'neutral')

    a  = accuracy_score(y_te, final_cv)
    f1 = f1_score(y_te, final_cv, average='macro')
    fn = f1_score(y_te, final_cv, labels=['negative'], average=None)[0]
    fu = f1_score(y_te, final_cv, labels=['neutral'],  average=None)[0]
    fp = f1_score(y_te, final_cv, labels=['positive'], average=None)[0]

    accs.append(a); f1s_macro.append(f1)
    f1s_neg.append(fn); f1s_neu.append(fu); f1s_pos.append(fp)

    print(f"Fold {fold}: acc={a:.4f}, f1_macro={f1:.4f}, "
          f"f1_neg={fn:.4f}, f1_neu={fu:.4f}, f1_pos={fp:.4f}")
    fold += 1

print(f"\n{'='*70}")
print(f" 5-FOLD CV RESULTS — TWO-STAGE LOGISTIC REGRESSION")
print(f"{'='*70}")
print(f"{'Metric':<20} {'Mean':>8} {'Std':>8}")
print('-' * 36)
print(f"{'Accuracy':<20} {np.mean(accs):>8.4f} {np.std(accs):>8.4f}")
print(f"{'F1 Macro':<20} {np.mean(f1s_macro):>8.4f} {np.std(f1s_macro):>8.4f}")
print(f"{'F1 Negative':<20} {np.mean(f1s_neg):>8.4f} {np.std(f1s_neg):>8.4f}")
print(f"{'F1 Neutral':<20} {np.mean(f1s_neu):>8.4f} {np.std(f1s_neu):>8.4f}")
print(f"{'F1 Positive':<20} {np.mean(f1s_pos):>8.4f} {np.std(f1s_pos):>8.4f}")

# Final test: predict_proba confidence on a few samples
print("\n" + "=" * 70)
print(" TEST PREDICT_PROBA — CONTOH SAMPEL")
print("=" * 70)
test_samples = [
    "Gold price rallies to new all-time high as investors seek safe haven",
    "Gold futures drop sharply amid stronger dollar and rising yields",
    "Gold prices steady as traders await Fed decision on interest rates",
]
for t in test_samples:
    cleaned = preprocess(t)
    X_t = vec.transform([cleaned])
    # Stage 1: proba Positive
    proba1 = lr_stage1.predict_proba(X_t)[0]  # [P(rest), P(positive)]
    p_pos = proba1[1]
    p1 = lr_stage1.predict(X_t)[0]

    if p1 == 1:
        label = 'positive'
        conf = p_pos * 100
    else:
        proba2 = lr_stage2.predict_proba(X_t)[0]  # [P(neutral), P(negative)]
        p_neg = proba2[1]
        if p_neg >= 0.5:
            label = 'negative'
            conf = p_neg * 100
        else:
            label = 'neutral'
            conf = (1 - p_neg) * 100
    print(f"[{label.upper():>8}] {conf:5.1f}%  {t[:60]}...")

print("\nTraining selesai. Model disimpan di models/lr_stage1.pkl dan models/lr_stage2.pkl")
