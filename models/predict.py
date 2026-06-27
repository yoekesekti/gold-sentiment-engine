import warnings
warnings.filterwarnings('ignore')

import re
import joblib
import numpy as np
import nltk
import os

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.stem import SnowballStemmer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

vec        = joblib.load(os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl'))
lr_stage1  = joblib.load(os.path.join(BASE_DIR, 'lr_stage1.pkl'))
lr_stage2  = joblib.load(os.path.join(BASE_DIR, 'lr_stage2.pkl'))
preproc    = joblib.load(os.path.join(BASE_DIR, 'preprocessing_config.pkl'))

stopwords_final = preproc['stopwords_final']

try:
    factory = StemmerFactory()
    indonesian_stemmer = factory.create_stemmer()
except Exception:
    indonesian_stemmer = None
english_stemmer = SnowballStemmer('english')


# =============================================================================
# PREPROCESSING
# =============================================================================

def lowercase(text):
    return text.lower()


def remove_noise(text):
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def protect_entities(text):
    finance_terms = preproc['finance_terms']
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
    return nltk.word_tokenize(text)


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
            if indonesian_stemmer:
                stemmed = indonesian_stemmer.stem(token)
                results.append(stemmed if len(stemmed) > 1 else token)
            else:
                results.append(token)
    return results


def preprocess(text):
    text = lowercase(text)
    text = remove_noise(text)
    text = protect_entities(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return ' '.join(tokens)


# =============================================================================
# CONFIDENCE SCORE — PREDICT_PROBA (Probabilitas Statistik)
#
# LogisticRegression menghasilkan probabilitas sesungguhnya via predict_proba(),
# bukan decision_function seperti LinearSVC.
#
# Stage 1 — predict_proba → [P(not_positive), P(positive)]
#   - Jika P(positive) > 0.5 → positive
#   - Confidence = P(positive)
#
# Stage 2 — predict_proba → [P(neutral), P(negative)]
#   - Jika P(negative) >= 0.5 → negative, confidence = P(negative)
#   - Jika P(negative) <  0.5 → neutral,  confidence = P(neutral)
#
# Tidak diperlukan Min-Max normalization maupun threshold manual.
# =============================================================================


def get_class_distribution(X):
    """
    Hitung skor probabilitas ketiga kelas dari Two-Stage Logistic Regression.

    Arsitektur Two-Stage:
      Stage 1 → memisahkan Positive vs (Negative + Neutral)
      Stage 2 → memisahkan Negative vs Neutral

    Strategi distribusi:
      p_positive = P(positive) dari Stage 1 predict_proba
      sisa       = 1 - p_positive
      p_negative = sisa x P(negative) dari Stage 2 predict_proba
      p_neutral  = sisa x P(neutral) dari Stage 2 predict_proba

    Returns:
        dict : {'positive': float, 'negative': float, 'neutral': float}
               masing-masing dalam satuan persen (0.0 - 100.0)
    """
    proba1 = lr_stage1.predict_proba(X)[0]   # [P(rest), P(positive)]
    proba2 = lr_stage2.predict_proba(X)[0]   # [P(neutral), P(negative)]

    p_positive    = float(proba1[1])
    p_not_pos     = 1.0 - p_positive
    p_neg_portion = float(proba2[1])

    p_negative = p_not_pos * p_neg_portion
    p_neutral  = p_not_pos * (1.0 - p_neg_portion)

    return {
        'positive': round(p_positive * 100, 1),
        'negative': round(p_negative * 100, 1),
        'neutral' : round(p_neutral  * 100, 1),
    }


# =============================================================================
# FUNGSI PREDIKSI
# =============================================================================

def predict_sentiment(text):
    """
    Prediksi label sentimen saja (tanpa confidence score).

    Returns:
        str : 'positive' | 'negative' | 'neutral'
    """
    cleaned = preprocess(text)
    X = vec.transform([cleaned])
    proba1 = lr_stage1.predict_proba(X)[0]
    if proba1[1] > 0.5:
        return 'positive'
    proba2 = lr_stage2.predict_proba(X)[0]
    return 'negative' if proba2[1] >= 0.5 else 'neutral'


def predict_title_desc(title, description=""):
    """
    Prediksi sentimen dari judul + isi berita (digabung).

    Args:
        title       (str) : judul berita
        description (str) : isi / snippet berita (opsional)

    Returns:
        tuple : (label, confidence, cleaned_text, distribution)
    """
    merged = str(title) + ' ' + str(description)
    return predict_sentiment_with_proba(merged)


def predict_sentiment_with_proba(text):
    """
    Prediksi sentimen beserta confidence score (probabilitas statistik).

    Alur Two-Stage Logistic Regression:
      1. Stage 1 → cek apakah P(positive) > 0.5
         - Jika ya  : label = positive, confidence = P(positive)
         - Jika tidak:
      2. Stage 2 → cek apakah P(negative) >= 0.5
         - Jika ya  : label = negative, confidence = P(negative)
         - Jika tidak: label = neutral,  confidence = P(neutral)

    Args:
        text (str) : teks berita mentah

    Returns:
        label        (str)   : 'positive' | 'negative' | 'neutral'
        confidence   (float) : probabilitas 0.0-100.0 (statistik)
        cleaned      (str)   : teks setelah preprocessing
        distribution (dict)  : {'positive': x, 'negative': y, 'neutral': z}
                               probabilitas ketiga kelas dalam persen
    """
    cleaned = preprocess(text)
    X = vec.transform([cleaned])

    # --- Stage 1: Positive vs (Negative + Neutral) ---
    proba1 = lr_stage1.predict_proba(X)[0]   # [P(rest), P(positive)]

    if proba1[1] > 0.5:
        label      = 'positive'
        confidence = float(proba1[1]) * 100
    else:
        # --- Stage 2: Negative vs Neutral ---
        proba2 = lr_stage2.predict_proba(X)[0]   # [P(neutral), P(negative)]

        if proba2[1] >= 0.5:
            label      = 'negative'
            confidence = float(proba2[1]) * 100
        else:
            label      = 'neutral'
            confidence = float(proba2[0]) * 100

    distribution = get_class_distribution(X)

    return label, round(confidence, 1), cleaned, distribution


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == '__main__':
    test_texts = [
        "Gold price rallies to new all-time high as investors seek safe haven",
        "Gold futures drop sharply amid stronger dollar and rising yields",
        "Gold prices steady as traders await Fed decision on interest rates",
        "Gold crash intensifies as market panic spreads worldwide",
        "Gold rally showing strong bullish momentum this week",
    ]

    print(f"\n{'Label':>10}  {'Conf':>6}  {'Pos%':>6}  {'Neu%':>6}  {'Neg%':>6}  Text")
    print("-" * 90)

    for t in test_texts:
        label, conf, _, dist = predict_sentiment_with_proba(t)
        print(
            f"[{label.upper():>8}]  {conf:5.1f}%"
            f"  pos={dist['positive']:4.1f}%"
            f"  neu={dist['neutral']:4.1f}%"
            f"  neg={dist['negative']:4.1f}%"
            f"  {t[:50]}..."
        )
