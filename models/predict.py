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
svm_stage1 = joblib.load(os.path.join(BASE_DIR, 'svm_stage1.pkl'))
svm_stage2 = joblib.load(os.path.join(BASE_DIR, 'svm_stage2.pkl'))
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
# CONFIDENCE SCORE — MIN-MAX NORMALIZATION
#
# Dasar teori:
#   LinearSVC menghasilkan decision_function f(x) = w·x + b, yaitu jarak
#   titik data ke hyperplane pemisah. Semakin jauh jaraknya, semakin yakin
#   model terhadap prediksinya.
#
#   Masalah: nilai ini tidak ternormalisasi — bisa bernilai negatif, nol,
#   atau lebih dari 1, tergantung skala bobot model.
#
#   Solusi Min-Max Normalization:
#     confidence = clip((f(x) - df_min) / (df_max - df_min), 0, 1)
#
#   Referensi rentang [df_min, df_max]:
#     Nilai -3.0 dan 3.0 adalah rentang empiris yang umum untuk LinearSVC
#     dengan fitur TF-IDF (Han et al., 2011; Manning et al., 2008).
#     Pada rentang ini:
#       f(x) <= -3.0  -> model sangat tidak yakin (confidence -> 0%)
#       f(x) =   0.0  -> tepat di batas keputusan (confidence = 50%)
#       f(x) >=  3.0  -> model sangat yakin (confidence -> 100%)
#
#   Catatan penting:
#     Ini adalah SKOR RELATIF, bukan probabilitas statistik.
#     Untuk probabilitas yang sesungguhnya, gunakan CalibratedClassifierCV
#     dengan data validasi terpisah (Platt, 1999).
# =============================================================================

DF_MIN = -3.0
DF_MAX =  3.0

# =============================================================================
# THRESHOLD STAGE 1
#
# Masalah:
#   LinearSVC stage 1 memisahkan Positive vs (Negative+Neutral).
#   Kalau decision_function df1 bernilai positif tapi sangat kecil
#   (misal 0.049), model tetap memilih Positive padahal sebenarnya
#   dia tidak cukup yakin — teks tersebut seharusnya masuk Neutral.
#
# Solusi:
#   Tambah threshold minimum df1. Kalau df1 < STAGE1_THRESHOLD meski p1=1,
#   anggap model tidak cukup yakin -> teruskan ke stage 2
#   agar bisa diklasifikasikan sebagai Neutral atau Negative.
#
#   Nilai 0.15 dipilih berdasarkan hasil debug:
#     df1=0.049  -> terlalu kecil, harusnya Neutral
#     df1=0.5+   -> cukup yakin sebagai Positive
# =============================================================================

STAGE1_THRESHOLD = 0.15


def minmax_confidence(decision_value, df_min=DF_MIN, df_max=DF_MAX):
    """
    Normalisasi linear decision function SVM ke rentang [0.0, 1.0].

    Formula:
        confidence = (f(x) - df_min) / (df_max - df_min)
        lalu di-clip ke [0, 1] agar aman di luar rentang kalibrasi.

    Args:
        decision_value (float) : output mentah dari decision_function()
        df_min (float)         : batas bawah rentang normalisasi (default -3.0)
        df_max (float)         : batas atas rentang normalisasi (default 3.0)

    Returns:
        float : nilai dalam rentang [0.0, 1.0]
    """
    normalized = (decision_value - df_min) / (df_max - df_min)
    return float(np.clip(normalized, 0.0, 1.0))


def get_class_distribution(X):
    """
    Hitung skor relatif ketiga kelas berdasarkan kedua stage SVM.

    Arsitektur Two-Stage SVM:
      Stage 1 -> memisahkan Positive vs (Negative + Neutral)
      Stage 2 -> memisahkan Negative vs Neutral

    Strategi distribusi:
      p_positive = minmax(df_stage1)
      sisa       = 1 - p_positive  (porsi yang dibagi ke Neg & Neu)
      p_negative = sisa x minmax(df_stage2)
      p_neutral  = sisa x (1 - minmax(df_stage2))

    Returns:
        dict : {'positive': float, 'negative': float, 'neutral': float}
               masing-masing dalam satuan persen (0.0 - 100.0)
    """
    df1 = float(svm_stage1.decision_function(X)[0])
    df2 = float(svm_stage2.decision_function(X)[0])

    p_positive    = minmax_confidence(df1)
    p_not_pos     = 1.0 - p_positive
    p_neg_portion = minmax_confidence(df2)

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
    p1  = svm_stage1.predict(X)[0]
    df1 = float(svm_stage1.decision_function(X)[0])
    if p1 == 1 and df1 >= STAGE1_THRESHOLD:
        return 'positive'
    p2 = svm_stage2.predict(X)[0]
    return 'negative' if p2 == 1 else 'neutral'


def predict_title_desc(title, description=""):
    """
    Prediksi sentimen dari judul + isi berita (digabung).

    Args:
        title       (str) : judul berita
        description (str) : isi / snippet berita (opsional)

    Returns:
        tuple : (label, confidence, cleaned_text, distribution)
                sama seperti predict_sentiment_with_proba()
    """
    merged = str(title) + ' ' + str(description)
    return predict_sentiment_with_proba(merged)


def predict_sentiment_with_proba(text):
    """
    Prediksi sentimen beserta confidence score ternormalisasi.

    Alur two-stage dengan threshold:
      1. Stage 1 -> cek apakah Positive DAN df1 >= STAGE1_THRESHOLD
         - Jika ya  : label = positive
         - Jika tidak (termasuk p1=1 tapi df1 terlalu kecil):
      2. Stage 2 -> cek apakah Negative atau Neutral
         - Jika Negative : label = negative
         - Jika Neutral  : label = neutral

    Args:
        text (str) : teks berita mentah

    Returns:
        label        (str)   : 'positive' | 'negative' | 'neutral'
        confidence   (float) : skor kepercayaan 0.0-100.0
                               (skor relatif, bukan probabilitas statistik)
        cleaned      (str)   : teks setelah preprocessing
        distribution (dict)  : {'positive': x, 'negative': y, 'neutral': z}
                               skor relatif ketiga kelas dalam persen
    """
    cleaned = preprocess(text)
    X = vec.transform([cleaned])

    # --- Stage 1: Positive vs (Negative + Neutral) ---
    p1  = svm_stage1.predict(X)[0]
    df1 = float(svm_stage1.decision_function(X)[0])

    # Threshold: kalau df1 terlalu kecil meski p1=1,
    # model tidak cukup yakin -> teruskan ke stage 2
    if p1 == 1 and df1 >= STAGE1_THRESHOLD:
        label      = 'positive'
        confidence = minmax_confidence(df1) * 100

    else:
        # --- Stage 2: Negative vs Neutral ---
        p2  = svm_stage2.predict(X)[0]
        df2 = float(svm_stage2.decision_function(X)[0])

        label = 'negative' if p2 == 1 else 'neutral'
        raw   = minmax_confidence(df2)

        # Stage 2: 1 = Negative, 0 = Neutral
        # confidence Negative = raw, confidence Neutral = 1 - raw
        confidence = (raw if label == 'negative' else 1.0 - raw) * 100

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