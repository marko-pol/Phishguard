"""
Text / NLP feature extractor (hand-crafted scalar features only).

TF-IDF lives in features/pipeline.py as a fitted corpus-level transformer.
This module produces per-email scalars that do not require a fitted corpus:

  - urgency_score      : count of urgency / pressure keywords
  - caps_ratio         : fraction of alphabetic chars that are uppercase
  - exclamation_count  : number of '!' characters
  - question_count     : number of '?' characters
  - punct_density      : punctuation chars / max(total chars, 1)
  - body_char_len      : raw character length of the body
  - body_word_count    : whitespace-split word count
  - avg_word_len       : mean word length (chars)
  - unique_word_ratio  : unique lowercase words / total words
  - digit_ratio        : digit chars / max(total chars, 1)
  - flesch_reading_ease: readability score via syllable/sentence heuristics
                         (higher = easier to read; phishing often uses very
                          simple or very fragmented language)

Provides both a standalone function and a sklearn-compatible transformer.
"""

import re
import string

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Urgency / social-engineering keywords frequently found in phishing
URGENCY_KEYWORDS = {
    "urgent", "immediately", "verify", "suspended", "limited", "expire",
    "expired", "expires", "confirm", "update", "validate", "reactivate",
    "deactivated", "locked", "unlock", "compromised", "unauthorized",
    "unusual activity", "suspicious", "alert", "warning", "required",
    "action required", "act now", "click here", "sign in", "log in",
    "login", "password", "credit card", "social security", "ssn",
    "bank account", "billing", "invoice", "refund", "won", "winner",
    "congratulations", "prize", "free offer", "limited time",
    "verify your", "confirm your", "update your", "restore access",
    "your account", "account will be", "account has been",
}

_PUNCT_RE = re.compile(r'[^\w\s]')
_SENTENCE_SPLIT_RE = re.compile(r'[.!?]+')

FEATURE_NAMES = [
    "urgency_score",
    "caps_ratio",
    "exclamation_count",
    "question_count",
    "punct_density",
    "body_char_len",
    "body_word_count",
    "avg_word_len",
    "unique_word_ratio",
    "digit_ratio",
    "flesch_reading_ease",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_syllables(word: str) -> int:
    """Heuristic syllable count: count vowel groups in a word."""
    word = word.lower().strip(string.punctuation)
    if not word:
        return 0
    vowels = re.findall(r'[aeiouy]+', word)
    count = len(vowels)
    # Subtract silent trailing 'e'
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _flesch_score(body: str) -> float:
    """
    Approximate Flesch Reading Ease score.
    Formula: 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
    Clamped to [0, 100].
    """
    words = body.split()
    word_count = len(words)
    if word_count == 0:
        return 50.0  # neutral default for empty bodies

    sentences = [s for s in _SENTENCE_SPLIT_RE.split(body) if s.strip()]
    sentence_count = max(len(sentences), 1)

    syllable_count = sum(_count_syllables(w) for w in words)

    score = (
        206.835
        - 1.015 * (word_count / sentence_count)
        - 84.6 * (syllable_count / word_count)
    )
    return max(0.0, min(100.0, round(score, 2)))


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------

def extract_text_features(body: str) -> dict:
    """
    Return a dict of hand-crafted text features for a single email body.

    Parameters
    ----------
    body : plain-text email body
    """
    body = body or ""
    total_chars = len(body)

    # --- urgency_score ---
    body_lower = body.lower()
    urgency_score = sum(1 for kw in URGENCY_KEYWORDS if kw in body_lower)

    # --- caps_ratio ---
    alpha_chars = [c for c in body if c.isalpha()]
    caps_ratio = (
        sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if alpha_chars else 0.0
    )

    # --- exclamation / question counts ---
    exclamation_count = body.count("!")
    question_count = body.count("?")

    # --- punct_density ---
    punct_chars = len(_PUNCT_RE.findall(body))
    punct_density = punct_chars / max(total_chars, 1)

    # --- body_char_len ---
    body_char_len = total_chars

    # --- word-level features ---
    words = body.split()
    body_word_count = len(words)
    avg_word_len = (
        sum(len(w) for w in words) / body_word_count if words else 0.0
    )
    unique_word_ratio = (
        len({w.lower() for w in words}) / body_word_count if words else 0.0
    )

    # --- digit_ratio ---
    digit_count = sum(1 for c in body if c.isdigit())
    digit_ratio = digit_count / max(total_chars, 1)

    # --- flesch_reading_ease ---
    flesch = _flesch_score(body)

    return {
        "urgency_score": urgency_score,
        "caps_ratio": round(caps_ratio, 4),
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "punct_density": round(punct_density, 4),
        "body_char_len": body_char_len,
        "body_word_count": body_word_count,
        "avg_word_len": round(avg_word_len, 4),
        "unique_word_ratio": round(unique_word_ratio, 4),
        "digit_ratio": round(digit_ratio, 4),
        "flesch_reading_ease": flesch,
    }


# ---------------------------------------------------------------------------
# Sklearn transformer
# ---------------------------------------------------------------------------

class TextFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer wrapping extract_text_features.

    Expects a Series or 1-D array of plain-text body strings.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        results = []
        for item in X:
            feats = extract_text_features(str(item) if item is not None else "")
            results.append([feats[k] for k in FEATURE_NAMES])
        return np.array(results, dtype=float)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(FEATURE_NAMES)
