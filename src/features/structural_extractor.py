"""
Structural feature extractor.

Analyses the raw HTML body (when present) to surface signals that indicate
HTML-based obfuscation or deceptive formatting:

  - has_html             : 1 if the email contains an HTML part
  - img_count            : number of <img> tags
  - img_to_word_ratio    : img_count / max(plain-text word count, 1)
  - hidden_element_count : elements hidden via inline CSS or zero-size tricks
  - form_count           : number of <form> tags (credential harvesting)
  - script_count         : number of <script> tags
  - external_link_count  : number of hrefs / src pointing to external domains
  - generic_salutation   : 1 if body opens with a generic greeting
                           ("dear customer", "dear user", "dear account holder")

Provides both a standalone function and a sklearn-compatible transformer.
"""

import re

import numpy as np
from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GENERIC_SALUTATION_RE = re.compile(
    r'\bdear\s+(customer|user|member|account\s*holder|client|valued\s+\w+)',
    re.IGNORECASE,
)

_HIDDEN_STYLE_RE = re.compile(
    r'display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0|'
    r'color\s*:\s*(white|#fff{1,3}|#ffffff)',
    re.IGNORECASE,
)

FEATURE_NAMES = [
    "has_html",
    "img_count",
    "img_to_word_ratio",
    "hidden_element_count",
    "form_count",
    "script_count",
    "external_link_count",
    "generic_salutation",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_external(url: str) -> bool:
    """Return True if the URL is an absolute http(s) reference."""
    return bool(url) and url.strip().lower().startswith(("http://", "https://"))


def _word_count(html_body: str) -> int:
    """Approximate word count from HTML by stripping tags."""
    text = BeautifulSoup(html_body, "html.parser").get_text(" ")
    return len(text.split())


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------

def extract_structural_features(html_body: str, plain_body: str = "") -> dict:
    """
    Return a dict of structural features for a single email.

    Parameters
    ----------
    html_body  : raw HTML body string, or empty string
    plain_body : plain-text body (used for salutation check and word count
                 fallback when html_body is empty)
    """
    html_body = html_body or ""
    plain_body = plain_body or ""

    has_html = int(bool(html_body.strip()))

    if not has_html:
        # No HTML — most features default to 0; check plain text for salutation
        plain_lower = plain_body.lower()
        generic_sal = int(bool(_GENERIC_SALUTATION_RE.search(plain_lower)))
        return {
            "has_html": 0,
            "img_count": 0,
            "img_to_word_ratio": 0.0,
            "hidden_element_count": 0,
            "form_count": 0,
            "script_count": 0,
            "external_link_count": 0,
            "generic_salutation": generic_sal,
        }

    soup = BeautifulSoup(html_body, "html.parser")

    # --- img_count and img_to_word_ratio ---
    img_count = len(soup.find_all("img"))
    word_count = _word_count(html_body) or len(plain_body.split()) or 1
    img_to_word_ratio = round(img_count / word_count, 4)

    # --- hidden_element_count ---
    hidden_count = 0
    for tag in soup.find_all(style=True):
        style_val = tag.get("style", "")
        if _HIDDEN_STYLE_RE.search(style_val):
            hidden_count += 1
    # Also check width=0 / height=0 on elements
    for tag in soup.find_all(True):
        if tag.get("width") == "0" or tag.get("height") == "0":
            hidden_count += 1

    # --- form_count ---
    form_count = len(soup.find_all("form"))

    # --- script_count ---
    script_count = len(soup.find_all("script"))

    # --- external_link_count ---
    external_link_count = 0
    for tag in soup.find_all(["a", "img", "link", "script"]):
        url = tag.get("href") or tag.get("src") or ""
        if _is_external(url):
            external_link_count += 1

    # --- generic_salutation ---
    # Check both the HTML text and the plain body
    combined_text = soup.get_text(" ").lower() + " " + plain_body.lower()
    generic_salutation = int(bool(_GENERIC_SALUTATION_RE.search(combined_text)))

    return {
        "has_html": has_html,
        "img_count": img_count,
        "img_to_word_ratio": img_to_word_ratio,
        "hidden_element_count": hidden_count,
        "form_count": form_count,
        "script_count": script_count,
        "external_link_count": external_link_count,
        "generic_salutation": generic_salutation,
    }


# ---------------------------------------------------------------------------
# Sklearn transformer
# ---------------------------------------------------------------------------

class StructuralFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer wrapping extract_structural_features.

    Expects a DataFrame with columns ['html_body', 'body'] or a
    two-column array-like in that order.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        results = []
        for i in range(len(X)):
            if hasattr(X, "iloc"):
                row = X.iloc[i]
                html_body = str(row.get("html_body", "") or "")
                plain_body = str(row.get("body", "") or "")
            else:
                html_body = str(X[i, 0]) if X.shape[1] > 0 else ""
                plain_body = str(X[i, 1]) if X.shape[1] > 1 else ""
            feats = extract_structural_features(html_body, plain_body)
            results.append([feats[k] for k in FEATURE_NAMES])
        return np.array(results, dtype=float)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(FEATURE_NAMES)
