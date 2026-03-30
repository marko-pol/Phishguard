"""
URL feature extractor.

Extracts per-email URL-based signals that correlate with phishing:
  - url_count             : total unique URLs across plain text + HTML
  - display_mismatch_count: anchors whose visible text is a URL pointing to a
                            different domain than the href
  - has_shortener         : 1 if any known URL shortener is present
  - suspicious_tld_count  : URLs with TLDs commonly abused in phishing
  - ip_url_count          : URLs using a raw IP address instead of a hostname
  - max_url_entropy       : Shannon entropy of the most complex URL path+query
                            (high entropy → obfuscated / randomised URLs)

Provides both a standalone function and a sklearn-compatible transformer.
"""

import math
import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "buff.ly",
    "is.gd", "cli.gs", "su.pr", "tiny.cc", "url4.eu", "tr.im",
    "youtu.be", "kl.am", "wp.me", "x.co", "budurl.com", "snurl.com",
    "short.ie", "u.nu", "rb.gy", "cutt.ly", "shorturl.at", "gg.gg",
}

SUSPICIOUS_TLDS = {
    ".xyz", ".tk", ".top", ".cc", ".pw", ".su", ".bid", ".loan",
    ".win", ".download", ".click", ".link", ".work", ".gq", ".ml",
    ".cf", ".ga", ".zip", ".mov", ".cam", ".icu",
}

_URL_RE = re.compile(r'https?://[^\s<>"\']+', re.IGNORECASE)
_IP_URL_RE = re.compile(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', re.IGNORECASE)

FEATURE_NAMES = [
    "url_count",
    "display_mismatch_count",
    "has_shortener",
    "suspicious_tld_count",
    "ip_url_count",
    "max_url_entropy",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_parse(url: str):
    """urlparse wrapper that returns an empty result on malformed URLs."""
    try:
        return urlparse(url)
    except ValueError:
        from urllib.parse import ParseResult
        return ParseResult("", "", "", "", "", "")


def _path_entropy(url: str) -> float:
    """Shannon entropy of the URL path + query string."""
    parsed = _safe_parse(url)
    s = parsed.path + parsed.query
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    return -sum((f / n) * math.log2(f / n) for f in freq.values())


def _normalise_domain(url: str) -> str:
    return _safe_parse(url).netloc.lower().lstrip("www.")


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------

def extract_url_features(body: str, html_body: str = "") -> dict:
    """
    Return a dict of URL-based features for a single email.

    Parameters
    ----------
    body      : plain-text email body (already stripped of HTML)
    html_body : raw HTML body, or empty string if unavailable
    """
    # --- Collect URLs from plain text ---
    plain_urls: list[str] = _URL_RE.findall(body)

    # --- Parse HTML for href-anchored URLs and display-text mismatch ---
    display_mismatch = 0
    html_hrefs: list[str] = []

    if html_body:
        soup = BeautifulSoup(html_body, "html.parser")
        for tag in soup.find_all("a", href=True):
            href = tag["href"].strip()
            if not href.startswith("http"):
                continue
            html_hrefs.append(href)
            # Check if anchor display text looks like a URL to a different domain
            display = tag.get_text(strip=True)
            if re.match(r'https?://', display, re.IGNORECASE):
                href_domain = _normalise_domain(href)
                disp_domain = _normalise_domain(display)
                if href_domain and disp_domain and href_domain != disp_domain:
                    display_mismatch += 1

    all_urls = list({u for u in plain_urls + html_hrefs})

    # --- Derive features ---
    url_count = len(all_urls)

    has_shortener = int(any(
        _normalise_domain(u) in SHORTENER_DOMAINS for u in all_urls
    ))

    suspicious_tld_count = sum(
        1 for u in all_urls
        if any(_safe_parse(u).netloc.lower().endswith(tld) for tld in SUSPICIOUS_TLDS)
    )

    combined_text = body + " " + html_body
    ip_url_count = len(_IP_URL_RE.findall(combined_text))

    max_entropy = max((_path_entropy(u) for u in all_urls), default=0.0)

    return {
        "url_count": url_count,
        "display_mismatch_count": display_mismatch,
        "has_shortener": has_shortener,
        "suspicious_tld_count": suspicious_tld_count,
        "ip_url_count": ip_url_count,
        "max_url_entropy": round(max_entropy, 4),
    }


# ---------------------------------------------------------------------------
# Sklearn transformer
# ---------------------------------------------------------------------------

class UrlFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that wraps extract_url_features.

    Expects a DataFrame with columns ['body', 'html_body'] or a
    two-column array-like in that order.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        results = []
        for i in range(len(X)):
            if hasattr(X, "iloc"):
                row = X.iloc[i]
                body = str(row.get("body", "") or "")
                html_body = str(row.get("html_body", "") or "")
            else:
                body = str(X[i, 0]) if X.shape[1] > 0 else ""
                html_body = str(X[i, 1]) if X.shape[1] > 1 else ""
            feats = extract_url_features(body, html_body)
            results.append([feats[k] for k in FEATURE_NAMES])
        return np.array(results, dtype=float)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(FEATURE_NAMES)
