"""
Email header feature extractor.

Extracts per-email header signals that indicate spoofing or impersonation:
  - has_reply_to          : 1 if a Reply-To header is present
  - reply_to_differs      : 1 if Reply-To domain ≠ From domain
  - sender_is_freemail    : 1 if sender uses a free consumer email service
  - display_name_mismatch : 1 if the From display name contains a known brand
                            but the sending domain does not match that brand
  - sender_domain_numeric : 1 if the sender domain contains digits in an
                            unusual pattern (e.g. paypa1.com, amaz0n.com)

Provides both a standalone function and a sklearn-compatible transformer.
"""

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FREEMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "live.com",
    "msn.com", "aol.com", "icloud.com", "me.com", "mac.com",
    "protonmail.com", "proton.me", "yandex.com", "yandex.ru",
    "mail.com", "gmx.com", "gmx.net", "zoho.com", "tutanota.com",
    "fastmail.com", "inbox.com", "rediffmail.com",
}

# Well-known brands whose names appearing in a display name but not in the
# actual sending domain is a strong phishing signal.
KNOWN_BRANDS = {
    "paypal", "amazon", "apple", "microsoft", "google", "facebook",
    "netflix", "ebay", "instagram", "twitter", "linkedin", "dropbox",
    "chase", "wellsfargo", "bankofamerica", "citibank", "hsbc",
    "fedex", "ups", "usps", "dhl", "irs", "intuit", "turbotax",
}

FEATURE_NAMES = [
    "has_reply_to",
    "reply_to_differs",
    "sender_is_freemail",
    "display_name_mismatch",
    "sender_domain_numeric",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ADDR_RE = re.compile(r'<([^>]+)>')


def _extract_address(field: str) -> str:
    """Pull the bare email address out of a header value like 'Name <addr>'."""
    m = _ADDR_RE.search(field)
    if m:
        return m.group(1).strip().lower()
    return field.strip().lower()


def _domain(address: str) -> str:
    """Return the domain portion of an email address."""
    if "@" in address:
        return address.split("@", 1)[1]
    return ""


def _display_name(field: str) -> str:
    """Return the display name portion of a 'Name <addr>' header value."""
    m = _ADDR_RE.search(field)
    if m:
        return field[:m.start()].strip().strip('"\'').lower()
    return ""


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------

def extract_header_features(sender: str, reply_to: str) -> dict:
    """
    Return a dict of header-based features for a single email.

    Parameters
    ----------
    sender   : raw value of the From header  (e.g. '"PayPal" <spoof@evil.ru>')
    reply_to : raw value of the Reply-To header, or empty string
    """
    sender = sender or ""
    reply_to = reply_to or ""

    from_addr = _extract_address(sender)
    from_domain = _domain(from_addr)

    # --- has_reply_to ---
    has_reply_to = int(bool(reply_to.strip()))

    # --- reply_to_differs ---
    reply_to_differs = 0
    if has_reply_to:
        rt_addr = _extract_address(reply_to)
        rt_domain = _domain(rt_addr)
        if from_domain and rt_domain and from_domain != rt_domain:
            reply_to_differs = 1

    # --- sender_is_freemail ---
    sender_is_freemail = int(from_domain in FREEMAIL_DOMAINS)

    # --- display_name_mismatch ---
    display_name_mismatch = 0
    display = _display_name(sender)
    if display:
        for brand in KNOWN_BRANDS:
            if brand in display:
                # Brand appears in display name — does the sending domain match?
                if brand not in from_domain:
                    display_name_mismatch = 1
                break

    # --- sender_domain_numeric ---
    # Flags domains that replace letters with digits (leet-speak substitution)
    # e.g. paypa1.com, amaz0n-security.com
    domain_name = from_domain.split(".")[0] if from_domain else ""
    sender_domain_numeric = int(bool(re.search(r'[0-9]', domain_name)))

    return {
        "has_reply_to": has_reply_to,
        "reply_to_differs": reply_to_differs,
        "sender_is_freemail": sender_is_freemail,
        "display_name_mismatch": display_name_mismatch,
        "sender_domain_numeric": sender_domain_numeric,
    }


# ---------------------------------------------------------------------------
# Sklearn transformer
# ---------------------------------------------------------------------------

class HeaderFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer wrapping extract_header_features.

    Expects a DataFrame with columns ['sender', 'reply_to'] or a
    two-column array-like in that order.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        results = []
        for i in range(len(X)):
            if hasattr(X, "iloc"):
                row = X.iloc[i]
                sender = str(row.get("sender", "") or "")
                reply_to = str(row.get("reply_to", "") or "")
            else:
                sender = str(X[i, 0]) if X.shape[1] > 0 else ""
                reply_to = str(X[i, 1]) if X.shape[1] > 1 else ""
            feats = extract_header_features(sender, reply_to)
            results.append([feats[k] for k in FEATURE_NAMES])
        return np.array(results, dtype=float)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(FEATURE_NAMES)
