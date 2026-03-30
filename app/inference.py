"""
Model inference module.
Loads best_model.joblib and runs predictions on a single parsed email.
"""
from __future__ import annotations

import email as _email_mod
import email.message
import logging
import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parents[1]))

log = logging.getLogger(__name__)

BEST_MODEL_PATH = Path(__file__).parents[1] / "models" / "artifacts" / "best_model.joblib"

_artifact = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model() -> dict:
    global _artifact
    if _artifact is None:
        import joblib
        if not BEST_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {BEST_MODEL_PATH}. "
                "Run `python -m src.models.train` first."
            )
        _artifact = joblib.load(BEST_MODEL_PATH)
        log.info("Loaded model: %s", _artifact["model_name"])
    return _artifact


# ---------------------------------------------------------------------------
# Email parsing
# ---------------------------------------------------------------------------

def _walk_mime(msg: email.message.Message) -> tuple[str, str]:
    """Return (plain_body, html_body) from a parsed MIME message."""
    plain, html = "", ""
    for part in msg.walk():
        ct   = part.get_content_type()
        disp = str(part.get("Content-Disposition", ""))
        if "attachment" in disp:
            continue
        raw = part.get_payload(decode=True)
        if not raw:
            continue
        text = raw.decode("utf-8", errors="replace")
        if ct == "text/plain" and not plain:
            plain = text
        elif ct == "text/html" and not html:
            html = text
    return plain, html


def parse_raw_input(text: str) -> dict:
    """
    Parse raw text as an RFC2822 email message.
    If parsing yields no usable body, treat the whole string as the plain body.
    """
    try:
        msg = _email_mod.message_from_string(text)
        subject  = msg.get("Subject", "") or ""
        sender   = msg.get("From", "")    or ""
        reply_to = msg.get("Reply-To", "") or ""
        plain, html = _walk_mime(msg)

        if not plain and html:
            plain = BeautifulSoup(html, "html.parser").get_text(" ")

        # If parsing produced nothing meaningful, fall back to raw text as body
        if not plain:
            plain = text

        # If the MIME parser found no HTML part but the plain body looks like HTML
        # (e.g. a pasted email with inline HTML), promote it so structural features fire.
        if not html and plain and re.search(r'<(html|body|form|table|div|script)', plain, re.IGNORECASE):
            html = plain

        return {
            "subject":   subject,
            "body":      plain,
            "sender":    sender,
            "html_body": html,
            "reply_to":  reply_to,
        }
    except Exception:
        return {"subject": "", "body": text, "sender": "", "html_body": "", "reply_to": ""}


def parse_file_input(file_path: str) -> dict:
    """Read an uploaded .eml or .txt file and parse it."""
    raw = Path(file_path).read_text(errors="replace")
    return parse_raw_input(raw)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def run_prediction(email_dict: dict) -> dict:
    """
    Run the full feature pipeline and model on a single email dict.

    Returns
    -------
    dict with:
        label        : int   0=ham, 1=phishing
        confidence   : float probability of phishing [0, 1]
        threshold    : float decision threshold used
        raw_features : dict  named hand-crafted feature values (pre-scaling)
    """
    art = load_model()
    row = pd.DataFrame([{
        "subject":   email_dict.get("subject", ""),
        "body":      email_dict.get("body", ""),
        "sender":    email_dict.get("sender", ""),
        "html_body": email_dict.get("html_body", ""),
        "reply_to":  email_dict.get("reply_to", ""),
    }])

    fp   = art["feature_pipeline"]
    X    = fp.transform(row)
    prob = float(art["model"].predict_proba(X)[0, 1])

    # Pre-scale hand-crafted features for the explainer (first 30 dimensions)
    from src.features.pipeline import get_feature_names
    X_raw = fp.named_steps["features"].transform(row)
    names = get_feature_names(fp)[:30]
    raw_features = dict(zip(names, map(float, X_raw[0, :30])))

    return {
        "label":        int(prob >= art["threshold"]),
        "confidence":   round(prob, 4),
        "threshold":    art["threshold"],
        "raw_features": raw_features,
    }
