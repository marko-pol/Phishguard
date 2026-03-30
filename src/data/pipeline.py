"""
Data loading, parsing, and deduplication pipeline.

Loads raw datasets from data/raw/, unifies schema, deduplicates,
balances classes, and writes data/processed/cleaned.csv.

Usage:
    python -m src.data.pipeline          # run full pipeline
    from src.data.pipeline import build_dataset
"""

import email
import email.message
import hashlib
import logging
import os
import random
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

# Target ham samples to draw from Enron to supplement SpamAssassin ham.
# Keeps memory usage manageable while providing enough legitimate examples.
ENRON_SAMPLE_TARGET = 2500

# Folders to skip when sampling Enron (noise / non-representative content)
ENRON_SKIP_FOLDERS = {
    "deleted_items", "trash", "_sent_mail", "sent", "sent_items",
    "notes_inbox", "calendar", "contacts", "tasks",
}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _extract_text_from_email(msg: email.message.Message) -> str:
    """Return plain-text body from a parsed email.Message."""
    body_parts = []
    for part in msg.walk():
        ct = part.get_content_type()
        disp = str(part.get("Content-Disposition", ""))
        if "attachment" in disp:
            continue
        if ct == "text/plain":
            payload = part.get_payload(decode=True)
            if payload:
                body_parts.append(payload.decode("utf-8", errors="replace"))
        elif ct == "text/html" and not body_parts:
            # Fall back to HTML-stripped text only if no plain-text part found
            payload = part.get_payload(decode=True)
            if payload:
                html = payload.decode("utf-8", errors="replace")
                body_parts.append(BeautifulSoup(html, "html.parser").get_text(" "))
    return " ".join(body_parts).strip()


def _extract_html_from_email(msg: email.message.Message) -> str:
    """Return the first text/html MIME part as a raw string, or empty string."""
    for part in msg.walk():
        ct = part.get_content_type()
        disp = str(part.get("Content-Disposition", ""))
        if "attachment" in disp:
            continue
        if ct == "text/html":
            payload = part.get_payload(decode=True)
            if payload:
                return payload.decode("utf-8", errors="replace")
    return ""


def _parse_email_file(path: Path) -> dict | None:
    """Parse a single RFC2822 email file. Returns None on failure."""
    try:
        raw = path.read_bytes()
        msg = email.message_from_bytes(raw)
        body = _extract_text_from_email(msg)
        if not body:
            return None
        return {
            "subject": msg.get("Subject", "") or "",
            "sender": msg.get("From", "") or "",
            "body": body,
            "html_body": _extract_html_from_email(msg),
            "reply_to": msg.get("Reply-To", "") or "",
        }
    except Exception:
        return None


def _body_hash(body: str) -> str:
    """MD5 of whitespace-normalised, lowercase body for deduplication."""
    normalised = " ".join(body.lower().split())
    return hashlib.md5(normalised.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_nazario(path: Path | None = None) -> pd.DataFrame:
    """Load Nazario phishing CSV. Returns DataFrame with unified schema."""
    path = path or RAW_DIR / "Nazario.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"body": "body", "subject": "subject", "sender": "sender"})
    df = df[["subject", "body", "sender"]].copy()
    df["label"] = 1
    df["source"] = "nazario"
    log.info("Nazario: %d rows loaded", len(df))
    return df


def load_spamassassin(archive_dir: Path | None = None) -> pd.DataFrame:
    """
    Load SpamAssassin corpus from the archive directory.
    easy_ham + hard_ham → label 0, spam_2 → label 1.
    """
    archive_dir = archive_dir or RAW_DIR / "archive"
    folder_labels = {
        "easy_ham": 0,
        "hard_ham": 0,
        "spam_2": 1,
    }
    records = []
    for folder, label in folder_labels.items():
        folder_path = archive_dir / folder
        if not folder_path.exists():
            log.warning("SpamAssassin folder not found: %s", folder_path)
            continue
        files = [f for f in folder_path.iterdir() if f.is_file() and not f.name.startswith(".")]
        for fp in files:
            parsed = _parse_email_file(fp)
            if parsed:
                parsed["label"] = label
                parsed["source"] = f"spamassassin_{folder}"
                records.append(parsed)
        log.info("SpamAssassin %s: %d/%d files parsed", folder, sum(1 for r in records if r["source"] == f"spamassassin_{folder}"), len(files))
    df = pd.DataFrame(records)
    log.info("SpamAssassin total: %d rows", len(df))
    return df


def load_enron(maildir: Path | None = None, sample_n: int = ENRON_SAMPLE_TARGET, seed: int = 42) -> pd.DataFrame:
    """
    Sample ham emails from the Enron maildir corpus.
    Walks all user folders, collects file paths, draws a random sample,
    then parses only those files (avoids loading all 500k into memory).
    """
    maildir = maildir or RAW_DIR / "maildir"
    if not maildir.exists():
        log.warning("Enron maildir not found: %s", maildir)
        return pd.DataFrame()

    # Collect all candidate file paths
    candidate_paths: list[Path] = []
    for user_dir in maildir.iterdir():
        if not user_dir.is_dir():
            continue
        for folder in user_dir.iterdir():
            if not folder.is_dir():
                continue
            if folder.name.lower() in ENRON_SKIP_FOLDERS:
                continue
            for fp in folder.iterdir():
                if fp.is_file():
                    candidate_paths.append(fp)

    log.info("Enron: %d candidate files found", len(candidate_paths))

    rng = random.Random(seed)
    sample_paths = rng.sample(candidate_paths, min(sample_n * 3, len(candidate_paths)))

    records = []
    for fp in sample_paths:
        if len(records) >= sample_n:
            break
        parsed = _parse_email_file(fp)
        if parsed and len(parsed["body"]) > 20:
            parsed["label"] = 0
            parsed["source"] = "enron"
            records.append(parsed)

    df = pd.DataFrame(records)
    log.info("Enron: %d usable emails sampled", len(df))
    return df


# ---------------------------------------------------------------------------
# Deduplication and balancing
# ---------------------------------------------------------------------------

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact-duplicate bodies (after whitespace normalisation)."""
    before = len(df)
    df = df.copy()
    df["_hash"] = df["body"].apply(_body_hash)
    df = df.drop_duplicates(subset="_hash").drop(columns="_hash")
    df = df.reset_index(drop=True)
    log.info("Deduplication: %d → %d rows (removed %d)", before, len(df), before - len(df))
    return df


def balance_classes(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Undersample the majority class to match the minority class size."""
    counts = df["label"].value_counts()
    min_count = counts.min()
    balanced = pd.concat(
        [group.sample(min_count, random_state=seed) for _, group in df.groupby("label")],
        ignore_index=True,
    )
    log.info(
        "Class balancing: %s → %d per class (%d total)",
        dict(counts),
        min_count,
        len(balanced),
    )
    return balanced


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_dataset(
    balance: bool = True,
    save: bool = True,
    enron_sample: int = ENRON_SAMPLE_TARGET,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run the full data pipeline:
      1. Load all sources
      2. Unify schema
      3. Deduplicate
      4. (Optionally) balance classes
      5. (Optionally) save to data/processed/cleaned.csv

    Returns the cleaned DataFrame.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parts = [
        load_nazario(),
        load_spamassassin(),
        load_enron(sample_n=enron_sample, seed=seed),
    ]
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)

    # Normalise types
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["sender"] = df["sender"].fillna("").astype(str)
    df["html_body"] = df["html_body"].fillna("").astype(str)
    df["reply_to"] = df["reply_to"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)

    df = deduplicate(df)

    if balance:
        df = balance_classes(df, seed=seed)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / "cleaned.csv"
        df.to_csv(out_path, index=False)
        log.info("Saved cleaned dataset → %s", out_path)

    return df


if __name__ == "__main__":
    build_dataset()
