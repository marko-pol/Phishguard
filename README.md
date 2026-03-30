---
title: PhishGuard
emoji: 🛡
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "6.9.0"
app_file: app.py
pinned: false
license: mit
---

# PhishGuard

Phishing Email Identification Tool — v1.0

A zero-budget, open-source phishing email classifier powered by traditional ML (scikit-learn + XGBoost). Paste or upload an email and receive an instant verdict, confidence score, and human-readable red-flag explanations.

## Setup

```bash
git clone https://github.com/your-username/phishguard.git
cd phishguard
python -m venv .venv && source .venv/bin/activate
make install
```

## Usage

```bash
make train   # Train the model
make test    # Run test suite
make app     # Launch Gradio UI
```

## Project Structure

- `data/` — raw datasets, processed features, train/val/test splits
- `src/features/` — individual feature extractors (URL, header, text, structural)
- `src/models/` — training, evaluation, and serialized model artifacts
- `app/` — Gradio UI, inference, and explanation modules
- `tests/` — pytest unit and integration tests
- `notebooks/` — EDA, feature exploration, and training experiments

## Datasets

Download and place in `data/raw/`:
- SpamAssassin Public Corpus
- Nazario Phishing Corpus
- Enron Email Dataset

See training documentation in `src/models/train.py` for preprocessing steps.
