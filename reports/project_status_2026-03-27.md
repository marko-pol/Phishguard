# PhishGuard — Project Status Report
**Date:** March 27, 2026 | **Phase:** Week 3 of 4 Complete

---

## What's Been Built

### Data Pipeline (Week 1)
The data pipeline ingests five email datasets (Nazario, SpamAssassin, Enron) into a unified schema, deduplicates by body hash, and balances to **5,452 rows (2,726 per class)**. The schema was later extended to carry `html_body` and `reply_to` through the full pipeline — enabling richer feature extraction downstream.

### Feature Engineering (Week 2)
Four feature extractors produce a **130-dimensional feature vector** per email:

| Extractor | Features | Highlights |
|---|---|---|
| URL | 6 | Shortener detection, IP URLs, path entropy |
| Header | 5 | Reply-To mismatch, freemail sender, numeric domain |
| Text (NLP) | 11 | Urgency keywords, caps ratio, Flesch reading ease |
| Structural (HTML) | 8 | Hidden elements, forms, external links, generic salutation |
| TF-IDF + LSA | 100 | 15k-vocab bigrams → TruncatedSVD |

### Model Training (Week 2)
Four classifiers were trained on an 80/20 stratified split with threshold tuning to hold phishing recall ≥ 0.95:

| Model | F1 | ROC-AUC |
|---|---|---|
| **XGBoost** *(best)* | **0.984** | **0.999** |
| Random Forest | 0.976 | 0.997 |
| Logistic Regression | 0.974 | 0.995 |
| Gaussian Naive Bayes | 0.734 | 0.960 |

The best model artifact is saved to `models/artifacts/best_model.joblib`.

### Web UI (Week 3)
A dark-themed Gradio 6 application (`app/gradio_app.py`) is fully operational:
- **Input**: paste raw email text or upload a `.eml`/`.txt` file
- **Output**: animated result card with verdict, confidence %, probability bar, red-flag explanations, and color-coded signal chips
- **UX**: result card auto-scrolls into focus on completion
- **Smoke test**: phishing example → 99.9% confidence phishing; ham example → 1.8% confidence safe

Launch: `.venv/bin/python app/gradio_app.py` → `http://localhost:7860`

---

## Open Items & Risks

| Item | Status |
|---|---|
| CLAIR fraud dataset | Blocked — requires Kaggle credentials |
| Threshold tuned on test set | Known concern — should use a held-out validation fold |
| Hyperparameter tuning (GridSearchCV) | Not started — Week 4 scope |
| 5-fold cross-validation & model calibration | Not started — Week 4 scope |

---

## Week 4 Priorities
1. Hyperparameter tuning on XGBoost and Random Forest
2. Proper threshold optimisation on a dedicated validation fold
3. 5-fold cross-validation for generalisation estimates
4. Model calibration (Platt scaling / isotonic regression)
5. Final packaging and documentation
