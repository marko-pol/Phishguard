# PhishGuard — Week 1 Deliverable Report
**Phase 1: Data Collection & EDA** | March 13–19, 2026

---

## What Was Completed

### Datasets Acquired and Verified
Three email corpora were downloaded, extracted, and validated in `data/raw/`:

| Dataset | Format | Rows / Files | Label |
|---|---|---|---|
| Nazario Phishing Corpus | CSV | 1,565 rows | Phishing (1) |
| SpamAssassin Corpus | RFC2822 email files | 4,148 files across 3 folders | Ham (0) and Spam (1) |
| Enron Maildir Corpus | RFC2822 email files | ~275,000 files across 150 users | Ham (0) |

The CLAIR Fraudulent Email dataset was attempted but could not be downloaded automatically — it requires Kaggle authentication. This is a known gap (see Concerns below).

### Data Pipeline (`src/data/pipeline.py`)
A reusable, importable pipeline module was built and verified end-to-end:
- **Loaders** for each dataset with a unified 5-column schema: `subject, body, sender, label, source`
- **HTML-to-text fallback** using BeautifulSoup for emails with no plain-text part
- **Deduplication** via MD5 hash of whitespace-normalised, lowercased body text
- **Class balancing** via majority-class undersampling (pandas 3.x compatible)
- **Output**: `data/processed/cleaned.csv` — 5,452 rows, perfectly balanced at 2,726 per class

**Pipeline flow summary:**

```
Raw combined: 8,196 rows
  → After dedup: 7,894 rows  (302 duplicates removed)
  → After balance: 5,452 rows (2,726 ham / 2,726 phishing)
```

**Source mix in final dataset:**

| Source | Count |
|---|---|
| Nazario (phishing) | 1,527 |
| SpamAssassin easy_ham | 1,333 |
| Enron (ham) | 1,272 |
| SpamAssassin spam_2 | 1,199 |
| SpamAssassin hard_ham | 121 |

### Exploratory Data Analysis (`notebooks/01_eda.ipynb`)
A 6-section EDA notebook was created covering:
1. Raw dataset inventory and per-source row counts
2. Class distribution and source breakdown (bar and stacked horizontal bar charts)
3. Body and subject length distributions by class (2×2 histogram grid, 97th-percentile capped)
4. Missing value analysis (subject, body, sender emptiness rates)
5. Before/after deduplication comparison with grouped bar chart
6. Final balanced dataset summary and class distribution chart

---

## Considerations for Future Deliverables

**Week 2 — Feature Engineering**
The pipeline outputs clean text in `body` and `subject`. Week 2 feature work should account for:
- Phishing emails tend to have shorter bodies (confirmed in EDA) — length itself is a weak signal worth including.
- The `sender` field has high empty rates in some sources; header-based features will need null-safe handling.
- Enron ham is genuine workplace email — it may use very different vocabulary than SpamAssassin ham, which could inflate or deflate TF-IDF weights. Monitor per-source performance during model evaluation.

**Week 3 — Model Training**
- The 50/50 class split was achieved by undersampling ham. This is intentional for clean baselines, but real-world phishing rates are far lower. Consider evaluating models on an imbalanced holdout set or reporting precision/recall separately from accuracy.
- The Enron corpus is older (early 2000s). If the final model is deployed against modern email, vocabulary drift may reduce performance on novel phishing tactics. Flag this as a known limitation in the final report.

**Week 4 — Deployment**
- The pipeline is designed to be re-run (`build_dataset()` is a single function call), which makes it straightforward to retrain on updated datasets.
- The `source` column is preserved throughout — this enables per-source error analysis in the Gradio UI or explainability layer if desired.

---

## Concerns

1. **CLAIR dataset is missing.** The CLAIR Fraudulent Email corpus (intended as a third phishing source) requires Kaggle credentials to download. It can be manually downloaded from Kaggle (`rtatman/fraudulent-email-corpus`) and placed in `data/raw/`. A loader function is ready to be added to `pipeline.py` once the file is present. Without it, phishing examples come only from Nazario and SpamAssassin spam_2.

2. **SpamAssassin hard_ham is underrepresented (121 examples).** This folder was small in the original corpus. It contributes only a minor fraction to the final dataset and is unlikely to affect training significantly, but it means "hard" legitimate emails (those resembling spam) may be underrepresented.

3. **Enron sampling is non-deterministic across runs only if seed changes.** The current seed is fixed at 42, so reruns are reproducible. If the seed is changed, the Enron sample changes — keep this in mind if reproducing results later.

---

## Plain-English Recap

We collected three large collections of real emails — some phishing, some legitimate — cleaned them up, removed duplicates, and trimmed them down to an even 50/50 split of about 5,400 emails. We also built an exploratory notebook that shows the shape of the data: how long the emails are, where they come from, and how many fields are missing. Everything is saved and ready for the next step, which is teaching the model what patterns to look for inside those emails. The one thing we couldn't finish was adding a fourth phishing dataset that requires a Kaggle account to download — that can be dropped in manually at any point without changing anything else.
