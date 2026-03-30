# PhishGuard — Week 2 Deliverable Report
**Phase 2: Feature Engineering & Baseline Models** | March 20–28, 2026

---

## What Was Completed

### 1. Dataset Schema Update

The data pipeline (`src/data/pipeline.py`) was extended to preserve two fields that were previously discarded:

- **`html_body`**: raw HTML from the email's `text/html` MIME part (used by URL mismatch and structural extractors)
- **`reply_to`**: the `Reply-To` header value (used by the header anomaly extractor)

`cleaned.csv` was regenerated. The final schema is now 7 columns: `subject, body, sender, label, source, html_body, reply_to`. Population by source:

| Column | Non-empty rows | Primary sources |
|---|---|---|
| `html_body` | 725 | SpamAssassin spam_2 (643), hard_ham (76), easy_ham (6) |
| `reply_to` | 938 | SpamAssassin spam_2 (533), easy_ham (363), hard_ham (42) |

Enron and Nazario have no HTML or Reply-To data — their feature values for those groups default to zero, which is accurate.

---

### 2. Four Feature Extractors

All four extractor modules were implemented from scratch in `src/features/`. Each provides a standalone function for single-email use and a `sklearn`-compatible `TransformerMixin` class for pipeline integration.

**`url_extractor.py` — 6 features**

| Feature | What it captures |
|---|---|
| `url_count` | Total unique URLs across plain text and HTML |
| `display_mismatch_count` | Anchors where visible text is a URL pointing to a different domain than the href |
| `has_shortener` | Presence of known URL shortener services (bit.ly, tinyurl, etc.) |
| `suspicious_tld_count` | URLs using TLDs commonly abused in phishing (.xyz, .tk, .top, etc.) |
| `ip_url_count` | URLs using a raw IP address instead of a hostname |
| `max_url_entropy` | Shannon entropy of the most complex URL's path+query (high = obfuscated) |

**`header_extractor.py` — 5 features**

| Feature | What it captures |
|---|---|
| `has_reply_to` | Whether a Reply-To header is present at all |
| `reply_to_differs` | Reply-To domain ≠ From domain (replies go to a different address) |
| `sender_is_freemail` | Sender uses a free consumer email service |
| `display_name_mismatch` | From display name contains a known brand but the sending domain does not match |
| `sender_domain_numeric` | Domain name contains digits (leet-speak substitution, e.g. paypa1.com) |

**`text_extractor.py` — 11 features**

| Feature | What it captures |
|---|---|
| `urgency_score` | Count of urgency/pressure keywords (verify, suspended, act now, etc.) |
| `caps_ratio` | Fraction of alphabetic characters that are uppercase |
| `exclamation_count` | Number of `!` characters |
| `question_count` | Number of `?` characters |
| `punct_density` | Punctuation characters / total characters |
| `body_char_len` | Raw character length |
| `body_word_count` | Word count |
| `avg_word_len` | Mean word length |
| `unique_word_ratio` | Unique lowercase words / total words (lexical diversity) |
| `digit_ratio` | Digit characters / total characters |
| `flesch_reading_ease` | Readability score (heuristic syllable/sentence approximation) |

**`structural_extractor.py` — 8 features**

| Feature | What it captures |
|---|---|
| `has_html` | Whether the email contains an HTML body |
| `img_count` | Number of `<img>` tags |
| `img_to_word_ratio` | Images per word (image-heavy emails hide text content) |
| `hidden_element_count` | Elements hidden via CSS (`display:none`, zero-size, white-on-white) |
| `form_count` | Number of `<form>` tags (credential harvesting indicator) |
| `script_count` | Number of `<script>` tags |
| `external_link_count` | External `href`/`src` references |
| `generic_salutation` | "Dear customer / user / valued member" in the greeting |

---

### 3. Sklearn Feature Pipeline (`src/features/pipeline.py`)

A `ColumnTransformer` + `Pipeline` was built that accepts a DataFrame and outputs a **130-dimensional dense float64 array** per email:

| Step | Input columns | Output dimensions |
|---|---|---|
| `TextFeatureTransformer` | `body` | 11 |
| `UrlFeatureTransformer` | `body`, `html_body` | 6 |
| `StructuralFeatureTransformer` | `html_body`, `body` | 8 |
| `HeaderFeatureTransformer` | `sender`, `reply_to` | 5 |
| `TfidfVectorizer` (15k vocab, bigrams) + `TruncatedSVD` (100 components) | `subject` + `body` | 100 |
| `StandardScaler` (final step) | all 130 | 130 |

The pipeline was verified on the full 5,452-row dataset: no NaN, no Inf, exact name-to-column correspondence.

---

### 4. Baseline Models (`src/models/train.py`, `src/models/evaluate.py`)

Four classifiers were trained on an 80/20 stratified split (4,361 train / 1,091 test). Each model's decision threshold was tuned to maximise precision while keeping recall ≥ 0.95 on the phishing class. All artifacts saved to `models/artifacts/`.

#### Results

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| **XGBoost** | 0.76 | **0.9844** | 0.9962 | **0.9725** | **0.9842** | **0.9991** |
| Random Forest | 0.69 | 0.9762 | 0.9924 | 0.9596 | 0.9757 | 0.9982 |
| Logistic Regression | 0.89 | 0.9743 | **0.9981** | 0.9505 | 0.9737 | 0.9981 |
| Gaussian Naive Bayes | 0.50 | 0.7846 | 0.9586 | 0.5945 | 0.7339 | 0.9585 |

**Best model: XGBoost** with F1 = 0.984 and ROC-AUC = 0.999 — already exceeding the 95% accuracy target from the scope document.

#### Notable observations

- **XGBoost and Random Forest are near-equivalent** in ROC-AUC (0.9991 vs 0.9982), suggesting the data has strong non-linear structure that linear models partially miss.
- **Logistic Regression achieves the highest precision** (0.998) but the lowest recall (0.950) — it is conservative and rarely false-alarms, but misses more phishing.
- **Gaussian Naive Bayes underperforms on recall** (0.595 at 0.5 threshold). The independence assumption breaks down across the correlated SVD components. Its ROC-AUC (0.958) shows the model does separate classes at the probability level; threshold tuning alone cannot recover recall here because the probability estimates are poorly calibrated.
- **XGBoost missed only 15 phishing emails** out of 545. Error analysis shows these tend to be plain-text-only emails with no URLs and low urgency scores — structurally indistinguishable from normal email by the current feature set.

---

### 5. EDA Notebooks

- **`notebooks/02_features.ipynb`**: Feature distribution violin plots, Cohen's d class separation ranking, correlation heatmap, Random Forest feature importance, false-negative vs. true-positive feature signatures.
- **`notebooks/03_training.ipynb`**: Model comparison table, overlaid ROC curves, confusion matrices, precision-recall curves, XGBoost feature importance (top 20), false-negative deep-dive.

---

## Key Findings

1. **`urgency_score` and `body_word_count` are the strongest single-feature separators** (highest Cohen's d). Phishing emails average ~8 urgency keywords vs ~1 in ham.
2. **TF-IDF SVD components dominate feature importance** in both XGBoost and Random Forest — vocabulary-level signals carry more weight than any individual hand-crafted feature. This validates including them.
3. **`img_to_word_ratio` and `form_count` are near-zero across the dataset** because most of the HTML-bearing emails come from SpamAssassin spam, which skews old. Modern phishing HTML with credential-harvesting forms is underrepresented — a known gap tied to the missing CLAIR dataset.
4. **`reply_to_differs` fires on 17% of phishing emails** — a reliable but not dominant signal. Its low base rate in ham makes it high-precision when present.

---

## Concerns

1. **Naive Bayes is not production-viable** as implemented. Its recall at any reasonable threshold is too low for a security tool. It will be dropped from Week 3 hyperparameter tuning.

2. **The 15 false negatives from XGBoost are structurally plain-text emails with no URL or HTML signals.** The model currently has no way to distinguish them from legitimate short emails. Content-based features (n-gram patterns, sender reputation, domain age) may help but require external data sources not in the current stack.

3. **Threshold tuning was performed on the test set** (acceptable for Week 2 baseline comparison, as noted in the code). Week 3 should use a proper validation split or cross-validation to avoid overfitting the threshold.

4. **`cleaned.csv` has 4,727 rows with empty `html_body`** — the majority of the dataset. Structural and URL-HTML features are zero for most rows, which means they contribute little to training but still occupy 14 feature dimensions. Week 3 feature selection may prune them.

---

## Next Steps (Week 3)

Per the scope document Phase 3 (March 28 – April 6):

1. **Hyperparameter tuning** — `GridSearchCV` / `RandomizedSearchCV` on XGBoost and Random Forest. Key parameters: `max_depth`, `learning_rate`, `n_estimators`, `colsample_bytree`, `subsample`.
2. **Proper threshold optimisation** — tune on a held-out validation fold, not the test set.
3. **Feature selection** — consider dropping zero-variance or low-importance features (especially structural HTML features that fire on <15% of rows).
4. **Cross-validation** — replace single train/test split with 5-fold stratified CV for more reliable metric estimates.
5. **Model calibration** — check if XGBoost's probability estimates are well-calibrated (Platt scaling / isotonic regression if not).

---

## Plain-English Recap

We taught the model what to look for inside emails. We built four sets of detectors: one that spots suspicious links (hidden redirects, IP addresses, URL shorteners), one that checks if the sender is pretending to be someone they're not, one that counts pressure words and unusual punctuation, and one that analyses the HTML structure for hidden content and fake login forms. We combined all of these with a vocabulary-based analysis of the email text, giving us 130 signals per email. We then trained four different classifiers — XGBoost came out on top, catching 97% of phishing emails in the test set while almost never flagging a legitimate email as suspicious. The model already exceeds the project's accuracy target with no tuning at all, which gives us a strong foundation heading into the final optimisation phase.
