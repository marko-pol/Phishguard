# PhishGuard — Week 3 Deliverable Report
**Phase 3: Web Application Build** | March 29 – April 6, 2026
**Status as of:** March 30, 2026

---

## What Was Completed

### 1. Gradio Web Application (`app/gradio_app.py`)

A complete single-page dark-themed UI was built and wired to the inference pipeline. All core UI features from scope §3.4 are implemented.

**Layout:**
- Branded header with PhishGuard shield logo and gradient typography
- Single-field email paste area (monospace font, 12-line textarea)
- "Analyze Email" primary button + "Clear" secondary button
- Auto-scrolling result card that fades in on submit
- "Try an example" section with two pre-loaded emails (phishing + ham)

**Result card (rendered on every analysis):**
| Section | Content |
|---|---|
| Verdict header | PHISHING DETECTED / LIKELY SAFE with colour-coded icon |
| Confidence badge | Phishing probability as a percentage (0–100%) |
| Confidence bar | Animated progress bar with threshold marker |
| Analysis section | Ordered bullet list of red-flag explanations |
| Signal breakdown | 8 colour-coded chips (URLs, Urgency, HTML, Shortener, Freemail, Reply-To, IP URLs, Generic Greeting) |

**Design decisions:**
- Full dark theme with CSS variable palette (`--bg`, `--surface`, `--accent`, etc.)
- Result card built as injected HTML so all styling is self-contained and survives Gradio re-renders
- Empty card collapses to zero height via CSS `:empty` selector — no layout shift between analyses

---

### 2. Inference Pipeline (`app/inference.py`)

`parse_raw_input()` handles RFC2822-formatted emails pasted as raw text:
- Extracts `Subject`, `From`, `Reply-To` headers via Python's `email` stdlib
- Walks MIME parts to extract `text/plain` and `text/html` parts separately
- Falls back to treating the full string as plain body when no headers are present
- **Bug fix (Week 3):** Detects HTML in the plain body and promotes it to `html_body` when the email was pasted without MIME wrapping — previously caused `form_count`, `script_count`, and `has_html` to read zero for pasted HTML emails

`run_prediction()` passes the parsed dict through the feature pipeline and model:
- Loads `models/artifacts/best_model.joblib` once (cached globally)
- Returns `label`, `confidence`, `threshold`, and the 30-dimensional `raw_features` dict used by the explainer

---

### 3. Red-Flag Explanation Generator (`app/explainer.py`)

`generate_explanation()` translates raw feature values into prioritised human-readable bullets. Signals are grouped into four categories and sorted by severity (priority 3 → 1):

| Priority | Examples |
|---|---|
| 3 (High) | URL shortener, IP address URL, Reply-To mismatch, display name spoofing, embedded form |
| 2 (Medium) | Suspicious TLD, high URL entropy, urgency language ≥5 keywords, caps ratio >12%, hidden HTML elements, numeric sender domain |
| 1 (Low) | URL count above average, moderate urgency, exclamation overuse, freemail sender (phishing only), embedded scripts |

`get_signal_chips()` returns 8 structured `{label, value, level}` dicts for the chip row. Levels (`danger`, `warn`, `neutral`) map to distinct colour schemes in the UI.

**Explainer improvements made during testing (Week 3):**

| Issue | Fix |
|---|---|
| URL chip warned at >2 URLs but text flag only fired at >4 | Lowered text flag threshold to match chip: `> 2` |
| Freemail flag fired on clearly safe personal emails | Suppressed freemail flag when `result["label"] == 0` |
| High-confidence verdict with ≤2 structural flags had no explanation | Added vocabulary/phrasing note when `confidence ≥ 0.85` and `len(flags) ≤ 2` |
| Pasted HTML emails showed 0 for `has_html`, `form_count` | Fixed in `parse_raw_input()` via HTML-in-body detection |

---

### 4. Feature Removed: File Upload

The `gr.File` upload component was removed from this version. Investigation confirmed the server-side upload endpoint and SSE progress tracking both work correctly, but Gradio 6.9.0's client-side JavaScript never resolves the `uploading` state in the browser — a known issue with Gradio 6 components inside `gr.Tabs`. Since pasting raw email text covers the same use case, the upload tab was removed to ship a working product. File upload is documented in the post-v1 backlog.

---

### 5. Unit Test Suite (`tests/`)

A full pytest suite was written from scratch covering all four feature extractors and the inference pipeline.

**Coverage summary:**

| File | Tests | What's covered |
|---|---|---|
| `test_features.py` | 72 | All four extractors: edge cases, true/false signal detection, transformer output shape and feature names |
| `test_inference.py` | 20 | `parse_raw_input` header parsing, fallback behaviour, HTML promotion fix, `run_prediction` output contract and model correctness |
| **Total** | **92** | **92 passed, 0 failed — 1.3 seconds** |

**Selected test highlights:**
- `test_html_in_plain_body_promoted_to_html_body` — regression test for the HTML detection fix
- `test_display_mismatch_not_triggered_when_text_is_not_url` — guards against false positives in link text detection
- `test_phishing_example_classified_as_phishing` / `test_ham_example_classified_as_safe` — end-to-end model correctness with confidence assertions
- `test_prediction_is_deterministic` — ensures no random state leaks across calls

---

## Current Project Status vs. Scope

### Phase 3 checklist (scope §3.4)

| Deliverable | Status |
|---|---|
| Text area for pasting raw email content | ✅ Complete |
| File upload (.eml / .txt) | ⛔ Removed — Gradio 6 client-side bug; post-v1 |
| "Analyze" button with loading feedback | ✅ Complete |
| Verdict (Phishing / Legitimate) | ✅ Complete |
| Confidence score (0–100%) | ✅ Complete |
| Red-flag bullet list with explanations | ✅ Complete |
| "Why is this suspicious?" educational context | ✅ Complete — signal chips + prioritised explanations |
| pytest unit tests for feature extractors | ✅ Complete — 92 tests |
| GitHub Actions CI pipeline | 🔄 In progress (Week 3 remainder) |

---

## Success Metrics Check (scope §7)

| Metric | Target | Current status |
|---|---|---|
| Accuracy on held-out test set | >95% | ✅ 98.4% (XGBoost) |
| False-negative rate (missed phishing) | <3% | ✅ 2.75% (15/545) |
| Inference time | <2 seconds | ✅ ~0.8s per email on CPU |
| 20+ real-world emails manually tested | Required for Phase 4 | 🔄 Pending |
| Public URL via Hugging Face Spaces | Required for Phase 4 | 🔄 Pending |
| Complete README | Required for Phase 4 | 🔄 Pending |

---

## Remaining Phase 3 Work

1. **GitHub Actions CI pipeline** — the stub at `.github/workflows/ci.yml` needs Python version correction, pip caching, and proper test flags before push
2. End-to-end manual testing with real-world email samples (starts Phase 4)

---

## Plain-English Recap

We built and polished the web interface that lets anyone paste an email and get a verdict in under a second. The result card tells you not just whether the email is suspicious, but exactly why — flagging things like hidden redirect links, fake sender names, and pressure language. Along the way we caught and fixed four bugs in the explanation engine (HTML forms going undetected, misleading warnings on personal emails, and a mismatch between the visual indicators and the text explanations). We also wrote 92 automated tests that confirm every part of the detection pipeline behaves correctly — from individual feature extractors all the way through to the final model prediction. The project is on track: all core features are working locally, the model exceeds its accuracy target, and deployment to Hugging Face Spaces is next.
