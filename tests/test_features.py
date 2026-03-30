"""
Unit tests for the four hand-crafted feature extractors.

Each extractor is tested via its standalone function (fast, no sklearn overhead)
and, where behaviour differs, via the sklearn transformer wrapper.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.url_extractor import extract_url_features, UrlFeatureTransformer
from src.features.text_extractor import extract_text_features, TextFeatureTransformer
from src.features.header_extractor import extract_header_features, HeaderFeatureTransformer
from src.features.structural_extractor import extract_structural_features, StructuralFeatureTransformer


# ── URL extractor ──────────────────────────────────────────────────────────────

class TestUrlExtractor:

    def test_empty_inputs_return_zeros(self):
        f = extract_url_features("", "")
        assert f["url_count"] == 0
        assert f["has_shortener"] == 0
        assert f["ip_url_count"] == 0
        assert f["suspicious_tld_count"] == 0
        assert f["display_mismatch_count"] == 0
        assert f["max_url_entropy"] == 0.0

    def test_single_clean_url_counted(self):
        f = extract_url_features("Visit https://example.com for details.", "")
        assert f["url_count"] == 1
        assert f["has_shortener"] == 0
        assert f["ip_url_count"] == 0

    def test_url_shortener_detected(self):
        f = extract_url_features("Click here: http://bit.ly/abc123", "")
        assert f["has_shortener"] == 1

    def test_url_shortener_not_triggered_by_normal_domain(self):
        f = extract_url_features("Go to https://example.com/path", "")
        assert f["has_shortener"] == 0

    def test_ip_url_detected_in_plain_body(self):
        f = extract_url_features("Verify: http://185.220.101.47/login", "")
        assert f["ip_url_count"] == 1

    def test_ip_url_detected_in_html_body(self):
        html = '<a href="http://192.168.1.1/steal">Click</a>'
        f = extract_url_features("", html)
        assert f["ip_url_count"] == 1

    def test_suspicious_tld_xyz(self):
        f = extract_url_features("Go to https://login.paypal-secure.xyz/verify", "")
        assert f["suspicious_tld_count"] == 1

    def test_suspicious_tld_tk(self):
        f = extract_url_features("https://free-prize.tk/claim", "")
        assert f["suspicious_tld_count"] == 1

    def test_legitimate_tld_not_flagged(self):
        f = extract_url_features("https://google.com https://github.com", "")
        assert f["suspicious_tld_count"] == 0

    def test_display_mismatch_detected(self):
        html = '<a href="http://evil.ru/steal">http://paypal.com/login</a>'
        f = extract_url_features("", html)
        assert f["display_mismatch_count"] == 1

    def test_display_mismatch_not_triggered_when_domains_match(self):
        html = '<a href="https://paypal.com/login">https://paypal.com/login</a>'
        f = extract_url_features("", html)
        assert f["display_mismatch_count"] == 0

    def test_display_mismatch_not_triggered_when_text_is_not_url(self):
        html = '<a href="https://evil.com/steal">Click here to verify</a>'
        f = extract_url_features("", html)
        assert f["display_mismatch_count"] == 0

    def test_high_entropy_url(self):
        # Long random token path → high Shannon entropy
        f = extract_url_features(
            "https://example.com/aB3xK9qZpW2mR8nTvL5jDcYeUoIsGfHt?t=xQzPmRnB", ""
        )
        assert f["max_url_entropy"] > 3.5

    def test_simple_path_has_low_entropy(self):
        f = extract_url_features("https://example.com/login", "")
        assert f["max_url_entropy"] < 3.5

    def test_urls_deduplicated(self):
        body = "https://example.com https://example.com https://example.com"
        f = extract_url_features(body, "")
        assert f["url_count"] == 1

    def test_urls_merged_across_plain_and_html(self):
        # Same URL in both sources should be counted once
        html = '<a href="https://example.com">link</a>'
        f = extract_url_features("https://example.com", html)
        assert f["url_count"] == 1

    def test_multiple_distinct_urls_counted(self):
        body = "https://a.com https://b.com https://c.com"
        f = extract_url_features(body, "")
        assert f["url_count"] == 3

    def test_transformer_output_shape(self):
        df = pd.DataFrame([
            {"body": "https://bit.ly/abc", "html_body": ""},
            {"body": "Hello world", "html_body": ""},
        ])
        t = UrlFeatureTransformer()
        out = t.fit_transform(df)
        assert out.shape == (2, 6)

    def test_transformer_feature_names(self):
        names = UrlFeatureTransformer().get_feature_names_out()
        assert "url_count" in names
        assert "has_shortener" in names
        assert len(names) == 6


# ── Text extractor ─────────────────────────────────────────────────────────────

class TestTextExtractor:

    def test_empty_body_returns_safe_defaults(self):
        f = extract_text_features("")
        assert f["urgency_score"] == 0
        assert f["exclamation_count"] == 0
        assert f["caps_ratio"] == 0.0
        assert f["body_char_len"] == 0
        assert f["flesch_reading_ease"] == 50.0  # neutral default

    def test_urgency_keywords_counted(self):
        f = extract_text_features("Your account has been suspended. Verify immediately.")
        assert f["urgency_score"] >= 3  # "suspended", "verify", "immediately"

    def test_urgency_case_insensitive(self):
        lower = extract_text_features("your account has been suspended")
        upper = extract_text_features("YOUR ACCOUNT HAS BEEN SUSPENDED")
        assert lower["urgency_score"] == upper["urgency_score"]

    def test_no_urgency_in_clean_email(self):
        f = extract_text_features("Hi Sarah, are we still on for lunch tomorrow?")
        assert f["urgency_score"] == 0

    def test_caps_ratio_all_caps(self):
        f = extract_text_features("URGENT ACTION REQUIRED NOW")
        assert f["caps_ratio"] > 0.9

    def test_caps_ratio_all_lower(self):
        f = extract_text_features("hello this is a test message")
        assert f["caps_ratio"] == 0.0

    def test_caps_ratio_mixed(self):
        # "Hello World" — 2 uppercase out of 10 alpha chars
        f = extract_text_features("Hello World")
        assert 0.1 < f["caps_ratio"] < 0.3

    def test_exclamation_count(self):
        f = extract_text_features("Act now!! Limited time offer! Don't miss out!")
        assert f["exclamation_count"] == 4

    def test_question_count(self):
        f = extract_text_features("Are you sure? Really? Why?")
        assert f["question_count"] == 3

    def test_body_char_len(self):
        body = "Hello world"
        f = extract_text_features(body)
        assert f["body_char_len"] == len(body)

    def test_body_word_count(self):
        f = extract_text_features("one two three four five")
        assert f["body_word_count"] == 5

    def test_unique_word_ratio_all_unique(self):
        f = extract_text_features("apple banana cherry")
        assert f["unique_word_ratio"] == 1.0

    def test_unique_word_ratio_all_repeated(self):
        f = extract_text_features("the the the the")
        assert f["unique_word_ratio"] == 0.25

    def test_digit_ratio(self):
        f = extract_text_features("1234")  # 4 digits out of 4 chars
        assert f["digit_ratio"] == 1.0

    def test_digit_ratio_no_digits(self):
        f = extract_text_features("hello world")
        assert f["digit_ratio"] == 0.0

    def test_flesch_simple_text_high_score(self):
        # Very simple text should score high (easier to read)
        f = extract_text_features("The cat sat on the mat. It was a big fat cat.")
        assert f["flesch_reading_ease"] > 60

    def test_flesch_clamped_to_valid_range(self):
        f = extract_text_features("Hi. OK. Go. Do. Be.")
        assert 0.0 <= f["flesch_reading_ease"] <= 100.0

    def test_transformer_output_shape(self):
        t = TextFeatureTransformer()
        out = t.fit_transform(pd.Series(["hello world", "urgent verify now"]))
        assert out.shape == (2, 11)

    def test_transformer_handles_none(self):
        t = TextFeatureTransformer()
        out = t.fit_transform(pd.Series([None]))
        assert out.shape == (1, 11)
        assert not np.any(np.isnan(out))


# ── Header extractor ───────────────────────────────────────────────────────────

class TestHeaderExtractor:

    def test_empty_inputs_return_zeros(self):
        f = extract_header_features("", "")
        assert f["has_reply_to"] == 0
        assert f["reply_to_differs"] == 0
        assert f["sender_is_freemail"] == 0
        assert f["display_name_mismatch"] == 0
        assert f["sender_domain_numeric"] == 0

    def test_freemail_gmail_detected(self):
        f = extract_header_features("john@gmail.com", "")
        assert f["sender_is_freemail"] == 1

    def test_freemail_yahoo_detected(self):
        f = extract_header_features("user@yahoo.com", "")
        assert f["sender_is_freemail"] == 1

    def test_corporate_domain_not_freemail(self):
        f = extract_header_features("support@company.com", "")
        assert f["sender_is_freemail"] == 0

    def test_reply_to_present_flag(self):
        f = extract_header_features("sender@company.com", "reply@other.com")
        assert f["has_reply_to"] == 1

    def test_reply_to_same_domain_no_mismatch(self):
        f = extract_header_features("sender@company.com", "other@company.com")
        assert f["reply_to_differs"] == 0

    def test_reply_to_different_domain_mismatch(self):
        f = extract_header_features("sender@legit.com", "collect@evil.tk")
        assert f["reply_to_differs"] == 1

    def test_no_reply_to_differs_when_field_empty(self):
        f = extract_header_features("sender@legit.com", "")
        assert f["reply_to_differs"] == 0

    def test_display_name_brand_mismatch(self):
        # Display says "PayPal" but sending domain is not paypal.com
        f = extract_header_features('"PayPal Security" <spoof@evil-domain.com>', "")
        assert f["display_name_mismatch"] == 1

    def test_display_name_brand_matches_domain(self):
        # Display says "PayPal" and sending domain contains "paypal"
        f = extract_header_features('"PayPal" <noreply@paypal.com>', "")
        assert f["display_name_mismatch"] == 0

    def test_display_name_no_brand_no_mismatch(self):
        f = extract_header_features('"John Smith" <john@random.com>', "")
        assert f["display_name_mismatch"] == 0

    def test_sender_domain_numeric_leet(self):
        # paypa1.com — digit substituting a letter
        f = extract_header_features("noreply@paypa1.com", "")
        assert f["sender_domain_numeric"] == 1

    def test_sender_domain_numeric_amaz0n(self):
        f = extract_header_features("support@amaz0n-security.com", "")
        assert f["sender_domain_numeric"] == 1

    def test_sender_domain_not_numeric_for_clean_domain(self):
        f = extract_header_features("noreply@amazon.com", "")
        assert f["sender_domain_numeric"] == 0

    def test_angle_bracket_address_parsing(self):
        # Address in "Display Name <addr@domain.com>" format
        f = extract_header_features('"Acme Corp" <billing@acme.com>', "")
        assert f["sender_is_freemail"] == 0
        assert f["sender_domain_numeric"] == 0

    def test_transformer_output_shape(self):
        df = pd.DataFrame([
            {"sender": "user@gmail.com", "reply_to": ""},
            {"sender": '"PayPal" <spoof@evil.com>', "reply_to": "collect@harvest.tk"},
        ])
        t = HeaderFeatureTransformer()
        out = t.fit_transform(df)
        assert out.shape == (2, 5)


# ── Structural extractor ───────────────────────────────────────────────────────

class TestStructuralExtractor:

    def test_empty_inputs_return_zeros(self):
        f = extract_structural_features("", "")
        assert f["has_html"] == 0
        assert f["form_count"] == 0
        assert f["script_count"] == 0
        assert f["img_count"] == 0
        assert f["hidden_element_count"] == 0
        assert f["generic_salutation"] == 0

    def test_has_html_detected(self):
        f = extract_structural_features("<html><body>Hello</body></html>", "")
        assert f["has_html"] == 1

    def test_no_html_body_returns_has_html_zero(self):
        f = extract_structural_features("", "Plain text only")
        assert f["has_html"] == 0

    def test_form_count_detected(self):
        html = '<html><body><form action="/steal"><input name="pass"></form></body></html>'
        f = extract_structural_features(html, "")
        assert f["form_count"] == 1

    def test_multiple_forms_counted(self):
        html = "<form></form><form></form><form></form>"
        f = extract_structural_features(html, "")
        assert f["form_count"] == 3

    def test_script_count_detected(self):
        html = "<html><body><script>alert(1)</script></body></html>"
        f = extract_structural_features(html, "")
        assert f["script_count"] == 1

    def test_hidden_element_display_none(self):
        html = '<div style="display:none">hidden content</div>'
        f = extract_structural_features(html, "")
        assert f["hidden_element_count"] >= 1

    def test_hidden_element_visibility_hidden(self):
        html = '<span style="visibility:hidden">invisible</span>'
        f = extract_structural_features(html, "")
        assert f["hidden_element_count"] >= 1

    def test_img_count(self):
        html = "<img src='a.png'><img src='b.png'><img src='c.png'>"
        f = extract_structural_features(html, "")
        assert f["img_count"] == 3

    def test_img_to_word_ratio_computed(self):
        html = "<img src='a.png'> one two three four"
        f = extract_structural_features(html, "")
        assert f["img_to_word_ratio"] > 0

    def test_generic_salutation_dear_customer_html(self):
        html = "<html><body><p>Dear Customer, your account...</p></body></html>"
        f = extract_structural_features(html, "")
        assert f["generic_salutation"] == 1

    def test_generic_salutation_dear_user_plain(self):
        # Salutation in plain body when no HTML is present
        f = extract_structural_features("", "Dear User, please verify your account.")
        assert f["generic_salutation"] == 1

    def test_generic_salutation_dear_valued_member(self):
        f = extract_structural_features("", "Dear valued member, your invoice is ready.")
        assert f["generic_salutation"] == 1

    def test_no_generic_salutation_personal_name(self):
        f = extract_structural_features("", "Hi Sarah, are you free tomorrow?")
        assert f["generic_salutation"] == 0

    def test_external_link_count(self):
        html = '''
            <a href="https://external.com">link1</a>
            <a href="https://another.org">link2</a>
            <a href="/relative/path">local</a>
        '''
        f = extract_structural_features(html, "")
        assert f["external_link_count"] == 2

    def test_no_external_links_for_relative_hrefs(self):
        html = '<a href="/page">relative</a><a href="#anchor">anchor</a>'
        f = extract_structural_features(html, "")
        assert f["external_link_count"] == 0

    def test_transformer_output_shape(self):
        df = pd.DataFrame([
            {"html_body": "<form></form>", "body": "Dear Customer"},
            {"html_body": "", "body": "Hi John, how are you?"},
        ])
        t = StructuralFeatureTransformer()
        out = t.fit_transform(df)
        assert out.shape == (2, 8)

    def test_transformer_feature_names(self):
        names = StructuralFeatureTransformer().get_feature_names_out()
        assert "has_html" in names
        assert "form_count" in names
        assert len(names) == 8
