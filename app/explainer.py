"""
Red-flag explanation generator.
Translates raw hand-crafted feature values into human-readable bullets and
signal chips for the UI's result card.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_explanation(raw_features: dict, result: dict) -> list[str]:
    """
    Return an ordered list of human-readable red-flag strings based on which
    feature signals fired. The list is sorted: highest-severity signals first.
    """
    f     = raw_features
    flags: list[tuple[int, str]] = []   # (priority, text)

    # ── URL signals ────────────────────────────────────────────────────────
    n_urls = int(f.get("url_count", 0))
    if n_urls > 8:
        flags.append((3, f"Contains {n_urls} unique URLs — far more than a typical legitimate email"))
    elif n_urls > 2:
        flags.append((1, f"Contains {n_urls} URLs, which is above average for a normal email"))

    if f.get("has_shortener", 0):
        flags.append((3, "Uses a URL shortener (e.g. bit.ly) to conceal the true destination"))

    n_mismatch = int(f.get("display_mismatch_count", 0))
    if n_mismatch:
        flags.append((3, (
            f"{n_mismatch} link(s) display one domain in the visible text "
            "but redirect to a completely different destination"
        )))

    n_bad_tld = int(f.get("suspicious_tld_count", 0))
    if n_bad_tld:
        flags.append((2, "Link(s) use TLDs frequently associated with phishing (.xyz, .tk, .top, .cc, etc.)"))

    n_ip = int(f.get("ip_url_count", 0))
    if n_ip:
        flags.append((3, (
            f"{n_ip} link(s) use a raw IP address instead of a domain name "
            "— a common obfuscation tactic"
        )))

    entropy = f.get("max_url_entropy", 0.0)
    if entropy > 4.5:
        flags.append((2, (
            "URL path contains highly randomised characters, suggesting "
            "an obfuscated or auto-generated link"
        )))

    # ── Header signals ─────────────────────────────────────────────────────
    if f.get("reply_to_differs", 0):
        flags.append((3, (
            "Reply-To address belongs to a different domain than the sender "
            "— replies would be silently redirected to a third party"
        )))

    if f.get("display_name_mismatch", 0):
        flags.append((3, (
            "Sender display name impersonates a known brand, "
            "but the actual sending domain does not match"
        )))

    if f.get("sender_is_freemail", 0) and result.get("label") == 1:
        flags.append((1, (
            "Sent from a free consumer email service (Gmail, Yahoo, Hotmail, etc.) "
            "— unusual for official business communications"
        )))

    if f.get("sender_domain_numeric", 0):
        flags.append((2, (
            "Sender domain contains digits substituting letters "
            "(e.g. paypa1.com) — a common brand-impersonation trick"
        )))

    # ── Text signals ───────────────────────────────────────────────────────
    urgency = int(f.get("urgency_score", 0))
    if urgency >= 5:
        flags.append((3, f"Very high urgency language ({urgency} pressure keywords such as 'verify', 'suspended', 'act now')"))
    elif urgency >= 2:
        flags.append((2, f"Elevated urgency language ({urgency} pressure keywords detected)"))

    caps = f.get("caps_ratio", 0.0)
    if caps > 0.12:
        flags.append((2, f"Unusually high proportion of uppercase text ({caps*100:.0f}%) — a common pressure tactic"))

    excl = int(f.get("exclamation_count", 0))
    if excl > 4:
        flags.append((1, f"Excessive use of exclamation marks ({excl})"))

    # ── Structural signals ─────────────────────────────────────────────────
    if f.get("generic_salutation", 0):
        flags.append((2, (
            "Uses a generic greeting ('Dear Customer', 'Dear User') "
            "rather than your actual name — a mass-phishing indicator"
        )))

    if f.get("has_html", 0):
        n_form   = int(f.get("form_count", 0))
        n_hidden = int(f.get("hidden_element_count", 0))
        n_script = int(f.get("script_count", 0))
        if n_form:
            flags.append((3, f"HTML body contains {n_form} embedded form(s) — a potential credential-harvesting structure"))
        if n_hidden:
            flags.append((2, f"HTML body contains {n_hidden} hidden element(s) used to conceal content"))
        if n_script:
            flags.append((1, f"HTML body contains {n_script} embedded script(s)"))

    # Sort by priority descending, then return text only
    flags.sort(key=lambda x: -x[0])
    result_flags = [text for _, text in flags]

    if not result_flags:
        if result["label"] == 1:
            result_flags.append(
                "No specific structural red flags were detected — "
                "the verdict is based on vocabulary and language patterns in the email body"
            )
        else:
            result_flags.append("No phishing indicators were detected in this email")
    elif result["label"] == 1 and result["confidence"] >= 0.85 and len(result_flags) <= 2:
        result_flags.append(
            "The model also detected suspicious vocabulary and phrasing patterns "
            "commonly found in phishing emails — these contributed significantly to the verdict"
        )

    return result_flags


def get_signal_chips(raw_features: dict) -> list[dict]:
    """
    Return a list of {label, value, level} dicts for the signal chip row.
    level: "danger" | "warn" | "ok" | "neutral"
    """
    f = raw_features

    def urgency_label(score: float) -> str:
        s = int(score)
        if s == 0:   return "None"
        if s <= 2:   return f"Low ({s})"
        if s <= 5:   return f"Med ({s})"
        return f"High ({s})"

    return [
        {
            "label": "URLs",
            "value": str(int(f.get("url_count", 0))),
            "level": "danger" if f.get("url_count", 0) > 5
                     else "warn" if f.get("url_count", 0) > 2
                     else "neutral",
        },
        {
            "label": "Urgency",
            "value": urgency_label(f.get("urgency_score", 0)),
            "level": "danger" if f.get("urgency_score", 0) >= 5
                     else "warn" if f.get("urgency_score", 0) >= 2
                     else "neutral",
        },
        {
            "label": "HTML",
            "value": "Yes" if f.get("has_html", 0) else "No",
            "level": "warn" if f.get("has_html", 0) else "neutral",
        },
        {
            "label": "Shortener",
            "value": "Detected" if f.get("has_shortener", 0) else "None",
            "level": "danger" if f.get("has_shortener", 0) else "neutral",
        },
        {
            "label": "Freemail",
            "value": "Yes" if f.get("sender_is_freemail", 0) else "No",
            "level": "warn" if f.get("sender_is_freemail", 0) else "neutral",
        },
        {
            "label": "Reply-To",
            "value": "Mismatch" if f.get("reply_to_differs", 0) else "Normal",
            "level": "danger" if f.get("reply_to_differs", 0) else "neutral",
        },
        {
            "label": "IP URLs",
            "value": str(int(f.get("ip_url_count", 0))),
            "level": "danger" if f.get("ip_url_count", 0) > 0 else "neutral",
        },
        {
            "label": "Generic Greeting",
            "value": "Yes" if f.get("generic_salutation", 0) else "No",
            "level": "warn" if f.get("generic_salutation", 0) else "neutral",
        },
    ]
