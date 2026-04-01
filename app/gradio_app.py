"""
PhishGuard Gradio UI.

Dark-themed single-page app:
  • Top    : branded header with shield icon + tagline
  • Middle : tab-based input — paste raw text | upload .eml / .txt
  • Bottom : result card that fades in and auto-scrolls into view on submit
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import gradio as gr
from app.inference import parse_raw_input, run_prediction
from app.explainer import generate_explanation, get_signal_chips


# ── CSS ─────────────────────────────────────────────────────────────────────

DARK_CSS = """
/* ── Palette variables ─────────────────────────────────────────────────── */
:root {
    --bg:        #0a0c14;
    --surface:   #12162a;
    --card:      #181c2e;
    --border:    #252d4a;
    --text:      #e2e8f4;
    --muted:     #7b88b5;
    --accent:    #6366f1;
    --red:       #ef4444;
    --red-dim:   #7f1d1d;
    --green:     #22c55e;
    --green-dim: #14532d;
    --warn:      #f59e0b;
    /* Strip Gradio theme block borders globally */
    --block-border-width: 0px;
    --block-border-color: transparent;
    --block-shadow: none;
}

/* ── Global ─────────────────────────────────────────────────────────────── */
body, .gradio-container, .main, .wrap,
.block, .form, .contain, .gap, .panel,
[data-testid="block"], [data-testid="textbox"],
.svelte-1gfkn6j, .input-text, .input-file {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}
/* Remove all block-level borders injected by the Gradio theme */
.block, [data-testid="block"], .gradio-group, .form > .block {
    border: none !important;
    box-shadow: none !important;
}
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 2.5rem !important;
    box-sizing: border-box !important;
}
footer, .footer { display: none !important; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Content width cap ─────────────────────────────────────────────────── */
.gradio-container > .main > .wrap,
.gradio-container > .main > .wrap > .contain,
.gradio-container > .app {
    max-width: 1100px !important;
    margin: 0 auto !important;
    width: 100% !important;
}

/* ── Header ─────────────────────────────────────────────────────────────── */
#pg-header {
    text-align: center;
    padding: 2.8rem 0 2rem;
    user-select: none;
}
#pg-header .shield {
    font-size: 3rem;
    display: block;
    margin-bottom: 0.5rem;
    filter: drop-shadow(0 0 18px rgba(99,102,241,0.6));
}
#pg-header h1 {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #a5b4fc 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1;
}
#pg-header p {
    color: var(--muted);
    margin: 0.5rem 0 0;
    font-size: 0.95rem;
    letter-spacing: 0.03em;
}

/* ── Input panel ─────────────────────────────────────────────────────────── */
#input-panel {
    background: var(--surface) !important;
    color: var(--text) !important;
}
#input-panel {
    border: none !important;
    border-radius: 16px !important;
    padding: 1.6rem !important;
    box-shadow: none !important;
}
#input-panel > .form { gap: 0.8rem !important; }


/* ── Textarea ───────────────────────────────────────────────────────────── */
#email-textarea,
#email-textarea label,
#email-textarea .wrap,
#email-textarea .wrap-inner,
#email-textarea .block {
    background: var(--bg) !important;
    color: var(--text) !important;
}
#email-textarea label > span:first-child {
    color: #818cf8 !important;
    font-size: 0.8rem !important;
}
#email-textarea textarea {
    background: var(--bg) !important;
    border: 1.5px solid #4f46e5 !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.55 !important;
    padding: 0.85rem !important;
    min-height: 220px !important;
    resize: vertical !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    caret-color: #a5b4fc !important;
    box-shadow: 0 0 0 1px rgba(99,102,241,0.15) !important;
}
#email-textarea textarea:focus {
    border-color: #a5b4fc !important;
    box-shadow: 0 0 0 3px rgba(165,180,252,0.15) !important;
    outline: none !important;
}
#email-textarea textarea::placeholder { color: #2e3561 !important; }


/* ── Analyze button ─────────────────────────────────────────────────────── */
#analyze-btn button {
    width: 100% !important;
    padding: 0.9rem 2rem !important;
    background: linear-gradient(135deg, #3730a3 0%, #4f46e5 50%, #7c3aed 100%) !important;
    color: #a5b4fc !important;
    font-size: 1rem !important;
    font-weight: 650 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    border-radius: 10px !important;
    border: 1px solid rgba(165,180,252,0.2) !important;
    cursor: pointer !important;
    transition: transform 0.15s, box-shadow 0.15s, color 0.15s !important;
    margin-top: 0.5rem !important;
    -webkit-font-smoothing: antialiased !important;
}
#analyze-btn button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 28px rgba(99,102,241,0.45) !important;
    color: #c084fc !important;
}
#analyze-btn button:active { transform: translateY(0) !important; }
#analyze-btn button.generating {
    background: linear-gradient(135deg, #1e1b4b, #3730a3) !important;
    color: #818cf8 !important;
}

/* ── Clear button ───────────────────────────────────────────────────────── */
#clear-btn button {
    background: transparent !important;
    color: #6366f1 !important;
    border: 1px solid #2e3561 !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    padding: 0.45rem 1rem !important;
    width: 100% !important;
    transition: color 0.15s, border-color 0.15s !important;
    -webkit-font-smoothing: antialiased !important;
}
#clear-btn button:hover {
    color: #a5b4fc !important;
    border-color: #6366f1 !important;
}

/* ── Result card ────────────────────────────────────────────────────────── */
#result-card { margin-top: 1.8rem !important; }
/* Collapse the card container when content is empty */
#result-card > div:empty,
#result-card:not(:has(.result-wrapper)) {
    margin: 0 !important;
    padding: 0 !important;
    min-height: 0 !important;
}

/* ── Status message ─────────────────────────────────────────────────────── */
#status-msg {
    text-align: center;
    color: var(--muted);
    font-size: 0.82rem;
    min-height: 1.2rem;
    margin-top: 0.6rem;
}

/* ── Example buttons ────────────────────────────────────────────────────── */
#ex-phish-btn,
#ex-ham-btn {
    background: rgb(129, 140, 248) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}
#ex-phish-btn button,
#ex-ham-btn button,
#ex-phish-btn > div > button,
#ex-ham-btn > div > button {
    background: rgb(129, 140, 248) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.15s, transform 0.15s, box-shadow 0.15s !important;
    -webkit-font-smoothing: antialiased !important;
    width: 100% !important;
}
#ex-phish-btn button *,
#ex-ham-btn button *,
#ex-phish-btn > div > button *,
#ex-ham-btn > div > button * {
    color: #ffffff !important;
    background: transparent !important;
}
#ex-phish-btn button:hover,
#ex-ham-btn button:hover,
#ex-phish-btn > div > button:hover,
#ex-ham-btn > div > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(129,140,248,0.5) !important;
}
#ex-phish-btn button:active,
#ex-ham-btn button:active {
    transform: translateY(0) !important;
    opacity: 1 !important;
}
"""

# ── Result card HTML builder ─────────────────────────────────────────────────

_CHIP_COLORS = {
    "danger":  ("rgba(239,68,68,0.15)",  "#ef4444", "#7f1d1d"),
    "warn":    ("rgba(245,158,11,0.12)", "#f59e0b", "#78350f"),
    "ok":      ("rgba(34,197,94,0.12)",  "#22c55e", "#14532d"),
    "neutral": ("rgba(123,136,181,0.1)", "#7b88b5", "#252d4a"),
}


def _chip_html(chip: dict) -> str:
    bg, fg, border = _CHIP_COLORS.get(chip["level"], _CHIP_COLORS["neutral"])
    return (
        f'<span style="display:inline-flex;align-items:center;gap:0.35rem;'
        f'background:{bg};color:{fg};border:1px solid {border};'
        f'border-radius:20px;padding:0.3rem 0.75rem;font-size:0.78rem;font-weight:500;'
        f'white-space:nowrap;">'
        f'<span style="color:{fg};opacity:0.7;font-size:0.7rem;">{chip["label"]}</span>'
        f'<span style="font-weight:700;">{chip["value"]}</span>'
        f'</span>'
    )


def _flag_html(flag: str) -> str:
    return (
        f'<li style="display:flex;align-items:flex-start;gap:0.6rem;'
        f'padding:0.55rem 0;border-bottom:1px solid rgba(37,45,74,0.5);">'
        f'<span style="color:#f59e0b;flex-shrink:0;margin-top:0.05rem;">⚑</span>'
        f'<span style="color:#d1d8f0;font-size:0.88rem;line-height:1.45;">{flag}</span>'
        f'</li>'
    )


def build_result_html(result: dict, flags: list[str], chips: list[dict]) -> str:
    is_phishing  = result["label"] == 1
    conf_pct     = result["confidence"] * 100
    threshold_pct = result["threshold"] * 100

    if is_phishing:
        header_bg   = "linear-gradient(135deg, #1a0505 0%, #2d0808 100%)"
        header_border = "#7f1d1d"
        icon        = "⚠"
        icon_color  = "#ef4444"
        icon_glow   = "rgba(239,68,68,0.4)"
        verdict     = "PHISHING DETECTED"
        sub         = "This email shows characteristics consistent with a phishing attempt"
        bar_color   = "#ef4444"
        bar_glow    = "rgba(239,68,68,0.35)"
        badge_bg    = "rgba(239,68,68,0.15)"
        badge_fg    = "#ef4444"
    else:
        header_bg   = "linear-gradient(135deg, #030f07 0%, #071a0e 100%)"
        header_border = "#14532d"
        icon        = "✓"
        icon_color  = "#22c55e"
        icon_glow   = "rgba(34,197,94,0.4)"
        verdict     = "LIKELY SAFE"
        sub         = "No significant phishing indicators were detected in this email"
        bar_color   = "#22c55e"
        bar_glow    = "rgba(34,197,94,0.3)"
        badge_bg    = "rgba(34,197,94,0.12)"
        badge_fg    = "#22c55e"

    flag_items  = "".join(_flag_html(f) for f in flags)
    chip_items  = "".join(_chip_html(c) for c in chips)

    # Inline scroll trigger via broken-image onerror — executes even via innerHTML
    scroll_js = (
        "this.closest('#result-card').scrollIntoView"
        "({behavior:'smooth',block:'center'});"
        "this.remove();"
    )

    return f"""
<div class="result-wrapper" style="
    background: #12162a;
    border: 1px solid {header_border};
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 6px 40px rgba(0,0,0,0.5);
    animation: pgFadeIn 0.3s ease forwards;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
">
<style>
@keyframes pgFadeIn {{
    from {{ opacity: 0; }}
    to   {{ opacity: 1; }}
}}
</style>

<!-- Scroll trigger (fires when HTML is injected) -->
<img src="x" onerror="{scroll_js}" style="display:none" aria-hidden="true">

<!-- ── Verdict header ── -->
<div style="
    background: {header_bg};
    border-bottom: 1px solid {header_border};
    padding: 1.4rem 1.6rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
">
    <div style="
        font-size: 2rem;
        color: {icon_color};
        filter: drop-shadow(0 0 10px {icon_glow});
        flex-shrink: 0;
        line-height: 1;
    ">{icon}</div>
    <div style="flex: 1; min-width: 0;">
        <div style="
            font-size: 1.3rem;
            font-weight: 800;
            color: {icon_color};
            letter-spacing: 0.06em;
            line-height: 1;
        ">{verdict}</div>
        <div style="
            font-size: 0.82rem;
            color: #7b88b5;
            margin-top: 0.3rem;
            line-height: 1.3;
        ">{sub}</div>
    </div>
    <div style="
        background: {badge_bg};
        color: {badge_fg};
        border: 1px solid {header_border};
        border-radius: 50px;
        padding: 0.4rem 1rem;
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        flex-shrink: 0;
        white-space: nowrap;
    ">{conf_pct:.0f}<span style="font-size:0.75rem;font-weight:600;opacity:0.7;">%</span></div>
</div>

<!-- ── Confidence bar ── -->
<div style="padding: 1.2rem 1.6rem 0.4rem;">
    <div style="
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.45rem;
    ">
        <span style="font-size:0.78rem;color:#7b88b5;font-weight:500;letter-spacing:0.04em;">
            PHISHING PROBABILITY
        </span>
        <span style="font-size:0.78rem;color:#7b88b5;">
            threshold {threshold_pct:.0f}%
        </span>
    </div>
    <div style="
        background: rgba(37,45,74,0.5);
        border-radius: 999px;
        height: 8px;
        overflow: hidden;
        position: relative;
    ">
        <div style="
            width: {conf_pct:.1f}%;
            height: 100%;
            background: {bar_color};
            border-radius: 999px;
            box-shadow: 0 0 10px {bar_glow};
            transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
        "></div>
    </div>
    <!-- Threshold marker -->
    <div style="
        position: relative;
        height: 14px;
        margin-top: -11px;
    ">
        <div style="
            position: absolute;
            left: {threshold_pct:.1f}%;
            transform: translateX(-50%);
            width: 2px;
            height: 14px;
            background: #363d5a;
            border-radius: 1px;
        "></div>
    </div>
</div>

<!-- ── Red flags ── -->
<div style="padding: 0.8rem 1.6rem 1rem;">
    <div style="
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        color: #7b88b5;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
    ">Analysis</div>
    <ul style="list-style:none;margin:0;padding:0;">
        {flag_items}
    </ul>
</div>

<!-- ── Signal chips ── -->
<div style="
    padding: 0.8rem 1.6rem 1.4rem;
    border-top: 1px solid rgba(37,45,74,0.5);
">
    <div style="
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        color: #7b88b5;
        margin-bottom: 0.6rem;
        text-transform: uppercase;
    ">Signal Breakdown</div>
    <div style="display:flex;flex-wrap:wrap;gap:0.45rem;">
        {chip_items}
    </div>
</div>

</div>
"""


# ── Example emails ───────────────────────────────────────────────────────────

_EXAMPLE_PHISHING = """\
From: security-alert@paypa1-secure.xyz
To: customer@example.com
Subject: URGENT: Your account has been suspended!
Reply-To: collect@evil-harvest.tk

Dear Customer,

URGENT ACTION REQUIRED!! Your PayPal account has been TEMPORARILY SUSPENDED due to suspicious activity.

You must verify your identity IMMEDIATELY or your account will be permanently closed within 24 hours.

Click here to verify now: http://bit.ly/verify-paypa1-now

Failure to verify will result in permanent account termination and loss of all funds.

Verify now: http://185.220.101.47/paypal/verify?token=a7f3k9xQzP2mR8nB

PayPal Security Team
"""

_EXAMPLE_HAM = """\
From: newsletter@company.com
To: markus@example.com
Subject: Your weekly project update

Hi Markus,

Here's your weekly update for the Q1 planning project.

Completed this week:
- Finished the onboarding documentation
- Scheduled the kick-off meeting for Thursday

Next steps:
- Review the revised timeline (attached)
- Confirm budget approval with finance

The meeting invite has been sent to the team. Let me know if you have any questions.

Best,
Sarah
"""


# ── Core analysis function ───────────────────────────────────────────────────

def run_analysis(text_input: str | None) -> tuple:
    """
    Called by the Gradio event. Returns (result_html, status_msg).
    result_html : gr.update for the #result-card HTML component
    status_msg  : gr.update for the small status text below the button
    """
    if text_input and text_input.strip():
        email_dict = parse_raw_input(text_input.strip())
    else:
        return (
            gr.update(value=""),
            gr.update(value="Error: Please paste an email first."),
        )

    # Run model
    try:
        result = run_prediction(email_dict)
    except FileNotFoundError as exc:
        return (
            gr.update(value=""),
            gr.update(value=f"⚠ {exc}"),
        )
    except Exception as exc:
        return (
            gr.update(value=""),
            gr.update(value=f"⚠ Analysis failed: {exc}"),
        )

    # Build explanation
    flags = generate_explanation(result["raw_features"], result)
    chips = get_signal_chips(result["raw_features"])
    html  = build_result_html(result, flags, chips)

    label_text = "phishing" if result["label"] == 1 else "safe"
    status = (
        f"Analysis complete — classified as {label_text} "
        f"({result['confidence']*100:.1f}% confidence)"
    )

    return (
        gr.update(value=html),
        gr.update(value=status),
    )


def clear_all() -> tuple:
    return (
        gr.update(value=""),  # text input
        gr.update(value=""),  # result card — empty collapses it via CSS
        gr.update(value=""),  # status msg
    )


# ── UI builder ───────────────────────────────────────────────────────────────

_LAUNCH_JS = """
function() {
    document.documentElement.style.background = '#0a0c14';
    document.body.style.background = '#0a0c14';
}
"""

_LAUNCH_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
)


def build_ui() -> gr.Blocks:

    with gr.Blocks(title="PhishGuard") as demo:

        # ── Header ──────────────────────────────────────────────────────────
        gr.HTML("""
        <div id="pg-header">
            <span class="shield">🛡</span>
            <h1>PhishGuard</h1>
            <p>Paste an email below and the AI will flag phishing attempts in seconds</p>
        </div>
        """)

        # ── Input panel ──────────────────────────────────────────────────────
        with gr.Group(elem_id="input-panel"):
            text_input = gr.Textbox(
                placeholder=(
                    "Paste the full email here — headers and body welcome.\n\n"
                    "Example: From: noreply@bank-secure.xyz\n"
                    "Subject: Urgent: Verify your account\n\n"
                    "Dear Customer, your account has been suspended…"
                ),
                lines=12,
                show_label=False,
                elem_id="email-textarea",
            )

            with gr.Row():
                with gr.Column(scale=4):
                    analyze_btn = gr.Button(
                        "⚡  Analyze Email",
                        variant="primary",
                        elem_id="analyze-btn",
                    )
                with gr.Column(scale=1, min_width=90):
                    clear_btn = gr.Button(
                        "Clear",
                        elem_id="clear-btn",
                    )

        # ── Status message ────────────────────────────────────────────────────
        status_msg = gr.HTML(
            value="",
            elem_id="status-msg",
        )

        # ── Result card — always mounted; empty string collapses it via CSS ──
        result_card = gr.HTML(
            value="",
            visible=True,
            elem_id="result-card",
        )

        # ── Example buttons ───────────────────────────────────────────────────
        gr.HTML("""
        <div style="
            margin-top: 2rem;
            padding-top: 1.2rem;
            border-top: 1px solid #1e2440;
            text-align: center;
        ">
            <span style="font-size:0.78rem;color:#3d4870;letter-spacing:0.08em;
                         text-transform:uppercase;font-weight:600;">
                Try an example
            </span>
        </div>
        """)

        with gr.Row():
            ex_phish_btn = gr.Button(
                "Load phishing example",
                size="sm",
                variant="secondary",
                elem_id="ex-phish-btn",
            )
            ex_ham_btn = gr.Button(
                "Load safe example",
                size="sm",
                variant="secondary",
                elem_id="ex-ham-btn",
            )

        # ── Event wiring ──────────────────────────────────────────────────────

        analyze_btn.click(
            fn=run_analysis,
            inputs=[text_input],
            outputs=[result_card, status_msg],
            show_progress="hidden",
        )

        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[text_input, result_card, status_msg],
        )

        ex_phish_btn.click(
            fn=lambda: gr.update(value=_EXAMPLE_PHISHING),
            inputs=[],
            outputs=[text_input],
        )

        ex_ham_btn.click(
            fn=lambda: gr.update(value=_EXAMPLE_HAM),
            inputs=[],
            outputs=[text_input],
        )

    return demo


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
        css=DARK_CSS,
        js=_LAUNCH_JS,
        theme=_LAUNCH_THEME,
    )
