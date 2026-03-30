"""
HF Spaces / local entry point for PhishGuard.

HF Spaces runs this file directly. The project root is inserted into sys.path
first so that 'app' resolves to the app/ package, not this file.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.gradio_app import build_ui, DARK_CSS, _LAUNCH_JS, _LAUNCH_THEME

demo = build_ui()

demo.launch(
    server_name="0.0.0.0",
    css=DARK_CSS,
    js=_LAUNCH_JS,
    theme=_LAUNCH_THEME,
)
