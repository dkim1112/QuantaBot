import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from quantabot.ui.streamlit_app import streamlit_ui

if __name__ == "__main__":
    streamlit_ui()
