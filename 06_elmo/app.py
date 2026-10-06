"""
06. ELMo (3-layer contextual) - Streamlit demo
Task: Topic Classification (20-Newsgroups, 4 classes)
Real architecture: 3 stacked TF-IDF->SVD(50) representations (char 3-5 / word 1 / word 2-3), fixed weights .3/.4/.3 + Logistic Regression
Framework: scikit-learn

This file is a genuine Streamlit entry point. On Streamlit Community Cloud set
"Main file path" to:  06_elmo/app.py
"""
import sys
from pathlib import Path

# Allow `import common...` when this file is the Streamlit entry point.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.app_base import run_app

FOLDER = "06_elmo"

run_app(FOLDER)
