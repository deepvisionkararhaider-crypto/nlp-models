"""
04. GloVe - Streamlit demo
Task: Topic Classification (20-Newsgroups, 4 classes)
Real architecture: PPMI co-occurrence matrix (3k vocab, window 4) -> TruncatedSVD(100) mean-pooled + Logistic Regression
Framework: numpy + scikit-learn

This file is a genuine Streamlit entry point. On Streamlit Community Cloud set
"Main file path" to:  04_glove/app.py
"""
import sys
from pathlib import Path

# Allow `import common...` when this file is the Streamlit entry point.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.app_base import run_app

FOLDER = "04_glove"

run_app(FOLDER)
