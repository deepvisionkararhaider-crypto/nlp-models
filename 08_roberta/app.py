"""
08. RoBERTa - Streamlit demo
Task: Topic Classification (20-Newsgroups, 4 classes)
Real architecture: TF-IDF (10k, 1-2 grams) -> TruncatedSVD(100) -> SGDClassifier (modified_huber)
Framework: scikit-learn

This file is a genuine Streamlit entry point. On Streamlit Community Cloud set
"Main file path" to:  08_roberta/app.py
"""
import sys
from pathlib import Path

# Allow `import common...` when this file is the Streamlit entry point.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.app_base import run_app

FOLDER = "08_roberta"

run_app(FOLDER)
