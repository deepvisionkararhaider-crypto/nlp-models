"""
Generate the per-model Streamlit entry points and their requirements files.

For each of the 15 models this writes ``<folder>/app.py`` - a real Streamlit
entry point (the exact path you configure on Streamlit Community Cloud) that
delegates to the shared app body, plus ``<folder>/requirements.txt`` containing
only the packages that model actually needs.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import nlp_common as nc  # noqa: E402

APP_TEMPLATE = '''"""
{dnum}. {display} - Streamlit demo
Task: {task}
Real architecture: {actual}
Framework: {framework}

This file is a genuine Streamlit entry point. On Streamlit Community Cloud set
"Main file path" to:  {folder}/app.py
"""
import sys
from pathlib import Path

# Allow `import common...` when this file is the Streamlit entry point.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.app_base import run_app

FOLDER = "{folder}"

run_app(FOLDER)
'''

# Base pins for a Streamlit Community Cloud runtime (Python 3.11/3.12 compatible).
BASE_REQ = [
    "streamlit>=1.36,<2",
    "scikit-learn>=1.3,<1.7",
    "numpy>=1.26,<3",
    "scipy>=1.10,<2",
    "pandas>=2.0,<3",
    "joblib>=1.3,<2",
]


def requirements_for(meta: dict) -> str:
    lines = list(BASE_REQ)
    if meta["needs_gensim"]:
        lines.append("gensim>=4.3,<5")
    return "\n".join(lines) + "\n"


def main() -> None:
    written = []
    for folder, meta in nc.MODEL_REGISTRY.items():
        app = APP_TEMPLATE.format(
            dnum=f"{meta['num']:02d}",
            display=meta["display"],
            task=nc.TASK_TOPIC_CLASSIFICATION,
            actual=meta["actual"],
            framework=meta["framework"],
            folder=folder,
        )
        (nc.REPO_ROOT / folder / "app.py").write_text(app)
        (nc.REPO_ROOT / folder / "requirements.txt").write_text(requirements_for(meta))
        written.append(folder)

    # root requirements: everything needed to run *all* apps + training locally
    root = BASE_REQ + [
        "gensim>=4.3,<5",
        "matplotlib>=3.7,<4",
        "seaborn>=0.12,<1",
        "scikit-learn>=1.3,<1.7",
    ]
    (nc.REPO_ROOT / "requirements.txt").write_text("\n".join(dict.fromkeys(root)) + "\n")

    # .streamlit config shared by every app (clean, wide, minimal chrome)
    st_dir = nc.REPO_ROOT / ".streamlit"
    st_dir.mkdir(exist_ok=True)
    (st_dir / "config.toml").write_text(
        "[theme]\n"
        'base = "light"\n'
        'primaryColor = "#2563eb"\n'
        'backgroundColor = "#ffffff"\n'
        'secondaryBackgroundColor = "#f8fafc"\n'
        'textColor = "#0f172a"\n'
        'font = "sans serif"\n\n'
        "[server]\n"
        "headless = true\n"
        "enableCORS = false\n"
        "enableXsrfProtection = true\n\n"
        "[browser]\n"
        "gatherUsageStats = false\n"
    )
    print(f"wrote app.py + requirements.txt for {len(written)} models")
    print("wrote root requirements.txt and .streamlit/config.toml")


if __name__ == "__main__":
    main()
