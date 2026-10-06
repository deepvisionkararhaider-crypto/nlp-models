"""
Generate per-model README files and Colab notebooks.

Every model in this repository is small, classical NLP (sparse features /
shallow embeddings / tree or linear classifiers). They train on CPU in seconds
to a couple of minutes - a GPU is *not* required. The notebooks therefore state
that plainly instead of promising free GPU time that the task does not need.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import nlp_common as nc  # noqa: E402

NOTEBOOK_DIR = nc.REPO_ROOT / "notebooks"
GITHUB = "https://github.com/deepvisionkararhaider-crypto/nlp-models"


def load_metrics(folder):
    p = nc.artifact_dir(folder) / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def model_readme(folder: str) -> str:
    m = nc.MODEL_REGISTRY[folder]
    met = load_metrics(folder)
    nb = f"{m['num']:02d}_{folder.split('_', 1)[1]}"
    metrics_block = "_(run `python training/train_all.py --only %d` to populate)_" % m["num"]
    if met:
        metrics_block = (
            f"| Metric | Value |\n|---|---|\n"
            f"| Test accuracy | {met['accuracy']*100:.2f}% |\n"
            f"| F1 (macro) | {met['f1_macro']*100:.2f}% |\n"
            f"| Precision (macro) | {met['precision_macro']*100:.2f}% |\n"
            f"| Recall (macro) | {met['recall_macro']*100:.2f}% |\n"
            f"| Test set size | {met['n_test']} |\n"
            f"| Train set size | {met['n_train']} |\n\n"
            "_Metrics are measured on the held-out 20-Newsgroups test split the "
            "model never trained on._"
        )
    honesty = (
        "✅ The folder name matches the technique actually implemented here."
        if m["faithful_family"] else
        "⚠️ **Naming note:** " + m.get("caveat", "")
    )
    return f"""# {m['num']:02d}. {m['display']}

**Task:** {nc.TASK_TOPIC_CLASSIFICATION}
**Framework:** {m['framework']}
**Real implementation:** {m['actual']}

{honesty}

---

## What it does

{m['description']}

## Files

| File | Purpose |
|------|---------|
| `model.py` | Original training script (unchanged from the original project). |
| `app.py` | **Streamlit entry point** - deploy this on Streamlit Community Cloud. |
| `requirements.txt` | Dependencies for this model only. |
| `artifacts/model.joblib` | Trained model bundle used by `app.py`. |
| `artifacts/metrics.json` | Verified held-out test metrics. |
| `artifacts/meta.json` | Reproducibility info (hyper-parameters, versions). |
| `data/newsgroups.csv` | Sample of the training data. |

## Results (held-out test set)

{metrics_block}

## Run locally

```bash
pip install -r requirements.txt
python model.py                        # retrain from scratch (original script)
python training/train_all.py --only {m['num']}   # rebuild artifacts/ from repo root
streamlit run app.py
```

## Deploy (free)

1. Push this repository to GitHub.
2. On **share.streamlit.io** → *New app* → *Deploy from GitHub*.
3. Repository `deepvisionkararhaider-crypto/nlp-models`, branch `main`.
4. **Main file path:** `{folder}/app.py`

## Reproducible training

- **Local / CPU:** `python training/train_all.py --only {m['num']}`
- **Colab / Kaggle:** open [`notebooks/{nb}.ipynb`](../notebooks/{nb}.ipynb).
  → No GPU is required for this model; it trains on CPU.

## Input & output

- **Input:** any English text (a sentence or a whole post).
- **Output:** one of four topic labels - `hockey`, `space`, `graphics`, `politics` -
  with a per-class score table.
  Models marked `decision_function_softmax` show softmaxed decision-function
  scores, which are **not** calibrated probabilities (stated in the app).

## Limitations

- Trained on only 4 of the 20-Newsgroups categories; other topics are forced into
  these four.
- 1990s Usenet text; modern or very short inputs are harder.
- Classification only - no generation, summarisation or translation.

---
[← Back to the portfolio](../README.md) · [Source folder]({GITHUB}/tree/main/{folder})
"""


def _src_lines(src: str) -> list[str]:
    """Jupyter joins `source` entries verbatim, so every line needs its newline."""
    return [line + "\n" for line in src.strip("\n").split("\n")]


def _code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _src_lines(src)}


def _md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src_lines(src)}


def notebook(folder: str) -> dict:
    m = nc.MODEL_REGISTRY[folder]
    nb_name = f"{m['num']:02d}_{folder.split('_', 1)[1]}"
    gensim_note = ("\nThis model additionally trains a **gensim** embedding model.\n"
                   if m["needs_gensim"] else "")
    transformer_note = ("" if m["faithful_family"] else
                        "\n> **Honesty note.** This folder is *named* after a "
                        "transformer, but the repository implements it as a "
                        "TF-IDF + SVD + classical scikit-learn pipeline. This "
                        "notebook trains exactly that - **no transformer is "
                        "downloaded or trained**.\n")
    cells = [
        _md(f"""# {m['num']:02d}. {m['display']} - reproducible training

**Task:** {nc.TASK_TOPIC_CLASSIFICATION}
**Framework:** {m['framework']}
**Real implementation:** {m['actual']}
{transformer_note}
## GPU?
**Not required.** This is a small classical NLP model that trains on CPU in
seconds to ~1 minute. A free Colab CPU runtime is enough; a GPU will not make it
meaningfully faster.
{gensim_note}
Run the cells in order. At the end you will have a fresh
`{folder}/artifacts/model.joblib` plus verified metrics and an unseen-text test.
"""),
        _md("## 1. Get the repository"),
        _code(f"""# Clone the portfolio (or upload your own copy and skip this cell).
import os
REPO_DIR = "nlp-models"
if not os.path.isdir(REPO_DIR):
    !git clone https://github.com/deepvisionkararhaider-crypto/nlp-models.git
%cd {{REPO_DIR}}
!git rev-parse --short HEAD"""),
        _md("## 2. Install dependencies"),
        _code(f"""!pip install -q -r {folder}/requirements.txt
import sklearn, numpy, pandas
print("scikit-learn", sklearn.__version__, "| numpy", numpy.__version__)"""),
        _md(f"""## 3. Data

The four-category dataset (`data/20newsgroups_4cat/`) is committed to the
repository. If it is missing, rebuild it from the original 20-Newsgroups archive:

```bash
python training/build_dataset.py --archive /path/to/20news-bydate.tar.gz
```
"""),
        _code("""import os
print("dataset present:", os.path.exists("data/20newsgroups_4cat/train.csv"))
import pandas as pd
if os.path.exists("data/20newsgroups_4cat/train.csv"):
    df = pd.read_csv("data/20newsgroups_4cat/train.csv")
    print(df.shape, "| classes:", df["category"].unique().tolist())"""),
        _md("## 4. Train (and save artifacts)"),
        _code(f"""# Trains this model's original architecture and writes artifacts/ + metrics.json
!python training/train_all.py --only {m['num']}"""),
        _md("## 5. Evaluate on the held-out test set"),
        _code(f"""import json
met = json.load(open("{folder}/artifacts/metrics.json"))
print("Test accuracy :", round(met["accuracy"], 4))
print("F1 (macro)    :", round(met["f1_macro"], 4))
print("N test        :", met["n_test"])
print()
for cls, v in met["per_class"].items():
    print(f"  {{cls:<10}} precision={{v['precision']:.3f}} recall={{v['recall']:.3f}} f1={{v['f1']:.3f}}")
print()
print("Confusion matrix (rows=true, cols=pred):")
for row in met["confusion_matrix"]:
    print(" ", row)"""),
        _md("## 6. Inference test on **new, unseen** text"),
        _code(f"""import sys; sys.path.insert(0, ".")
from common import nlp_common as nc

UNSEEN = [
    ("hockey",   "The goalie made an incredible save in the third period and the team reached the playoffs."),
    ("space",    "NASA launched a new rocket to the space station to study the atmosphere of Mars."),
    ("graphics", "The 3D rendering engine uses ray tracing for realistic shadows and textures."),
    ("politics", "The senator debated the tax policy and voting reform bill before the election."),
]
correct = 0
for expected, text in UNSEEN:
    r = nc.predict_text("{folder}", text)
    ok = r["predicted_short"] == expected
    correct += ok
    print(f"{{'OK ' if ok else '== '}} expected={{expected:<9}} got={{r['predicted_short']:<9}} "
          f"score={{r['scores'][0]['score']:.3f}} ({{r['score_type']}})")
print(f"\\nunseen accuracy: {{correct}}/{{len(UNSEEN)}}")"""),
        _md("## 7. Export the artifacts"),
        _code(f"""import shutil
shutil.make_archive("artifacts_{nb_name}", "zip", f"{folder}/artifacts")
print("wrote artifacts_{nb_name}.zip - download it from the Files panel.")

# Optional: commit straight back to your fork
# !git add {folder}/artifacts && git commit -m "Retrain {m['display']}" && git push"""),
        _md(f"""## 8. Where the artifacts go

Place the produced `model.joblib` (and `metrics.json`, `meta.json`) in
**`{folder}/artifacts/`** in the GitHub repository, then deploy
`{folder}/app.py` on Streamlit Community Cloud.

> Some models (Word2Vec / FastText) also write a large gensim model file
> (`w2v.model` / `ft.model`) inside `artifacts/`. Keep it next to `model.joblib`
> - the app loads both.
"""),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": [], "toc_visible": True},
        },
        "nbformat": 4, "nbformat_minor": 0,
    }


def main() -> None:
    NOTEBOOK_DIR.mkdir(exist_ok=True)
    for folder, m in nc.MODEL_REGISTRY.items():
        (nc.REPO_ROOT / folder / "README.md").write_text(model_readme(folder))
        nb_name = f"{m['num']:02d}_{folder.split('_', 1)[1]}"
        (NOTEBOOK_DIR / f"{nb_name}.ipynb").write_text(
            json.dumps(notebook(folder), indent=1))

    (NOTEBOOK_DIR / "README.md").write_text(
        "# Training notebooks\n\n"
        "Reproducible Colab / Kaggle notebooks - one per model. Each clones the\n"
        "repository, trains the model's original architecture, verifies metrics on\n"
        "the held-out test split, tests inference on unseen text and exports the\n"
        "artifact.\n\n"
        "> **No GPU needed.** All 15 models are small classical NLP pipelines that\n"
        "> train on CPU in seconds to ~1 minute.\n\n"
        "| # | Model | Notebook |\n|---|-------|----------|\n" +
        "\n".join(f"| {m['num']:02d} | {m['display']} | "
                  f"[`{m['num']:02d}_{f.split('_',1)[1]}.ipynb`]"
                  f"({m['num']:02d}_{f.split('_',1)[1]}.ipynb) |"
                  for f, m in nc.MODEL_REGISTRY.items()) + "\n"
    )
    print(f"wrote {len(nc.MODEL_REGISTRY)} READMEs and "
          f"{len(nc.MODEL_REGISTRY)} notebooks + notebooks/README.md")


if __name__ == "__main__":
    main()
