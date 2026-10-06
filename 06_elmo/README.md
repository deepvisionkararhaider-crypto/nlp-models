# 06. ELMo (3-layer contextual)

**Task:** Topic Classification (20-Newsgroups, 4 classes)
**Framework:** scikit-learn
**Real implementation:** 3 stacked TF-IDF->SVD(50) representations (char 3-5 / word 1 / word 2-3), fixed weights .3/.4/.3 + Logistic Regression

⚠️ **Naming note:** This is NOT a real ELMo biLM (the 8 GB AllenNLP model is not used). It is a 3-layer feature stack inspired by ELMo's multi-level idea.

---

## What it does

ELMo layers several representation levels and weights them. This project mimics that idea with three TF-IDF+SVD 'layers' (character, word, bigram) combined with fixed weights.

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

| Metric | Value |
|---|---|
| Test accuracy | 90.15% |
| F1 (macro) | 89.90% |
| Precision (macro) | 89.85% |
| Recall (macro) | 89.97% |
| Test set size | 1492 |
| Train set size | 2242 |

_Metrics are measured on the held-out 20-Newsgroups test split the model never trained on._

## Run locally

```bash
pip install -r requirements.txt
python model.py                        # retrain from scratch (original script)
python training/train_all.py --only 6   # rebuild artifacts/ from repo root
streamlit run app.py
```

## Deploy (free)

1. Push this repository to GitHub.
2. On **share.streamlit.io** → *New app* → *Deploy from GitHub*.
3. Repository `deepvisionkararhaider-crypto/nlp-models`, branch `main`.
4. **Main file path:** `06_elmo/app.py`

## Reproducible training

- **Local / CPU:** `python training/train_all.py --only 6`
- **Colab / Kaggle:** open [`notebooks/06_elmo.ipynb`](../notebooks/06_elmo.ipynb).
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
[← Back to the portfolio](../README.md) · [Source folder](https://github.com/deepvisionkararhaider-crypto/nlp-models/tree/main/06_elmo)
