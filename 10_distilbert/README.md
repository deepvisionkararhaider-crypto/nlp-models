# 10. DistilBERT

**Task:** Topic Classification (20-Newsgroups, 4 classes)
**Framework:** scikit-learn
**Real implementation:** TF-IDF (5k, 1-2 grams) -> TruncatedSVD(100) -> Calibrated LinearSVC

⚠️ **Naming note:** No transformer is used. The name references DistilBERT's distillation; the implementation is classical.

---

## What it does

Named after DistilBERT's compression; implemented with a smaller TF-IDF vocabulary (5k) + SVD + a calibrated linear SVM.

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
| Test accuracy | 90.68% |
| F1 (macro) | 90.43% |
| Precision (macro) | 90.44% |
| Recall (macro) | 90.44% |
| Test set size | 1492 |
| Train set size | 2242 |

_Metrics are measured on the held-out 20-Newsgroups test split the model never trained on._

## Run locally

```bash
pip install -r requirements.txt
python model.py                        # retrain from scratch (original script)
python training/train_all.py --only 10   # rebuild artifacts/ from repo root
streamlit run app.py
```

## Deploy (free)

1. Push this repository to GitHub.
2. On **share.streamlit.io** → *New app* → *Deploy from GitHub*.
3. Repository `deepvisionkararhaider-crypto/nlp-models`, branch `main`.
4. **Main file path:** `10_distilbert/app.py`

## Reproducible training

- **Local / CPU:** `python training/train_all.py --only 10`
- **Colab / Kaggle:** open [`notebooks/10_distilbert.ipynb`](../notebooks/10_distilbert.ipynb).
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
[← Back to the portfolio](../README.md) · [Source folder](https://github.com/deepvisionkararhaider-crypto/nlp-models/tree/main/10_distilbert)
