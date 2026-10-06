# 01. Bag of Words

**Task:** Topic Classification (20-Newsgroups, 4 classes)
**Framework:** scikit-learn
**Real implementation:** CountVectorizer (5k features, 1-2 grams, English stop-words) + Logistic Regression

✅ The folder name matches the technique actually implemented here.

---

## What it does

Classic bag-of-words representation: text becomes a sparse matrix of unigram+bigram counts, then a logistic-regression classifier predicts the newsgroup topic.

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
| Test accuracy | 87.06% |
| F1 (macro) | 86.82% |
| Precision (macro) | 86.91% |
| Recall (macro) | 86.80% |
| Test set size | 1492 |
| Train set size | 2242 |

_Metrics are measured on the held-out 20-Newsgroups test split the model never trained on._

## Run locally

```bash
pip install -r requirements.txt
python model.py                        # retrain from scratch (original script)
python training/train_all.py --only 1   # rebuild artifacts/ from repo root
streamlit run app.py
```

## Deploy (free)

1. Push this repository to GitHub.
2. On **share.streamlit.io** → *New app* → *Deploy from GitHub*.
3. Repository `deepvisionkararhaider-crypto/nlp-models`, branch `main`.
4. **Main file path:** `01_bag_of_words/app.py`

## Reproducible training

- **Local / CPU:** `python training/train_all.py --only 1`
- **Colab / Kaggle:** open [`notebooks/01_bag_of_words.ipynb`](../notebooks/01_bag_of_words.ipynb).
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
[← Back to the portfolio](../README.md) · [Source folder](https://github.com/deepvisionkararhaider-crypto/nlp-models/tree/main/01_bag_of_words)
