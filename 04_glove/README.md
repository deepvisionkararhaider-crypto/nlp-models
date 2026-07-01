# 04. GloVe

## Overview
Global Vectors for Word Representation using co-occurrence statistics and PPMI matrix factorization.

**Dataset**: 20 Newsgroups (4 categories: hockey, space, graphics, politics)
**Task**: Multi-class text classification
**Implementation**: GloVe-style (PPMI + SVD) + Logistic Regression

## Key Concept
`PPMI co-occurrence + TruncatedSVD(100)`

## Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.8693 |
| Precision | 0.8700 |
| F1-Score  | 0.8676 |

## Files
| File | Description |
|------|-------------|
| `model.py` | Main training and evaluation script |
| `predictions.csv` | Model predictions on test set |
| `data/newsgroups.csv` | Sample of training data |
| `plots/glove_analysis.png` | Confusion matrix, PCA, per-class accuracy |
| `requirements.txt` | Python dependencies |

## Run
```bash
pip install -r requirements.txt
python model.py
```

## Categories
- `rec.sport.hockey` → hockey
- `sci.space` → space
- `comp.graphics` → graphics
- `talk.politics.misc` → politics
