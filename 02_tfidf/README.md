# 02. TF-IDF

## Overview
Term Frequency-Inverse Document Frequency reduces the weight of common words, improving over plain BoW.

**Dataset**: 20 Newsgroups (4 categories: hockey, space, graphics, politics)
**Task**: Multi-class text classification
**Implementation**: TF-IDF + LinearSVC

## Key Concept
`10000 TF-IDF features, sublinear_tf, ngram(1,2)`

## Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.8753 |
| Precision | 0.8749 |
| F1-Score  | 0.8719 |

## Files
| File | Description |
|------|-------------|
| `model.py` | Main training and evaluation script |
| `predictions.csv` | Model predictions on test set |
| `data/newsgroups.csv` | Sample of training data |
| `plots/tfidf_analysis.png` | Confusion matrix, PCA, per-class accuracy |
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
