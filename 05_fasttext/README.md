# 05. FastText

## Overview
Facebook's extension of Word2Vec using character n-gram subwords to handle OOV words.

**Dataset**: 20 Newsgroups (4 categories: hockey, space, graphics, politics)
**Task**: Multi-class text classification
**Implementation**: FastText (gensim) + Logistic Regression

## Key Concept
`100-dim, char n-grams (3-5), handles OOV via subwords`

## Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.7790 |
| Precision | 0.7800 |
| F1-Score  | 0.7800 |

## Files
| File | Description |
|------|-------------|
| `model.py` | Main training and evaluation script |
| `predictions.csv` | Model predictions on test set |
| `data/newsgroups.csv` | Sample of training data |
| `plots/fasttext_analysis.png` | Confusion matrix, PCA, per-class accuracy |
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
