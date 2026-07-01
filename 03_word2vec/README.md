# 03. Word2Vec

## Overview
Neural word embeddings trained with skip-gram/CBOW to capture semantic similarity between words.

**Dataset**: 20 Newsgroups (4 categories: hockey, space, graphics, politics)
**Task**: Multi-class text classification
**Implementation**: Word2Vec (gensim) + Logistic Regression

## Key Concept
`100-dim embeddings, window=5, mean pooling`

## Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.8009 |
| Precision | 0.8032 |
| F1-Score  | 0.7999 |

## Files
| File | Description |
|------|-------------|
| `model.py` | Main training and evaluation script |
| `predictions.csv` | Model predictions on test set |
| `data/newsgroups.csv` | Sample of training data |
| `plots/word2vec_analysis.png` | Confusion matrix, PCA, per-class accuracy |
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
