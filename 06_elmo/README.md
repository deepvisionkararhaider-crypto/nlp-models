# 06. ELMo

## Overview
Embeddings from Language Models: contextual representations from a 3-level (char/word/context) architecture.

**Dataset**: 20 Newsgroups (4 categories: hockey, space, graphics, politics)
**Task**: Multi-class text classification
**Implementation**: ELMo-style (3-layer: char+word+context) + Logistic Regression

## Key Concept
`Char n-grams + word + bigram TF-IDF SVD(50 each)`

## Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.8760 |
| Precision | 0.8747 |
| F1-Score  | 0.8724 |

## Files
| File | Description |
|------|-------------|
| `model.py` | Main training and evaluation script |
| `predictions.csv` | Model predictions on test set |
| `data/newsgroups.csv` | Sample of training data |
| `plots/elmo_analysis.png` | Confusion matrix, PCA, per-class accuracy |
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
