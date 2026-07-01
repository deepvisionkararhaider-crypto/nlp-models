# 09. ALBERT

## Overview
A Lite BERT: parameter sharing and factorized embeddings for efficient pretraining.

**Dataset**: 20 Newsgroups (4 categories: hockey, space, graphics, politics)
**Task**: Multi-class text classification
**Implementation**: ALBERT-style (TF-IDF + SVD + Passive-Aggressive)

## Key Concept
`10000 TF-IDF → SVD(100), PA parameter-efficient`

## Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.8693 |
| Precision | 0.8771 |
| F1-Score  | 0.8681 |

## Files
| File | Description |
|------|-------------|
| `model.py` | Main training and evaluation script |
| `predictions.csv` | Model predictions on test set |
| `data/newsgroups.csv` | Sample of training data |
| `plots/albert_analysis.png` | Confusion matrix, PCA, per-class accuracy |
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
