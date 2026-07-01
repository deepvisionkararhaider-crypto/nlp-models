# 11. XLNet

## Overview
Generalized Autoregressive Pretraining: permutation language modeling captures bidirectional context.

**Dataset**: 20 Newsgroups (4 categories: hockey, space, graphics, politics)
**Task**: Multi-class text classification
**Implementation**: XLNet-style (TF-IDF + SVD + Gradient Boosting)

## Key Concept
`10000 TF-IDF → SVD(100), autoregressive-style GB`

## Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.8686 |
| Precision | 0.8671 |
| F1-Score  | 0.8666 |

## Files
| File | Description |
|------|-------------|
| `model.py` | Main training and evaluation script |
| `predictions.csv` | Model predictions on test set |
| `data/newsgroups.csv` | Sample of training data |
| `plots/xlnet_analysis.png` | Confusion matrix, PCA, per-class accuracy |
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
