# 08. RoBERTa

## Overview
Robustly Optimized BERT: trained longer, on more data, with dynamic masking and no NSP objective.

**Dataset**: 20 Newsgroups (4 categories: hockey, space, graphics, politics)
**Task**: Multi-class text classification
**Implementation**: RoBERTa-style (TF-IDF + SVD + SGD Classifier)

## Key Concept
`10000 TF-IDF → SVD(100), modified_huber loss`

## Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.8747 |
| Precision | 0.8735 |
| F1-Score  | 0.8713 |

## Files
| File | Description |
|------|-------------|
| `model.py` | Main training and evaluation script |
| `predictions.csv` | Model predictions on test set |
| `data/newsgroups.csv` | Sample of training data |
| `plots/roberta_analysis.png` | Confusion matrix, PCA, per-class accuracy |
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
