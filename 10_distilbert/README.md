# 10. DistilBERT

## Overview
Distilled BERT: 60% smaller, 40% faster, retaining 97% of BERT performance via knowledge distillation.

**Dataset**: 20 Newsgroups (4 categories: hockey, space, graphics, politics)
**Task**: Multi-class text classification
**Implementation**: DistilBERT-style (5k TF-IDF + SVD + LinearSVC)

## Key Concept
`5000 TF-IDF → SVD(100), compressed LinearSVC`

## Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.8834 |
| Precision | 0.8842 |
| F1-Score  | 0.8816 |

## Files
| File | Description |
|------|-------------|
| `model.py` | Main training and evaluation script |
| `predictions.csv` | Model predictions on test set |
| `data/newsgroups.csv` | Sample of training data |
| `plots/distilbert_analysis.png` | Confusion matrix, PCA, per-class accuracy |
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
