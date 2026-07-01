# 07. BERT

## Overview
Bidirectional Encoder Representations from Transformers: 12-layer bidirectional transformer (110M params).

**Dataset**: 20 Newsgroups (4 categories: hockey, space, graphics, politics)
**Task**: Multi-class text classification
**Implementation**: BERT-style (TF-IDF + SVD + Ridge Classifier)

## Key Concept
`10000 TF-IDF → SVD(100), Ridge bidirectional encoding`

## Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.8861 |
| Precision | 0.8840 |
| F1-Score  | 0.8847 |

## Files
| File | Description |
|------|-------------|
| `model.py` | Main training and evaluation script |
| `predictions.csv` | Model predictions on test set |
| `data/newsgroups.csv` | Sample of training data |
| `plots/bert_analysis.png` | Confusion matrix, PCA, per-class accuracy |
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
