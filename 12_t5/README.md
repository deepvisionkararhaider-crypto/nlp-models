# 12. T5

## Overview
Text-to-Text Transfer Transformer: frames all NLP tasks as text generation.

**Dataset**: 20 Newsgroups (4 categories: hockey, space, graphics, politics)
**Task**: Multi-class text classification
**Implementation**: T5-style (TF-IDF + SVD + Random Forest)

## Key Concept
`10000 TF-IDF → SVD(100), 200 trees`

## Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.8619 |
| Precision | 0.8640 |
| F1-Score  | 0.8613 |

## Files
| File | Description |
|------|-------------|
| `model.py` | Main training and evaluation script |
| `predictions.csv` | Model predictions on test set |
| `data/newsgroups.csv` | Sample of training data |
| `plots/t5_analysis.png` | Confusion matrix, PCA, per-class accuracy |
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
