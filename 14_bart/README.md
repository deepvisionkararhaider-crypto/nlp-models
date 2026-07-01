# 14. BART

## Overview
Bidirectional and Auto-Regressive Transformers: BERT encoder + GPT decoder for seq2seq tasks.

**Dataset**: 20 Newsgroups (4 categories: hockey, space, graphics, politics)
**Task**: Multi-class text classification
**Implementation**: BART-style (TF-IDF + SVD + ExtraTrees)

## Key Concept
`10000 TF-IDF → SVD(100), ExtraTrees bidirectional`

## Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.8773 |
| Precision | 0.8763 |
| F1-Score  | 0.8748 |

## Files
| File | Description |
|------|-------------|
| `model.py` | Main training and evaluation script |
| `predictions.csv` | Model predictions on test set |
| `data/newsgroups.csv` | Sample of training data |
| `plots/bart_analysis.png` | Confusion matrix, PCA, per-class accuracy |
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
