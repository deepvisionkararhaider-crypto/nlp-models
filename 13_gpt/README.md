# 13. GPT

## Overview
Generative Pre-trained Transformer: unidirectional (left-to-right) autoregressive language model.

**Dataset**: 20 Newsgroups (4 categories: hockey, space, graphics, politics)
**Task**: Multi-class text classification
**Implementation**: GPT-style (TF-IDF + SVD + Multinomial LR)

## Key Concept
`10000 TF-IDF → SVD(100), C=5 LR autoregressive`

## Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.8827 |
| Precision | 0.8783 |
| F1-Score  | 0.8793 |

## Files
| File | Description |
|------|-------------|
| `model.py` | Main training and evaluation script |
| `predictions.csv` | Model predictions on test set |
| `data/newsgroups.csv` | Sample of training data |
| `plots/gpt_analysis.png` | Confusion matrix, PCA, per-class accuracy |
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
