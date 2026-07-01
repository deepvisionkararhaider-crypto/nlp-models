# 15. ELECTRA

## Overview
Efficiently Learning an Encoder via replaced token detection, a more efficient pretraining objective than MLM.

**Dataset**: 20 Newsgroups (4 categories: hockey, space, graphics, politics)
**Task**: Multi-class text classification
**Implementation**: ELECTRA-style (TF-IDF + SVD + Perceptron)

## Key Concept
`10000 TF-IDF → SVD(100), discriminative Perceptron`

## Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.8760 |
| Precision | 0.8740 |
| F1-Score  | 0.8722 |

## Files
| File | Description |
|------|-------------|
| `model.py` | Main training and evaluation script |
| `predictions.csv` | Model predictions on test set |
| `data/newsgroups.csv` | Sample of training data |
| `plots/electra_analysis.png` | Confusion matrix, PCA, per-class accuracy |
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
