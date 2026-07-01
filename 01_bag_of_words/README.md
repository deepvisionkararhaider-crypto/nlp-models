# 01. Bag of Words

## Overview
CountVectorizer converts text into token count matrices. Simple, interpretable baseline for NLP classification.

**Dataset**: 20 Newsgroups (4 categories: hockey, space, graphics, politics)
**Task**: Multi-class text classification
**Implementation**: CountVectorizer + Logistic Regression

## Key Concept
`5000 BoW features, unigrams+bigrams`

## Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.8318 |
| Precision | 0.8301 |
| F1-Score  | 0.8255 |

## Files
| File | Description |
|------|-------------|
| `model.py` | Main training and evaluation script |
| `predictions.csv` | Model predictions on test set |
| `data/newsgroups.csv` | Sample of training data |
| `plots/bag_of_words_analysis.png` | Confusion matrix, PCA, per-class accuracy |
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
