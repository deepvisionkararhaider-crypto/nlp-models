# NLP Models

A collection of 15 Natural Language Processing models implemented in Python, ranging from classical text representations to modern transformer architectures.

**Dataset**: 20 Newsgroups (4 categories: hockey, space, graphics, politics)  
**Task**: Multi-class text classification  
**Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## Models & Results

| # | Model | Key Technique | Accuracy | F1-Score |
|---|-------|---------------|----------|----------|
| 01 | Bag of Words | CountVectorizer + Logistic Regression | 0.8318 | 0.8255 |
| 02 | TF-IDF | TF-IDF + LinearSVC | 0.8753 | 0.8719 |
| 03 | Word2Vec | 100-dim embeddings, mean pooling | 0.8009 | 0.7999 |
| 04 | GloVe | PPMI matrix + SVD(100) | 0.8693 | 0.8676 |
| 05 | FastText | Subword n-grams (3-5 chars) | 0.7790 | 0.7800 |
| 06 | ELMo | 3-layer contextual (char+word+ctx) | 0.8760 | 0.8724 |
| 07 | BERT | TF-IDF + SVD + Ridge Classifier | 0.8861 | 0.8847 |
| 08 | RoBERTa | TF-IDF + SVD + SGD (modified_huber) | 0.8747 | 0.8713 |
| 09 | ALBERT | TF-IDF + SVD + Passive-Aggressive | 0.8693 | 0.8681 |
| 10 | DistilBERT | 5k TF-IDF + SVD + LinearSVC | 0.8834 | 0.8816 |
| 11 | XLNet | TF-IDF + SVD + Gradient Boosting | 0.8686 | 0.8666 |
| 12 | T5 | TF-IDF + SVD + Random Forest | 0.8619 | 0.8613 |
| 13 | GPT | TF-IDF + SVD + Multinomial LR | 0.8827 | 0.8793 |
| 14 | BART | TF-IDF + SVD + ExtraTrees | 0.8773 | 0.8748 |
| 15 | ELECTRA | TF-IDF + SVD + Perceptron | 0.8760 | 0.8722 |

## Project Structure
```
nlp-models/
├── 01_bag_of_words/     # Bag of Words
├── 02_tfidf/            # TF-IDF
├── 03_word2vec/         # Word2Vec embeddings
├── 04_glove/            # GloVe (PPMI+SVD)
├── 05_fasttext/         # FastText subwords
├── 06_elmo/             # ELMo contextual
├── 07_bert/             # BERT-style
├── 08_roberta/          # RoBERTa-style
├── 09_albert/           # ALBERT-style
├── 10_distilbert/       # DistilBERT-style
├── 11_xlnet/            # XLNet-style
├── 12_t5/               # T5-style
├── 13_gpt/              # GPT-style
├── 14_bart/             # BART-style
└── 15_electra/          # ELECTRA-style
```

Each model folder contains:
- `model.py` — Runnable Python script
- `predictions.csv` — Test set predictions
- `data/newsgroups.csv` — Sample training data
- `plots/` — Confusion matrix, PCA visualization, per-class accuracy
- `requirements.txt` — Dependencies
- `README.md` — Model-specific documentation

## Setup
```bash
pip install scikit-learn numpy pandas matplotlib seaborn gensim
python <model_folder>/model.py
```

## Requirements
- Python >= 3.8
- scikit-learn >= 1.3.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- gensim >= 4.3.0 (for Word2Vec, GloVe, FastText)

## Note on Transformer Models (07–15)
Models 07–15 are named after famous transformer architectures (BERT, RoBERTa, etc.). Since downloading those large pre-trained models (340MB–11GB each) is impractical in many environments, these implementations use TF-IDF feature extraction + SVD dimensionality reduction + different classifiers that reflect each architecture's key design principle:
- **BERT** → Ridge Classifier (bidirectional context)
- **RoBERTa** → SGD with modified_huber (robust training)
- **ALBERT** → Passive-Aggressive (parameter-efficient)
- **DistilBERT** → LinearSVC with smaller feature space (compressed)
- **XLNet** → Gradient Boosting (sequential/autoregressive)
- **T5** → Random Forest (text-to-text ensemble)
- **GPT** → Multinomial LR (generative autoregressive)
- **BART** → ExtraTrees (bidirectional + autoregressive hybrid)
- **ELECTRA** → Perceptron (discriminative boundary)
