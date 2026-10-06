# Training notebooks

Reproducible Colab / Kaggle notebooks - one per model. Each clones the
repository, trains the model's original architecture, verifies metrics on
the held-out test split, tests inference on unseen text and exports the
artifact.

> **No GPU needed.** All 15 models are small classical NLP pipelines that
> train on CPU in seconds to ~1 minute.

| # | Model | Notebook |
|---|-------|----------|
| 01 | Bag of Words | [`01_bag_of_words.ipynb`](01_bag_of_words.ipynb) |
| 02 | TF-IDF | [`02_tfidf.ipynb`](02_tfidf.ipynb) |
| 03 | Word2Vec | [`03_word2vec.ipynb`](03_word2vec.ipynb) |
| 04 | GloVe | [`04_glove.ipynb`](04_glove.ipynb) |
| 05 | FastText | [`05_fasttext.ipynb`](05_fasttext.ipynb) |
| 06 | ELMo (3-layer contextual) | [`06_elmo.ipynb`](06_elmo.ipynb) |
| 07 | BERT | [`07_bert.ipynb`](07_bert.ipynb) |
| 08 | RoBERTa | [`08_roberta.ipynb`](08_roberta.ipynb) |
| 09 | ALBERT | [`09_albert.ipynb`](09_albert.ipynb) |
| 10 | DistilBERT | [`10_distilbert.ipynb`](10_distilbert.ipynb) |
| 11 | XLNet | [`11_xlnet.ipynb`](11_xlnet.ipynb) |
| 12 | T5 | [`12_t5.ipynb`](12_t5.ipynb) |
| 13 | GPT | [`13_gpt.ipynb`](13_gpt.ipynb) |
| 14 | BART | [`14_bart.ipynb`](14_bart.ipynb) |
| 15 | ELECTRA | [`15_electra.ipynb`](15_electra.ipynb) |
