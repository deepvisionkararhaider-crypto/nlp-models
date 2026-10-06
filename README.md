# NLP Model Portfolio — 15 models, 15 live demos

A teaching portfolio of **15 natural-language-processing projects**, each with a
**real trained model** behind a polished [Streamlit](https://streamlit.io) front
end. Students enter new text, run the actual model, and see the real output — no
hard-coded, simulated, or randomised predictions anywhere.

**Task (all 15):** multi-class **topic classification** on four
[20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset)
categories — `hockey`, `space`, `graphics`, `politics`.

---

## ⚠️ Read this first: names vs. real implementations

Folders **07–15** are *named* after famous transformer architectures
(BERT, RoBERTa, ALBERT, DistilBERT, XLNet, T5, GPT, BART, ELECTRA). **No
transformer is downloaded or trained in this repository.** Those projects are
implemented as **TF-IDF → TruncatedSVD → classical scikit-learn classifiers** —
the original code says so in its own header comments, and we kept the original
architectures as requested.

The portfolio therefore labels each model **truthfully** everywhere it appears:
the card, the app hero, the sidebar and a "What this model actually is" panel all
state the *real* architecture. **6 of 15** folder names match their implementation
(01–05, plus ELMo-style 06 is described as a feature-stack, not a biLM);
**9 of 15** (07–15) are name-only.

---

## Models at a glance

| # | Model | Task | Framework | Real implementation | Name matches? | Test acc. | F1 (macro) | Deploy status |
|---|-------|------|-----------|---------------------|:-------------:|:---------:|:----------:|---------------|
| 01 | Bag of Words | Topic classification | scikit-learn | CountVectorizer(5k, 1-2g) + LogisticRegression | ✅ | 87.06% | 86.82% | ⏳ Ready to deploy |
| 02 | TF-IDF | Topic classification | scikit-learn | TfidfVectorizer(10k, 1-2g) + Calibrated LinearSVC | ✅ | 91.55% | 91.36% | ⏳ Ready to deploy |
| 03 | Word2Vec | Topic classification | gensim + sklearn | Word2Vec(100d) mean-pool + LogisticRegression | ✅ | 76.41% | 75.87% | ⏳ Ready to deploy |
| 04 | GloVe | Topic classification | numpy + sklearn | PPMI co-occurrence → SVD(100) mean-pool + LR | ✅ | 89.81% | 89.65% | ⏳ Ready to deploy |
| 05 | FastText | Topic classification | gensim + sklearn | FastText(100d, subwords 3-5g) mean-pool + LR | ✅ | 72.45% | 71.94% | ⏳ Ready to deploy |
| 06 | ELMo (3-layer contextual) | Topic classification | scikit-learn | 3× TF-IDF→SVD(50) stack (char/word/bigram), weights .3/.4/.3 + LR | ⚠️ feature stack, not a biLM | 90.15% | 89.90% | ⏳ Ready to deploy |
| 07 | BERT | Topic classification | scikit-learn | TF-IDF→SVD(100) + RidgeClassifier | ❌ no transformer | 91.35% | 91.14% | ⏳ Ready to deploy |
| 08 | RoBERTa | Topic classification | scikit-learn | TF-IDF→SVD(100) + SGDClassifier(modified_huber) | ❌ no transformer | 90.62% | 90.41% | ⏳ Ready to deploy |
| 09 | ALBERT | Topic classification | scikit-learn | TF-IDF→SVD(100) + PassiveAggressive | ❌ no transformer | 90.35% | 90.15% | ⏳ Ready to deploy |
| 10 | DistilBERT | Topic classification | scikit-learn | TF-IDF(5k)→SVD(100) + Calibrated LinearSVC | ❌ no transformer | 90.68% | 90.43% | ⏳ Ready to deploy |
| 11 | XLNet | Topic classification | scikit-learn | TF-IDF→SVD(100) + GradientBoosting | ❌ no transformer | 89.48% | 89.23% | ⏳ Ready to deploy |
| 12 | T5 | Topic classification | scikit-learn | TF-IDF→SVD(100) + RandomForest(200) | ❌ no transformer | 89.48% | 89.22% | ⏳ Ready to deploy |
| 13 | GPT | Topic classification | scikit-learn | TF-IDF→SVD(100) + Multinomial LogisticRegression | ❌ no transformer | 91.02% | 90.83% | ⏳ Ready to deploy |
| 14 | BART | Topic classification | scikit-learn | TF-IDF→SVD(100) + ExtraTrees(200) | ❌ no transformer | 90.62% | 90.35% | ⏳ Ready to deploy |
| 15 | ELECTRA | Topic classification | scikit-learn | TF-IDF→SVD(100) + Calibrated Perceptron | ❌ no transformer | 91.02% | 90.79% | ⏳ Ready to deploy |

**Test accuracy** is measured on the held-out 20-Newsgroups **test split**
(1,492 documents never seen during training). Training-set accuracy is never
shown, because it would overstate performance.

> **Deploy status legend.** `⏳ Ready to deploy` = app, artifacts and entry point
> all exist and were verified locally; the public app has **not** been created
> yet, because Streamlit Community Cloud deploys require your own GitHub-linked
> Streamlit account (see [DEPLOYMENT.md](DEPLOYMENT.md)). No `*.streamlit.app`
> URL is claimed anywhere in this repository until it is actually created.

---

## How it works

```
user text → normalise → tokenise / vectorise → trained model → decoded label → UI
```

- `app.py` (repo root) — **portfolio dashboard**: all 15 cards, verified metrics,
  deployment table, and a "compare all 15 models" consensus tool.
- `<folder>/app.py` — **each model's own Streamlit entry point** (deploy path).
- `common/nlp_common.py` — one honest model registry, artifact loader, tokeniser
  and inference path shared by training, tests and every app.
- `common/app_base.py` + `common/ui.py` — the shared, consistent UI.
- `training/` — reproducible dataset build, training, app/README/notebook
  generators.
- `notebooks/` — 15 resumable Colab/Kaggle notebooks (no GPU required).
- `tests/test_inference.py` — inference tests on unseen text for all 15 models.

### Truthful scoring

Where a classifier genuinely exposes probabilities we show them. Where it does
not (Ridge 07, Passive-Aggressive 09), the app shows a **softmax of the real
`decision_function`** and labels it `decision_function_softmax` — explicitly
"*not* calibrated probabilities". No confidence value is ever invented.

---

## Data

- **Source:** original `20news-bydate` archive (20-Newsgroups).
- **Categories:** `rec.sport.hockey`, `sci.space`, `comp.graphics`,
  `talk.politics.misc` (label order 0-3, same as the original scripts).
- **Splits:** 2,242 train / 1,492 test, with headers/footers/quotes removed
  (mirrors `fetch_20newsgroups(remove=('headers','footers','quotes'))`).
- **Committed at:** `data/20newsgroups_4cat/{train.csv,test.csv,meta.json}`.
- **Rebuild:** `python training/build_dataset.py --archive /path/to/20news-bydate.tar.gz`

## Artifacts

Each model writes `<folder>/artifacts/`:

| File | Contents |
|------|----------|
| `model.joblib` | fitted model bundle (pickled) |
| `metrics.json` | verified held-out metrics |
| `meta.json` | hyper-parameters, versions, timestamp |
| `w2v.model` / `ft.model` | gensim embeddings for models 03 / 05 |

Rebuild everything: `python training/train_all.py` (CPU only, ~4 min total).

---

## Run locally

```bash
pip install -r requirements.txt
python training/train_all.py            # build all 15 artifacts (~4 min, CPU)
python tests/test_inference.py          # verify real predictions on unseen text
streamlit run app.py                    # dashboard
streamlit run 01_bag_of_words/app.py    # any single model
```

## Deploy (free, public)

Streamlit Community Cloud, one app per model, from this single repository —
**no 15 repositories needed**. Full instructions and the entry-point table are in
**[DEPLOYMENT.md](DEPLOYMENT.md)**. Short version:

1. share.streamlit.io → New app → deploy from `deepvisionkararhaider-crypto/nlp-models`, branch `main`.
2. **Main file path:** `01_bag_of_words/app.py` (change per model).
3. Deploy. Repeat for the other 14.

---

## Verified status (final report)

| Status | Count | Models |
|--------|:-----:|--------|
| READY FOR TRAINING | 15 | all (each has a Colab notebook) |
| READY FOR INFERENCE | 15 | all (artifacts built, apps load them) |
| READY FOR DEPLOYMENT | 16 | 15 models + dashboard (entry points + `requirements.txt` verified) |
| DEPLOYED AND VERIFIED | 0 | requires your Streamlit account (see DEPLOYMENT.md) |

**Verified locally:** all 15 apps launched (health check `ok`, no errors); all 15
passed the unseen-text inference test (empty-input and long-input guards
included).

## Repository layout

```
nlp-models/
├── app.py                     # portfolio dashboard (Streamlit entry point)
├── DEPLOYMENT.md              # per-model deploy table + checklist
├── requirements.txt           # everything for local dev
├── common/                    # nlp_common.py, app_base.py, ui.py
├── data/20newsgroups_4cat/    # committed dataset (train/test/meta)
├── training/                  # build_dataset, train_all, generators
├── tests/test_inference.py    # inference tests (unseen text)
├── notebooks/                 # 15 reproducible training notebooks
└── 01_bag_of_words/ … 15_electra/
    ├── model.py               # original training script (preserved)
    ├── app.py                 # Streamlit entry point
    ├── requirements.txt
    ├── README.md
    ├── artifacts/             # trained model + metrics + meta
    ├── data/newsgroups.csv    # original sample data (preserved)
    ├── plots/                 # original plots (preserved)
    └── predictions.csv        # original predictions (preserved)
```

## Honesty commitments

- No fake, random, or hard-coded predictions — every output comes from a trained model.
- Original `model.py`, `data/`, `plots/`, `predictions.csv` are preserved untouched.
- No dataset, weight, metric or URL is fabricated.
- Name/architecture mismatches are disclosed on every surface a student sees.

---
*Built with scikit-learn, gensim and Streamlit. All models are small classical
NLP pipelines — **no GPU required**.*
