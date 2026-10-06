# Deployment — 15 free public Streamlit apps

Every model has its own Streamlit entry point, so each can be deployed as its own
public app on **Streamlit Community Cloud** (free), all from this single
repository. No 15 separate repositories are needed.

---

## TL;DR — deploy one model

1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. **New app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `deepvisionkararhaider-crypto/nlp-models`
   - **Branch:** `main`
   - **Main file path:** `01_bag_of_words/app.py`  *(change per model — see table)*
4. **Deploy.** The first build installs that model's `requirements.txt`
   (Streamlit reads a `requirements.txt` next to the entry point — every model
   folder has one).
5. Repeat for the other 14 models, changing only **Main file path**.

The dashboard is deployed the same way with **Main file path** = `app.py`.

---

## Entry points

| # | Model | Main file path |
|---|-------|----------------|
| – | **Portfolio dashboard** | `app.py` |
| 01 | Bag of Words | `01_bag_of_words/app.py` |
| 02 | TF-IDF | `02_tfidf/app.py` |
| 03 | Word2Vec | `03_word2vec/app.py` |
| 04 | GloVe | `04_glove/app.py` |
| 05 | FastText | `05_fasttext/app.py` |
| 06 | ELMo (3-layer contextual) | `06_elmo/app.py` |
| 07 | BERT | `07_bert/app.py` |
| 08 | RoBERTa | `08_roberta/app.py` |
| 09 | ALBERT | `09_albert/app.py` |
| 10 | DistilBERT | `10_distilbert/app.py` |
| 11 | XLNet | `11_xlnet/app.py` |
| 12 | T5 | `12_t5/app.py` |
| 13 | GPT | `13_gpt/app.py` |
| 14 | BART | `14_bart/app.py` |
| 15 | ELECTRA | `15_electra/app.py` |

All 15 share the same `common/` package, so a shared change (e.g. the UI) applies
everywhere.

---

## Important: artifact sizes

Each app loads **only its own** `artifacts/model.joblib`. Most are 0.2–15 MB.
Two are larger because they store a gensim embedding model:

| Model | artifact on disk |
|-------|------------------|
| 03 Word2Vec | ~13 MB |
| 05 FastText | ~91 MB |

- **Streamlit Community Cloud** clones the repo and installs dependencies from
  `requirements.txt`; committing these artifacts (already done) is the simplest
  way to keep the apps self-contained and reliable.
- If you prefer not to commit the large gensim files, host them on a model hub
  and download them at startup (see *Optional: load artifacts from a hub* below).
  **Never** add GitHub Actions or an external download that silently swaps in a
  different model.

### What is committed

`<folder>/artifacts/model.joblib`, `metrics.json`, `meta.json` are committed.
The 91 MB FastText model is intentionally included; if your Git host rejects it,
move that one file to a hub (below). Nothing else needs Git LFS.

---

## Optional: load artifacts from a hub

If a model file is too big for Git, edit that model's app to fetch it before load:

```python
# in <folder>/app.py, before run_app(FOLDER)
import os, urllib.request
from pathlib import Path
ART = Path(__file__).resolve().parent / "artifacts"
ART.mkdir(exist_ok=True)
target = ART / "model.joblib"
if not target.exists():
    urllib.request.urlretrieve("https://<your-hub>/model.joblib", target)
```

The app already raises a clear setup error (and never a fake prediction) if an
artifact is missing.

---

## Runtime & resources

- **Python:** Streamlit Community Cloud lets you pick 3.9–3.12. The code is
  compatible with **3.9–3.12**. Do **not** select 3.13 — pinning scikit-learn /
  numpy wheels for 3.13 on that platform is not guaranteed.
- **CPU only** — every model is small classical NLP; none needs a GPU.
- **First prediction** for models 03/05 loads a gensim model (a few seconds).
  Streamlit caches the bundle (`st.cache_resource`), so later predictions are
  instant.
- Memory footprint per app is small (tens of MB), well within the free tier.

---

## Environment variables / secrets

None are required. These demos call no external APIs and need no keys.

---

## Verifying a deployment

After a deploy, open the app and:

1. Click an example button, then **Predict topic** — you should see a topic and a
   score table.
2. Confirm the sidebar shows the correct model number and the "Real
   architecture" line.
3. Confirm no browser console errors.

Do **not** record a URL as live in the README until the app actually built
successfully on Streamlit Community Cloud.
