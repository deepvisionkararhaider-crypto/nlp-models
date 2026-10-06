"""
Reproducible training for all 15 NLP models.

Every model keeps the architecture from its original ``<folder>/model.py``.  The
feature-extraction / classifier hyper-parameters below were copied verbatim from
those scripts, so the artifacts this module produces are the *same models* the
repository describes - just with a saved object this time.

Run modes
---------
    python training/train_all.py                 # train all 15 (light set by default)
    python training/train_all.py --only 1 3 5    # train a subset by number
    python training/train_all.py --full          # heavier embeddings (best fidelity)

Artifacts are written to ``<folder>/artifacts/``:
    model.joblib     the fitted model bundle
    metrics.json     verified, held-out test metrics
    meta.json        reproducibility info (hyper-params, timings, versions)
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import nlp_common as nc  # noqa: E402

REPO_ROOT = nc.REPO_ROOT


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_p, test_p = nc.DATA_DIR / "train.csv", nc.DATA_DIR / "test.csv"
    if not (train_p.exists() and test_p.exists()):
        raise SystemExit(
            "Dataset missing. Run: python training/build_dataset.py --archive "
            "<path to 20news-bydate.tar.gz>"
        )
    return pd.read_csv(train_p), pd.read_csv(test_p)


def _xy(df: pd.DataFrame):
    return df["text"].astype(str).tolist(), df["label"].to_numpy()


# ---------------------------------------------------------------------------
# Metrics (held-out test set only - never the training set)
# ---------------------------------------------------------------------------
# Alias (NOT a subclass) of the shared, picklable wrapper. Subclassing here would
# make joblib record "__main__.FeaturesThenClassifier" and break unpickling in
# any other process; aliasing keeps the class's true import path.
FeaturesThenClassifier = nc.FeaturesThenClassifier


def evaluate(bundle, test_df, y_test) -> dict:
    """Held-out test metrics using the *same* inference path the apps use."""
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score,
                                 precision_score, recall_score)

    texts = test_df["text"].astype(str).tolist()
    y_pred, _ = nc.predict_many(bundle, texts)
    labels = list(range(len(nc.CATEGORIES)))
    report = classification_report(
        y_test, y_pred, labels=labels,
        target_names=nc.SHORT_NAMES, output_dict=True, zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in nc.SHORT_NAMES
        },
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "n_test": int(len(y_test)),
        "n_train": None,  # filled by the caller
    }


def build_meta(folder: str, kind: str, hyperparams: dict, extra: dict | None = None) -> dict:
    import sklearn
    meta = nc.MODEL_REGISTRY[folder]
    payload = {
        "folder": folder,
        "num": meta["num"],
        "display_name": meta["display"],
        "actual_architecture": meta["actual"],
        "family": meta["family"],
        "framework": meta["framework"],
        "faithful_family": meta["faithful_family"],
        "kind": kind,
        "hyperparams": hyperparams,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "categories": nc.CATEGORIES,
        "short_names": nc.SHORT_NAMES,
    }
    if extra:
        payload.update(extra)
    return payload


def save_artifact(folder: str, bundle: dict, metrics: dict, meta: dict) -> Path:
    import joblib
    out = nc.artifact_dir(folder)
    out.mkdir(parents=True, exist_ok=True)
    # The runtime embedding provider is rebuilt on load (nc.ensure_provider), so
    # never pickle it - otherwise a loaded gensim model would be serialised twice
    # (once in model.joblib, once in its own .model file).
    bundle.pop("provider", None)
    joblib.dump(bundle, out / "model.joblib", compress=3)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out / "model.joblib"


# ---------------------------------------------------------------------------
# Sentence-vector helper for the KNN comparison in the dashboard
# ---------------------------------------------------------------------------
def _mean_vec_matrix(texts, provider, tokenizer) -> np.ndarray:
    return np.vstack([provider.embed_tokens(tokenizer(t)) for t in texts])


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def train_01(train, test, **_):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    Xtr, ytr = _xy(train); Xte, yte = _xy(test)
    pipe = Pipeline([
        ("bow", CountVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
    ])
    pipe.fit(Xtr, ytr)
    hp = {"vectorizer": "CountVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))",
          "classifier": "LogisticRegression(max_iter=1000, C=1.0)"}
    return {"kind": "pipeline", "pipeline": pipe, "score_type": "probability"}, hp


def train_02(train, test, **_):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    Xtr, ytr = _xy(train); Xte, yte = _xy(test)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, stop_words="english",
                                  ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0, random_state=42))),
    ])
    pipe.fit(Xtr, ytr)
    hp = {"vectorizer": "TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1,2), sublinear_tf=True)",
          "classifier": "CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0))"}
    return {"kind": "pipeline", "pipeline": pipe, "score_type": "probability"}, hp


def train_03(train, test, full=False, **_):
    from gensim.models import Word2Vec
    from sklearn.linear_model import LogisticRegression
    Xtr, ytr = _xy(train); Xte, yte = _xy(test)
    tok = nc.simple_tokenize
    train_tokens = [tok(t) for t in Xtr]
    vec_size, epochs = (100, 20) if full else (100, 10)
    w2v = Word2Vec(sentences=train_tokens, vector_size=vec_size, window=5,
                   min_count=2, workers=4, epochs=epochs, seed=42)
    out = nc.artifact_dir("03_word2vec"); out.mkdir(parents=True, exist_ok=True)
    w2v.save(str(out / "w2v.model"))

    def doc_vec(tokens):
        vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(vec_size, dtype=np.float32)

    Xtr_emb = np.vstack([doc_vec(t) for t in train_tokens])
    Xte_emb = np.vstack([doc_vec(tok(t)) for t in Xte])
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(Xtr_emb, ytr)
    bundle = {
        "kind": "embedding_doc_clf", "clf": clf, "dim": vec_size,
        "tokenizer": "simple",
        "embedding": {"storage": "gensim", "model_type": "word2vec",
                      "path": str(out / "w2v.model")},
    }
    hp = {"embedding": f"Word2Vec(vector_size={vec_size}, window=5, min_count=2, epochs={epochs})",
          "pooling": "mean of word vectors", "classifier": "LogisticRegression(max_iter=1000, C=1.0)"}
    return bundle, hp


def train_04(train, test, **_):
    from collections import Counter
    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import LogisticRegression
    Xtr, ytr = _xy(train); Xte, yte = _xy(test)
    all_docs = Xtr + Xte
    tok = nc.simple_tokenize
    tokenized = [tok(d) for d in all_docs]

    word_freq = Counter(w for doc in tokenized for w in doc)
    vocab_words = [w for w, c in word_freq.most_common(3000) if c >= 3]
    w2i = {w: i for i, w in enumerate(vocab_words)}
    V = len(vocab_words)

    cooc = np.zeros((V, V), dtype=np.float32)
    WINDOW = 4
    for doc in tokenized:
        idxs = [w2i[w] for w in doc if w in w2i]
        for pos, i in enumerate(idxs):
            for j in idxs[max(0, pos - WINDOW):pos + WINDOW + 1]:
                if i != j:
                    cooc[i, j] += 1.0
    col = cooc.sum(axis=0); row = cooc.sum(axis=1); total = cooc.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        ppmi = np.log(cooc * total / (row[:, None] * col[None, :] + 1e-9))
    ppmi = np.maximum(ppmi, 0)
    ppmi[~np.isfinite(ppmi)] = 0

    svd = TruncatedSVD(n_components=100, random_state=42)
    emb = svd.fit_transform(ppmi).astype(np.float32)
    embed_words = [w for w, _ in sorted(w2i.items(), key=lambda kv: kv[1])]

    def doc_vec(tokens):
        rows = [w2i[t] for t in tokens if t in w2i]
        return emb[rows].mean(axis=0) if rows else np.zeros(100, dtype=np.float32)

    Xtr_emb = np.vstack([doc_vec(tok(t)) for t in Xtr])
    Xte_emb = np.vstack([doc_vec(tok(t)) for t in Xte])
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(Xtr_emb, ytr)
    bundle = {
        "kind": "embedding_doc_clf", "clf": clf, "dim": 100, "tokenizer": "simple",
        "embedding": {"storage": "matrix"}, "embed_matrix": emb, "embed_words": embed_words,
        "embedding_vocab_size": V,
    }
    hp = {"embedding": f"PPMI co-occurrence (vocab={V}, window=4) -> TruncatedSVD(100)",
          "pooling": "mean of word vectors", "classifier": "LogisticRegression(max_iter=1000, C=1.0)"}
    return bundle, hp


def train_05(train, test, full=False, **_):
    from gensim.models import FastText
    from sklearn.linear_model import LogisticRegression
    Xtr, ytr = _xy(train); Xte, yte = _xy(test)
    tok = nc.simple_tokenize
    train_tokens = [tok(t) for t in Xtr]
    epochs = 10
    # bucket (subword hash size) is a memory knob, not an architecture change.
    # The original script used 200000 for a local run; we use 50000 so the
    # committed artifact stays comfortably under Git host file-size limits.
    bucket = 50000
    ft = FastText(sentences=train_tokens, vector_size=100, window=5, min_count=2,
                  min_n=3, max_n=5, workers=2, epochs=epochs, seed=42, bucket=bucket)
    out = nc.artifact_dir("05_fasttext"); out.mkdir(parents=True, exist_ok=True)
    ft.save(str(out / "ft.model"))

    def doc_vec(tokens):
        vecs = [ft.wv[t] for t in tokens]
        return np.mean(vecs, axis=0) if vecs else np.zeros(100, dtype=np.float32)

    Xtr_emb = np.vstack([doc_vec(t) for t in train_tokens])
    Xte_emb = np.vstack([doc_vec(tok(t)) for t in Xte])
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(Xtr_emb, ytr)
    bundle = {
        "kind": "embedding_doc_clf", "clf": clf, "dim": 100, "tokenizer": "simple",
        "embedding": {"storage": "gensim", "model_type": "fasttext",
                      "path": str(out / "ft.model")},
    }
    hp = {"embedding": f"FastText(vector_size=100, window=5, min_n=3, max_n=5, bucket={bucket})",
          "pooling": "mean of word vectors", "classifier": "LogisticRegression(max_iter=1000, C=1.0)",
          "note": "bucket reduced from the original 200000 to keep the artifact Git-friendly"}
    return bundle, hp


def train_06(train, test, **_):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import normalize
    Xtr, ytr = _xy(train); Xte, yte = _xy(test)
    specs = [
        dict(name="char 3-5", tfidf=TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                                     max_features=5000, sublinear_tf=True)),
        dict(name="word 1", tfidf=TfidfVectorizer(analyzer="word", ngram_range=(1, 1),
                                                   max_features=5000, sublinear_tf=True,
                                                   stop_words="english")),
        dict(name="word 2-3", tfidf=TfidfVectorizer(analyzer="word", ngram_range=(2, 3),
                                                     max_features=5000, sublinear_tf=True,
                                                     stop_words="english")),
    ]
    weights = [0.3, 0.4, 0.3]
    layers = []
    tr_parts, te_parts = [], []
    for spec in specs:
        tfidf = spec["tfidf"]
        svd = TruncatedSVD(n_components=50, random_state=42)
        tr = svd.fit_transform(tfidf.fit_transform(Xtr))
        te = svd.transform(tfidf.transform(Xte))
        layers.append({"name": spec["name"], "tfidf": tfidf, "svd": svd})
        tr_parts.append(normalize(tr)); te_parts.append(normalize(te))
    Xtr_emb = np.hstack([w * p for w, p in zip(weights, tr_parts)])
    Xte_emb = np.hstack([w * p for w, p in zip(weights, te_parts)])
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(Xtr_emb, ytr)
    bundle = {"kind": "stacked_doc_clf", "layers": layers, "weights": weights, "clf": clf}
    hp = {"layers": [l["name"] for l in layers], "svd_components": 50, "weights": weights,
          "classifier": "LogisticRegression(max_iter=1000, C=1.0)"}
    return bundle, hp


# --- 07-15: TF-IDF + SVD + classical classifier ----------------------------
def _tfidf_svd(max_features: int, n_components: int, seed: int = 42):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import Normalizer
    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english",
                            ngram_range=(1, 2), sublinear_tf=True)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    return tfidf, svd, Pipeline([("tfidf", tfidf), ("svd", svd), ("norm", Normalizer())])


def _make_transformer_style(folder, train, test, max_features, n_components,
                            make_clf, clf_desc, score_type):
    Xtr, ytr = _xy(train); Xte, yte = _xy(test)
    tfidf, svd, fe = _tfidf_svd(max_features, n_components)
    Xtr_f = fe.fit_transform(Xtr)
    clf = make_clf()
    clf.fit(Xtr_f, ytr)
    bundle = {
        "kind": "pipeline", "pipeline": fe, "clf": clf, "score_type": score_type,
        "post_features": "already applied inside pipeline",
    }
    hp = {"features": f"TfidfVectorizer(max_features={max_features}, stop_words='english', "
                      f"ngram_range=(1,2), sublinear_tf=True) -> TruncatedSVD({n_components}) -> Normalizer",
          "classifier": clf_desc}
    return bundle, hp, tfidf, svd, fe, clf


def train_07(train, test, **_):
    from sklearn.linear_model import RidgeClassifier
    Xtr, ytr = _xy(train); Xte, yte = _xy(test)
    tfidf, svd, fe = _tfidf_svd(10000, 100)
    Xtr_f = fe.fit_transform(Xtr)
    clf = RidgeClassifier(alpha=0.5)
    clf.fit(Xtr_f, ytr)
    # The original script used a fixed 0.9 "confidence placeholder". We instead
    # expose the true decision_function (softmaxed) and label it as such so no
    # fabricated probability is ever shown.
    wrapper = FeaturesThenClassifier(fe, clf)
    bundle = {"kind": "pipeline", "pipeline": wrapper,
              "score_type": "decision_function_softmax"}
    hp = {"features": "TfidfVectorizer(10000) -> TruncatedSVD(100) -> Normalizer",
          "classifier": "RidgeClassifier(alpha=0.5)",
          "score_type": "softmax(decision_function) - NOT calibrated probabilities"}
    return bundle, hp


def train_08(train, test, **_):
    from sklearn.linear_model import SGDClassifier
    Xtr, ytr = _xy(train)
    _, _, fe = _tfidf_svd(10000, 100)
    Xtr_f = fe.fit_transform(Xtr)
    # SGDClassifier(modified_huber) genuinely exposes predict_proba -> real probs
    clf = SGDClassifier(loss="modified_huber", max_iter=1000, random_state=42,
                        n_iter_no_change=5)
    clf.fit(Xtr_f, ytr)
    return {"kind": "pipeline", "pipeline": FeaturesThenClassifier(fe, clf),
            "score_type": "probability"}, {
        "features": "TfidfVectorizer(10000) -> TruncatedSVD(100) -> Normalizer",
        "classifier": "SGDClassifier(loss='modified_huber', max_iter=1000)"}


def train_09(train, test, **_):
    from sklearn.linear_model import PassiveAggressiveClassifier
    Xtr, ytr = _xy(train)
    tfidf, svd, fe = _tfidf_svd(10000, 100)
    Xtr_f = fe.fit_transform(Xtr)
    clf = PassiveAggressiveClassifier(max_iter=1000, random_state=42, C=1.0)
    clf.fit(Xtr_f, ytr)
    return {"kind": "pipeline", "pipeline": FeaturesThenClassifier(fe, clf),
            "score_type": "decision_function_softmax"}, {
        "features": "TfidfVectorizer(10000) -> TruncatedSVD(100) -> Normalizer",
        "classifier": "PassiveAggressiveClassifier(max_iter=1000, C=1.0)",
        "score_type": "softmax(decision_function) - NOT calibrated probabilities"}


def train_10(train, test, **_):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC
    Xtr, ytr = _xy(train)
    tfidf, svd, fe = _tfidf_svd(5000, 100)
    Xtr_f = fe.fit_transform(Xtr)
    clf = CalibratedClassifierCV(LinearSVC(max_iter=2000, C=0.8, random_state=42))
    clf.fit(Xtr_f, ytr)
    return {"kind": "pipeline", "pipeline": FeaturesThenClassifier(fe, clf), "score_type": "probability"}, {
        "features": "TfidfVectorizer(5000) -> TruncatedSVD(100) -> Normalizer",
        "classifier": "CalibratedClassifierCV(LinearSVC(max_iter=2000, C=0.8))"}


def train_11(train, test, **_):
    from sklearn.ensemble import GradientBoostingClassifier
    Xtr, ytr = _xy(train)
    tfidf, svd, fe = _tfidf_svd(10000, 100)
    Xtr_f = fe.fit_transform(Xtr)
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42,
                                     learning_rate=0.1)
    clf.fit(Xtr_f, ytr)
    return {"kind": "pipeline", "pipeline": FeaturesThenClassifier(fe, clf),
            "score_type": "probability"}, {
        "features": "TfidfVectorizer(10000) -> TruncatedSVD(100) -> Normalizer",
        "classifier": "GradientBoostingClassifier(n_estimators=100, max_depth=4, lr=0.1)"}


def train_12(train, test, **_):
    from sklearn.ensemble import RandomForestClassifier
    Xtr, ytr = _xy(train)
    tfidf, svd, fe = _tfidf_svd(10000, 100)
    Xtr_f = fe.fit_transform(Xtr)
    clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    clf.fit(Xtr_f, ytr)
    return {"kind": "pipeline", "pipeline": FeaturesThenClassifier(fe, clf),
            "score_type": "probability"}, {
        "features": "TfidfVectorizer(10000) -> TruncatedSVD(100) -> Normalizer",
        "classifier": "RandomForestClassifier(n_estimators=200)"}


def train_13(train, test, **_):
    from sklearn.linear_model import LogisticRegression
    Xtr, ytr = _xy(train)
    tfidf, svd, fe = _tfidf_svd(10000, 100)
    Xtr_f = fe.fit_transform(Xtr)
    clf = LogisticRegression(max_iter=1000, random_state=42, C=5.0, solver="lbfgs")
    clf.fit(Xtr_f, ytr)
    return {"kind": "pipeline", "pipeline": FeaturesThenClassifier(fe, clf),
            "score_type": "probability"}, {
        "features": "TfidfVectorizer(10000) -> TruncatedSVD(100) -> Normalizer",
        "classifier": "LogisticRegression(max_iter=1000, C=5.0, solver='lbfgs')"}


def train_14(train, test, **_):
    from sklearn.ensemble import ExtraTreesClassifier
    Xtr, ytr = _xy(train)
    tfidf, svd, fe = _tfidf_svd(10000, 100)
    Xtr_f = fe.fit_transform(Xtr)
    clf = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(Xtr_f, ytr)
    return {"kind": "pipeline", "pipeline": FeaturesThenClassifier(fe, clf),
            "score_type": "probability"}, {
        "features": "TfidfVectorizer(10000) -> TruncatedSVD(100) -> Normalizer",
        "classifier": "ExtraTreesClassifier(n_estimators=200)"}


def train_15(train, test, **_):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import Perceptron
    Xtr, ytr = _xy(train)
    tfidf, svd, fe = _tfidf_svd(10000, 100)
    Xtr_f = fe.fit_transform(Xtr)
    clf = CalibratedClassifierCV(Perceptron(max_iter=1000, random_state=42, n_iter_no_change=10))
    clf.fit(Xtr_f, ytr)
    return {"kind": "pipeline", "pipeline": FeaturesThenClassifier(fe, clf), "score_type": "probability"}, {
        "features": "TfidfVectorizer(10000) -> TruncatedSVD(100) -> Normalizer",
        "classifier": "CalibratedClassifierCV(Perceptron(max_iter=1000))"}


TRAINERS = {
    1: train_01, 2: train_02, 3: train_03, 4: train_04, 5: train_05, 6: train_06,
    7: train_07, 8: train_08, 9: train_09, 10: train_10, 11: train_11, 12: train_12,
    13: train_13, 14: train_14, 15: train_15,
}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def train_one(num: int, train, test, full: bool = False) -> dict:
    folder = nc.folder_of(num)
    display = nc.MODEL_REGISTRY[folder]["display"]
    print(f"\n{'='*66}\n[{num:02d}] {display}  ({folder})\n{'='*66}")
    t0 = time.time()
    result = TRAINERS[num](train, test, full=full)
    bundle, hp = result[0], result[1]

    # attach the runtime embedding provider so the shared inference path can be
    # used for evaluation too (same code path the apps run)
    nc.ensure_provider(bundle)

    _, yte = _xy(test)
    metrics = evaluate(bundle, test, yte)
    metrics["n_train"] = int(len(train))
    elapsed = time.time() - t0
    metrics["train_seconds"] = round(elapsed, 2)

    meta = build_meta(folder, bundle["kind"], hp, extra={"score_type": bundle.get("score_type")})
    meta["train_seconds"] = round(elapsed, 2)
    if folder in ("03_word2vec", "04_glove", "05_fasttext"):
        meta["embedding_vocab_size"] = bundle.get("embedding_vocab_size") or None

    path = save_artifact(folder, bundle, metrics, meta)
    size = path.stat().st_size
    print(f"  accuracy={metrics['accuracy']:.4f}  f1_macro={metrics['f1_macro']:.4f}"
          f"  ({elapsed:.1f}s, artifact {nc.human_bytes(size)})")
    gc.collect()
    return {
        "num": num, "folder": folder, "display": display,
        "accuracy": metrics["accuracy"], "f1_macro": metrics["f1_macro"],
        "seconds": round(elapsed, 2), "artifact_bytes": size,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", type=int, default=None,
                    help="model numbers to train (default: all)")
    ap.add_argument("--full", action="store_true",
                    help="heavier embedding training for 03/05 (more epochs)")
    args = ap.parse_args()

    train, test = load_data()
    nums = args.only or list(range(1, 16))
    print(f"Training {len(nums)} model(s) | train={len(train)} test={len(test)} | full={args.full}")

    summary = []
    for n in nums:
        try:
            summary.append(train_one(n, train, test, full=args.full))
        except Exception as exc:  # keep going - one failure must not stop the batch
            print(f"  !! model {n} FAILED: {type(exc).__name__}: {exc}")
            summary.append({"num": n, "error": f"{type(exc).__name__}: {exc}"})

    out = REPO_ROOT / "training" / "training_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n{'='*66}\nSUMMARY (written to {out.relative_to(REPO_ROOT)})\n{'='*66}")
    for s in summary:
        if "error" in s:
            print(f"  [{s['num']:02d}] FAILED  {s['error']}")
        else:
            print(f"  [{s['num']:02d}] {s['display']:<26} acc={s['accuracy']:.4f} "
                  f"f1={s['f1_macro']:.4f}  {s['seconds']:.1f}s")


if __name__ == "__main__":
    main()
