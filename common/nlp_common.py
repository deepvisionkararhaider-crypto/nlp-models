"""
Shared, framework-light helpers for all 15 NLP demos.

This module is intentionally importable **without** Streamlit so that the
training scripts, the test harness and the Streamlit apps can all share exactly
one definition of:

* the class / label mapping used by every model,
* each model's *honest* metadata (what it actually is vs. what it is named),
* how to load an artifact bundle from disk,
* how to turn raw user text into a prediction.

Nothing here fabricates outputs.  When a model cannot produce calibrated
probabilities we expose the real decision function and say so.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COMMON_DIR = Path(__file__).resolve().parent
REPO_ROOT = COMMON_DIR.parent
DATA_DIR = REPO_ROOT / "data" / "20newsgroups_4cat"

# ---------------------------------------------------------------------------
# Label space (identical to the original model scripts)
# ---------------------------------------------------------------------------
CATEGORIES = [
    "rec.sport.hockey",
    "sci.space",
    "comp.graphics",
    "talk.politics.misc",
]
SHORT_NAMES = ["hockey", "space", "graphics", "politics"]
CATEGORY_DESCRIPTIONS = {
    "rec.sport.hockey": "Ice-hockey / NHL discussion (rec.sport.hockey)",
    "sci.space": "Space exploration, astronomy, NASA (sci.space)",
    "comp.graphics": "Computer graphics, rendering, image formats (comp.graphics)",
    "talk.politics.misc": "General political discussion (talk.politics.misc)",
}
LABEL_MAP = {i: c for i, c in enumerate(CATEGORIES)}

TASK_TOPIC_CLASSIFICATION = "Topic Classification (20-Newsgroups, 4 classes)"

# ---------------------------------------------------------------------------
# Tokenisation (a re-implementation of ``gensim.utils.simple_preprocess``)
# ---------------------------------------------------------------------------
# We use THIS tokenizer both when training the embedding models and when running
# inference so the two are always consistent, and so the deployed apps do not
# need gensim at runtime.  Semantics mirror gensim's simple_preprocess:
# lowercase, optional de-accenting, word-character runs, length filtering.
_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)


def simple_tokenize(text: str, min_len: int = 2, max_len: int = 15,
                    deacc: bool = True) -> list[str]:
    if not text:
        return []
    text = str(text).lower()
    if deacc:
        text = unicodedata.normalize("NFKD", text)
    out = []
    for tok in _TOKEN_RE.findall(text):
        tok = tok.strip("'")
        if min_len <= len(tok) <= max_len:
            out.append(tok)
    return out


TOKENIZERS = {"simple": simple_tokenize}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# IMPORTANT / HONESTY NOTE
# ------------------------
# Folders 07-15 are *named* after transformer architectures (BERT, RoBERTa, ...)
# but the code in this repository never downloads or trains a transformer.  They
# are TF-IDF -> TruncatedSVD -> classical-classifier pipelines.  We keep the
# original folder names and original architectures (as requested) but label them
# truthfully everywhere a student can see them: `display` vs `actual`.
#
# `faithful_family` is True only where the folder name matches the technique
# that is actually implemented.
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "01_bag_of_words": {
        "num": 1, "display": "Bag of Words", "short": "BoW",
        "actual": "CountVectorizer (5k features, 1-2 grams, English stop-words) + Logistic Regression",
        "family": "Classical sparse features",
        "faithful_family": True,
        "framework": "scikit-learn",
        "needs_gensim": False,
        "training_required": True,
        "accent": "#2563eb",
        "description": (
            "Classic bag-of-words representation: text becomes a sparse matrix of "
            "unigram+bigram counts, then a logistic-regression classifier predicts "
            "the newsgroup topic."
        ),
    },
    "02_tfidf": {
        "num": 2, "display": "TF-IDF", "short": "TF-IDF",
        "actual": "TfidfVectorizer (10k features, 1-2 grams, sublinear) + Calibrated LinearSVC",
        "family": "Classical sparse features",
        "faithful_family": True,
        "framework": "scikit-learn",
        "needs_gensim": False,
        "training_required": True,
        "accent": "#0ea5e9",
        "description": (
            "TF-IDF down-weights common words and up-weights topic-specific ones; "
            "a calibrated linear SVM then separates the four topics."
        ),
    },
    "03_word2vec": {
        "num": 3, "display": "Word2Vec", "short": "Word2Vec",
        "actual": "gensim Word2Vec (100-d, window 5, skip-gram-family CBOW default) mean-pooled + Logistic Regression",
        "family": "Static word embeddings",
        "faithful_family": True,
        "framework": "gensim + scikit-learn",
        "needs_gensim": True,
        "training_required": True,
        "accent": "#7c3aed",
        "description": (
            "A Word2Vec embedding is trained on the corpus; each document becomes "
            "the mean of its word vectors, and a logistic regression classifies "
            "that dense vector. Inspect nearest neighbours of any word."
        ),
    },
    "04_glove": {
        "num": 4, "display": "GloVe", "short": "GloVe",
        "actual": "PPMI co-occurrence matrix (3k vocab, window 4) -> TruncatedSVD(100) mean-pooled + Logistic Regression",
        "family": "Static word embeddings",
        "faithful_family": True,
        "framework": "numpy + scikit-learn",
        "needs_gensim": False,
        "training_required": True,
        "accent": "#db2777",
        "description": (
            "GloVe learns word vectors by factorising a global word-word "
            "co-occurrence matrix. This project builds the positive-PMI matrix and "
            "factorises it with SVD - the standard GloVe formulation - then "
            "mean-pools the vectors per document."
        ),
    },
    "05_fasttext": {
        "num": 5, "display": "FastText", "short": "FastText",
        "actual": "gensim FastText (100-d, subword 3-5 grams) mean-pooled + Logistic Regression",
        "family": "Static word embeddings",
        "faithful_family": True,
        "framework": "gensim + scikit-learn",
        "needs_gensim": True,
        "training_required": True,
        "accent": "#ea580c",
        "description": (
            "FastText extends Word2Vec with subword (character n-gram) vectors, so "
            "it can embed words it has never seen. Mean-pooled document vectors "
            "feed a logistic regression."
        ),
    },
    "06_elmo": {
        "num": 6, "display": "ELMo (3-layer contextual)", "short": "ELMo",
        "actual": "3 stacked TF-IDF->SVD(50) representations (char 3-5 / word 1 / word 2-3), fixed weights .3/.4/.3 + Logistic Regression",
        "family": "Contextual feature stacking (not a real biLM)",
        "faithful_family": False,
        "framework": "scikit-learn",
        "needs_gensim": False,
        "training_required": True,
        "accent": "#059669",
        "description": (
            "ELMo layers several representation levels and weights them. This "
            "project mimics that idea with three TF-IDF+SVD 'layers' (character, "
            "word, bigram) combined with fixed weights."
        ),
        "caveat": (
            "This is NOT a real ELMo biLM (the 8 GB AllenNLP model is not used). "
            "It is a 3-layer feature stack inspired by ELMo's multi-level idea."
        ),
    },
    "07_bert": {
        "num": 7, "display": "BERT", "short": "BERT",
        "actual": "TF-IDF (10k, 1-2 grams) -> TruncatedSVD(100) -> RidgeClassifier",
        "family": "TF-IDF + SVD + classical classifier (NOT a transformer)",
        "faithful_family": False,
        "framework": "scikit-learn",
        "needs_gensim": False,
        "training_required": True,
        "accent": "#1d4ed8",
        "description": (
            "Named after BERT's bidirectional encoder; implemented here as TF-IDF "
            "features compressed with SVD, then a Ridge classifier."
        ),
        "caveat": (
            "No transformer is used. The folder name references BERT's design idea "
            "(bidirectional context); the implementation is a classical TF-IDF + "
            "SVD + Ridge pipeline."
        ),
    },
    "08_roberta": {
        "num": 8, "display": "RoBERTa", "short": "RoBERTa",
        "actual": "TF-IDF (10k, 1-2 grams) -> TruncatedSVD(100) -> SGDClassifier (modified_huber)",
        "family": "TF-IDF + SVD + classical classifier (NOT a transformer)",
        "faithful_family": False,
        "framework": "scikit-learn",
        "needs_gensim": False,
        "training_required": True,
        "accent": "#b91c1c",
        "description": (
            "Named after RoBERTa's robust training; implemented as TF-IDF + SVD "
            "with an SGD classifier using the modified-huber loss."
        ),
        "caveat": (
            "No transformer is used. The name references RoBERTa's robust training "
            "objective; the implementation is classical."
        ),
    },
    "09_albert": {
        "num": 9, "display": "ALBERT", "short": "ALBERT",
        "actual": "TF-IDF (10k, 1-2 grams) -> TruncatedSVD(100) -> PassiveAggressiveClassifier",
        "family": "TF-IDF + SVD + classical classifier (NOT a transformer)",
        "faithful_family": False,
        "framework": "scikit-learn",
        "needs_gensim": False,
        "training_required": True,
        "accent": "#0891b2",
        "description": (
            "Named after ALBERT's parameter-efficiency; implemented as TF-IDF + SVD "
            "with a Passive-Aggressive online learner."
        ),
        "caveat": (
            "No transformer is used. The name references ALBERT's efficient design; "
            "the implementation is classical."
        ),
    },
    "10_distilbert": {
        "num": 10, "display": "DistilBERT", "short": "DistilBERT",
        "actual": "TF-IDF (5k, 1-2 grams) -> TruncatedSVD(100) -> Calibrated LinearSVC",
        "family": "TF-IDF + SVD + classical classifier (NOT a transformer)",
        "faithful_family": False,
        "framework": "scikit-learn",
        "needs_gensim": False,
        "training_required": True,
        "accent": "#065f46",
        "description": (
            "Named after DistilBERT's compression; implemented with a smaller "
            "TF-IDF vocabulary (5k) + SVD + a calibrated linear SVM."
        ),
        "caveat": (
            "No transformer is used. The name references DistilBERT's distillation; "
            "the implementation is classical."
        ),
    },
    "11_xlnet": {
        "num": 11, "display": "XLNet", "short": "XLNet",
        "actual": "TF-IDF (10k, 1-2 grams) -> TruncatedSVD(100) -> GradientBoostingClassifier",
        "family": "TF-IDF + SVD + classical classifier (NOT a transformer)",
        "faithful_family": False,
        "framework": "scikit-learn",
        "needs_gensim": False,
        "training_required": True,
        "accent": "#9333ea",
        "description": (
            "Named after XLNet's autoregressive modelling; implemented as TF-IDF + "
            "SVD with a gradient-boosted tree ensemble."
        ),
        "caveat": (
            "No transformer is used. The name references XLNet's design; the "
            "implementation is classical."
        ),
    },
    "12_t5": {
        "num": 12, "display": "T5", "short": "T5",
        "actual": "TF-IDF (10k, 1-2 grams) -> TruncatedSVD(100) -> RandomForestClassifier",
        "family": "TF-IDF + SVD + classical classifier (NOT a transformer)",
        "faithful_family": False,
        "framework": "scikit-learn",
        "needs_gensim": False,
        "training_required": True,
        "accent": "#ca8a04",
        "description": (
            "Named after T5's text-to-text framing; implemented as TF-IDF + SVD "
            "with a random-forest ensemble."
        ),
        "caveat": (
            "No transformer is used. The name references T5's design; the "
            "implementation is classical."
        ),
    },
    "13_gpt": {
        "num": 13, "display": "GPT", "short": "GPT",
        "actual": "TF-IDF (10k, 1-2 grams) -> TruncatedSVD(100) -> Multinomial Logistic Regression",
        "family": "TF-IDF + SVD + classical classifier (NOT a transformer)",
        "faithful_family": False,
        "framework": "scikit-learn",
        "needs_gensim": False,
        "training_required": True,
        "accent": "#16a34a",
        "description": (
            "Named after GPT's autoregressive decoding; implemented as TF-IDF + SVD "
            "with a multinomial logistic regression."
        ),
        "caveat": (
            "No transformer is used. The name references GPT's design; the "
            "implementation is classical."
        ),
    },
    "14_bart": {
        "num": 14, "display": "BART", "short": "BART",
        "actual": "TF-IDF (10k, 1-2 grams) -> TruncatedSVD(100) -> ExtraTreesClassifier",
        "family": "TF-IDF + SVD + classical classifier (NOT a transformer)",
        "faithful_family": False,
        "framework": "scikit-learn",
        "needs_gensim": False,
        "training_required": True,
        "accent": "#e11d48",
        "description": (
            "Named after BART's encoder-decoder hybrid; implemented as TF-IDF + SVD "
            "with an extremely-randomised tree ensemble."
        ),
        "caveat": (
            "No transformer is used. The name references BART's design; the "
            "implementation is classical."
        ),
    },
    "15_electra": {
        "num": 15, "display": "ELECTRA", "short": "ELECTRA",
        "actual": "TF-IDF (10k, 1-2 grams) -> TruncatedSVD(100) -> Calibrated Perceptron",
        "family": "TF-IDF + SVD + classical classifier (NOT a transformer)",
        "faithful_family": False,
        "framework": "scikit-learn",
        "needs_gensim": False,
        "training_required": True,
        "accent": "#0d9488",
        "description": (
            "Named after ELECTRA's discriminative token-detection objective; "
            "implemented as TF-IDF + SVD with a calibrated perceptron."
        ),
        "caveat": (
            "No transformer is used. The name references ELECTRA's discriminator "
            "idea; the implementation is classical."
        ),
    },
}


def list_models() -> list[dict[str, Any]]:
    """Return the registry as a list sorted by model number."""
    return sorted(MODEL_REGISTRY.values(), key=lambda m: m["num"])


def folder_of(num_or_key: str | int) -> str:
    """Resolve a model number (7 / '7' / '07_bert') to its folder name."""
    if isinstance(num_or_key, str) and num_or_key in MODEL_REGISTRY:
        return num_or_key
    n = int(num_or_key)
    for folder, meta in MODEL_REGISTRY.items():
        if meta["num"] == n:
            return folder
    raise KeyError(f"unknown model: {num_or_key}")


def artifact_dir(folder: str) -> Path:
    folder = folder if folder in MODEL_REGISTRY else folder_of(folder)
    return REPO_ROOT / folder / "artifacts"


# ---------------------------------------------------------------------------
# Embedding providers (make 03/04/05 give a uniform, portable interface)
# ---------------------------------------------------------------------------
class MatrixEmbeddingProvider:
    """Numpy-only word vectors (used for the GloVe-style model)."""

    def __init__(self, matrix: np.ndarray, words: list[str]):
        self.matrix = np.asarray(matrix, dtype=np.float32)
        self.words = list(words)
        self.vocab = {w: i for i, w in enumerate(self.words)}
        self.dim = int(self.matrix.shape[1])
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.normed = self.matrix / norms

    def has(self, word: str) -> bool:
        return word in self.vocab

    def embed_tokens(self, tokens: list[str]) -> np.ndarray:
        rows = [self.vocab[t] for t in tokens if t in self.vocab]
        if not rows:
            return np.zeros(self.dim, dtype=np.float32)
        return self.matrix[rows].mean(axis=0)

    def vector(self, word: str) -> np.ndarray | None:
        i = self.vocab.get(word)
        return None if i is None else self.matrix[i]

    def most_similar(self, word: str, topn: int = 8) -> list[tuple[str, float]]:
        i = self.vocab.get(word)
        if i is None:
            return []
        sims = self.normed @ self.normed[i]
        order = np.argsort(-sims)
        out: list[tuple[str, float]] = []
        for r in order:
            r = int(r)
            if r == i:
                continue
            out.append((self.words[r], float(sims[r])))
            if len(out) >= topn:
                break
        return out


class FeaturesThenClassifier:
    """Picklable ``features -> classifier`` wrapper.

    Some of the transformer-named projects feed TF-IDF->SVD features into a
    classifier that has *no* native ``predict_proba`` (Ridge / Passive-Aggressive).
    Keeping both steps inside one module-level object means the artifact can be
    unpickled from any process (training script, test harness, Streamlit app) and
    lets the app label the numbers honestly - probabilities vs a softmaxed
    decision function.

    It lives here (a stable import path, ``common.nlp_common``) rather than in the
    training entry point so that unpickling never depends on ``__main__``.
    """

    def __init__(self, fe, clf):
        self.fe = fe
        self.clf = clf

    def predict(self, X):
        return self.clf.predict(self.fe.transform(X))

    def predict_proba(self, X):
        if not hasattr(self.clf, "predict_proba"):
            raise AttributeError("underlying classifier has no predict_proba")
        return self.clf.predict_proba(self.fe.transform(X))

    def decision_function(self, X):
        if not hasattr(self.clf, "decision_function"):
            raise AttributeError("underlying classifier has no decision_function")
        return self.clf.decision_function(self.fe.transform(X))


class GensimEmbeddingProvider:
    """gensim Word2Vec / FastText model loaded lazily from disk.

    FastText's provider can embed out-of-vocabulary words via subwords, which is
    the whole point of that model, so we keep the native gensim artifact.
    """

    def __init__(self, path: str, model_type: str):
        self.path = path
        self.model_type = model_type
        self._model = None

    def _load(self):
        if self._model is None:
            from gensim.models import FastText, Word2Vec
            cls = FastText if self.model_type == "fasttext" else Word2Vec
            self._model = cls.load(self.path)
        return self._model

    @property
    def wv(self):
        return self._load().wv

    @property
    def dim(self) -> int:
        return int(self.wv.vector_size)

    def has(self, word: str) -> bool:
        return word in self.wv.key_to_index

    def embed_tokens(self, tokens: list[str]) -> np.ndarray:
        vecs = [self.wv[t] for t in tokens if t in self.wv.key_to_index]
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        return np.asarray(vecs, dtype=np.float32).mean(axis=0)

    def vector(self, word: str) -> np.ndarray | None:
        if word not in self.wv.key_to_index:
            return None
        return np.asarray(self.wv[word], dtype=np.float32)

    def most_similar(self, word: str, topn: int = 8) -> list[tuple[str, float]]:
        if word not in self.wv.key_to_index:
            return []
        return [(w, float(s)) for w, s in self.wv.most_similar(word, topn=topn)]


def ensure_provider(bundle: dict) -> dict:
    """Attach the runtime embedding provider (idempotent)."""
    if bundle.get("kind") == "embedding_doc_clf" and "provider" not in bundle:
        spec = bundle.get("embedding", {})
        if spec.get("storage") == "gensim":
            bundle["provider"] = GensimEmbeddingProvider(spec["path"], spec["model_type"])
        else:
            bundle["provider"] = MatrixEmbeddingProvider(bundle["embed_matrix"],
                                                         bundle["embed_words"])
    return bundle


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------
_cached_bundles: dict[str, dict[str, Any]] = {}


class ArtifactMissingError(FileNotFoundError):
    """Raised when a model artifact has not been built/trained yet."""

    def __init__(self, folder: str, path: Path):
        self.folder = folder
        self.path = path
        meta = MODEL_REGISTRY.get(folder, {})
        self.message = (
            f"Model artifact not found for **{meta.get('display', folder)}**.\n\n"
            f"Expected file: `{path}`\n\n"
            "Build it locally with:\n"
            f"```bash\npython training/train_all.py --only {meta.get('num', '')}\n```\n"
            "or run the matching Colab notebook in `notebooks/` and place the "
            "exported `model.joblib` in this folder's `artifacts/` directory."
        )
        super().__init__(self.message)


def load_bundle(folder: str) -> dict[str, Any]:
    """Load ``<folder>/artifacts/model.joblib`` and attach its runtime provider.

    Never silently substitutes a different model - if the artifact is absent an
    :class:`ArtifactMissingError` is raised with setup instructions.
    """
    folder = folder if folder in MODEL_REGISTRY else folder_of(folder)
    if folder in _cached_bundles:
        return _cached_bundles[folder]

    path = artifact_dir(folder) / "model.joblib"
    if not path.exists():
        raise ArtifactMissingError(folder, path)

    import joblib  # local import keeps module import cheap

    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "kind" not in bundle:
        raise ValueError(f"{path} is not a recognised artifact bundle")

    ensure_provider(bundle)
    _cached_bundles[folder] = bundle
    return bundle


def artifacts_available(folder: str) -> bool:
    return (artifact_dir(folder) / "model.joblib").exists()


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    scores = scores - scores.max()
    exp = np.exp(scores)
    return exp / exp.sum()


def normalise_input(text: str) -> str:
    """Light, training-safe normalisation shared by every app.

    We deliberately do NOT lowercase / stem / remove stop-words here because the
    vectorizers were trained on the raw text (their own ``stop_words`` /
    ``analyzer`` settings handle that). We only collapse obviously broken
    whitespace so a pasted paragraph is handled gracefully.
    """
    if text is None:
        return ""
    text = str(text).replace("\t", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def _classifier_scores(clf, X) -> np.ndarray:
    """A 2-D ``(n_samples, n_classes)`` matrix of per-class scores.

    ``predict_proba`` when genuinely available; otherwise the real
    ``decision_function`` squashed through a softmax (flagged as such by the
    caller so the UI can say these are *not* calibrated probabilities).
    """
    if hasattr(clf, "predict_proba"):
        try:
            return np.asarray(clf.predict_proba(X))
        except Exception:
            pass
    if hasattr(clf, "decision_function"):
        df = np.asarray(clf.decision_function(X))
        if df.ndim == 1:
            df = np.hstack([-df.reshape(-1, 1), df.reshape(-1, 1)])
        return np.vstack([_softmax(row) for row in np.atleast_2d(df)])
    raise ValueError("estimator exposes neither predict_proba nor decision_function")


def _document_vector(bundle: dict, text: str) -> np.ndarray:
    provider = bundle["provider"]
    tokenize = TOKENIZERS[bundle.get("tokenizer", "simple")]
    return provider.embed_tokens(tokenize(text))


def _stack_document(bundle: dict, text: str) -> np.ndarray:
    layers, weights, parts = bundle["layers"], bundle["weights"], []
    for layer, weight in zip(layers, weights):
        mat = layer["tfidf"].transform([text])
        dense = layer["svd"].transform(mat)
        norm = np.linalg.norm(dense)
        if norm > 0:
            dense = dense / norm
        parts.append(weight * dense)
    return np.hstack(parts)


def _feature_matrix(bundle: dict, texts: list[str]):
    """Return ``(X, estimator)``: X is whatever the estimator actually consumes."""
    kind = bundle["kind"]
    if kind == "embedding_doc_clf":
        X = np.vstack([_document_vector(bundle, t) for t in texts])
        return X, bundle["clf"]
    if kind == "stacked_doc_clf":
        X = np.vstack([_stack_document(bundle, t) for t in texts])
        return X, bundle["clf"]
    return texts, bundle["pipeline"]  # 'pipeline' bundles fe+clf


def predict_many(bundle: dict, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised inference: returns ``(predicted_indices, score_matrix)``."""
    texts = [normalise_input(t) for t in texts]
    if not any(texts):
        raise ValueError("empty input")
    X, est = _feature_matrix(bundle, texts)
    idx = np.asarray(est.predict(X)).astype(int)
    scores = _classifier_scores(est, X)
    return idx, scores


def predict_text(folder: str, text: str, bundle: dict | None = None) -> dict[str, Any]:
    """Run genuine inference for ``folder`` on ``text``.

    Returns the predicted class, a full score table and a ``score_type`` stating
    exactly what those numbers are (``probability`` vs
    ``decision_function_softmax``). No hard-coded or random outputs exist on this
    path.
    """
    bundle = bundle or load_bundle(folder)
    if not normalise_input(text):
        raise ValueError("empty input")
    idx_arr, scores = predict_many(bundle, [text])
    idx = int(idx_arr[0])
    row = scores[0]
    rows = [
        {
            "label": LABEL_MAP.get(i, f"class_{i}"),
            "short": SHORT_NAMES[i] if i < len(SHORT_NAMES) else str(i),
            "score": float(row[i]),
        }
        for i in range(len(row))
    ]
    ranked = sorted(rows, key=lambda r: r["score"], reverse=True)
    return {
        "predicted_index": idx,
        "predicted_label": LABEL_MAP.get(idx, f"class_{idx}"),
        "predicted_short": SHORT_NAMES[idx] if idx < len(SHORT_NAMES) else str(idx),
        "scores": ranked,
        "score_type": bundle.get("score_type", "probability"),
        "model": bundle.get("meta", {}),
    }


# ---------------------------------------------------------------------------
# Word-similarity helpers (03 / 04 / 05 live demos)
# ---------------------------------------------------------------------------
def nearest_words(bundle: dict, word: str, topn: int = 8) -> list[tuple[str, float]]:
    provider = bundle.get("provider")
    if provider is None:
        return []
    return provider.most_similar(str(word).lower().strip(), topn=topn)


def word_vector(bundle: dict, word: str):
    provider = bundle.get("provider")
    if provider is None:
        return None
    return provider.vector(str(word).lower().strip())


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def load_metrics(folder: str) -> dict[str, Any] | None:
    path = artifact_dir(folder) / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def pretty_percent(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"


def human_bytes(n: int | float) -> str:
    n = float(n)
    if n < 1024:
        return f"{n:.0f} B"
    for unit in ("KB", "MB", "GB"):
        n /= 1024.0
        if n < 1024:
            return f"{n:.1f} {unit}"
    return f"{n:.1f} TB"
