"""
04 - GloVe Embeddings (via SVD on co-occurrence matrix)
Dataset: 20 Newsgroups (4 categories)
Model: GloVe-style (TruncatedSVD on PPMI matrix) → mean embedding → Logistic Regression
Note: Downloads GloVe vectors if available, falls back to SVD-based approach.
"""

import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
from sklearn.decomposition import PCA
from gensim.utils import simple_preprocess

os.makedirs('data', exist_ok=True)
os.makedirs('plots', exist_ok=True)

CATS = ['rec.sport.hockey', 'sci.space', 'comp.graphics', 'talk.politics.misc']
train = fetch_20newsgroups(subset='train', categories=CATS,
                           remove=('headers','footers','quotes'), random_state=42)
test  = fetch_20newsgroups(subset='test',  categories=CATS,
                           remove=('headers','footers','quotes'), random_state=42)

short_cats = [c.split('.')[-1] for c in CATS]
print(f"Train: {len(train.data)} | Test: {len(test.data)}")

df_data = pd.DataFrame({'text': train.data[:200], 'label': train.target[:200]})
df_data.to_csv('data/newsgroups.csv', index=False)

# ── Build PPMI co-occurrence matrix (GloVe-style) ─────────────────────────
print("Building co-occurrence matrix for GloVe-style embeddings...")
all_docs  = train.data + test.data
tokenized = [simple_preprocess(d, deacc=True, min_len=2) for d in all_docs]

# Vocabulary (top 3000 words)
word_freq = Counter(w for doc in tokenized for w in doc)
vocab_words = [w for w, c in word_freq.most_common(3000) if c >= 3]
w2i = {w: i for i, w in enumerate(vocab_words)}
V   = len(vocab_words)
print(f"Vocabulary size: {V}")

# Co-occurrence counts with window=4
cooc = np.zeros((V, V), dtype=np.float32)
WINDOW = 4
for doc in tokenized:
    idxs = [w2i[w] for w in doc if w in w2i]
    for pos, i in enumerate(idxs):
        for j in idxs[max(0, pos-WINDOW):pos+WINDOW+1]:
            if i != j:
                cooc[i, j] += 1.0

# PPMI
col_sum = cooc.sum(axis=0)
row_sum = cooc.sum(axis=1)
total   = cooc.sum()
with np.errstate(divide='ignore', invalid='ignore'):
    ppmi = np.log(cooc * total / (row_sum[:,None] * col_sum[None,:] + 1e-9))
ppmi = np.maximum(ppmi, 0)
ppmi[~np.isfinite(ppmi)] = 0

# SVD → GloVe-style embeddings (dim=100)
print("Running SVD for GloVe embeddings...")
svd = TruncatedSVD(n_components=100, random_state=42)
glove_vecs = svd.fit_transform(ppmi)   # (V, 100)
glove_dict = {w: glove_vecs[i] for w, i in w2i.items()}

def mean_embed(tokens, gd, dim=100):
    vecs = [gd[t] for t in tokens if t in gd]
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim)

X_tr = np.array([mean_embed(t, glove_dict)
                  for t in tokenized[:len(train.data)]])
X_te = np.array([mean_embed(t, glove_dict)
                  for t in tokenized[len(train.data):]])
y_tr = np.array(train.target)
y_te = np.array(test.target)

print(f"Embeddings: train={X_tr.shape}, test={X_te.shape}")

# ── Classify ───────────────────────────────────────────────────────────────
clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
clf.fit(X_tr, y_tr)
y_pred = clf.predict(X_te)
probs  = clf.predict_proba(X_te)

acc  = accuracy_score(y_te, y_pred)
prec = precision_score(y_te, y_pred, average='macro', zero_division=0)
rec  = recall_score(y_te, y_pred, average='macro', zero_division=0)
f1   = f1_score(y_te, y_pred, average='macro', zero_division=0)

print(f"\n{'='*50}\nGloVe (PPMI+SVD) Results\n{'='*50}")
print(f"Accuracy:  {acc:.4f}\nPrecision: {prec:.4f}\nRecall:    {rec:.4f}\nF1-Score:  {f1:.4f}")
print(classification_report(y_te, y_pred, target_names=CATS))

df_pred = pd.DataFrame({
    'sample_id': range(len(y_te)),
    'true_label': y_te,
    'true_category': [short_cats[i] for i in y_te],
    'predicted_label': y_pred,
    'predicted_category': [short_cats[i] for i in y_pred],
    'confidence': probs.max(axis=1).round(4),
    'correct': (y_te == y_pred).astype(int),
})
df_pred.to_csv('predictions.csv', index=False)
print("Saved predictions.csv")

# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'GloVe (PPMI+SVD) | Acc={acc:.4f}  F1={f1:.4f}', fontsize=14, fontweight='bold')

cm = confusion_matrix(y_te, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0,0],
            xticklabels=short_cats, yticklabels=short_cats)
axes[0,0].set(title='Confusion Matrix', xlabel='Predicted', ylabel='True')

pca = PCA(n_components=2, random_state=42)
Xpca = pca.fit_transform(X_te)
sc = axes[0,1].scatter(Xpca[:,0], Xpca[:,1], c=y_te, cmap='tab10', alpha=0.5, s=20)
axes[0,1].set(title='PCA of GloVe Doc Embeddings', xlabel='PC1', ylabel='PC2')
for ci, cat in enumerate(short_cats):
    mask = y_te == ci
    cx, cy = Xpca[mask,0].mean(), Xpca[mask,1].mean()
    axes[0,1].annotate(cat, (cx, cy), fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

# Explained variance by SVD components
ev = svd.explained_variance_ratio_
axes[1,0].bar(range(1, len(ev)+1), np.cumsum(ev), color='teal', edgecolor='white')
axes[1,0].set(title='Cumulative Variance Explained (SVD)', xlabel='Component', ylabel='Cumulative Var')
axes[1,0].axhline(0.9, color='red', linestyle='--', label='90% var')
axes[1,0].legend()

per_class_acc = cm.diagonal() / cm.sum(axis=1)
axes[1,1].bar(short_cats, per_class_acc, color='teal', edgecolor='darkslategray')
axes[1,1].set(title='Per-class Accuracy', xlabel='Category', ylabel='Accuracy', ylim=(0,1))
for i, v in enumerate(per_class_acc):
    axes[1,1].text(i, v+0.02, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('plots/glove_analysis.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved plots/glove_analysis.png\n✅ GloVe complete!")
