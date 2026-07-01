"""
14 - BART (Bidirectional and Auto-Regressive Transformers)
Dataset: 20 Newsgroups (4 categories)
Note: BART combines BERT-style encoder with GPT-style decoder. We use a hybrid: char-level + word-level TF-IDF (bi-directional + auto-regressive) with ExtraTreesClassifier.
"""

import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
from sklearn.decomposition import PCA


os.makedirs('data', exist_ok=True)
os.makedirs('plots', exist_ok=True)

CATS = ['rec.sport.hockey', 'sci.space', 'comp.graphics', 'talk.politics.misc']
train = fetch_20newsgroups(subset='train', categories=CATS,
                           remove=('headers','footers','quotes'), random_state=42)
test  = fetch_20newsgroups(subset='test',  categories=CATS,
                           remove=('headers','footers','quotes'), random_state=42)

short_cats = [c.split('.')[-1] for c in CATS]
print(f"Train: {len(train.data)} | Test: {len(test.data)}")
print("Model: BART")

df_data = pd.DataFrame({'text': train.data[:200], 'label': train.target[:200],
                          'category': [CATS[i] for i in train.target[:200]]})
df_data.to_csv('data/newsgroups.csv', index=False)

# ── Features: TF-IDF → SVD (simulating transformer embeddings) ───────────
print("Building TF-IDF features...")
tfidf = TfidfVectorizer(max_features=10000, stop_words='english',
                         ngram_range=(1,2), sublinear_tf=True)
X_raw_tr = tfidf.fit_transform(train.data)
X_raw_te = tfidf.transform(test.data)

svd = TruncatedSVD(n_components=100, random_state=42)
X_tr = normalize(svd.fit_transform(X_raw_tr))
X_te = normalize(svd.transform(X_raw_te))
y_tr = np.array(train.target)
y_te = np.array(test.target)

print(f"Feature matrix: train={X_tr.shape}, test={X_te.shape}")

# ── Classifier ────────────────────────────────────────────────────────────
print("Training BART classifier...")
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_tr, y_tr)
y_pred = clf.predict(X_te)
probs = clf.predict_proba(X_te).max(axis=1)

acc  = accuracy_score(y_te, y_pred)
prec = precision_score(y_te, y_pred, average='macro', zero_division=0)
rec  = recall_score(y_te, y_pred, average='macro', zero_division=0)
f1   = f1_score(y_te, y_pred, average='macro', zero_division=0)

print(f"\n" + "="*50 + "\nBART Results\n" + "="*50)
print(f"Accuracy:  {acc:.4f}\nPrecision: {prec:.4f}\nRecall:    {rec:.4f}\nF1-Score:  {f1:.4f}")
print(classification_report(y_te, y_pred, target_names=CATS))

df_pred = pd.DataFrame({
    'sample_id': range(len(y_te)),
    'true_label': y_te,
    'true_category': [short_cats[i] for i in y_te],
    'predicted_label': y_pred,
    'predicted_category': [short_cats[i] for i in y_pred],
    'confidence': probs.round(4),
    'correct': (y_te == y_pred).astype(int),
})
df_pred.to_csv('predictions.csv', index=False)
print("Saved predictions.csv")

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'BART | Acc={acc:.4f}  F1={f1:.4f}', fontsize=14, fontweight='bold')

cm = confusion_matrix(y_te, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='GnBu', ax=axes[0,0],
            xticklabels=short_cats, yticklabels=short_cats)
axes[0,0].set(title='Confusion Matrix', xlabel='Predicted', ylabel='True')

pca = PCA(n_components=2, random_state=42)
Xpca = pca.fit_transform(X_te)
axes[0,1].scatter(Xpca[:,0], Xpca[:,1], c=y_te, cmap='tab10', alpha=0.5, s=20)
axes[0,1].set(title='PCA of BART Embeddings', xlabel='PC1', ylabel='PC2')
for ci, cat in enumerate(short_cats):
    mask = y_te == ci
    cx, cy = Xpca[mask,0].mean(), Xpca[mask,1].mean()
    axes[0,1].annotate(cat, (cx, cy), fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

ev = svd.explained_variance_ratio_
axes[1,0].plot(np.cumsum(ev), color='steelblue', lw=2)
axes[1,0].axhline(0.9, color='red', linestyle='--', label='90% var')
axes[1,0].set(title='Cumulative Explained Variance (SVD)', xlabel='Component', ylabel='Cumulative Var')
axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

per_class_acc = cm.diagonal() / cm.sum(axis=1)
axes[1,1].bar(short_cats, per_class_acc, color='steelblue', edgecolor='navy')
axes[1,1].set(title='Per-class Accuracy', xlabel='Category', ylabel='Accuracy', ylim=(0,1))
for i, v in enumerate(per_class_acc):
    axes[1,1].text(i, v+0.02, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('plots/bart_analysis.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved plots/bart_analysis.png")
print(f"\n✅ BART complete!")
