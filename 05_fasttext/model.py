"""
05 - FastText Embeddings + Text Classification
Dataset: 20 Newsgroups (4 categories)
Model: FastText (gensim) with subword n-grams → mean embedding → Logistic Regression
"""

import os, warnings, gc
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
from sklearn.decomposition import PCA
from gensim.models import FastText
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

df_data = pd.DataFrame({'text': train.data[:200], 'label': train.target[:200],
                         'category': [CATS[i] for i in train.target[:200]]})
df_data.to_csv('data/newsgroups.csv', index=False)

# ── Tokenize (lightweight) ─────────────────────────────────────────────────
def tokenize(docs):
    return [simple_preprocess(doc, deacc=True, min_len=2) for doc in docs]

print("Tokenizing...")
train_tokens = tokenize(train.data)
test_tokens  = tokenize(test.data)

# ── Train FastText (smaller model to save memory) ─────────────────────────
print("Training FastText (subword n-grams: min_n=3, max_n=5, vector_size=50)...")
ft = FastText(sentences=train_tokens, vector_size=100, window=5,
              min_count=2, min_n=3, max_n=5, workers=2, epochs=10, seed=42,
              bucket=200000)  # Reduced bucket size for memory
print(f"Vocabulary size: {len(ft.wv)}")
gc.collect()

def mean_embed(tokens, model, dim=100):
    vecs = [model.wv[t] for t in tokens]  # FastText handles OOV
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim)

print("Building embeddings...")
X_tr = np.array([mean_embed(t, ft) for t in train_tokens])
X_te = np.array([mean_embed(t, ft) for t in test_tokens])
y_tr = np.array(train.target)
y_te = np.array(test.target)
del ft; gc.collect()

print(f"Embeddings: train={X_tr.shape}, test={X_te.shape}")

clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
clf.fit(X_tr, y_tr)
y_pred = clf.predict(X_te)
probs  = clf.predict_proba(X_te)

acc  = accuracy_score(y_te, y_pred)
prec = precision_score(y_te, y_pred, average='macro', zero_division=0)
rec  = recall_score(y_te, y_pred, average='macro', zero_division=0)
f1   = f1_score(y_te, y_pred, average='macro', zero_division=0)

print(f"\n{'='*50}\nFastText Results\n{'='*50}")
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
fig.suptitle(f'FastText | Acc={acc:.4f}  F1={f1:.4f}', fontsize=14, fontweight='bold')

cm = confusion_matrix(y_te, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', ax=axes[0,0],
            xticklabels=short_cats, yticklabels=short_cats)
axes[0,0].set(title='Confusion Matrix', xlabel='Predicted', ylabel='True')

pca = PCA(n_components=2, random_state=42)
Xpca = pca.fit_transform(X_te)
sc = axes[0,1].scatter(Xpca[:,0], Xpca[:,1], c=y_te, cmap='tab10', alpha=0.5, s=20)
axes[0,1].set(title='PCA of FastText Embeddings', xlabel='PC1', ylabel='PC2')
for ci, cat in enumerate(short_cats):
    mask = y_te == ci
    cx, cy = Xpca[mask,0].mean(), Xpca[mask,1].mean()
    axes[0,1].annotate(cat, (cx, cy), fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

# Confidence distribution
axes[1,0].hist(df_pred.loc[df_pred['correct']==1,'confidence'], bins=25,
               color='green', alpha=0.6, label='Correct')
axes[1,0].hist(df_pred.loc[df_pred['correct']==0,'confidence'], bins=25,
               color='red', alpha=0.6, label='Wrong')
axes[1,0].set(title='Confidence: Correct vs Wrong', xlabel='Confidence', ylabel='Count')
axes[1,0].legend()

per_class_acc = cm.diagonal() / cm.sum(axis=1)
axes[1,1].bar(short_cats, per_class_acc, color='crimson', edgecolor='darkred')
axes[1,1].set(title='Per-class Accuracy', xlabel='Category', ylabel='Accuracy', ylim=(0,1))
for i, v in enumerate(per_class_acc):
    axes[1,1].text(i, v+0.02, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('plots/fasttext_analysis.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved plots/fasttext_analysis.png\n✅ FastText complete!")
