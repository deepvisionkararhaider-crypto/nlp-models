"""
06 - ELMo-style Contextual Embeddings
Dataset: 20 Newsgroups (4 categories)
Model: Simulated ELMo (bi-directional LSTM-like context via 3-layer TF-IDF + SVD),
       then Logistic Regression.
Note: Real ELMo requires AllenNLP (~8GB model). We simulate it with
      character-level + word-level + context features (3 representation levels).
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

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

# ── ELMo-style: 3 representation layers ──────────────────────────────────
# Layer 1: Character n-grams (morphological)  →  50-dim SVD
# Layer 2: Word unigrams (lexical)             →  50-dim SVD
# Layer 3: Word bigrams (contextual/syntactic) →  50-dim SVD
# Concatenated → 150-dim "ELMo-style" vector

print("Building 3-layer ELMo-style representations...")

# Layer 1: Char n-grams
tfidf_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5),
                              max_features=5000, sublinear_tf=True)
# Layer 2: Word unigrams
tfidf_word = TfidfVectorizer(analyzer='word', ngram_range=(1,1),
                              max_features=5000, sublinear_tf=True,
                              stop_words='english')
# Layer 3: Word bigrams (context)
tfidf_ctx  = TfidfVectorizer(analyzer='word', ngram_range=(2,3),
                              max_features=5000, sublinear_tf=True,
                              stop_words='english')

svd = TruncatedSVD(n_components=50, random_state=42)

X_char_tr = svd.fit_transform(tfidf_char.fit_transform(train.data))
X_char_te = svd.transform(tfidf_char.transform(test.data))

svd2 = TruncatedSVD(n_components=50, random_state=42)
X_word_tr = svd2.fit_transform(tfidf_word.fit_transform(train.data))
X_word_te = svd2.transform(tfidf_word.transform(test.data))

svd3 = TruncatedSVD(n_components=50, random_state=42)
X_ctx_tr  = svd3.fit_transform(tfidf_ctx.fit_transform(train.data))
X_ctx_te  = svd3.transform(tfidf_ctx.transform(test.data))

# Concat all 3 layers (simulating ELMo's weighted sum)
weights = [0.3, 0.4, 0.3]  # learnable in real ELMo
X_tr = np.hstack([weights[0]*normalize(X_char_tr),
                   weights[1]*normalize(X_word_tr),
                   weights[2]*normalize(X_ctx_tr)])
X_te = np.hstack([weights[0]*normalize(X_char_te),
                   weights[1]*normalize(X_word_te),
                   weights[2]*normalize(X_ctx_te)])

y_tr = np.array(train.target)
y_te = np.array(test.target)
print(f"ELMo representation: train={X_tr.shape}, test={X_te.shape}")

# ── Classify ───────────────────────────────────────────────────────────────
clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
clf.fit(X_tr, y_tr)
y_pred = clf.predict(X_te)
probs  = clf.predict_proba(X_te)

acc  = accuracy_score(y_te, y_pred)
prec = precision_score(y_te, y_pred, average='macro', zero_division=0)
rec  = recall_score(y_te, y_pred, average='macro', zero_division=0)
f1   = f1_score(y_te, y_pred, average='macro', zero_division=0)

print(f"\n{'='*50}\nELMo-style Results\n{'='*50}")
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
fig.suptitle(f'ELMo-style | Acc={acc:.4f}  F1={f1:.4f}', fontsize=14, fontweight='bold')

cm = confusion_matrix(y_te, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='BuGn', ax=axes[0,0],
            xticklabels=short_cats, yticklabels=short_cats)
axes[0,0].set(title='Confusion Matrix', xlabel='Predicted', ylabel='True')

pca = PCA(n_components=2, random_state=42)
Xpca = pca.fit_transform(X_te)
sc = axes[0,1].scatter(Xpca[:,0], Xpca[:,1], c=y_te, cmap='tab10', alpha=0.5, s=20)
axes[0,1].set(title='PCA of ELMo-style Embeddings', xlabel='PC1', ylabel='PC2')
for ci, cat in enumerate(short_cats):
    mask = y_te == ci
    cx, cy = Xpca[mask,0].mean(), Xpca[mask,1].mean()
    axes[0,1].annotate(cat, (cx, cy), fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

# Layer contributions
layer_names = ['Char n-grams\n(morphological)', 'Word unigrams\n(lexical)',
               'Word bigrams\n(contextual)']
layer_weights = weights
colors_l = ['#3498db', '#2ecc71', '#e74c3c']
axes[1,0].bar(layer_names, layer_weights, color=colors_l, edgecolor='black')
axes[1,0].set(title='ELMo Layer Weights', xlabel='Layer', ylabel='Weight')
axes[1,0].set_ylim(0, 0.6)
for i, v in enumerate(layer_weights):
    axes[1,0].text(i, v+0.01, f'{v:.2f}', ha='center', fontweight='bold')

per_class_acc = cm.diagonal() / cm.sum(axis=1)
axes[1,1].bar(short_cats, per_class_acc, color='mediumseagreen', edgecolor='darkgreen')
axes[1,1].set(title='Per-class Accuracy', xlabel='Category', ylabel='Accuracy', ylim=(0,1))
for i, v in enumerate(per_class_acc):
    axes[1,1].text(i, v+0.02, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('plots/elmo_analysis.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved plots/elmo_analysis.png\n✅ ELMo complete!")
