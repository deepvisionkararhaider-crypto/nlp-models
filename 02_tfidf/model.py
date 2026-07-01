"""
02 - TF-IDF Text Classification
Dataset: 20 Newsgroups (4 categories)
Model: TF-IDF Vectorizer + Linear SVM
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
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
from sklearn.calibration import CalibratedClassifierCV

os.makedirs('data', exist_ok=True)
os.makedirs('plots', exist_ok=True)

CATS = ['rec.sport.hockey', 'sci.space', 'comp.graphics', 'talk.politics.misc']
train = fetch_20newsgroups(subset='train', categories=CATS,
                           remove=('headers','footers','quotes'), random_state=42)
test  = fetch_20newsgroups(subset='test',  categories=CATS,
                           remove=('headers','footers','quotes'), random_state=42)

print(f"Train: {len(train.data)} | Test: {len(test.data)}")

df_data = pd.DataFrame({'text': train.data[:200], 'label': train.target[:200],
                         'category': [CATS[i] for i in train.target[:200]]})
df_data.to_csv('data/newsgroups.csv', index=False)

# ── Model: TF-IDF + LinearSVC ─────────────────────────────────────────────
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english',
                              ngram_range=(1,2), sublinear_tf=True)),
    ('clf',   CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0, random_state=42))),
])
pipe.fit(train.data, train.target)
y_pred = pipe.predict(test.data)
probs   = pipe.predict_proba(test.data)

acc  = accuracy_score(test.target, y_pred)
prec = precision_score(test.target, y_pred, average='macro', zero_division=0)
rec  = recall_score(test.target, y_pred, average='macro', zero_division=0)
f1   = f1_score(test.target, y_pred, average='macro', zero_division=0)

print(f"\n{'='*50}\nTF-IDF Results\n{'='*50}")
print(f"Accuracy:  {acc:.4f}\nPrecision: {prec:.4f}\nRecall:    {rec:.4f}\nF1-Score:  {f1:.4f}")
short_cats = [c.split('.')[-1] for c in CATS]
print("\nClassification Report:")
print(classification_report(test.target, y_pred, target_names=CATS))

df_pred = pd.DataFrame({
    'sample_id': range(len(test.target)),
    'true_label': test.target,
    'true_category': [short_cats[i] for i in test.target],
    'predicted_label': y_pred,
    'predicted_category': [short_cats[i] for i in y_pred],
    'confidence': probs.max(axis=1).round(4),
    'correct': (test.target == y_pred).astype(int),
})
df_pred.to_csv('predictions.csv', index=False)
print("Saved predictions.csv")

# ── IDF weights for top features ──────────────────────────────────────────
tfidf = pipe.named_steps['tfidf']
feature_names = tfidf.get_feature_names_out()
idf_vals = tfidf.idf_
top_idx = np.argsort(idf_vals)[-30:]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'TF-IDF | Acc={acc:.4f}  F1={f1:.4f}', fontsize=14, fontweight='bold')

cm = confusion_matrix(test.target, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[0,0],
            xticklabels=short_cats, yticklabels=short_cats)
axes[0,0].set(title='Confusion Matrix', xlabel='Predicted', ylabel='True')

# Top IDF features
top30_names  = feature_names[top_idx]
top30_idf    = idf_vals[top_idx]
axes[0,1].barh(top30_names, top30_idf, color='darkorange', edgecolor='white')
axes[0,1].set(title='Top 30 Terms by IDF Weight', xlabel='IDF Value')
axes[0,1].tick_params(axis='y', labelsize=7)

# Confidence dist
axes[1,0].hist(df_pred.loc[df_pred['correct']==1,'confidence'], bins=25,
               color='green', alpha=0.6, label='Correct')
axes[1,0].hist(df_pred.loc[df_pred['correct']==0,'confidence'], bins=25,
               color='red', alpha=0.6, label='Wrong')
axes[1,0].set(title='Confidence: Correct vs Wrong', xlabel='Confidence', ylabel='Count')
axes[1,0].legend()

per_class_acc = cm.diagonal() / cm.sum(axis=1)
axes[1,1].bar(short_cats, per_class_acc, color='steelblue', edgecolor='navy')
axes[1,1].set(title='Per-class Accuracy', xlabel='Category', ylabel='Accuracy', ylim=(0,1))
for i, v in enumerate(per_class_acc):
    axes[1,1].text(i, v+0.02, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('plots/tfidf_analysis.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved plots/tfidf_analysis.png\n✅ TF-IDF complete!")
