"""
01 - Bag of Words (BoW) Text Classification
Dataset: 20 Newsgroups (4 categories)
Model: CountVectorizer + Logistic Regression
"""

import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

os.makedirs('data', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────
CATS = ['rec.sport.hockey', 'sci.space', 'comp.graphics', 'talk.politics.misc']
train = fetch_20newsgroups(subset='train', categories=CATS,
                           remove=('headers','footers','quotes'), random_state=42)
test  = fetch_20newsgroups(subset='test',  categories=CATS,
                           remove=('headers','footers','quotes'), random_state=42)

print(f"Train: {len(train.data)} | Test: {len(test.data)}")
print("Categories:", CATS)

# Save dataset sample
df_data = pd.DataFrame({'text': train.data[:200], 'label': train.target[:200],
                         'category': [CATS[i] for i in train.target[:200]]})
df_data.to_csv('data/newsgroups.csv', index=False)
print("Saved data/newsgroups.csv")

# ── Model: Bag of Words ──────────────────────────────────────────────────
pipe = Pipeline([
    ('bow', CountVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
])
pipe.fit(train.data, train.target)
y_pred = pipe.predict(test.data)
probs   = pipe.predict_proba(test.data)

# ── Metrics ──────────────────────────────────────────────────────────────
acc  = accuracy_score(test.target, y_pred)
prec = precision_score(test.target, y_pred, average='macro', zero_division=0)
rec  = recall_score(test.target, y_pred, average='macro', zero_division=0)
f1   = f1_score(test.target, y_pred, average='macro', zero_division=0)

print(f"\n{'='*50}")
print(f"Bag of Words Results")
print(f"{'='*50}")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("\nClassification Report:")
print(classification_report(test.target, y_pred, target_names=CATS))

# ── Predictions CSV ───────────────────────────────────────────────────────
short_cats = [c.split('.')[-1] for c in CATS]
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
print("\nSaved predictions.csv")

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Bag of Words | Acc={acc:.4f}  F1={f1:.4f}', fontsize=14, fontweight='bold')

# Confusion Matrix
cm = confusion_matrix(test.target, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
            xticklabels=short_cats, yticklabels=short_cats)
axes[0,0].set(title='Confusion Matrix', xlabel='Predicted', ylabel='True')

# Top feature words per class
bow = pipe.named_steps['bow']
clf = pipe.named_steps['clf']
feature_names = bow.get_feature_names_out()
top_n = 10
for ci, cat in enumerate(short_cats):
    coef = clf.coef_[ci]
    top_idx = np.argsort(coef)[-top_n:]
    axes[0,1].barh([f"{cat}: {feature_names[i]}" for i in top_idx],
                   coef[top_idx], alpha=0.7)
axes[0,1].set(title=f'Top {top_n} BoW Features per Class', xlabel='Coefficient')
axes[0,1].tick_params(axis='y', labelsize=7)

# Confidence distribution
axes[1,0].hist(df_pred['confidence'], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
axes[1,0].set(title='Prediction Confidence Distribution', xlabel='Confidence', ylabel='Count')
axes[1,0].axvline(df_pred['confidence'].mean(), color='red', linestyle='--',
                  label=f"Mean={df_pred['confidence'].mean():.3f}")
axes[1,0].legend()

# Per-class accuracy
per_class_acc = cm.diagonal() / cm.sum(axis=1)
axes[1,1].bar(short_cats, per_class_acc, color='darkorange', edgecolor='black')
axes[1,1].set(title='Per-class Accuracy', xlabel='Category', ylabel='Accuracy', ylim=(0,1))
for i, v in enumerate(per_class_acc):
    axes[1,1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('plots/bow_analysis.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved plots/bow_analysis.png")
print("\n✅ Bag of Words complete!")
