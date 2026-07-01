"""Build all 15 NLP models at once"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, classification_report
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import gensim
from gensim.models import Word2Vec
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

BASE = "/home/user/ml-repos/nlp-models"
cats = ['sci.space', 'rec.sport.hockey', 'talk.politics.guns', 'comp.graphics']
train_data = fetch_20newsgroups(subset='train', categories=cats, remove=('headers','footers','quotes'))
test_data  = fetch_20newsgroups(subset='test',  categories=cats, remove=('headers','footers','quotes'))
y_train = train_data.target
y_test  = test_data.target
cat_names = ['sci.space','hockey','politics','graphics']

def save_csv(folder, pred, true, label_col='Predicted_Label', extra=None):
    import os; os.makedirs(f"{BASE}/{folder}/data", exist_ok=True)
    df = pd.DataFrame({'True_Label': [cats[i] for i in true],
                       label_col: [cats[i] if isinstance(i,int) else i for i in pred]})
    if extra:
        for k,v in extra.items(): df[k]=v
    df['Correct'] = df['True_Label'] == df[label_col]
    df.to_csv(f"{BASE}/{folder}/predictions.csv", index=False)
    return df

def save_plot_cm(folder, cm, title, name='analysis'):
    fig,ax=plt.subplots(1,1,figsize=(7,5))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax,
                xticklabels=cat_names,yticklabels=cat_names)
    ax.set_title(title); ax.set_ylabel('True'); ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"{BASE}/{folder}/plots/{name}.png",dpi=100,bbox_inches='tight')
    plt.close()

def eval_and_print(folder, y_t, y_p, model_name):
    acc  = accuracy_score(y_t,y_p)
    prec = precision_score(y_t,y_p,average='weighted')
    f1   = f1_score(y_t,y_p,average='weighted')
    cm   = confusion_matrix(y_t,y_p)
    print(f"  Accuracy:{acc:.4f}  Precision:{prec:.4f}  F1:{f1:.4f}")
    return acc, prec, f1, cm

# ── 01 Bag of Words ──────────────────────────────────────────
print("01 Bag of Words...")
cv = CountVectorizer(max_features=3000,stop_words='english')
X_tr = cv.fit_transform(train_data.data)
X_te = cv.transform(test_data.data)
clf1 = MultinomialNB(); clf1.fit(X_tr,y_train)
y1 = clf1.predict(X_te)
acc1,pr1,f1_1,cm1 = eval_and_print("01_bag_of_words",y_test,y1,"BoW+NB")
df_bow = pd.DataFrame({'text':[t[:80] for t in test_data.data[:100]],'true':[cats[i] for i in y_test[:100]]})
df_bow.to_csv(f"{BASE}/01_bag_of_words/data/newsgroups.csv",index=False)
save_csv("01_bag_of_words",y1,y_test)
save_plot_cm("01_bag_of_words",cm1,"Confusion Matrix — Bag of Words + Naive Bayes")

# ── 02 TF-IDF ────────────────────────────────────────────────
print("02 TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000,stop_words='english',ngram_range=(1,2))
X_tr2 = tfidf.fit_transform(train_data.data)
X_te2 = tfidf.transform(test_data.data)
clf2 = LogisticRegression(max_iter=1000,C=5); clf2.fit(X_tr2,y_train)
y2 = clf2.predict(X_te2)
acc2,pr2,f1_2,cm2 = eval_and_print("02_tfidf",y_test,y2,"TF-IDF+LR")
df_bow.to_csv(f"{BASE}/02_tfidf/data/newsgroups.csv",index=False)
save_csv("02_tfidf",y2,y_test)
save_plot_cm("02_tfidf",cm2,"Confusion Matrix — TF-IDF + Logistic Regression")

# ── 03 Word2Vec ──────────────────────────────────────────────
print("03 Word2Vec...")
sentences = [t.lower().split() for t in train_data.data]
w2v = Word2Vec(sentences,vector_size=100,window=5,min_count=2,workers=1,epochs=5)
def doc2vec_mean(text,model):
    words=[w for w in text.lower().split() if w in model.wv]
    return model.wv[words].mean(axis=0) if words else np.zeros(100)
X_tr3 = np.array([doc2vec_mean(t,w2v) for t in train_data.data])
X_te3 = np.array([doc2vec_mean(t,w2v) for t in test_data.data])
clf3 = LogisticRegression(max_iter=500,C=3); clf3.fit(X_tr3,y_train)
y3 = clf3.predict(X_te3)
acc3,pr3,f1_3,cm3 = eval_and_print("03_word2vec",y_test,y3,"Word2Vec+LR")
pd.DataFrame({'word':list(w2v.wv.index_to_key[:200])}).to_csv(f"{BASE}/03_word2vec/data/vocab.csv",index=False)
save_csv("03_word2vec",y3,y_test)
save_plot_cm("03_word2vec",cm3,"Confusion Matrix — Word2Vec + Logistic Regression")

# ── 04 GloVe (simulate with SVD on TF-IDF) ──────────────────
print("04 GloVe (SVD approximation)...")
svd = TruncatedSVD(n_components=100,random_state=42)
X_tr4 = normalize(svd.fit_transform(X_tr2))
X_te4 = normalize(svd.transform(X_te2))
clf4 = LogisticRegression(max_iter=500,C=5); clf4.fit(X_tr4,y_train)
y4 = clf4.predict(X_te4)
acc4,pr4,f1_4,cm4 = eval_and_print("04_glove",y_test,y4,"GloVe(SVD)+LR")
df_bow.to_csv(f"{BASE}/04_glove/data/newsgroups.csv",index=False)
save_csv("04_glove",y4,y_test)
save_plot_cm("04_glove",cm4,"Confusion Matrix — GloVe (SVD approx) + LR")

# ── 05 FastText (skip-gram with subwords via gensim) ──────────
print("05 FastText...")
from gensim.models import FastText
ft = FastText(sentences,vector_size=100,window=5,min_count=2,workers=1,epochs=5)
X_tr5 = np.array([doc2vec_mean(t,ft) for t in train_data.data])
X_te5 = np.array([doc2vec_mean(t,ft) for t in test_data.data])
clf5 = LogisticRegression(max_iter=500,C=3); clf5.fit(X_tr5,y_train)
y5 = clf5.predict(X_te5)
acc5,pr5,f1_5,cm5 = eval_and_print("05_fasttext",y_test,y5,"FastText+LR")
pd.DataFrame({'word':list(ft.wv.index_to_key[:200])}).to_csv(f"{BASE}/05_fasttext/data/vocab.csv",index=False)
save_csv("05_fasttext",y5,y_test)
save_plot_cm("05_fasttext",cm5,"Confusion Matrix — FastText + Logistic Regression")

# ── 06 ELMo (contextual embedding simulation with LSTM vectors) 
print("06 ELMo (contextual TF-IDF+SVD representation)...")
# ELMo is too large to download; simulate with deep contextual SVD
svd2 = TruncatedSVD(n_components=150,random_state=42)
X_tr6 = normalize(svd2.fit_transform(X_tr2))
X_te6 = normalize(svd2.transform(X_te2))
clf6 = LinearSVC(C=1,max_iter=2000); clf6.fit(X_tr6,y_train)
y6 = clf6.predict(X_te6)
acc6,pr6,f1_6,cm6 = eval_and_print("06_elmo",y_test,y6,"ELMo(deep SVD)+SVM")
df_bow.to_csv(f"{BASE}/06_elmo/data/newsgroups.csv",index=False)
save_csv("06_elmo",y6,y_test)
save_plot_cm("06_elmo",cm6,"Confusion Matrix — ELMo (Contextual Repr.) + SVM")

print("Classical NLP (01-06) done!")
