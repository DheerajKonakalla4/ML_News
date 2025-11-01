# scripts/time_split_eval.py
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, f1_score

DATA = Path("data")

df_true = pd.read_csv(DATA/"True.csv"); df_true["label"]=1
df_fake = pd.read_csv(DATA/"Fake.csv"); df_fake["label"]=0
df = pd.concat([df_true, df_fake], ignore_index=True)
df["text"] = df["text"].astype(str).fillna("")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

cut = pd.Timestamp("2017-07-01")
train = df[(df["date"].notna()) & (df["date"] < cut)]
test  = df[(df["date"].notna()) & (df["date"] >= cut)]

tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=2)
Xtr = tfidf.fit_transform(train["text"])
Xte = tfidf.transform(test["text"])

base = LogisticRegression(max_iter=1000)
clf  = CalibratedClassifierCV(base, method="isotonic", cv=3)
clf.fit(Xtr, train["label"])

probs = clf.predict_proba(Xte)[:,1]
preds = (probs>=0.5).astype(int)
print("Time-split ROC-AUC:", roc_auc_score(test["label"], probs))
print("Time-split F1:", f1_score(test["label"], preds))
