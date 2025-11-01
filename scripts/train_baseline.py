# scripts/train_baseline.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
import joblib

DATA = Path("data")
MODELS = Path("models"); MODELS.mkdir(parents=True, exist_ok=True)

df_true = pd.read_csv(DATA/"True.csv"); df_true["label"]=1
df_fake = pd.read_csv(DATA/"Fake.csv"); df_fake["label"]=0
df = pd.concat([df_true, df_fake], ignore_index=True)
df["text"] = df["text"].astype(str).fillna("")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=2)
Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)

base = LogisticRegression(max_iter=1000)
clf  = CalibratedClassifierCV(base, method="isotonic", cv=5)
clf.fit(Xtr, y_train)

probs = clf.predict_proba(Xte)[:,1]
preds = (probs>=0.5).astype(int)
print(classification_report(y_test, preds))
print("ROC-AUC:", roc_auc_score(y_test, probs))

joblib.dump(tfidf, MODELS/"tfidf.joblib")
joblib.dump(clf,   MODELS/"logreg_calibrated.joblib")
print("Saved model to", MODELS.resolve())
