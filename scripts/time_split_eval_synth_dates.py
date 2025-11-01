# scripts/time_split_eval_synth_dates.py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import matplotlib.pyplot as plt
import warnings

RANDOM_STATE = 42
DATA = Path("data")
FIGS = Path("figures"); FIGS.mkdir(parents=True, exist_ok=True)

print("Using scikit-learn version:")
import sklearn; print(sklearn.__version__)  # sanity

# ---------- 1) Load ----------
df_true = pd.read_csv(DATA/"True.csv"); df_true["label"] = 1
df_fake = pd.read_csv(DATA/"Fake.csv"); df_fake["label"] = 0

# Parse dates
df_true["date"] = pd.to_datetime(df_true["date"], errors="coerce")
df_fake["date"] = pd.to_datetime(df_fake["date"], errors="coerce")  # likely NaT for most fake rows

# Ensure text exists
for d in (df_true, df_fake):
    d["text"] = d["text"].astype(str).fillna("")

# ---------- 2) Assign synthetic dates to Fake rows ----------
# Sample from TRUE date distribution (keeps seasonality)
rng = np.random.RandomState(RANDOM_STATE)
valid_true_dates = df_true["date"].dropna().values
if len(valid_true_dates) == 0:
    raise RuntimeError("No valid dates in True.csv, cannot synthesize from distribution.")

mask_missing = df_fake["date"].isna()
synth = rng.choice(valid_true_dates, size=mask_missing.sum(), replace=True)
df_fake.loc[mask_missing, "date"] = synth

# ---------- 3) Combine ----------
df = pd.concat([df_true, df_fake], ignore_index=True)

# ---------- 4) Helper: calibrated training with hold-out validation ----------
def fit_calibrated_clf(X_train, y_train):
    # avoid CV folds that might lose a class; use a stratified hold-out
    X_in, X_val, y_in, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )
    base = LogisticRegression(max_iter=1000)
    base.fit(X_in, y_in)
    # Newer sklearn uses 'estimator' (not 'base_estimator')
    try:
        cal = CalibratedClassifierCV(estimator=base, method="isotonic", cv="prefit")
        cal.fit(X_val, y_val)
    except Exception as e:
        warnings.warn(f"Isotonic calibration failed ({e}); falling back to sigmoid.")
        cal = CalibratedClassifierCV(estimator=base, method="sigmoid", cv="prefit")
        cal.fit(X_val, y_val)
    return cal

# ---------- 5) TIME-SPLIT evaluation ----------
dfd = df.dropna(subset=["date"]).copy()
candidate = dfd["date"].quantile(0.80)

def choose_time_split(dt, min_per_class=50):
    left  = dfd[dfd["date"] <  dt]
    right = dfd[dfd["date"] >= dt]
    ok = (
        left["label"].nunique()==2 and right["label"].nunique()==2 and
        left["label"].value_counts().min()  >= min_per_class and
        right["label"].value_counts().min() >= min_per_class
    )
    return ok, left, right

ok, train_df, test_df = choose_time_split(candidate)

if not ok:
    for q in [0.7, 0.6, 0.9, 0.5]:
        c = dfd["date"].quantile(q)
        ok, train_df, test_df = choose_time_split(c, min_per_class=20)
        if ok:
            candidate = c
            break

if not ok:
    candidate = dfd["date"].median()
    ok, train_df, test_df = choose_time_split(candidate, min_per_class=5)

print(f"\n[Time-split] Using cut date: {candidate.date()}")
print("TRAIN label counts:\n", train_df["label"].value_counts().rename({0:'Fake',1:'True'}))
print("TEST  label counts:\n",  test_df["label"].value_counts().rename({0:'Fake',1:'True'}))

# Vectorize on TRAIN period only (no peeking)
tfidf_time = TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=2)
X_tr_time = tfidf_time.fit_transform(train_df["text"])
X_te_time = tfidf_time.transform(test_df["text"])

cal_time = fit_calibrated_clf(X_tr_time, train_df["label"].values)
probs_time = cal_time.predict_proba(X_te_time)[:,1]
preds_time = (probs_time >= 0.5).astype(int)

auc_time = roc_auc_score(test_df["label"], probs_time)
f1_time  = f1_score(test_df["label"], preds_time)
print("\n[Time-split] ROC-AUC:", round(auc_time, 4))
print("[Time-split] F1     :", round(f1_time, 4))
print("\n[Time-split] Classification report:\n",
      classification_report(test_df["label"], preds_time, digits=4))

# ---------- 6) RANDOM-SPLIT for comparison ----------
tfidf_rand = TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=2)
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
)
Xtr = tfidf_rand.fit_transform(X_train)
Xte = tfidf_rand.transform(X_test)

cal_rand = fit_calibrated_clf(Xtr, y_train.values)
probs_rand = cal_rand.predict_proba(Xte)[:,1]
preds_rand = (probs_rand >= 0.5).astype(int)

auc_rand = roc_auc_score(y_test, probs_rand)
f1_rand  = f1_score(y_test, preds_rand)
print("\n[Random-split] ROC-AUC:", round(auc_rand, 4))
print("[Random-split] F1     :", round(f1_rand, 4))

# ---------- 7) Save a tiny bar chart for your PPT ----------
labels = ["Random", "Time"]
auc_vals = [auc_rand, auc_time]
f1_vals  = [f1_rand,  f1_time]

plt.figure(figsize=(6.5,4.2))
x = np.arange(len(labels))
w = 0.35
plt.bar(x - w/2, auc_vals, width=w, label="ROC-AUC")
plt.bar(x + w/2, f1_vals,  width=w, label="F1")
plt.xticks(x, labels)
plt.title("Random vs Time-split (with synthetic Fake dates)")
plt.legend()
plt.tight_layout()
out_chart = FIGS/"random_vs_time_split.png"
plt.savefig(out_chart, dpi=160)
plt.close()
print(f"\nSaved comparison chart → {out_chart.resolve()}")
