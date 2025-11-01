# scripts/ensembles_practice.py
# Practice 9 — Ensemble learning for text classification (True vs Fake)
# Outputs:
#   figures/ensemble_auc_compare.png
#   figures/ensemble_roc.png
#   figures/ensemble_confusion_<model>.png
#   figures/ensemble_metrics.csv
# Run:
#   python scripts/ensembles_practice.py
#   python scripts/ensembles_practice.py --time-split  # train on early months, test on later

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# -----------------------------
# CLI
# -----------------------------
parser = argparse.ArgumentParser(description="Practice 9: Ensemble models for Fake/True news")
parser.add_argument("--time-split", action="store_true",
                    help="Use chronological split by date instead of random split")
args = parser.parse_args()

# -----------------------------
# Paths & constants
# -----------------------------
DATA = Path("data")
FIGS = Path("figures"); FIGS.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

# -----------------------------
# 1) Load data
# -----------------------------
print("Loading dataset ...")
df_true = pd.read_csv(DATA / "True.csv"); df_true["label"] = 1
df_fake = pd.read_csv(DATA / "Fake.csv"); df_fake["label"] = 0
df = pd.concat([df_true, df_fake], ignore_index=True)

# Ensure text & (optional) date
df["text"] = df["text"].astype(str).fillna("")
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
else:
    df["date"] = pd.NaT

print(f"Total samples: {len(df):,}")

# -----------------------------
# 2) Split (random or time)
# -----------------------------
if args.time_split and df["date"].notna().sum() > 0:
    # sort by date; 80% earliest for train, 20% latest for test
    dfx = df.dropna(subset=["date"]).sort_values("date")
    cutoff_idx = int(0.8 * len(dfx))
    train_df = dfx.iloc[:cutoff_idx]
    test_df  = dfx.iloc[cutoff_idx:]
    split_desc = f"time-split: train ≤ {train_df['date'].max().date()} | test > that"
else:
    train_df, test_df = train_test_split(
        df[["text","label"]], test_size=0.20, random_state=RANDOM_STATE, stratify=df["label"]
    )
    split_desc = "random 80/20 split (stratified)"

print("Using", split_desc)

# -----------------------------
# 3) Vectorize (TF-IDF)
# -----------------------------
print("Vectorizing text (TF-IDF) ...")
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1,2),
    min_df=5
)
Xtr = vectorizer.fit_transform(train_df["text"])
Xte = vectorizer.transform(test_df["text"])
ytr = train_df["label"].values
yte = test_df["label"].values

# -----------------------------
# 4) Define models (ensembles)
# -----------------------------
# Tip: keep configs moderate so it runs fast on laptops
rf  = RandomForestClassifier(n_estimators=400, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1)
et  = ExtraTreesClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
gb  = GradientBoostingClassifier(random_state=RANDOM_STATE)
ada = AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=RANDOM_STATE)

# Strong linear baseline; calibrated for better probs
lr_base = LogisticRegression(max_iter=1000, n_jobs=None, random_state=RANDOM_STATE)
lr  = CalibratedClassifierCV(lr_base, method="sigmoid", cv=3)

# Bagging (default base estimator = DecisionTree)
bag = BaggingClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)

# Soft voting (averages probabilities)
vote = VotingClassifier(
    estimators=[("rf", rf), ("gb", gb), ("ada", ada), ("lr", lr)],
    voting="soft", n_jobs=-1
)

models = {
    "RandomForest": rf,
    "ExtraTrees": et,
    "GradBoost": gb,
    "AdaBoost": ada,
    "CalibLogReg": lr,
    "Bagging": bag,
    "SoftVote": vote
}

# -----------------------------
# 5) Train, evaluate, collect metrics
# -----------------------------
def get_scores(model, X, y):
    # prefer predict_proba; fall back to decision_function if available
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        p = model.decision_function(X)
        # scale to [0,1] for plots (not required for AUC, but nicer)
        p = (p - p.min()) / (p.max() - p.min() + 1e-12)
    else:
        # last resort: use predictions (degrades AUC fidelity)
        p = model.predict(X).astype(float)
    return p

metrics_rows = []
roc_curves = {}  # name -> (fpr, tpr, auc)

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    model.fit(Xtr, ytr)
    probs = get_scores(model, Xte, yte)
    preds = (probs >= 0.5).astype(int)

    # reports
    print(classification_report(yte, preds, digits=4))
    auc_val = roc_auc_score(yte, probs)
    print("ROC-AUC:", round(auc_val, 4))

    # confusion matrix plot
    cm = confusion_matrix(yte, preds)
    fig = plt.figure(figsize=(5.0, 4.0))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{name} – Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, int(val), ha="center", va="center")
    plt.tight_layout()
    out_cm = FIGS / f"ensemble_confusion_{name.lower()}.png"
    plt.savefig(out_cm, dpi=160); plt.close(fig)
    print("Saved →", out_cm)

    # ROC curve (store for combined plot)
    fpr, tpr, _ = roc_curve(yte, probs)
    roc_curves[name] = (fpr, tpr, auc_val)

    metrics_rows.append({
        "model": name,
        "ROC_AUC": auc_val,
        "TP": int(cm[1,1]),
        "FP": int(cm[0,1]),
        "TN": int(cm[0,0]),
        "FN": int(cm[1,0])
    })

# -----------------------------
# 6) Comparison plots
# -----------------------------
# AUC bar chart
metrics_df = pd.DataFrame(metrics_rows).sort_values("ROC_AUC", ascending=False)
metrics_df.to_csv(FIGS / "ensemble_metrics.csv", index=False)

plt.figure(figsize=(7.5, 4.6))
plt.bar(metrics_df["model"], metrics_df["ROC_AUC"])
plt.title("Ensemble Model Comparison (ROC-AUC)")
plt.ylabel("ROC-AUC"); plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(FIGS / "ensemble_auc_compare.png", dpi=160)
plt.close()
print("Saved → figures/ensemble_auc_compare.png")

# Combined ROC curves
plt.figure(figsize=(7.5, 5.5))
for name, (fpr, tpr, auc_val) in roc_curves.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
plt.plot([0,1], [0,1], "k--", linewidth=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Ensemble Models")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(FIGS / "ensemble_roc.png", dpi=160)
plt.close()
print("Saved → figures/ensemble_roc.png")

print("\nEnsemble Practice complete ✅")
print("Top models:\n", metrics_df.head(3).to_string(index=False))
