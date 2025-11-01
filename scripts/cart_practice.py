# scripts/cart_practice.py
# Practice 8 — Implement CART (Decision Tree) to perform categorization on True/Fake news text
# Outputs:
#   figures/cart_confusion_matrix.png
#   figures/cart_roc_curve.png
#   figures/cart_tree_top.png
#   figures/cart_top_features.png
#   figures/cart_tree_top.txt  (textual tree rules)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve,
                             auc, RocCurveDisplay)

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
df["text"] = df["text"].astype(str).fillna("")
print(f"Total samples: {len(df):,}")

# -----------------------------
# 2) Train / Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
)

# -----------------------------
# 3) TF-IDF vectorization
# -----------------------------
print("Vectorizing text (TF-IDF) ...")
vectorizer = TfidfVectorizer(
    max_features=30000,  # keep modest for speed
    ngram_range=(1, 2),
    min_df=5
)
Xtr = vectorizer.fit_transform(X_train)
Xte = vectorizer.transform(X_test)
feature_names = vectorizer.get_feature_names_out()

# -----------------------------
# 4) CART model + small grid search
# -----------------------------
print("Training Decision Tree (GridSearchCV) ...")
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [6, 10, 14],
    "min_samples_split": [2, 10, 50],
    "min_samples_leaf": [1, 5, 10]
}
base = DecisionTreeClassifier(random_state=RANDOM_STATE)
grid = GridSearchCV(base, param_grid=param_grid, cv=3, n_jobs=-1, verbose=0)
grid.fit(Xtr, y_train)

clf: DecisionTreeClassifier = grid.best_estimator_
print("Best params:", grid.best_params_)

# -----------------------------
# 5) Evaluation
# -----------------------------
print("Evaluating ...")
probs = clf.predict_proba(Xte)[:, 1]
preds = (probs >= 0.5).astype(int)

print("\nClassification Report:\n")
print(classification_report(y_test, preds, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, preds)
fig = plt.figure(figsize=(5.2, 4.2))
plt.imshow(cm, cmap="Blues")
plt.title("CART Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
for (i, j), val in np.ndenumerate(cm):
    plt.text(j, i, int(val), ha="center", va="center")
plt.tight_layout()
plt.savefig(FIGS / "cart_confusion_matrix.png", dpi=160)
plt.close(fig)
print("Saved → figures/cart_confusion_matrix.png")

# ROC Curve & AUC
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)
fig = plt.figure(figsize=(5.2, 4.2))
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="CART").plot()
plt.title(f"CART ROC curve (AUC = {roc_auc:.3f})")
plt.tight_layout()
plt.savefig(FIGS / "cart_roc_curve.png", dpi=160)
plt.close(fig)
print("Saved → figures/cart_roc_curve.png")

# -----------------------------
# 6) Tree visualization (top)
# -----------------------------
# Trees on text are huge; show only the top part to keep the plot readable
print("Rendering top of the tree ...")
fig = plt.figure(figsize=(12, 6))
plot_tree(clf, max_depth=3, filled=True, fontsize=8, feature_names=None, class_names=["Fake","True"])
plt.title("Decision Tree (Top 3 Levels)")
plt.tight_layout()
plt.savefig(FIGS / "cart_tree_top.png", dpi=160)
plt.close(fig)
print("Saved → figures/cart_tree_top.png")

# Also export text rules (limit features for readability)
tree_text = export_text(clf, feature_names=list(feature_names[:2000]))
(Path(FIGS) / "cart_tree_top.txt").write_text(tree_text, encoding="utf-8")
print("Saved → figures/cart_tree_top.txt")

# -----------------------------
# 7) Top features by importance
# -----------------------------
# Many importances will be zero in sparse text. Show top 20 non-zero.
importances = clf.feature_importances_
nz = np.where(importances > 0)[0]
if nz.size > 0:
    topk = min(20, nz.size)
    top_idx = nz[np.argsort(importances[nz])[::-1][:topk]]
    top_feats = feature_names[top_idx]
    top_vals = importances[top_idx]

    order = np.argsort(top_vals)
    fig = plt.figure(figsize=(8, 6))
    plt.barh(range(topk), top_vals[order])
    plt.yticks(range(topk), top_feats[order])
    plt.xlabel("Importance")
    plt.title("Top CART Features")
    plt.tight_layout()
    plt.savefig(FIGS / "cart_top_features.png", dpi=160)
    plt.close(fig)
    print("Saved → figures/cart_top_features.png")
else:
    print("No non-zero feature importances found (tree very shallow or pruned).")

print("\nCART Practice complete ✅")
