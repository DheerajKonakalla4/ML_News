# scripts/evidence_demo.py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
import textwrap

RANDOM_STATE = 42
DATA = Path("data")
OUT  = Path("figures"); OUT.mkdir(parents=True, exist_ok=True)

# ---------- load ----------
df_true = pd.read_csv(DATA/"True.csv"); df_true["label"] = 1
df_fake = pd.read_csv(DATA/"Fake.csv"); df_fake["label"] = 0
df = pd.concat([df_true, df_fake], ignore_index=True)
df["text"] = df["text"].astype(str).fillna("")

# prefer titles for display; if missing, make a short text snippet
def make_title(row):
    if "title" in row and isinstance(row["title"], str) and row["title"].strip():
        return row["title"]
    return textwrap.shorten(row["text"], width=120, placeholder="…")
df["disp_title"] = df.apply(make_title, axis=1)

# ---------- split (random, stratified) ----------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
)
titles_train = df.loc[X_train.index, "disp_title"].reset_index(drop=True)
titles_test  = df.loc[X_test.index,  "disp_title"].reset_index(drop=True)

# ---------- vectorize on TRAIN only ----------
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=2)
Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)

# ---------- simple calibrated classifier (for probabilities) ----------
base = LogisticRegression(max_iter=1000)
cal  = CalibratedClassifierCV(estimator=base, method="isotonic", cv=5)
cal.fit(Xtr, y_train)
probs = cal.predict_proba(Xte)[:, 1]
preds = (probs >= 0.5).astype(int)

# quick quality print (optional)
try:
    print("Random-split ROC-AUC:", roc_auc_score(y_test, probs))
except Exception:
    pass

# ---------- fit nearest neighbors on TRAIN set ----------
nn = NearestNeighbors(n_neighbors=5, metric="cosine")
nn.fit(Xtr)

def evidence_for_index(i, tag):
    """Create a small markdown table of 5 nearest neighbors for test sample i."""
    # neighbors
    dists, idxs = nn.kneighbors(Xte[i], n_neighbors=5, return_distance=True)
    idxs = idxs[0]; dists = dists[0]
    sims = 1.0 - dists  # cosine similarity

    # collect rows
    rows = []
    neigh_labels = y_train.iloc[idxs].values
    support = neigh_labels.mean()  # 1=True, 0=Fake
    for rank, (j, sim) in enumerate(zip(idxs, sims), start=1):
        rows.append({
            "Rank": rank,
            "Similarity": f"{sim:.3f}",
            "Train Label": "True" if y_train.iloc[j] == 1 else "Fake",
            "Neighbor Title / Snippet": titles_train.iloc[j]
        })

    # make markdown
    md_lines = []
    md_lines.append(f"### Evidence for {tag}")
    md_lines.append("")
    md_lines.append(f"**Test predicted**: {'True' if preds[i]==1 else 'Fake'}  "
                    f"(prob={probs[i]:.3f})")
    md_lines.append("")
    md_lines.append(f"**Test title/snippet:** {titles_test.iloc[i]}")
    md_lines.append("")
    md_lines.append(f"**Neighbor support score (fraction True among neighbors): {support:.2f}**")
    md_lines.append("")
    md_lines.append("| Rank | Similarity | Train Label | Neighbor Title / Snippet |")
    md_lines.append("|-----:|-----------:|-------------|---------------------------|")
    for r in rows:
        md_lines.append(f"| {r['Rank']} | {r['Similarity']} | {r['Train Label']} | {r['Neighbor Title / Snippet'].replace('|','/')} |")
    md = "\n".join(md_lines)
    return md

# ---------- pick one confident Fake and one confident True ----------
idx_fake = np.where((preds == 0) & (probs < 0.30))[0]
idx_true = np.where((preds == 1) & (probs > 0.70))[0]
if len(idx_fake) == 0: idx_fake = np.where(preds == 0)[0]
if len(idx_true) == 0: idx_true = np.where(preds == 1)[0]

if len(idx_fake) == 0 or len(idx_true) == 0:
    raise SystemExit("Could not find both a predicted Fake and True sample. Try rerunning or changing thresholds.")

md_fake = evidence_for_index(int(idx_fake[0]), "Predicted Fake")
md_true = evidence_for_index(int(idx_true[0]), "Predicted True")

# ---------- save to files you can open in VS Code and paste into PPT ----------
(fake_path := OUT/"evidence_fake.md").write_text(md_fake, encoding="utf-8")
(true_path := OUT/"evidence_true.md").write_text(md_true, encoding="utf-8")

print("\nSaved:")
print(" -", fake_path.resolve())
print(" -", true_path.resolve())
print("\nOpen these .md files in VS Code (Markdown preview).")
