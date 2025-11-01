# scripts/pca_practice.py
# Practice 6: Perform PCA on TF-IDF text features (True/Fake dataset)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Paths
# -----------------------------
DATA = Path("data")
FIGS = Path("figures")
FIGS.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1️⃣ Load Data
# -----------------------------
print("Loading dataset ...")
df_true = pd.read_csv(DATA / "True.csv");  df_true["label"] = 1
df_fake = pd.read_csv(DATA / "Fake.csv");  df_fake["label"] = 0
df = pd.concat([df_true, df_fake], ignore_index=True)
df["text"] = df["text"].astype(str).fillna("")
print(f"Loaded {len(df):,} articles")

# -----------------------------
# 2️⃣ TF-IDF Vectorization
# -----------------------------
print("Vectorizing text using TF-IDF ...")
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=5)
X_tfidf = vectorizer.fit_transform(df["text"])
print(f"TF-IDF shape: {X_tfidf.shape}")

# -----------------------------
# 3️⃣ PCA via TruncatedSVD
# -----------------------------
# For sparse text matrices, TruncatedSVD acts as PCA.
print("Performing TruncatedSVD (PCA) ...")
n_components = 50
svd = TruncatedSVD(n_components=n_components, random_state=42)
X_reduced = svd.fit_transform(X_tfidf)
explained_var = svd.explained_variance_ratio_

# -----------------------------
# 4️⃣ Explained Variance Plot
# -----------------------------
cum_var = np.cumsum(explained_var)
plt.figure(figsize=(7, 4.5))
plt.plot(range(1, n_components+1), cum_var, marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA (TruncatedSVD) - Explained Variance Curve")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(FIGS / "pca_cumulative_variance.png", dpi=160)
plt.close()
print("Saved → figures/pca_cumulative_variance.png")

# -----------------------------
# 5️⃣ 2D Scatter Plot (First 2 PCs)
# -----------------------------
labels = df["label"].values
colors = np.where(labels == 1, "tab:blue", "tab:orange")

plt.figure(figsize=(7, 5.5))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=colors, s=10, alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA (2D projection of TF-IDF features)")
plt.tight_layout()
plt.savefig(FIGS / "pca_2d_scatter.png", dpi=160)
plt.close()
print("Saved → figures/pca_2d_scatter.png")

# -----------------------------
# 6️⃣ Print Results
# -----------------------------
print(f"Top 5 component variances: {explained_var[:5]}")
print(f"Total variance explained by {n_components} components: {cum_var[-1]*100:.2f}%")
print("PCA Practice complete ✅")
