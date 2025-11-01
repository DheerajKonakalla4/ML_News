# ...existing code...
# scripts/clustering_fast.py
# Run:
#   python scripts/clustering_fast.py           # FAST mode (default)
#   python scripts/clustering_fast.py --full    # FULL mode (slower, higher fidelity)

from pathlib import Path
import os, time, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# ---------------------------
# CLI & config
# ---------------------------
parser = argparse.ArgumentParser(description="Clustering (KMeans/GMM/Hierarchical) with FAST/FULL modes")
parser.add_argument("--full", action="store_true", help="Run FULL mode (slower, higher fidelity)")
args = parser.parse_args()

FAST = not args.full  # default fast
RANDOM_STATE = 42

if FAST:
    MAX_FEATS = 20000
    N_SVD = 30
    SIL_SAMPLES = 4000
    HIER_SAMPLES = 3000
    PLOT_SAMPLES = 8000
    GMM_COV = "diag"
    GMM_MAX_ITER = 100
    GMM_TOL = 1e-2
    KM_BATCH = 2048
    KM_N_INIT = 5
else:
    MAX_FEATS = 50000
    N_SVD = 50
    SIL_SAMPLES = None  # full silhouette on all points (can be slow!)
    HIER_SAMPLES = None  # cluster all (can be slow!)
    PLOT_SAMPLES = None  # plot all
    GMM_COV = "full"
    GMM_MAX_ITER = 200
    GMM_TOL = 1e-3
    KM_BATCH = None      # MiniBatch still used; batch adapts
    KM_N_INIT = 10

DATA = Path("data")
FIGS = Path("figures"); FIGS.mkdir(parents=True, exist_ok=True)

def tstamp(msg, t0):
    t1 = time.time()
    print(f"[{t1 - t0:7.2f}s] {msg}")
    return t1

def safe_silhouette(X, labels, sample_size=None, random_state=RANDOM_STATE):
    # need at least 2 clusters present
    if len(np.unique(labels)) < 2 or X.shape[0] < 2:
        return float("nan")
    # clamp sample_size to n_samples
    if sample_size is not None:
        sample_size = min(sample_size, X.shape[0])
        if sample_size < 2:
            return float("nan")
        return silhouette_score(X, labels, sample_size=sample_size, random_state=random_state)
    return silhouette_score(X, labels)

def scatter_2d(points, labels, title, outpath):
    plt.figure(figsize=(7,5.4))
    plt.scatter(points[:,0], points[:,1], c=labels, s=10, alpha=0.85)
    plt.title(title)
    plt.xlabel("SVD-1"); plt.ylabel("SVD-2")
    plt.tight_layout(); plt.savefig(outpath, dpi=160); plt.close()

def maybe_sample(X, n_samples, rng):
    if n_samples is None or n_samples >= X.shape[0]:
        return np.arange(X.shape[0])
    return rng.choice(X.shape[0], size=n_samples, replace=False)

def main():
    print("Mode:", "FAST" if FAST else "FULL")
    rng = np.random.RandomState(RANDOM_STATE)
    t0 = time.time()

    # ---------------------------
    # 1) Load data
    # ---------------------------
    df_true = pd.read_csv(DATA/"True.csv"); df_true["label"] = 1
    df_fake = pd.read_csv(DATA/"Fake.csv"); df_fake["label"] = 0
    df = pd.concat([df_true, df_fake], ignore_index=True)
    df["text"] = df["text"].astype(str).fillna("")
    t = tstamp("Loaded data", t0)

    # ---------------------------
    # 2) TF-IDF -> SVD (dense)
    # ---------------------------
    vec = TfidfVectorizer(max_features=MAX_FEATS, ngram_range=(1,2), min_df=5 if FAST else 2)
    X_tfidf = vec.fit_transform(df["text"])
    t = tstamp(f"TF-IDF vectorized (features={X_tfidf.shape[1]:,})", t)

    svd = TruncatedSVD(n_components=N_SVD, random_state=RANDOM_STATE)
    X = svd.fit_transform(X_tfidf)
    t = tstamp(f"SVD reduced to {N_SVD} dims", t)

    scaler = StandardScaler()
    X2 = scaler.fit_transform(X)  # good for GMM
    t = tstamp("Standardized SVD features", t)

    # Prepare indices for plotting (to keep figures nimble)
    plot_idx = maybe_sample(X, PLOT_SAMPLES, rng)
    X_plot = X[plot_idx]

    # ---------------------------
    # 3) MiniBatch K-Means
    # ---------------------------
    # compute batch size if adaptive
    km_batch = KM_BATCH
    if km_batch is None:
        km_batch = max(2048, min(8192, max(1, X.shape[0]//4)))

    kmeans = MiniBatchKMeans(
        n_clusters=2,
        random_state=RANDOM_STATE,
        n_init=KM_N_INIT,
        batch_size=km_batch,
        max_no_improvement=10
    )
    km_labels = kmeans.fit_predict(X)
    km_sil = safe_silhouette(X, km_labels, sample_size=SIL_SAMPLES)
    scatter_2d(X_plot[:, :2], km_labels[plot_idx], f"MiniBatchKMeans (k=2)  sil={km_sil:.3f}", FIGS/"cluster_kmeans.png")
    t = tstamp("KMeans done", t)

    # ---------------------------
    # 4) Gaussian Mixture (Mixture of Gaussians)
    # ---------------------------
    gmm = GaussianMixture(
        n_components=2,
        covariance_type=GMM_COV,
        max_iter=GMM_MAX_ITER,
        tol=GMM_TOL,
        random_state=RANDOM_STATE
    )
    gmm_labels = gmm.fit_predict(X2)
    gmm_sil = safe_silhouette(X2, gmm_labels, sample_size=SIL_SAMPLES)
    scatter_2d(X_plot[:, :2], gmm_labels[plot_idx], f"GMM ({GMM_COV}) (k=2)  sil={gmm_sil:.3f}", FIGS/"cluster_gmm.png")
    t = tstamp("GMM done", t)

    # ---------------------------
    # 5) Hierarchical (Agglomerative) on subset
    # ---------------------------
    h_idx = maybe_sample(X, HIER_SAMPLES, rng)
    X_h = X[h_idx]
    agg = AgglomerativeClustering(n_clusters=2, linkage="ward")
    agg_labels = agg.fit_predict(X_h)
    agg_sil = safe_silhouette(X_h, agg_labels, sample_size=SIL_SAMPLES if FAST else None)
    scatter_2d(X_h[:, :2], agg_labels, f"Agglomerative (subset) (k=2)  sil={agg_sil:.3f}", FIGS/"cluster_agg.png")
    t = tstamp("Agglomerative done", t)

    # ---------------------------
    # 6) Report
    # ---------------------------
    print("\nSilhouette scores (higher ≈ better):")
    print(f" - MiniBatchKMeans: {km_sil:.3f}")
    print(f" - GMM ({GMM_COV}): {gmm_sil:.3f}")
    print(f" - Agglomerative (subset): {agg_sil:.3f}")
    print("Saved plots → figures/cluster_kmeans.png, cluster_gmm.png, cluster_agg.png")
    print(f"TOTAL time: {time.time() - t0:0.2f}s")

if __name__ == "__main__":
    main()