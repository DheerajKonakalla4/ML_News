# scripts/hmm_practice.py
# Practice 7: HMM for sequential data + 12-month forecast

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

DATA = Path("data")
FIGS = Path("figures")
FIGS.mkdir(parents=True, exist_ok=True)

print("Loading dataset ...")
df_true = pd.read_csv(DATA / "True.csv"); df_true["label"] = 1
df_fake = pd.read_csv(DATA / "Fake.csv"); df_fake["label"] = 0
df = pd.concat([df_true, df_fake], ignore_index=True)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

monthly = (
    df.assign(month=df["date"].dt.to_period("M").dt.to_timestamp())
      .groupby("month").size().sort_index()
)
print(f"Prepared monthly counts: {len(monthly)} months")

y = monthly.values.reshape(-1, 1)
months = monthly.index

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    print("\n⚠️ Please install hmmlearn first:\n   pip install hmmlearn\n")
    raise SystemExit

print("Training GaussianHMM ...")
best_model = None
best_k = None
best_bic = float("inf")

# Keep k small for short sequences
for k in [2, 3]:
    try:
        model = GaussianHMM(n_components=k, covariance_type="diag", n_iter=300, random_state=42)
        model.fit(y)
        logL = model.score(y)
        n_params = k * 3 + k - 1  # rough count
        bic = -2 * logL + n_params * np.log(len(y))
        if bic < best_bic:
            best_bic = bic
            best_model = model
            best_k = k
    except ValueError as e:
        print(f"⚠️ Skipping k={k}: {e}")
        continue

hmm = best_model
print(f"✅ Best model has {best_k} hidden states (lowest BIC={best_bic:.2f})")

states = hmm.predict(y)

# ---- Next-month mean/std (cast to float) ----
last_state = int(states[-1])
pred_mean = float(hmm.means_[last_state, 0])
pred_std  = float(np.sqrt(hmm.covars_[last_state, 0]))
print(f"Last detected state: {last_state}")
print(f"Predicted next-month article count ≈ {pred_mean:.1f} ± {pred_std:.1f}")

# ---- Plot states ----
plt.figure(figsize=(10, 5))
plt.plot(months, y[:, 0], linewidth=2, color="black", label="Article Count")
for s in range(best_k):
    plt.scatter(months[states == s], y[states == s, 0], s=40, label=f"State {s}")
plt.title(f"HMM (k={best_k}) on Monthly Article Counts")
plt.xlabel("Month"); plt.ylabel("Number of Articles")
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "hmm_monthly_states.png", dpi=160)
plt.close()
print("Saved → figures/hmm_monthly_states.png")

# ---- Forecast next 12 months (cast everything to float; flatten) ----
print("Simulating next 12 months ...")
current_state = last_state
synthetic = []
rng = np.random.default_rng(42)

for _ in range(12):
    trans_probs = hmm.transmat_[current_state]
    # if a row is zero (rare), use uniform probs
    if float(trans_probs.sum()) == 0.0:
        trans_probs = np.ones(best_k, dtype=float) / best_k
    # choose next state
    next_state = int(rng.choice(np.arange(best_k), p=trans_probs))
    # sample from Gaussian of that state
    mean = float(hmm.means_[next_state, 0])
    std  = float(np.sqrt(hmm.covars_[next_state, 0]))
    synthetic.append(float(rng.normal(loc=mean, scale=std)))
    current_state = next_state

# make 1-D array and clip negatives
synthetic = np.asarray(synthetic, dtype=float).ravel()
synthetic = np.maximum(synthetic, 0.0)

future_months = pd.date_range(start=months[-1] + pd.offsets.MonthBegin(1),
                              periods=12, freq="MS")
forecast_series = pd.Series(synthetic, index=future_months)

plt.figure(figsize=(10, 5))
plt.plot(months, y[:, 0], "b-", label="Actual")
plt.plot(forecast_series.index, forecast_series.values, "r--o", label="Forecast (Next 12 Months)")
plt.title("HMM Forecast – Next 12 Months of Article Counts")
plt.xlabel("Month"); plt.ylabel("Predicted Articles")
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "hmm_forecast_next12.png", dpi=160)
plt.close()
print("Saved → figures/hmm_forecast_next12.png")

# ---- State means bar ----
plt.figure(figsize=(6, 4))
plt.bar(range(best_k), hmm.means_[:, 0].astype(float), color="skyblue")
plt.xlabel("Hidden State"); plt.ylabel("Mean Article Count")
plt.title("HMM State Means (Avg Activity per Hidden State)")
plt.tight_layout()
plt.savefig(FIGS / "hmm_state_means.png", dpi=160)
plt.close()
print("Saved → figures/hmm_state_means.png")

print("\nHMM Practice complete ✅")
