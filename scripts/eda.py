# scripts/eda.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA = Path("data")
FIGS = Path("figures"); FIGS.mkdir(parents=True, exist_ok=True)

df_true = pd.read_csv(DATA/"True.csv"); df_true["label"]="True"
df_fake = pd.read_csv(DATA/"Fake.csv"); df_fake["label"]="Fake"
df = pd.concat([df_true, df_fake], ignore_index=True)
df["text"] = df["text"].astype(str).fillna("")
df["text_len"] = df["text"].str.len()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

plt.rcParams.update({"figure.dpi":130,"savefig.dpi":160,"axes.grid":True,"grid.alpha":0.25})

# 1) Histogram
x = df["text_len"].clip(upper=df["text_len"].quantile(0.99))
plt.figure(figsize=(8.5,5))
plt.hist(x, bins=40, edgecolor="black", alpha=0.8)
plt.title("Histogram of Article Lengths"); plt.xlabel("Text length (characters)"); plt.ylabel("Frequency")
plt.tight_layout(); plt.savefig(FIGS/"hist_article_lengths.png"); plt.close()

# 2) Bar: True vs Fake
counts = df["label"].value_counts().reindex(["Fake","True"])
plt.figure(figsize=(6.5,5))
bars = plt.bar(counts.index, counts.values, alpha=0.85)
plt.title("Bar Chart: True vs Fake Count"); plt.xlabel("Label"); plt.ylabel("Number of Articles")
for b in bars: plt.text(b.get_x()+b.get_width()/2, b.get_height(), f"{int(b.get_height()):,}", ha="center", va="bottom")
plt.tight_layout(); plt.savefig(FIGS/"bar_true_vs_fake.png"); plt.close()

# 3) Pie: True vs Fake
plt.figure(figsize=(6,6))
plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor":"white"})
plt.title("Pie Chart: True vs Fake Proportion")
plt.axis("equal"); plt.tight_layout(); plt.savefig(FIGS/"pie_true_fake.png"); plt.close()

# 4) Boxplot by label
data = [df.loc[df["label"]=="Fake","text_len"], df.loc[df["label"]=="True","text_len"]]
plt.figure(figsize=(7.5,5.2))
plt.boxplot(data, labels=["Fake","True"], showfliers=True, patch_artist=True)
plt.title("Boxplot of Text Length by Label"); plt.xlabel("Label"); plt.ylabel("Text length (characters)")
plt.tight_layout(); plt.savefig(FIGS/"boxplot_text_length_by_label.png"); plt.close()

# 5) Line: Articles over time (monthly)
monthly = (df.dropna(subset=["month"]).groupby("month").size().sort_index())
plt.figure(figsize=(9.5,5.2))
plt.plot(monthly.index, monthly.values, marker="o", linewidth=2)
plt.title("Line Chart: Articles Over Time"); plt.xlabel("Month"); plt.ylabel("Number of Articles")
plt.gcf().autofmt_xdate(); plt.tight_layout(); plt.savefig(FIGS/"line_articles_over_time.png"); plt.close()

print("Saved charts to", FIGS.resolve())
