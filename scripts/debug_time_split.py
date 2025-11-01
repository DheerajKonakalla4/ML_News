# scripts/debug_time_split.py
from pathlib import Path
import pandas as pd

DATA = Path("data")
df_true = pd.read_csv(DATA/"True.csv"); df_true["label"]=1
df_fake = pd.read_csv(DATA/"Fake.csv"); df_fake["label"]=0
df = pd.concat([df_true, df_fake], ignore_index=True)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

print("Total rows:", len(df))
print("Rows with valid date:", df["date"].notna().sum())
print(df["label"].value_counts().rename({0:"Fake",1:"True"}), "\n")

# try a tentative cut and show class counts on each side
cut = pd.Timestamp("2017-07-01")
left  = df[(df["date"].notna()) & (df["date"] <  cut)]
right = df[(df["date"].notna()) & (df["date"] >= cut)]
print("Cut:", cut.date())
print("TRAIN counts by label:\n", left["label"].value_counts())
print("TEST  counts by label:\n", right["label"].value_counts())
