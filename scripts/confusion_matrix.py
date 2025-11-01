from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

DATA, FIGS = Path("data"), Path("figures"); FIGS.mkdir(exist_ok=True, parents=True)

df_true = pd.read_csv(DATA/"True.csv"); df_true["label"]=1
df_fake = pd.read_csv(DATA/"Fake.csv"); df_fake["label"]=0
df = pd.concat([df_true, df_fake], ignore_index=True)
df["text"] = df["text"].astype(str).fillna("")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

vec = TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=2)
Xtr = vec.fit_transform(X_train); Xte = vec.transform(X_test)

base = LogisticRegression(max_iter=1000)
cal  = CalibratedClassifierCV(estimator=base, method="isotonic", cv=5)
cal.fit(Xtr, y_train)
pred = cal.predict(Xte)

cm = confusion_matrix(y_test, pred, labels=[0,1])
disp = ConfusionMatrixDisplay(cm, display_labels=["Fake(0)","True(1)"])
disp.plot(values_format="d")
plt.title("Confusion Matrix (Random Split)")
plt.tight_layout()
plt.savefig(FIGS/"confusion_matrix.png", dpi=160)
print("Saved →", (FIGS/"confusion_matrix.png").resolve())
