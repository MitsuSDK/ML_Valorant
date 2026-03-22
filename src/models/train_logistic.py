import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]

# Load dataset
input_path = BASE_DIR / "data" / "processed" / "model_dataset.csv"
df = pd.read_csv(input_path)

df["date"] = pd.to_datetime(df["date"])

# --- Time-based split ---
train_df = df[df["date"] < "2025-10-01"]
test_df  = df[df["date"] >= "2025-10-01"]

X_train = train_df[["rating_diff", "kd_diff"]]
y_train = train_df["target"]

X_test = test_df[["rating_diff", "kd_diff"]]
y_test = test_df["target"]

# --- Train model ---
model = LogisticRegression()
model.fit(X_train, y_train)

# --- Predictions ---
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# --- Evaluation ---
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nModel Coefficients:")
print("rating_diff:", model.coef_[0][0])
print("kd_diff:", model.coef_[0][1])

print("Train size:", len(train_df))
print("Test size:", len(test_df))
print("Test positive ratio:", y_test.mean())
print("Mean predicted prob when team1 wins:",
      y_proba[y_test == 1].mean())

print("Mean predicted prob when team1 loses:",
      y_proba[y_test == 0].mean())