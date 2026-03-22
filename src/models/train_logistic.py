import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parents[2]

# Load dataset
input_path = BASE_DIR / "data" / "processed" / "model_dataset.csv"
df = pd.read_csv(input_path)

df["date"] = pd.to_datetime(df["date"])

# --- Time-based split ---
train_df = df[df["date"] < "2025-10-01"]
test_df  = df[df["date"] >= "2025-10-01"]

# -----------------------------
# Feature sets
# -----------------------------
feature_sets = {
    "Model_A_baseline": [
        "rating_decay_map_diff",
        "rating_decay_global_diff",
        "kd_decay_global_diff"
    ],
    "Model_B_plus_experience": [
        "rating_decay_map_diff",
        "rating_decay_global_diff",
        "kd_decay_global_diff",
        "experience_diff"
    ],
    "Model_C_full_with_winrate": [
        "rating_decay_map_diff",
        "kd_decay_map_diff",
        "rating_decay_global_diff",
        "kd_decay_global_diff",
        "experience_diff",
        "winrate_map_decay_diff",
        "winrate_global_decay_diff"
    ]
}

print("\n==============================")
print("LOGISTIC REGRESSION MODELS")
print("==============================")

for model_name, feature_cols in feature_sets.items():

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\n{model_name}")
    print(f"Features: {feature_cols}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

# -----------------------------
# Random Forest (best feature set)
# -----------------------------
print("\n==============================")
print("RANDOM FOREST")
print("==============================")

rf_features = feature_sets["Model_A_baseline"]

X_train = train_df[rf_features]
y_train = train_df["target"]
X_test = test_df[rf_features]
y_test = test_df["target"]

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\nRandomForest (Baseline Features)")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nFeature Importances:")
for name, importance in zip(rf_features, rf.feature_importances_):
    print(f"{name}: {importance}")

print("\nTrain size:", len(train_df))
print("Test size:", len(test_df))
print("Test positive ratio:", y_test.mean())