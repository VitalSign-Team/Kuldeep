import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = (
    BASE_DIR
    / "diabetes+130-us+hospitals+for+years+1999-2008"
    / "diabetic_data.csv"
)

OUTPUT_DIR = BASE_DIR / "eda_output"
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("VITALSIGN - DECISION TREE")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

print("\nDataset Shape:", df.shape)

# ============================================================
# TARGET
# ============================================================

df["readmitted_target"] = (
    df["readmitted"].astype(str).str.strip() == "<30"
).astype(int)

print("\nTarget Distribution:")
print(df["readmitted_target"].value_counts())

print("\n1 = Readmitted within 30 days")
print("0 = Not readmitted within 30 days")

# ============================================================
# FEATURES
# ============================================================

drop_columns = [
    "readmitted",
    "readmitted_target",
    "encounter_id",
    "patient_nbr"
]

X = df.drop(
    columns=[c for c in drop_columns if c in df.columns]
)

y = df["readmitted_target"]

X = X.replace("?", np.nan)

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

print("\nTotal Features:", X.shape[1])
print("Numeric Features:", len(numeric_features))
print("Categorical Features:", len(categorical_features))

# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# ============================================================
# MODEL
# ============================================================

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=6,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42
)

pipeline = Pipeline([
    (
        "preprocessing",
        preprocessor
    ),
    (
        "model",
        model
    )
])

print("\nTraining Decision Tree...")

pipeline.fit(X_train, y_train)

print("Training Completed.")

# ============================================================
# PREDICTION
# ============================================================

y_pred = pipeline.predict(X_test)

y_prob = pipeline.predict_proba(X_test)[:, 1]

# ============================================================
# PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_prob
)

print("\n" + "=" * 70)
print("DECISION TREE PERFORMANCE")
print("=" * 70)

print("\nAccuracy:", round(accuracy, 4))
print("ROC-AUC:", round(auc, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Not Readmitted",
        "Readmitted <30"
    ],
    yticklabels=[
        "Not Readmitted",
        "Readmitted <30"
    ]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("VitalSign - Decision Tree Confusion Matrix")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "decision_tree_confusion_matrix.png",
    dpi=150
)

plt.show()
plt.close()

# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Decision Tree (AUC = {auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("VitalSign - Decision Tree ROC Curve")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "decision_tree_roc_curve.png",
    dpi=150
)

plt.show()
plt.close()

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

trained_preprocessor = pipeline.named_steps[
    "preprocessing"
]

trained_model = pipeline.named_steps[
    "model"
]

feature_names = (
    trained_preprocessor
    .get_feature_names_out()
)

importance = trained_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
    .head(15)
)

print("\nTop 15 Important Features:")
print(importance_df)

plt.figure(figsize=(10, 7))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title(
    "VitalSign - Decision Tree Feature Importance"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "decision_tree_feature_importance.png",
    dpi=150
)

plt.show()
plt.close()

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("DECISION TREE COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGraphs saved in:")
print(OUTPUT_DIR)