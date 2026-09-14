# ============================================================
# VITALSIGN HEALTHCARE PROJECT
# LINEAR REGRESSION - LENGTH OF STAY PREDICTION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. FILE PATH
# ============================================================

DATA_PATH = r"C:\Users\ASUS\PycharmProjects\MainPro1\diabetes+130-us+hospitals+for+years+1999-2008\diabetic_data.csv"

OUTPUT_DIR = Path("static/eda_output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("VITALSIGN - LINEAR REGRESSION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

# Replace ? with missing values
df.replace("?", np.nan, inplace=True)

print("\nDataset Shape:", df.shape)


# ============================================================
# 3. TARGET VARIABLE
# ============================================================

TARGET = "time_in_hospital"

print("\nTarget Variable:", TARGET)


# ============================================================
# 4. REMOVE UNNECESSARY ID COLUMNS
# ============================================================

drop_columns = [
    TARGET,
    "encounter_id",
    "patient_nbr"
]

X = df.drop(
    columns=[col for col in drop_columns if col in df.columns]
)

y = df[TARGET]


# ============================================================
# 5. IDENTIFY NUMERIC AND CATEGORICAL COLUMNS
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()


print("\nNumber of Features:", len(X.columns))
print("Numeric Features:", len(numeric_features))
print("Categorical Features:", len(categorical_features))


# ============================================================
# 6. PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)


categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ============================================================
# 7. CREATE MODEL PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regression", LinearRegression())
    ]
)


# ============================================================
# 8. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ============================================================
# 9. TRAIN MODEL
# ============================================================

print("\nTraining Linear Regression Model...")

model.fit(X_train, y_train)


# ============================================================
# 10. PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 11. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R² Score : {r2:.4f}")


# ============================================================
# 12. GRAPH 1 - ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.4
)

plt.xlabel("Actual Time in Hospital (Days)")
plt.ylabel("Predicted Time in Hospital (Days)")

plt.title(
    "VitalSign: Actual vs Predicted Length of Stay"
)

plt.grid(True)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    label="Perfect Prediction"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "linear_actual_vs_predicted.png",
    dpi=150
)

plt.show()

plt.close()


# ============================================================
# 13. GRAPH 2 - RESIDUAL PLOT
# ============================================================

residuals = y_test - y_pred

plt.figure(figsize=(10, 6))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.4
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Length of Stay")
plt.ylabel("Residuals")

plt.title(
    "VitalSign: Residual Plot"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "linear_residual_plot.png",
    dpi=150
)

plt.show()

plt.close()


# ============================================================
# 14. GRAPH 3 - DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    y,
    bins=14,
    kde=True
)

plt.xlabel("Time in Hospital (Days)")

plt.title(
    "Distribution of Length of Stay"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "length_of_stay_distribution.png",
    dpi=150
)

plt.show()

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("LINEAR REGRESSION COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nTarget:")
print("time_in_hospital")

print("\nGoal:")
print("Predict how many days a patient stays in hospital.")

print("\nGraphs saved in:")
print(OUTPUT_DIR)