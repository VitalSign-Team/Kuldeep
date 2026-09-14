# ============================================================
# VITALSIGN HEALTHCARE PROJECT
# FAST LOGISTIC REGRESSION - READMISSION PREDICTION
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
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)


# ============================================================
# 1. FILE PATH
# ============================================================

DATA_PATH = r"C:\Users\ASUS\PycharmProjects\MainPro1\diabetes+130-us+hospitals+for+years+1999-2008\diabetic_data.csv"

OUTPUT_DIR = Path(
    r"/static/eda_output"
)

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("VITALSIGN - LOGISTIC REGRESSION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

# Replace ? with NaN
df = df.replace("?", np.nan)

print("\nDataset Shape:", df.shape)


# ============================================================
# 3. REMOVE UNNECESSARY COLUMNS
# ============================================================

columns_to_drop = [

    # Identifier columns
    "encounter_id",
    "patient_nbr",

    # Very high missing-value columns
    "weight",
    "payer_code",
    "medical_specialty"
]

df = df.drop(
    columns=[
        col for col in columns_to_drop
        if col in df.columns
    ]
)

print("\nShape After Removing Unnecessary Columns:")
print(df.shape)


# ============================================================
# 4. CREATE TARGET VARIABLE
# ============================================================

print("\nOriginal Readmission Categories:")

print(
    df["readmitted"].value_counts()
)


# <30 = 1
# >30 and NO = 0

df["readmitted_target"] = (

    df["readmitted"] == "<30"

).astype(int)


print("\nBinary Target Distribution:")

print(
    df["readmitted_target"].value_counts()
)


# ============================================================
# 5. FEATURES AND TARGET
# ============================================================

TARGET = "readmitted_target"


drop_columns = [

    "readmitted",

    "readmitted_target"

]


X = df.drop(

    columns=[
        col for col in drop_columns
        if col in df.columns
    ]

)


y = df[TARGET]


# ============================================================
# 6. FEATURE TYPES
# ============================================================

numeric_features = X.select_dtypes(

    include=[np.number]

).columns.tolist()


categorical_features = X.select_dtypes(

    include=[
        "object",
        "string",
        "category"
    ]

).columns.tolist()


print("\nTotal Features:", len(X.columns))

print("Numeric Features:", len(numeric_features))

print("Categorical Features:", len(categorical_features))


# ============================================================
# 7. TRAIN TEST SPLIT
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
# 8. PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(

    steps=[

        (

            "imputer",

            SimpleImputer(
                strategy="median"
            )

        )

    ]

)


categorical_transformer = Pipeline(

    steps=[

        (

            "imputer",

            SimpleImputer(
                strategy="most_frequent"
            )

        ),

        (

            "onehot",

            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=10
            )

        )

    ]

)


preprocessor = ColumnTransformer(

    transformers=[

        (

            "num",

            numeric_transformer,

            numeric_features

        ),

        (

            "cat",

            categorical_transformer,

            categorical_features

        )

    ]

)


# ============================================================
# 9. CREATE FAST LOGISTIC REGRESSION MODEL
# ============================================================

model = Pipeline(

    steps=[

        (

            "preprocessor",

            preprocessor

        ),

        (

            "classifier",

            LogisticRegression(

                max_iter=300,

                solver="lbfgs",

                class_weight="balanced"

            )

        )

    ]

)


# ============================================================
# 10. TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression Model...")

print("Please wait for model training to complete...")


model.fit(

    X_train,

    y_train

)


print("\nModel Training Completed Successfully!")


# ============================================================
# 11. PREDICTIONS
# ============================================================

print("\nMaking Predictions...")

y_pred = model.predict(

    X_test

)


y_probability = model.predict_proba(

    X_test

)[:, 1]


# ============================================================
# 12. MODEL PERFORMANCE
# ============================================================

accuracy = accuracy_score(

    y_test,

    y_pred

)


auc_score = roc_auc_score(

    y_test,

    y_probability

)


print("\n" + "=" * 60)

print("MODEL PERFORMANCE")

print("=" * 60)


print(

    f"Accuracy: {accuracy:.4f}"

)


print(

    f"ROC-AUC Score: {auc_score:.4f}"

)


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT")

print(

    classification_report(

        y_test,

        y_pred

    )

)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,

    y_pred

)


print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 15. GRAPH 1 - CONFUSION MATRIX
# ============================================================

print("\nDisplaying Confusion Matrix Graph...")


plt.figure(figsize=(8, 6))


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=[

        "Not Readmitted <30",

        "Readmitted <30"

    ],

    yticklabels=[

        "Not Readmitted <30",

        "Readmitted <30"

    ]

)


plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("VitalSign - Logistic Regression Confusion Matrix")


plt.tight_layout()


plt.savefig(

    OUTPUT_DIR /
    "logistic_confusion_matrix.png",

    dpi=150,

    bbox_inches="tight"

)


plt.show()

plt.close()


# ============================================================
# 16. GRAPH 2 - ROC CURVE
# ============================================================

print("\nDisplaying ROC Curve...")


fpr, tpr, thresholds = roc_curve(

    y_test,

    y_probability

)


plt.figure(figsize=(8, 6))


plt.plot(

    fpr,

    tpr,

    label=f"Logistic Regression (AUC = {auc_score:.3f})"

)


plt.plot(

    [0, 1],

    [0, 1],

    linestyle="--",

    label="Random Classifier"

)


plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("VitalSign - ROC Curve")


plt.legend()

plt.grid(True)


plt.tight_layout()


plt.savefig(

    OUTPUT_DIR /
    "logistic_roc_curve.png",

    dpi=150,

    bbox_inches="tight"

)


plt.show()

plt.close()


# ============================================================
# 17. GRAPH 3 - TARGET DISTRIBUTION
# ============================================================

print("\nDisplaying Readmission Distribution...")


plt.figure(figsize=(8, 6))


sns.countplot(

    data=df,

    x="readmitted",

    order=["NO", ">30", "<30"]

)


plt.xlabel("Readmission Status")

plt.ylabel("Number of Patients")

plt.title("VitalSign - Readmission Distribution")


plt.tight_layout()


plt.savefig(

    OUTPUT_DIR /
    "readmission_distribution.png",

    dpi=150,

    bbox_inches="tight"

)


plt.show()

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)

print("LOGISTIC REGRESSION COMPLETED SUCCESSFULLY!")

print("=" * 60)


print("\nTARGET VARIABLE:")

print("readmitted_target")


print("\nTARGET MEANING:")

print("1 = Patient readmitted within 30 days")

print("0 = Patient not readmitted within 30 days")


print("\nPROJECT GOAL:")

print(
    "Predict whether a diabetic patient "
    "will be readmitted within 30 days."
)


print("\nGRAPHS SAVED AT:")

print(OUTPUT_DIR)