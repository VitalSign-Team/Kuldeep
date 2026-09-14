# ============================================================
# VitalSign - Healthcare Analytics & Machine Learning
# Flask Application
# ============================================================

from flask import Flask, render_template, jsonify
import os
import io
import base64

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from Clustering.kmeans_simple import run_kmeans_simple
from Clustering.kmeans_elbow import run_kmeans_elbow


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# BASE DIRECTORY & DATASET PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET = os.path.join(
    BASE_DIR,
    "datasets",
    "diabetic_data.csv"
)

PLACEMENT = os.path.join(
    BASE_DIR,
    "datasets",
    "placement_preprocessed.csv"
)


# ============================================================
# GENERAL SETTINGS
# ============================================================

MAX_ROWS = 30000


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_healthcare_data():
    """
    Load the Diabetes 130-US Hospitals dataset.
    '?' values are treated as missing values.
    """

    if not os.path.exists(DATASET):
        raise FileNotFoundError(
            f"Healthcare dataset not found:\n{DATASET}"
        )

    df = pd.read_csv(
        DATASET,
        na_values=["?"]
    )

    return df


def fig_to_base64(fig):
    """
    Convert matplotlib figure into base64 string
    for displaying inside HTML.
    """

    buffer = io.BytesIO()

    fig.tight_layout()

    fig.savefig(
        buffer,
        format="png",
        dpi=120,
        bbox_inches="tight"
    )

    plt.close(fig)

    buffer.seek(0)

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


def get_summary(df):
    """
    Generate dataset summary.

    Missing data is calculated both as:
    1. Actual missing cells
    2. Missing percentage
    """

    rows = int(df.shape[0])
    columns = int(df.shape[1])

    missing_cells = int(
        df.isna().sum().sum()
    )

    total_cells = int(
        rows * columns
    )

    missing_percentage = (
        (missing_cells / total_cells) * 100
        if total_cells > 0
        else 0
    )

    duplicates = int(
        df.duplicated().sum()
    )

    return {
        "rows": rows,
        "columns": columns,
        "missing": missing_cells,
        "missing_percentage": round(
            missing_percentage,
            2
        ),
        "duplicates": duplicates
    }


def page(active, **kwargs):
    """
    Common template renderer.
    """

    return render_template(
        "index.html",
        active=active,
        title="VitalSign",
        **kwargs
    )


# ============================================================
# CLASSIFICATION MODEL FUNCTION
# ============================================================

def run_classification_model(model, model_name):
    """
    Train and evaluate a classification model.

    Target:
        readmitted == '<30'

    Features:
        time_in_hospital
        num_lab_procedures
        num_medications
        number_diagnoses
    """

    df = load_healthcare_data()

    features = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_medications",
        "number_diagnoses"
    ]

    target = "readmitted"

    required = features + [target]

    missing_columns = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    data = df[required].copy()

    # Convert numerical columns
    for col in features:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    # Target
    data["target"] = (
        data[target] == "<30"
    ).astype(int)

    # Remove rows with missing feature values
    data = data.dropna(
        subset=features
    )

    # Limit data for faster execution
    if len(data) > MAX_ROWS:
        data = data.sample(
            n=MAX_ROWS,
            random_state=42
        )

    X = data[features]
    y = data["target"]

    if y.nunique() < 2:
        raise ValueError(
            "Classification target contains only one class."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # Train
    model.fit(
        X_train_scaled,
        y_train
    )

    # Predict
    y_pred = model.predict(
        X_test_scaled
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    importance = {}

    if hasattr(model, "feature_importances_"):

        values = model.feature_importances_

        importance = {
            feature: round(
                float(value),
                4
            )
            for feature, value
            in zip(features, values)
        }

    elif hasattr(model, "coef_"):

        values = np.abs(
            model.coef_[0]
        )

        importance = {
            feature: round(
                float(value),
                4
            )
            for feature, value
            in zip(features, values)
        }

    # Sort feature importance
    importance = dict(
        sorted(
            importance.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    # ========================================================
    # CONFUSION MATRIX PLOT
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.imshow(cm)

    ax.set_title(
        f"{model_name} - Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Not <30", "<30"]
    )

    ax.set_yticklabels(
        ["Not <30", "<30"]
    )

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    ax.grid(False)

    image = fig_to_base64(
        fig
    )

    return {
        "name": model_name,

        "accuracy": round(
            float(accuracy * 100),
            2
        ),

        "train": int(
            len(X_train)
        ),

        "test": int(
            len(X_test)
        ),

        "features": features,

        "image": image,

        "details": {
            "confusion_matrix": cm.tolist(),

            "feature_importance": importance,

            "classification_report": report
        }
    }


# ============================================================
# LINEAR REGRESSION
# ============================================================

def run_linear_regression():
    """
    Predict time_in_hospital using other numerical
    healthcare features.
    """

    df = load_healthcare_data()

    target = "time_in_hospital"

    candidate_features = [
        "num_lab_procedures",
        "num_medications",
        "number_diagnoses",
        "number_inpatient",
        "number_emergency",
        "number_outpatient",
        "num_procedures"
    ]

    features = [
        col
        for col in candidate_features
        if col in df.columns
    ]

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found."
        )

    if not features:
        raise ValueError(
            "No valid numerical features found."
        )

    data = df[
        features + [target]
    ].copy()

    for col in features + [target]:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = data.dropna()

    if len(data) > MAX_ROWS:

        data = data.sample(
            n=MAX_ROWS,
            random_state=42
        )

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    # ========================================================
    # ACTUAL VS PREDICTED
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.scatter(
        y_test,
        y_pred,
        alpha=0.5,
        s=25
    )

    min_value = min(
        y_test.min(),
        y_pred.min()
    )

    max_value = max(
        y_test.max(),
        y_pred.max()
    )

    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        linewidth=2
    )

    ax.set_xlabel(
        "Actual Time in Hospital"
    )

    ax.set_ylabel(
        "Predicted Time in Hospital"
    )

    ax.set_title(
        "Linear Regression - Actual vs Predicted"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    image = fig_to_base64(
        fig
    )

    return {
        "name": "Linear Regression",

        "mae": round(
            float(mae),
            3
        ),

        "mse": round(
            float(mse),
            3
        ),

        "r2": round(
            float(r2),
            3
        ),

        "features": features,

        "target": target,

        "train": int(
            len(X_train)
        ),

        "test": int(
            len(X_test)
        ),

        "image": image
    }


# ============================================================
# PREPROCESSING
# ============================================================

def run_preprocessing():
    """
    Preprocess placement dataset.

    Operations:
    - Select numerical features
    - Fill missing values
    - Min-Max normalization
    - IQR outlier detection
    """

    if not os.path.exists(PLACEMENT):
        raise FileNotFoundError(
            f"Placement dataset not found:\n{PLACEMENT}"
        )

    df = pd.read_csv(
        PLACEMENT
    )

    # Candidate columns
    candidates = [
        "CGPA",
        "AttendancePercent",
        "AptitudeTestScore",
        "CodingTestScore",

        "numerical__CGPA",
        "numerical__AttendancePercent",
        "numerical__AptitudeTestScore",
        "numerical__CodingTestScore"
    ]

    features = [
        col
        for col in candidates
        if col in df.columns
    ]

    # Remove duplicate feature names
    features = list(
        dict.fromkeys(features)
    )

    # Fallback: numerical columns
    if not features:

        features = df.select_dtypes(
            include=np.number
        ).columns.tolist()

    if not features:
        raise ValueError(
            "No numerical columns available in placement dataset."
        )

    data = df[features].copy()

    # Convert to numerical
    for col in features:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    # Missing values before filling
    missing_before = int(
        data.isna().sum().sum()
    )

    # Median imputation
    data = data.fillna(
        data.median(numeric_only=True)
    )

    # ========================================================
    # MIN-MAX NORMALIZATION
    # ========================================================

    normalized = pd.DataFrame(
        index=data.index
    )

    for col in features:

        minimum = data[col].min()
        maximum = data[col].max()

        if maximum != minimum:

            normalized[col] = (
                (data[col] - minimum)
                /
                (maximum - minimum)
            )

        else:

            normalized[col] = 0.0

    # ========================================================
    # IQR OUTLIERS
    # ========================================================

    outlier_mask = pd.Series(
        False,
        index=data.index
    )

    outlier_counts = {}

    for col in features:

        q1 = data[col].quantile(
            0.25
        )

        q3 = data[col].quantile(
            0.75
        )

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        current_outliers = (
            (data[col] < lower)
            |
            (data[col] > upper)
        )

        outlier_mask = (
            outlier_mask
            |
            current_outliers
        )

        outlier_counts[col] = int(
            current_outliers.sum()
        )

    total_outliers = int(
        outlier_mask.sum()
    )

    return {
        "rows": int(len(data)),

        "columns": int(len(features)),

        "features": features,

        "missing_before": missing_before,

        "missing_after": int(
            normalized.isna().sum().sum()
        ),

        "outliers": total_outliers,

        "outlier_counts": outlier_counts
    }


# ============================================================
# HOME / DASHBOARD
# ============================================================

@app.route("/")
def home():

    df = load_healthcare_data()

    summary = get_summary(
        df
    )

    preview_df = df.head(
        10
    )

    preview = preview_df.to_dict(
        orient="records"
    )

    columns = preview_df.columns.tolist()

    return page(
        "dashboard",
        summary=summary,
        preview=preview,
        columns=columns
    )


# ============================================================
# DATASET
# ============================================================

@app.route("/dataset")
def dataset():

    df = load_healthcare_data()

    summary = get_summary(
        df
    )

    preview = df.head(
        30
    )

    data_html = preview.to_html(
        classes="data-table",
        index=False,
        border=0
    )

    columns = df.columns.tolist()

    return page(
        "dataset",
        summary=summary,
        data=data_html,
        columns=columns
    )


# ============================================================
# DATA LOADING
# ============================================================

@app.route("/data-loading")
def data_loading():

    df = load_healthcare_data()

    summary = get_summary(
        df
    )

    preview_df = df.head(
        15
    )

    preview = preview_df.to_dict(
        orient="records"
    )

    columns = preview_df.columns.tolist()

    return page(
        "data-loading",
        summary=summary,
        preview=preview,
        columns=columns
    )


# ============================================================
# EDA
# ============================================================

@app.route("/eda")
def eda():

    df = load_healthcare_data()

    images = {}

    # --------------------------------------------------------
    # 1. Age Distribution
    # --------------------------------------------------------

    if "age" in df.columns:

        age_data = (
            df["age"]
            .astype(str)
            .str.extract(r"\[(\d+)-(\d+)\]")
        )

        if not age_data.empty:

            age_labels = (
                age_data[0]
                + "-"
                + age_data[1]
            )

            counts = (
                age_labels
                .value_counts()
                .sort_index()
            )

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            ax.bar(
                counts.index,
                counts.values
            )

            ax.set_title(
                "Patient Age Distribution"
            )

            ax.set_xlabel(
                "Age Group"
            )

            ax.set_ylabel(
                "Number of Encounters"
            )

            ax.tick_params(
                axis="x",
                rotation=45
            )

            ax.grid(
                axis="y",
                alpha=0.3
            )

            images["Age Distribution"] = (
                fig_to_base64(fig)
            )

    # --------------------------------------------------------
    # 2. Readmission Distribution
    # --------------------------------------------------------

    if "readmitted" in df.columns:

        counts = (
            df["readmitted"]
            .fillna("Missing")
            .value_counts()
        )

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        ax.bar(
            counts.index.astype(str),
            counts.values
        )

        ax.set_title(
            "Patient Readmission Distribution"
        )

        ax.set_xlabel(
            "Readmission Status"
        )

        ax.set_ylabel(
            "Number of Encounters"
        )

        ax.grid(
            axis="y",
            alpha=0.3
        )

        images["Readmission Distribution"] = (
            fig_to_base64(fig)
        )

    # --------------------------------------------------------
    # 3. Time in Hospital
    # --------------------------------------------------------

    if "time_in_hospital" in df.columns:

        values = pd.to_numeric(
            df["time_in_hospital"],
            errors="coerce"
        ).dropna()

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        ax.hist(
            values,
            bins=15,
            alpha=0.75
        )

        ax.set_title(
            "Time in Hospital Distribution"
        )

        ax.set_xlabel(
            "Days"
        )

        ax.set_ylabel(
            "Frequency"
        )

        ax.grid(
            axis="y",
            alpha=0.3
        )

        images["Time in Hospital"] = (
            fig_to_base64(fig)
        )

    # --------------------------------------------------------
    # 4. Numerical Correlation
    # --------------------------------------------------------

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.shape[1] >= 2:

        corr = numeric_df.corr()

        fig, ax = plt.subplots(
            figsize=(9, 7)
        )

        image_plot = ax.imshow(
            corr,
            aspect="auto"
        )

        ax.set_title(
            "Numerical Feature Correlation"
        )

        ax.set_xticks(
            range(len(corr.columns))
        )

        ax.set_yticks(
            range(len(corr.columns))
        )

        ax.set_xticklabels(
            corr.columns,
            rotation=90
        )

        ax.set_yticklabels(
            corr.columns
        )

        fig.colorbar(
            image_plot,
            ax=ax
        )

        images["Correlation Matrix"] = (
            fig_to_base64(fig)
        )

    summary = get_summary(
        df
    )

    return page(
        "eda",
        summary=summary,
        images=images
    )


# ============================================================
# PREPROCESSING ROUTE
# ============================================================

@app.route("/preprocessing")
def preprocessing():

    data = run_preprocessing()

    return page(
        "preprocessing",
        data=data
    )


# ============================================================
# LINEAR REGRESSION ROUTE
# ============================================================

@app.route("/linear-regression")
def linear_regression():

    result = run_linear_regression()

    return page(
        "linear-regression",
        result=result
    )


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

@app.route("/logistic-regression")
def logistic_regression():

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    result = run_classification_model(
        model,
        "Logistic Regression"
    )

    return page(
        "logistic-regression",
        result=result
    )


# ============================================================
# DECISION TREE
# ============================================================

@app.route("/decision-tree")
def decision_tree():

    model = DecisionTreeClassifier(
        max_depth=8,
        random_state=42
    )

    result = run_classification_model(
        model,
        "Decision Tree"
    )

    return page(
        "decision-tree",
        result=result
    )


# ============================================================
# GRADIENT BOOSTING
# ============================================================

@app.route("/gradient-boosting")
def gradient_boosting():

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    result = run_classification_model(
        model,
        "Gradient Boosting"
    )

    return page(
        "gradient-boosting",
        result=result
    )


# ============================================================
# ADABOOST
# ============================================================

@app.route("/adaboost")
def adaboost():

    model = AdaBoostClassifier(
        n_estimators=100,
        random_state=42
    )

    result = run_classification_model(
        model,
        "AdaBoost"
    )

    return page(
        "adaboost",
        result=result
    )


# ============================================================
# BAGGING
# ============================================================

@app.route("/bagging")
def bagging():

    base_model = DecisionTreeClassifier(
        max_depth=8,
        random_state=42
    )

    model = BaggingClassifier(
        estimator=base_model,
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    )

    result = run_classification_model(
        model,
        "Bagging"
    )

    return page(
        "bagging",
        result=result
    )


# ============================================================
# K-MEANS
# ============================================================

@app.route("/k-means")
def kmeans():

    result = run_kmeans_simple(
        PLACEMENT
    )

    return page(
        "k-means",
        result=result
    )


# ============================================================
# K-MEANS ELBOW
# ============================================================

@app.route("/k-means-elbow")
def kmeans_elbow():

    result = run_kmeans_elbow(
        PLACEMENT
    )

    return page(
        "k-means-elbow",
        result=result
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

@app.route("/model-comparison")
def model_comparison():

    models = [

        (
            LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
            "Logistic Regression"
        ),

        (
            DecisionTreeClassifier(
                max_depth=8,
                random_state=42
            ),
            "Decision Tree"
        ),

        (
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ),
            "Gradient Boosting"
        ),

        (
            AdaBoostClassifier(
                n_estimators=100,
                random_state=42
            ),
            "AdaBoost"
        ),

        (
            BaggingClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=8,
                    random_state=42
                ),
                n_estimators=50,
                random_state=42,
                n_jobs=-1
            ),
            "Bagging"
        )
    ]

    results = []

    for model, name in models:

        result = run_classification_model(
            model,
            name
        )

        results.append({
            "name": result["name"],
            "accuracy": result["accuracy"],
            "train": result["train"],
            "test": result["test"],
            "image": result["image"]
        })

    # Highest accuracy first
    results.sort(
        key=lambda x: x["accuracy"],
        reverse=True
    )

    return page(
        "model-comparison",
        results=results
    )


# ============================================================
# ABOUT
# ============================================================

@app.route("/about")
def about():

    return page(
        "about"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "online",
        "application": "VitalSign",
        "message": "VitalSign ML Engine is running"
    })


# ============================================================
# ERROR HANDLER
# ============================================================

@app.errorhandler(Exception)
def handle_error(error):

    print("\n========================================")
    print("VITALSIGN ERROR")
    print("========================================")
    print(error)
    print("========================================\n")

    return f"""
    <div style="
        font-family: Arial;
        padding: 40px;
        background: #f8fafc;
        color: #1e293b;
    ">

        <h1 style="color:#dc2626;">
            VitalSign Error
        </h1>

        <p>
            Something went wrong while processing this page.
        </p>

        <pre style="
            background:#ffffff;
            padding:20px;
            border-radius:10px;
            border:1px solid #e2e8f0;
            overflow:auto;
        ">{error}</pre>

        <a href="/" style="
            display:inline-block;
            margin-top:20px;
            padding:12px 20px;
            background:#2563eb;
            color:white;
            text-decoration:none;
            border-radius:8px;
        ">
            ← Back to Dashboard
        </a>

    </div>
    """, 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("==============================================")
    print("        VITALSIGN HEALTHCARE ANALYTICS")
    print("==============================================")
    print(f"Dataset   : {DATASET}")
    print(f"Placement : {PLACEMENT}")
    print("Server    : http://127.0.0.1:5000")
    print("==============================================")
    print("\n")

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )