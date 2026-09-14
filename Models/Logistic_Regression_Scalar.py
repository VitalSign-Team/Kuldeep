import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

def load_data(filename):
    data = pd.read_csv(filename)

    features = [
        "CGPA",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore"
    ]

    # Select input features and target
    X = data[features]
    y = data["PlacementStatus"]

    # Remove missing values
    data = pd.concat([X, y], axis=1).dropna()

    X = data[features]
    y = data["PlacementStatus"]

    return X, y


# --------------------------------------------------
# 2. Split Dataset
# --------------------------------------------------

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# --------------------------------------------------
# 3. Train and Evaluate
# --------------------------------------------------

def train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
        name
):
    model = LogisticRegression(max_iter=1000)

    # Train model
    model.fit(X_train, y_train)

    # Training prediction
    train_pred = model.predict(X_train)

    # Testing prediction
    test_pred = model.predict(X_test)

    # Accuracy
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    print()
    print(name)
    print("-----------------------------")
    print("Train Accuracy:", round(train_accuracy, 4))
    print("Test Accuracy :", round(test_accuracy, 4))

    return model


# --------------------------------------------------
# 4. Main Program
# --------------------------------------------------

def main():
    # Your CSV file path
    filename = r"E:\PythonProject2\placement_predict_50k Dataset (3)(in).csv"

    # Load data
    X, y = load_data(filename)

    print("Dataset loaded successfully!")
    print("Number of rows:", len(X))
    print("Number of features:", X.shape[1])

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training samples:", len(X_train))
    print("Testing samples :", len(X_test))

    # --------------------------------------------------
    # Unscaled Logistic Regression
    # --------------------------------------------------

    model_unscaled = train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
        "Unscaled Logistic Regression"
    )

    # --------------------------------------------------
    # StandardScaler
    # --------------------------------------------------

    scaler = StandardScaler()

    X_train_standard = scaler.fit_transform(X_train)
    X_test_standard = scaler.transform(X_test)

    model_standard = train_and_evaluate(
        X_train_standard,
        X_test_standard,
        y_train,
        y_test,
        "StandardScaler Logistic Regression"
    )

    # --------------------------------------------------
    # MinMaxScaler
    # --------------------------------------------------

    minmax = MinMaxScaler()

    X_train_minmax = minmax.fit_transform(X_train)
    X_test_minmax = minmax.transform(X_test)

    model_minmax = train_and_evaluate(
        X_train_minmax,
        X_test_minmax,
        y_train,
        y_test,
        "MinMaxScaler Logistic Regression"
    )

    # StandardScalar
    scaler = StandardScaler()

    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    train_and_evaluate(
        X_train_std,
        X_test_std,
        y_train,
        y_test,
        "StandardScaler"
    )
    # MiniMaxScaler
    scaler = MinMaxScaler()

    X_train_minmax = scaler.fit_transform(X_train)
    X_test_minmax = scaler.transform(X_test)

    train_and_evaluate(
        X_train_minmax,
        X_test_minmax,
        y_train,
        y_test,
        "MinMaxScaler"
    )

    '''# --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------

    y_pred = model_standard.predict(X_test_standard)

    cm = confusion_matrix(y_test, y_pred)

    print()
    print("Confusion Matrix:")
    print(cm)

    ConfusionMatrixDisplay(
        confusion_matrix=cm
    ).plot()

    plt.title("Confusion Matrix - StandardScaler")
    plt.show()

    # --------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------

    y_probability = model_standard.predict_proba(
        X_test_standard
    )[:, 1]

    # Works when PlacementStatus is binary/numerically encoded
    try:

        auc_score = roc_auc_score(
            y_test,
            y_probability
        )

        print()
        print("ROC-AUC Score:", round(auc_score, 4))

        fpr, tpr, thresholds = roc_curve(
            y_test,
            y_probability
        )

        plt.plot(
            fpr,
            tpr,
            label="ROC Curve"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()

    except ValueError:
        print()
        print("ROC-AUC could not be calculated.")
        print("Check whether PlacementStatus is binary.")'''


# --------------------------------------------------
# Program Entry Point

if __name__ == "__main__":
    main()
