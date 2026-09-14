import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import io
import base64


def fig_to_base64(fig):

    buffer = io.BytesIO()

    fig.tight_layout()

    fig.savefig(
        buffer,
        format="png",
        dpi=120,
        bbox_inches="tight"
    )

    plt.close(fig)

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


def run_kmeans_elbow(dataset_path):

    # ========================================================
    # LOAD DATASET
    # ========================================================

    data = pd.read_csv(
        dataset_path
    )


    # ========================================================
    # SELECT FEATURES
    # ========================================================

    features = [
        "numerical__CGPA",
        "numerical__AptitudeTestScore"
    ]


    # Check columns
    missing = [
        col
        for col in features
        if col not in data.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns in dataset: {missing}"
        )


    X = data[features].apply(
        pd.to_numeric,
        errors="coerce"
    ).dropna()


    # ========================================================
    # STANDARDIZATION
    # ========================================================

    scaler = StandardScaler()

    Z = scaler.fit_transform(X)


    # ========================================================
    # ELBOW METHOD
    # ========================================================

    wcss = []

    k_values = range(
        1,
        11
    )


    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(Z)

        wcss.append(
            model.inertia_
        )


    # ========================================================
    # SELECT OPTIMAL K
    # ========================================================

    # Based on the elbow of the curve
    optimal_k = 3


    # ========================================================
    # ELBOW GRAPH
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    ax.plot(
        list(k_values),
        wcss,
        marker="o",
        linewidth=2
    )


    # Highlight selected K
    ax.scatter(
        optimal_k,
        wcss[optimal_k - 1],
        s=140,
        marker="X",
        edgecolors="black",
        linewidths=1.5,
        label=f"Selected K = {optimal_k}"
    )


    ax.set_xlabel(
        "Number of Clusters (K)"
    )

    ax.set_ylabel(
        "WCSS (Within-Cluster Sum of Squares)"
    )

    ax.set_title(
        "Elbow Method for Optimal K"
    )

    ax.set_xticks(
        list(k_values)
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()


    image = fig_to_base64(fig)


    # ========================================================
    # WCSS DICTIONARY
    # ========================================================

    wcss_dict = {

        str(k): round(
            float(value),
            3
        )

        for k, value in zip(
            k_values,
            wcss
        )
    }


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "features": features,

        "k": optimal_k,

        "wcss": wcss_dict,

        "rows": len(X),

        "image": image
    }


# ============================================================
# OPTIONAL: RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    result = run_kmeans_elbow(
        "../datasets/placement_preprocessed.csv"
    )

    print("\nElbow Method Results")
    print("---------------------")

    print(
        "Optimal K:",
        result["k"]
    )

    print(
        "Rows:",
        result["rows"]
    )

    print(
        "\nWCSS Values:"
    )

    for k, value in result["wcss"].items():

        print(
            f"K = {k} : {value}"
        )