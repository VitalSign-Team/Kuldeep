import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
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


def run_kmeans_simple(dataset_path):

    # ========================================================
    # LOAD DATASET
    # ========================================================

    data = pd.read_csv(dataset_path)


    # ========================================================
    # SELECT FEATURES
    # ========================================================

    features = [
        "numerical__CGPA",
        "numerical__AptitudeTestScore"
    ]

    # Check columns
    missing = [
        col for col in features
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
    # K-MEANS
    # ========================================================

    k = 3

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(Z)


    # ========================================================
    # CENTROIDS
    # ========================================================

    centers = scaler.inverse_transform(
        model.cluster_centers_
    )


    # ========================================================
    # CLUSTER COUNTS
    # ========================================================

    counts = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
        .to_dict()
    )


    # ========================================================
    # SILHOUETTE SCORE
    # ========================================================

    if len(set(labels)) > 1:

        silhouette = silhouette_score(
            Z,
            labels
        )

    else:

        silhouette = 0


    # ========================================================
    # CLUSTER GRAPH
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )


    scatter = ax.scatter(
        X.iloc[:, 0],
        X.iloc[:, 1],
        c=labels,
        alpha=0.7,
        s=35
    )


    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="X",
        s=220,
        edgecolors="black",
        linewidths=1.5,
        label="Centroids"
    )


    ax.set_xlabel(
        "CGPA"
    )

    ax.set_ylabel(
        "Aptitude Test Score"
    )

    ax.set_title(
        "K-Means Clustering (K=3)"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.3
    )


    image = fig_to_base64(fig)


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "features": features,

        "k": k,

        "centers": centers.round(
            3
        ).tolist(),

        "counts": {
            str(key): int(value)
            for key, value in counts.items()
        },

        "silhouette": round(
            float(silhouette),
            3
        ),

        "rows": len(X),

        "image": image
    }


# ============================================================
# OPTIONAL: RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    result = run_kmeans_simple(
        "../datasets/placement_preprocessed.csv"
    )

    print("\nK-Means Results")
    print("----------------")

    print(
        "K:",
        result["k"]
    )

    print(
        "Rows:",
        result["rows"]
    )

    print(
        "Silhouette:",
        result["silhouette"]
    )

    print(
        "\nCluster Centers:"
    )

    for center in result["centers"]:
        print(center)

    print(
        "\nCluster Counts:"
    )

    print(result["counts"])