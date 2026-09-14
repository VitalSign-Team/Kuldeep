
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def load_data():

    data = pd.read_csv("placement_preprocessed.csv")

    X = data[
        [
            "numerical__CGPA",
            "numerical__AptitudeTestScore"
        ]
    ]

    return data, X




def final_kmeans(X, k):

    print("\nRunning Final K-Means...")
    print("Selected K =", k)

    # Standardize data
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # Create K-Means model
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    # Predict clusters
    labels = kmeans.fit_predict(X_scaled)

    return kmeans, scaler, labels



def display_results(data, X, kmeans, scaler, labels):

    # Add cluster labels
    data["Cluster"] = labels

    print("\n======================================")
    print("FINAL K-MEANS RESULTS")
    print("======================================")

    print("\nFirst 10 Students:")

    print(
        data[
            [
                "numerical__CGPA",
                "numerical__AptitudeTestScore",
                "Cluster"
            ]
        ].head(10)
    )

    centers = scaler.inverse_transform(
        kmeans.cluster_centers_
    )

    print("\n======================================")
    print("CLUSTER CENTERS")
    print("======================================")

    centers_df = pd.DataFrame(
        centers,
        columns=[
            "CGPA",
            "AptitudeTestScore"
        ]
    )

    print(centers_df)


    print("\n======================================")
    print("NUMBER OF STUDENTS IN EACH CLUSTER")
    print("======================================")

    print(
        data["Cluster"]
        .value_counts()
        .sort_index()
    )

    return data, centers



def visualize_clusters(data, centers, k):

    plt.figure(figsize=(9, 6))

    # Plot students
    plt.scatter(
        data["numerical__CGPA"],
        data["numerical__AptitudeTestScore"],
        c=data["Cluster"],
        cmap="viridis",
        s=50,
        alpha=0.7
    )

    # Plot centroids
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="X",
        s=250,
        c="red",
        edgecolors="black",
        linewidths=2,
        label="Centroids"
    )

    plt.xlabel("CGPA")
    plt.ylabel("Aptitude Test Score")

    plt.title(
        f"Final K-Means Clustering (K = {k})"
    )

    plt.colorbar(
        label="Cluster"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


def save_results(data):

    data.to_csv(
        "kmeans_final_results.csv",
        index=False
    )

    print(
        "\nFinal results saved as "
        "kmeans_final_results.csv"
    )




def main():

    print("======================================")
    print("          FINAL K-MEANS")
    print("======================================")



    data, X = load_data()

    print("\nDataset loaded successfully.")

    print("\nFeatures used:")

    print(
        [
            "numerical__CGPA",
            "numerical__AptitudeTestScore"
        ]
    )


    print("\nK Selection")
    print("--------------------------------------")

    print("Enter the K value obtained from")
    print("Elbow Method / Silhouette Method.")

    k = int(
        input("Enter final K: ")
    )

    # Validate K
    if k < 2:

        print(
            "\nK must be at least 2."
        )

        return

    if k >= len(X):

        print(
            "\nK must be smaller than "
            "the number of students."
        )

        return


    kmeans, scaler, labels = final_kmeans(
        X,
        k
    )


    data, centers = display_results(
        data,
        X,
        kmeans,
        scaler,
        labels
    )



    save_results(data)


    visualize_clusters(
        data,
        centers,
        k
    )

    print("\n======================================")
    print("       PROGRAM COMPLETED")
    print("======================================")




if __name__ == "__main__":

    main()

