import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_data():

    data = pd.read_csv("placement_preprocessed.csv")

    # Select only numerical features
    x = data[
        [
            "numerical__CGPA",
            "numerical__AptitudeTestScore"
        ]
    ]

    return x


def manual_k():

    print("\nManual K selection ....")

    k = int(input("Enter the value of K: "))

    return k



def elbow_method(x):

    print("\nElbow Method ....")

    wcss = []

    k_values = range(1, 11)

    for k in k_values:

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        kmeans.fit(x)

        wcss.append(kmeans.inertia_)

    # Plot elbow graph
    plt.figure(figsize=(8, 5))

    plt.plot(
        k_values,
        wcss,
        marker="o"
    )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS / Inertia")
    plt.title("Elbow Method")

    plt.grid(True)

    plt.show()

    # Ask user to choose K after seeing graph
    k = int(input("\nEnter K from the elbow graph: "))

    return k




def silhouette_method(x):

    print("\nSilhouette Method ....")

    silhouette_scores = []

    k_values = range(2, 11)

    for k in k_values:

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(x)

        score = silhouette_score(
            x,
            labels
        )

        silhouette_scores.append(score)

        print(
            "K =", k,
            "Silhouette Score =", round(score, 4)
        )

    # Find best K
    best_k = k_values[
        silhouette_scores.index(
            max(silhouette_scores)
        )
    ]

    # Plot silhouette scores
    plt.figure(figsize=(8, 5))

    plt.plot(
        k_values,
        silhouette_scores,
        marker="o"
    )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Method")
    plt.xticks(k_values)
    plt.grid(True)

    plt.show()

    print(
        "\nBest K according to Silhouette Method:",
        best_k
    )

    return best_k



def run_kmeans(x, k):

    print("\nRunning K-Means with K =", k)

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(x)

    print("\nCluster Centers:")
    print(kmeans.cluster_centers_)

    print("\nCluster Count:")

    cluster_counts = pd.Series(labels).value_counts().sort_index()

    print(cluster_counts)

    return kmeans, labels




def main():

    print("======================================")
    print("       K VALUE CALCULATION")
    print("======================================")

    # Load data
    x = load_data()

    print("\nDataset Loaded Successfully!")

    print("\nSelected Features:")
    print(x.head())

    print("\nChoose K selection method:")

    print("1. Manual K")
    print("2. Elbow Method")
    print("3. Silhouette Method")

    choice = int(
        input("\nEnter your choice: ")
    )


    if choice == 1:

        k = manual_k()


    elif choice == 2:

        k = elbow_method(x)


    elif choice == 3:

        k = silhouette_method(x)

    else:

        print("Invalid choice!")

        return

    # Run K-Means
    kmeans, labels = run_kmeans(
        x,
        k
    )

    # Add cluster labels
    result = x.copy()

    result["Cluster"] = labels

    print("\n======================================")
    print("FINAL CLUSTER RESULTS")
    print("======================================")

    print(result.head(10))

    # Save results
    result.to_csv(
        "kmeans_results.csv",
        index=False
    )

    print(
        "\nResults saved as kmeans_results.csv"
    )

if __name__ == "__main__":

    main()