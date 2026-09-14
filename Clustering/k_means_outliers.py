import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# ==========================================================
# 1. LOAD DATASET
# ==========================================================

try:
    data = pd.read_csv("placement_preprocessed.csv")

    # Check required columns
    required_columns = [
        "numerical__CGPA",
        "numerical__AptitudeTestScore"
    ]

    for column in required_columns:
        if column not in data.columns:
            raise ValueError(f"Column not found: {column}")

except (FileNotFoundError, ValueError):

    print("Creating sample dataset...")

    np.random.seed(42)

    # Create 3 student groups
    cgpa1 = np.random.normal(8.5, 0.4, 70)
    apt1 = np.random.normal(85, 8, 70)

    cgpa2 = np.random.normal(7.0, 0.5, 70)
    apt2 = np.random.normal(65, 10, 70)

    cgpa3 = np.random.normal(5.5, 0.5, 60)
    apt3 = np.random.normal(45, 10, 60)

    # Combine groups
    cgpa = np.concatenate([
        cgpa1,
        cgpa2,
        cgpa3
    ])

    aptitude = np.concatenate([
        apt1,
        apt2,
        apt3
    ])

    # Create DataFrame
    data = pd.DataFrame({
        "numerical__CGPA":
            np.round(cgpa.clip(4.0, 10.0), 2),

        "numerical__AptitudeTestScore":
            np.round(
                aptitude.clip(0, 100),
                0
            ).astype(int)
    })

    # Shuffle data
    data = data.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # Save dataset
    data.to_csv(
        "placement_preprocessed.csv",
        index=False
    )

    print("Dataset created and saved!")


# ==========================================================
# 2. SELECT FEATURES
# ==========================================================

X = data[
    [
        "numerical__CGPA",
        "numerical__AptitudeTestScore"
    ]
]

print("\n======================================")
print("SELECTED FEATURES")
print("======================================")

print(X.head())


# ==========================================================
# 3. STANDARDIZE DATA
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nData standardized successfully.")


# ==========================================================
# 4. ELBOW METHOD
# ==========================================================

wcss = []

for k in range(1, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)


# ==========================================================
# 5. PLOT ELBOW CURVE
# ==========================================================

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)

plt.plot(
    range(1, 11),
    wcss,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.title("Elbow Method")

plt.grid(
    True,
    alpha=0.3
)


# ==========================================================
# 6. APPLY K-MEANS
# ==========================================================

# Based on the elbow method
optimal_k = 3

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

# Assign clusters
data["Cluster"] = kmeans.fit_predict(X_scaled)


# ==========================================================
# 7. DISPLAY CLUSTER RESULTS
# ==========================================================

print("\n======================================")
print("CLUSTER RESULTS")
print("======================================")

print(
    data[
        [
            "numerical__CGPA",
            "numerical__AptitudeTestScore",
            "Cluster"
        ]
    ].head(10)
)


# ==========================================================
# 8. CLUSTER CENTERS
# ==========================================================

print("\n======================================")
print("CLUSTER CENTERS")
print("======================================")

print(kmeans.cluster_centers_)


# Convert cluster centers back to original scale
cluster_centers_original = scaler.inverse_transform(
    kmeans.cluster_centers_
)

print("\nCluster Centers in Original Scale:")

print(
    pd.DataFrame(
        cluster_centers_original,
        columns=[
            "CGPA",
            "AptitudeTestScore"
        ]
    )
)


# ==========================================================
# 9. NUMBER OF STUDENTS IN EACH CLUSTER
# ==========================================================

print("\n======================================")
print("STUDENTS IN EACH CLUSTER")
print("======================================")

print(
    data["Cluster"]
    .value_counts()
    .sort_index()
)


# ==========================================================
# 10. CALCULATE DISTANCE FROM CENTROID
# ==========================================================

# Calculate distance from every point
# to every cluster centroid

distances = kmeans.transform(X_scaled)

# Get distance from student's assigned centroid

data["Distance_From_Centroid"] = np.min(
    distances,
    axis=1
)


# ==========================================================
# 11. DEFINE OUTLIER THRESHOLD
# ==========================================================

mean_distance = data[
    "Distance_From_Centroid"
].mean()

std_distance = data[
    "Distance_From_Centroid"
].std()

threshold = (
    mean_distance
    + 2 * std_distance
)

print("\n======================================")
print("OUTLIER DETECTION")
print("======================================")

print("\nMean Distance:")
print(mean_distance)

print("\nStandard Deviation:")
print(std_distance)

print("\nOutlier Threshold:")
print(threshold)


# ==========================================================
# 12. MARK OUTLIERS
# ==========================================================

data["Outlier"] = (
    data["Distance_From_Centroid"]
    > threshold
)


# ==========================================================
# 13. GET OUTLIERS
# ==========================================================

outliers = data[
    data["Outlier"] == True
]


print("\nNumber of Outliers:")
print(len(outliers))


print("\n======================================")
print("OUTLIER STUDENTS")
print("======================================")

print(
    outliers[
        [
            "numerical__CGPA",
            "numerical__AptitudeTestScore",
            "Cluster",
            "Distance_From_Centroid"
        ]
    ]
)


# ==========================================================
# 14. SAVE OUTLIERS
# ==========================================================

outliers.to_csv(
    "kmeans_outliers.csv",
    index=False
)

print("\nOutliers saved to:")
print("kmeans_outliers.csv")


# ==========================================================
# 15. SAVE COMPLETE RESULTS
# ==========================================================

data.to_csv(
    "kmeans_cluster_results.csv",
    index=False
)

print("\nComplete clustering results saved to:")
print("kmeans_cluster_results.csv")


# ==========================================================
# 16. VISUALIZATION
# ==========================================================

plt.subplot(1, 2, 2)

# Normal students
normal = data[
    data["Outlier"] == False
]

plt.scatter(
    normal["numerical__CGPA"],
    normal["numerical__AptitudeTestScore"],
    c=normal["Cluster"],
    cmap="viridis",
    alpha=0.7,
    s=50,
    label="Normal Students"
)


# Plot outliers
plt.scatter(
    outliers["numerical__CGPA"],
    outliers["numerical__AptitudeTestScore"],
    marker="x",
    s=150,
    c="red",
    linewidths=3,
    label="Outliers"
)


# Convert centroids back to original scale
centers = scaler.inverse_transform(
    kmeans.cluster_centers_
)


# Plot centroids
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=250,
    c="black",
    edgecolors="white",
    linewidths=2,
    label="Centroids"
)


plt.xlabel("CGPA")
plt.ylabel("Aptitude Test Score")

plt.title(
    "K-Means Clustering with Outlier Detection"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)


# ==========================================================
# 17. DISPLAY BOTH GRAPHS
# ==========================================================

plt.tight_layout()

plt.show()


# ==========================================================
# 18. FINAL SUMMARY
# ==========================================================

print("\n======================================")
print("FINAL SUMMARY")
print("======================================")

print("Total Students:", len(data))

print("Number of Clusters:", optimal_k)

print("Number of Outliers:", len(outliers))

print("Outlier Percentage:",
      round(
          len(outliers) / len(data) * 100,
          2
      ),
      "%"
)

print("\nProgram completed successfully!")