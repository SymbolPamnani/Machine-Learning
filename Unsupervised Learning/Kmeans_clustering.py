import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [1, 1],
    [2, 1],
    [1, 2],
    [8, 8],
    [9, 8],
    [8, 9]
])

# INITIALIZE THE CENTROIDS

# I want 2 clusters, so I need 2 centroids.

centroids = np.array([
    [1, 1],    # Centroid 0
    [8, 8]     # Centroid 1
])

# CALCULATE DISTANCE AND ASSIGN CLUSTERS

# This list will store the cluster number assigned to each data point.
labels = []

for point in X:

    # Store the distance from this point to every centroid.
    distances = []

    # Calculate distance from the current point to each centroid.

    for centroid in centroids:

        # Euclidean distance formula:
        # distance = sqrt((x1 - x2)^2 + (y1 - y2)^2)

        # NumPy allows to write this as:
        distance = np.sqrt(np.sum((point - centroid) ** 2))

        # Store the calculated distance.
        distances.append(distance)

    # Find the index of the smallest distance.
    # If distances = [2.0, 9.0]
    # np.argmin() returns 0.
    # Therefore, the point belongs to Cluster 0.

    cluster = np.argmin(distances)

    # Store the cluster assignment.
    labels.append(cluster)

# Convert the list into a NumPy array.

labels = np.array(labels)

print("Cluster assignments:")
print(labels)

# CALCULATE NEW CENTROIDS

# I now have something like: [0, 0, 0, 1, 1, 1]
# Points 0, 1, 2 belongs to Cluster 0 & Points 3, 4, 5 belongs to Cluster 1
# Now I calculate the mean of the points belonging to each cluster.

new_centroids = []

# I have 2 clusters, so check Cluster 0 and Cluster 1.

for cluster_number in range(2):

    # Select only the points that belong to the current cluster.

    cluster_points = X[labels == cluster_number]

    # Calculate the mean of those points.
    # axis=0 means: mean of all x & all y values

    new_centroid = cluster_points.mean(axis=0)

    # Store the new centroid.

    new_centroids.append(new_centroid)

# Convert the list into a NumPy array.

new_centroids = np.array(new_centroids)

print("\nOld centroids:")
print(centroids)

print("\nNew centroids:")
print(new_centroids)

# VISUALIZE THE CLUSTERS

# Plot Cluster 0 points.
plt.scatter(
    X[labels == 0, 0],
    X[labels == 0, 1],
    label="Cluster 0"
)


# Plot Cluster 1 points.
plt.scatter(
    X[labels == 1, 0],
    X[labels == 1, 1],
    label="Cluster 1"
)

# Plot the OLD centroids.
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=200,
    label="Old Centroids"
)

# Plot the NEW centroids.
plt.scatter(
    new_centroids[:, 0],
    new_centroids[:, 1],
    marker="*",
    s=250,
    label="New Centroids"
)

# Add labels and title.
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means: One Assignment and Centroid Update")

# Show legend.
plt.legend()

# Display the plot.
plt.show()
