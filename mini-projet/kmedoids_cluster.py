import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import kmedoids 


# open 'your_dataset.csv' dataset file
data = pd.read_csv('diabetes.csv')

# Data Cleaning
# Convert categorical attributes into numerical
cat_columns = []
for col in data.columns:
    if data[col].dtype == 'object':
        cat_columns.append(col)
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])


# Handle missing values by filling them with mean of the column
data.fillna(data.mean(), inplace=True)

# Normalization
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Display normalized dataset as Pandas DataFrame
df_normalized = pd.DataFrame(data_scaled, columns=data.columns)

# Apply kmedoids
diss = euclidean_distances(df_normalized)
kp = kmedoids.KMedoids(n_clusters=2, random_state=0, max_iter=100).fit(diss)

# Display cluster labels
print("Cluster labels:")
print(kp.labels_)


# Visualize using matplotlib
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=kp.labels_, cmap='rainbow')
plt.title('K-Medoids Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Calculate intra-inertia for each cluster
cluster_inertia = []
for i in range(kp.n_clusters):
    cluster_indices = np.where(kp.labels_ == i)[0]
    cluster_distances = diss[cluster_indices][:, cluster_indices]
    cluster_inertia.append(np.sum(cluster_distances))
    print(f"Inertia of cluster {i}: {cluster_inertia[-1]}")



# Calculate intra-inertia clusters
# Calculate TSS
tss = np.sum(diss ** 2)

# Calculate WSS
wss = np.sum(np.min(diss[:, kp.medoid_indices_], axis=1) ** 2)

inter_inertia_clusters = tss - wss

print("Inter-inertia clusters:", inter_inertia_clusters)







"""# Visualize the results
plt.scatter(df_normalized.iloc[:, 0], df_normalized.iloc[:, 1], c=kmedoids_instance.labels_, cmap='rainbow')
plt.scatter(kmedoids_instance.medoids[:, 0], kmedoids_instance.medoids[:, 1], marker='*', s=300, c='black')
plt.show()
"""















"""import kmedoids
def kmedoids(data, k, max_iter=100):
    # Step 1: Initialize medoids randomly
    medoids = np.random.choice(data.shape[0], size=k, replace=False)
    
    for i in range(max_iter):
        # Step 2: Assign each non-medoid point to the nearest medoid
        distances = np.sum((data - data[medoids][:, np.newaxis])**2, axis=2)
        labels = np.argmin(distances, axis=0)
        
        # Step 3: Calculate the cost of swapping each medoid with each non-medoid point
        costs = np.zeros((k, data.shape[0]))
        for j in range(k):
            mask = (labels == j)
            for l in range(data.shape[0]):
                if not mask[l]:
                    new_medoid = data[l]
                    old_medoid = data[medoids[j]]
                    new_cost = np.sum((data[mask] - new_medoid)**2)
                    old_cost = np.sum((data[mask] - old_medoid)**2)
                    costs[j, l] = new_cost - old_cost
        
        # Step 4: Swap the medoid with the lowest cost
        indices = np.argmin(costs, axis=1)
        medoids_old = medoids.copy()
        for j in range(k):
            medoids[j] = np.where(labels == j)[0][indices[j]]
        
        # Step 5: Check for convergence
        if np.array_equal(medoids, medoids_old):
            break
    
    # Return the final medoids and labels
    return data[medoids], labels
"""