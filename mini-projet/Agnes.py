import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt


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


# Compute the linkage matrix
Z = linkage(df_normalized, metric='euclidean',method='average')

# Apply AGNES clustering with 6 clusters
agg = AgglomerativeClustering(n_clusters=2, metric='euclidean',linkage='average')
agnes_clusters = agg.fit_predict(df_normalized)


# Plot the dendrogram
plt.title('Dendrogram')
plt.xlabel('Data points')
plt.ylabel('Distance')
dendrogram(Z)
plt.show()


# Compute pairwise distances between data points and cluster centers
distances = pairwise_distances(data)
inertias = np.zeros(2)

# For each cluster, compute the sum of squared distances between each point and the cluster center
for i in range(2):
    indices = np.where(agnes_clusters == i)[0]
    cluster_distances = distances[indices[:, np.newaxis], indices]
    center = np.mean(cluster_distances, axis=1)
    inertias[i] = np.sum((cluster_distances - center[:, np.newaxis])**2)

print(inertias)


# Calculate the Silhouette score instead of interclasse inertia cluster
silhouette_avg = silhouette_score(df_normalized, agnes_clusters)
print("Silhouette score:", silhouette_avg)

