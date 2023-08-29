import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

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

#---------------------------------------------------------------------
# DBSCAN clustering with different values of epsilon and min_samples
eps_values = [0.1, 0.5, 1, 1.5, 2]
min_samples_values = [2, 5, 10, 15, 20]
silhouette_scores = []
num_clusters_list = []

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data_scaled)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        num_clusters_list.append(num_clusters)
        if num_clusters ==2:
            silhouette_scores.append(silhouette_score(data_scaled, labels))
        else:
            silhouette_scores.append(-1)

# Plot the number of clusters for each combination of parameters
X, Y = np.meshgrid(min_samples_values, eps_values)
Z = np.array(num_clusters_list).reshape(len(eps_values), len(min_samples_values))
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, Z)
ax.set_xlabel('MinPts')
ax.set_ylabel('Epsilon')
ax.set_zlabel('Number of clusters')
plt.show()

# Find the optimal parameters and calculate the clustering performance
optimal_index = np.argmax(silhouette_scores)
optimal_eps = eps_values[optimal_index // len(min_samples_values)]
optimal_min_samples = min_samples_values[optimal_index % len(min_samples_values)]
optimal_dbscan = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples)
optimal_labels = optimal_dbscan.fit_predict(data_scaled)
optimal_num_clusters = len(set(optimal_labels)) - (1 if -1 in optimal_labels else 0)
optimal_silhouette_score = silhouette_score(data_scaled, optimal_labels)

print(f"Optimal number of clusters: {optimal_num_clusters}")
print(f"Optimal epsilon: {optimal_eps}")
print(f"Optimal MinPts: {optimal_min_samples}")
print(f"Optimal silhouette score: {optimal_silhouette_score}")
