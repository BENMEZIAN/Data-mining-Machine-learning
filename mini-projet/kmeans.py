import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# open the dataset
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

print(df_normalized)

wcss = []

# Loop over a range of cluster numbers
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df_normalized)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, 10), wcss)
plt.title('Elbow Curve')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# K-means clustering
k = 2
kmeans = KMeans(n_clusters=k, n_init=10)
kmeans.fit(df_normalized)

# Visualize the results
plt.scatter(df_normalized.iloc[:, 0], df_normalized.iloc[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*', s=300, c='black')
plt.show()

# calculate intra-classe inertia of each cluster
inertias = []
for i in range(1, k+1):
    kmeans = KMeans(n_clusters=i, n_init=10, random_state=42)
    kmeans.fit(df_normalized)
    inertias.append(kmeans.inertia_)
    
print("l\'inertie intra-classe est: ",inertias)

# calculate inter-classe cluster inertia
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
kmeans.fit(df_normalized)
centroids = kmeans.cluster_centers_
overall_mean = np.mean(df_normalized, axis=0)
inter_inertia = sum([np.sum((centroid - overall_mean)**2) for centroid in centroids])

print("The inter-class inertia is:", inter_inertia)
