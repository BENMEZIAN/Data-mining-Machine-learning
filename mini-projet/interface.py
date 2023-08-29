import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import kmedoids 
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from tkinter.scrolledtext import ScrolledText


class Interface:
    def __init__(self, master):
        self.master = master
        master.title("Unsupervised clustering")
        master.geometry("900x550")

        self.label = tk.Label(master, text="Unsupervised clustering", font=('helvetica', 10,'bold'))
        self.label.place(relx=0.5, rely=0.05, anchor="center")

        self.open_button = tk.Button(master, text="Import CSV file", command=self.open_dataset, bg='green', fg='white', font=('helvetica', 10,'bold'))
        self.open_button.place(x=200, y=50)

        self.clean_and_normalize_button = tk.Button(master, text="Preprocessing", command=self.clean_and_normalize_data, bg='red', fg='white', font=('helvetica', 10,'bold'))
        self.clean_and_normalize_button.place(x=320, y=50)

        self.display_button = tk.Button(master, text="Display dataset", command=self.display_normalized_data, bg='blue', fg='white', font=('helvetica', 10,'bold'))
        self.display_button.place(x=430, y=50)
        
        self.elbow_method_button = tk.Button(master, text="Elbow curve", command=self.elbow_curve, bg='magenta', fg='white', font=('helvetica', 10,'bold'))
        self.elbow_method_button.place(x=550, y=50)
        
        self.kmean_label = tk.Label(master, text="KMeans", font=('helvetica', 10, 'bold'))
        self.kmean_label.place(relx=0.5, rely=0.2, anchor="center")
        
        self.kmeanButton = tk.Button(master, text="Apply kmeans", command=self.Kmeans_clustering, bg='green', fg='white', font=('helvetica', 10,'bold'))
        self.kmeanButton.place(x=200, y=130)
        
        self.intraClusterInertia = tk.Button(master, text="intra class inertia", command=self.intra_classe_inertia, bg='black', fg='white', font=('helvetica', 10,'bold'))
        self.intraClusterInertia.place(x=320, y=130)
        
        self.interClusterInertia = tk.Button(master, text="inter class inertia", command=self.inter_classe_inertia, bg='black', fg='white', font=('helvetica', 10,'bold'))
        self.interClusterInertia.place(x=450, y=130)
        
        self.kmedoid_label = tk.Label(master, text="Kmedoids", font=('helvetica', 10, 'bold'))
        self.kmedoid_label.place(relx=0.5, rely=0.35, anchor="center")
        
        self.kmeanButton = tk.Button(master, text="Apply kmedoids", command=self.Kmedoids_clustering, bg='green', fg='white', font=('helvetica', 10,'bold'))
        self.kmeanButton.place(x=200, y=210)
        
        self.kmedoidsintraInertia = tk.Button(master, text="intra class inertia", command=self.kmedoids_intra_class_inertia, bg='black', fg='white', font=('helvetica', 10,'bold'))
        self.kmedoidsintraInertia.place(x=320, y=210)
        
        self.kmedoidsinterInertia = tk.Button(master, text="inter class inertia", command=self.kmedoids_inter_class_inertia, bg='black', fg='white', font=('helvetica', 10,'bold'))
        self.kmedoidsinterInertia.place(x=450, y=210)
        
        self.diana_label = tk.Label(master, text="DIANA", font=('helvetica', 10, 'bold'))
        self.diana_label.place(relx=0.5, rely=0.5, anchor="center")
        
        self.dianaButton = tk.Button(master, text="Apply DIANA", command=self.Diana_clustering, bg='green', fg='white', font=('helvetica', 10,'bold'))
        self.dianaButton.place(x=200, y=300)
        
        self.dianaintraInertia = tk.Button(master, text="intra class inertia", command=self.diana_intra_class_inertia, bg='black', fg='white', font=('helvetica', 10,'bold'))
        self.dianaintraInertia.place(x=320, y=300)
        
        self.dianasilhouette = tk.Button(master, text="silhouette score", command=self.diana_silhouette_score, bg='black', fg='white', font=('helvetica', 10,'bold'))
        self.dianasilhouette.place(x=450, y=300)
        
        self.agnes_label = tk.Label(master, text="AGNES", font=('helvetica', 10, 'bold'))
        self.agnes_label.place(relx=0.5, rely=0.65, anchor="center")
        
        self.agnesButton = tk.Button(master, text="Apply AGNES", command=self.Agnes_clustering, bg='green', fg='white', font=('helvetica', 10,'bold'))
        self.agnesButton.place(x=200, y=370)
        
        self.agnesintraInertia = tk.Button(master, text="intra class inertia", command=self.agnes_intra_class_inertia, bg='black', fg='white', font=('helvetica', 10,'bold'))
        self.agnesintraInertia.place(x=320, y=370)
        
        self.agnessilhouette = tk.Button(master, text="silhouette score", command=self.agnes_silhouette_score, bg='black', fg='white', font=('helvetica', 10,'bold'))
        self.agnessilhouette.place(x=450, y=370)
        
        self.agnes_label = tk.Label(master, text="DBSCAN", font=('helvetica', 10, 'bold'))
        self.agnes_label.place(relx=0.5, rely=0.77, anchor="center")
        
        self.dbscanButton = tk.Button(master, text="Apply DBSCAN", command=self.Dbscan_clustering, bg='green', fg='white', font=('helvetica', 10,'bold'))
        self.dbscanButton.place(x=200, y=450)
        
        self.performancesButton = tk.Button(master, text="performances", command=self.show_performances, bg='black', fg='white', font=('helvetica', 10,'bold'))
        self.performancesButton.place(x=320, y=450)
        
        self.data = None
        self.df_normalized = None

    def open_dataset(self):
        filename = filedialog.askopenfilename()
        if filename:
            self.data = pd.read_csv(filename)
            messagebox.showinfo("Info","Dataset has been uploaded successfully")
            self.clean_and_normalize_button.config(state="normal")

    def clean_and_normalize_data(self):
        # Convert categorical attributes into numerical
        cat_columns = []
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                cat_columns.append(col)
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col])

        # Handle missing values by filling them with mean of the column
        self.data.fillna(self.data.mean(), inplace=True)

        # Normalization
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(self.data)

        # Display normalized dataset as Pandas DataFrame
        self.df_normalized = pd.DataFrame(data_scaled, columns=self.data.columns)
        messagebox.showinfo("Info","Dataset has been preprocessed")
        self.display_button.config(state="normal")

    def display_normalized_data(self):
        if self.df_normalized is not None:
            top = tk.Toplevel()
            top.title("Normalized Data")
            top.geometry("900x550")
            text = ScrolledText(top)
            text.pack(expand=True, fill='both')
            text.insert('1.0', self.df_normalized.to_string())
    
    def elbow_curve(self):
        wcss = []

        # Loop over a range of cluster numbers
        for i in range(1, 10):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(self.df_normalized)
            wcss.append(kmeans.inertia_)

        # Plot the elbow curve
        plt.plot(range(1, 10), wcss)
        plt.title('Elbow Curve')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()
    
    def Kmeans_clustering(self,k = 2):
        # K-means clustering
        
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(self.df_normalized)

        # Visualize the results
        plt.scatter(self.df_normalized.iloc[:, 0], self.df_normalized.iloc[:, 1], c=kmeans.labels_, cmap='rainbow')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*', s=300, c='black')
        plt.show()
    
    def intra_classe_inertia(self,k = 2):
        inertias = []
        for i in range(1, k+1):
            kmeans = KMeans(n_clusters=i, n_init=10, random_state=42)
            kmeans.fit(self.df_normalized)
            inertias.append(kmeans.inertia_)
        messagebox.showinfo("Intra class Inertias", "\n".join([f"Inertia for cluster number {i}: {inertias[i-1]}" for i in range(1, len(inertias)+1)]))
    
    def inter_classe_inertia(self,k = 2):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(self.df_normalized)
        centroids = kmeans.cluster_centers_
        overall_mean = np.mean(self.df_normalized, axis=0)
        inter_inertia = sum([np.sum((centroid - overall_mean)**2) for centroid in centroids])
        messagebox.showinfo("Inter class Inertia", inter_inertia)
    
    def Kmedoids_clustering(self,k = 2):
        diss = euclidean_distances(self.df_normalized)
        kp = kmedoids.KMedoids(n_clusters=k, random_state=0, max_iter=100).fit(diss)

        # Display cluster labels
        print("Cluster labels:")
        print(kp.labels_)
        # Visualize using matplotlib
        n_cols = len(self.df_normalized.columns)
        if n_cols >= 2:
            plt.scatter(self.df_normalized.iloc[:,0], self.df_normalized.iloc[:,1], c=kp.labels_, cmap='rainbow')
        elif n_cols == 1:
            plt.scatter(self.df_normalized.iloc[:,0], [0] * len(self.df_normalized), c=kp.labels_, cmap='rainbow')
        plt.title('K-Medoids Clustering')
        plt.show()
        
    def kmedoids_intra_class_inertia(self,k = 2):
        diss = euclidean_distances(self.df_normalized)
        kp = kmedoids.KMedoids(n_clusters=k, random_state=0, max_iter=100).fit(diss)
        
        cluster_inertia = []
        for i in range(kp.n_clusters):
            cluster_indices = np.where(kp.labels_ == i)[0]
            cluster_distances = diss[cluster_indices][:, cluster_indices]
            cluster_inertia.append(np.sum(cluster_distances))
        messagebox.showinfo("Intra class Inertias", "\n".join([f"Inertia for cluster number {i}: {cluster_inertia[i-1]}" for i in range(1, k+1)]))
    
    def kmedoids_inter_class_inertia(self,k = 2):
        diss = euclidean_distances(self.df_normalized)
        kp = kmedoids.KMedoids(n_clusters=k, random_state=0, max_iter=100).fit(diss)

        tss = np.sum(diss ** 2) # Calculate TSS
        wss = np.sum(np.min(diss[:, kp.medoid_indices_], axis=1) ** 2) # Calculate WSS
        inter_inertia_clusters = tss - wss
        messagebox.showinfo("Inter class Inertias",inter_inertia_clusters)
    
    def Diana_clustering(self,k = 2):
        
        # Compute the linkage matrix
        Z = linkage(self.df_normalized, metric='euclidean',method='ward')
        agg = AgglomerativeClustering(n_clusters=k, metric='euclidean',linkage='ward')
        diana_clusters = agg.fit_predict(self.df_normalized) 

        # Plot the dendrogram
        plt.title('Dendrogram')
        plt.xlabel('Data points')
        plt.ylabel('Distance')
        dendrogram(Z)
        plt.show()
    
    def diana_intra_class_inertia(self,k =2):
        
        # Compute the linkage matrix
        Z = linkage(self.df_normalized, metric='euclidean',method='ward')
        agg = AgglomerativeClustering(n_clusters=k, metric='euclidean',linkage='ward')
        diana_clusters = agg.fit_predict(self.df_normalized) 
        # Compute pairwise distances between data points and cluster centers
        distances = pairwise_distances(self.data)
        inertias = np.zeros(2)

        # For each cluster (intra-classe), compute the sum of squared distances between each point and the cluster center
        for i in range(2):
            indices = np.where(diana_clusters == i)[0]
            cluster_distances = distances[indices[:, np.newaxis], indices]
            center = np.mean(cluster_distances, axis=1)
            inertias[i] = np.sum((cluster_distances - center[:, np.newaxis])**2)
        messagebox.showinfo("Intra class Inertias", "\n".join([f"Inertia for cluster number {i}: {inertias[i-1]}" for i in range(1, k+1)]))
    
    def diana_silhouette_score(self,k =2):
        
        Z = linkage(self.df_normalized, metric='euclidean',method='ward')
        agg = AgglomerativeClustering(n_clusters=k, metric='euclidean',linkage='ward')
        diana_clusters = agg.fit_predict(self.df_normalized) 
        # Calculate the Silhouette score instead of interclasse inertia cluster
        silhouette_avg = silhouette_score(self.df_normalized, diana_clusters)
        messagebox.showinfo("Silhouette score",silhouette_avg)
    
    def Agnes_clustering(self,k = 2):
        # Compute the linkage matrix
        Z = linkage(self.df_normalized, metric='euclidean',method='average')
        # Apply AGNES clustering with 2 clusters
        agg = AgglomerativeClustering(n_clusters=k, metric='euclidean',linkage='average')
        agnes_clusters = agg.fit_predict(self.df_normalized)
        
        # Plot the dendrogram
        plt.title('Dendrogram')
        plt.xlabel('Data points')
        plt.ylabel('Distance')
        dendrogram(Z)
        plt.show()
        
    def agnes_intra_class_inertia(self,k = 2):
        
        Z = linkage(self.df_normalized, metric='euclidean',method='average')
        # Apply AGNES clustering with 2 clusters
        agg = AgglomerativeClustering(n_clusters=k, metric='euclidean',linkage='average')
        agnes_clusters = agg.fit_predict(self.df_normalized)
        
        # Compute pairwise distances between data points and cluster centers
        distances = pairwise_distances(self.data)
        inertias = np.zeros(2)

        # For each cluster, compute the sum of squared distances between each point and the cluster center
        for i in range(2):
            indices = np.where(agnes_clusters == i)[0]
            cluster_distances = distances[indices[:, np.newaxis], indices]
            center = np.mean(cluster_distances, axis=1)
            inertias[i] = np.sum((cluster_distances - center[:, np.newaxis])**2)
        messagebox.showinfo("Intra class Inertias", "\n".join([f"Inertia for cluster number {i}: {inertias[i-1]}" for i in range(1, k+1)]))
    
    def agnes_silhouette_score(self,k = 2):
        
        Z = linkage(self.df_normalized, metric='euclidean',method='average')
        # Apply AGNES clustering with 2 clusters
        agg = AgglomerativeClustering(n_clusters=k, metric='euclidean',linkage='average')
        agnes_clusters = agg.fit_predict(self.df_normalized)
        silhouette_avg = silhouette_score(self.df_normalized, agnes_clusters)
        messagebox.showinfo("Silhouette score",silhouette_avg)
    
    def Dbscan_clustering(self,k = 2):
        # DBSCAN clustering with different values of epsilon and min_samples
        eps_values = [0.1, 0.5, 1, 1.5, 2]
        min_samples_values = [2, 5, 10, 15, 20]
        silhouette_scores = []
        num_clusters_list = []

        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(self.df_normalized)
                num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                num_clusters_list.append(num_clusters)
                if num_clusters == k:
                    silhouette_scores.append(silhouette_score(self.df_normalized, labels))
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
        
    def show_performances(self,k = 2):
        
        eps_values = [0.1, 0.5, 1, 1.5, 2]
        min_samples_values = [2, 5, 10, 15, 20]
        silhouette_scores = []
        num_clusters_list = []

        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(self.df_normalized)
                num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                num_clusters_list.append(num_clusters)
                if num_clusters == k:
                    silhouette_scores.append(silhouette_score(self.df_normalized, labels))
                else:
                    silhouette_scores.append(-1)
        
        
        # Find the optimal parameters and calculate the clustering performance
        optimal_index = np.argmax(silhouette_scores)
        optimal_eps = eps_values[optimal_index // len(min_samples_values)]
        optimal_min_samples = min_samples_values[optimal_index % len(min_samples_values)]
        optimal_dbscan = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples)
        optimal_labels = optimal_dbscan.fit_predict(self.df_normalized)
        optimal_num_clusters = len(set(optimal_labels)) - (1 if -1 in optimal_labels else 0)
        optimal_silhouette_score = silhouette_score(self.df_normalized, optimal_labels)

        messagebox.showinfo("Optimal Clustering Results", 
                    f"Optimal number of clusters: {optimal_num_clusters}\n"
                    f"Optimal epsilon: {optimal_eps}\n"
                    f"Optimal MinPts: {optimal_min_samples}\n"
                    f"Optimal silhouette score: {optimal_silhouette_score}")


root = tk.Tk()
interface = Interface(root)
root.mainloop()
