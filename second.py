# Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Loading and Initial Exploration of the Dataset
df = pd.read_csv("iris.csv")

# Display the first few rows of the dataset
print("First 5 Rows of the Dataset:")
print(df.head())

# General information about the dataset
print("\nDataset Information:")
print(df.info())

# 2. Data Preprocessing
# Remove the "Id" and "Species" columns
iris_data_cleaned = df.drop(columns=["Id", "Species"])

# Normalize the data
scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris_data_cleaned)

# 3. K-means Clustering: Determining Optimal K with the Elbow Method
inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(iris_scaled)
    inertia.append(kmeans.inertia_)

# Visualizing the Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method: Optimal K Value')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()

# Assume the optimal K is 3
optimal_k = 3
kmeans_model = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans_model.fit_predict(iris_scaled)

# 4. Evaluation and Analysis
# Assess clustering performance with Silhouette Score
silhouette_avg = silhouette_score(iris_scaled, clusters)
print(f"\nSilhouette Score for K={optimal_k}: {silhouette_avg}")

# Add clusters to the original dataset
df['Cluster'] = clusters

# Visualize the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=df['PetalLengthCm'], 
    y=df['PetalWidthCm'], 
    hue=df['Cluster'], 
    palette='viridis',
    style=df['Species'],
    s=100
)
plt.title('K-means Clustering Results')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend(title='Cluster', loc='upper left')
plt.show()
