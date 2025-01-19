import pandas as pd
import numpy as np
from sklearn.cluster import k_means, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

#input dataset
data = pd.read_csv('market_ds.csv')
data_scaled =  (data - data.mean()) / data.std()

# k-distance for eps
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(data_scaled)
distances, indices = neighbors_fit.kneighbors(data_scaled)
distances = np.sort(distances[:, 4])
plt.plot(distances)
plt.title('K-distance Graph')
plt.xlabel('Data Points')
plt.ylabel('Distance to 5th Nearest Neighbor')
plt.grid(True)
plt.show()

#detect outlier for DBSCAN
dbscan = DBSCAN(eps=0.6, min_samples=5)
labels = dbscan.fit_predict(data_scaled)
data['Cluster'] = labels
outliers = data[data['Cluster'] == -1]
cleaned_data = data[data['Cluster'] != -1]
cleaned_data = cleaned_data.drop('Cluster',axis=1)

#standard scale cleaned_data
scaled_data = (cleaned_data - cleaned_data.mean()) / cleaned_data.std()

#k-means and silhouette for optimal_k
silhouette_scores = []
best_score = 0
optimal_cluster = 0
wcss =[]
k_values = range(2,11)
k = 0
for k in k_values:
    k_means = KMeans(n_clusters=k)
    k_means.fit(scaled_data)
    wcss.append(k_means.inertia_)
    score = silhouette_score(scaled_data, k_means.labels_)
    silhouette_scores.append(score)
    if score > best_score:
        best_score = score
        optimal_cluster = k
    k += 1
print('optimal_cluster=',optimal_cluster)
print('silhouette_score=',best_score)

#Plotting the Elbow chart for k-means
plt.figure(figsize=(10,5))
plt.plot(k_values, wcss)
plt.title('Elbow Method for Optimal k')
plt.xlabel('k_values')
plt.ylabel('wcss')
plt.grid(True)
plt.show()

#Optimal number of clusters for k-means
model = KMeans(optimal_cluster) #optimal_cluster
model.fit(scaled_data)

#Income and spending for scatter
clusters = {}
i = 0
for i in range(optimal_cluster):  #optimal_cluster
    clusters[i] = scaled_data.loc[model.labels_ == i, :]
    plt.scatter(clusters[i].loc[:, 'Income'], clusters[i].loc[:, 'Spending'])

plt.show()

for i in range(optimal_cluster): #optimal_cluster
    clusters[i] = scaled_data.loc[model.labels_ == i,:]
    plt.scatter(clusters[i].loc[:, 'Income'], clusters[i].loc[:, 'Age'])
plt.show()

#define names according to different clusters
name_clusters = {}
name_model = KMeans(optimal_cluster)
name_model.fit(cleaned_data)
name_data = pd.DataFrame()
i = 0
for i in range(optimal_cluster):
    name_clusters[i] = cleaned_data.loc[model.labels_ == i, :]
    mean_age = name_clusters[i].loc[:, 'Age'].mean()
    mean_income = name_clusters[i].loc[:, 'Income'].mean()
    mean_spending = name_clusters[i].loc[:, 'Spending'].mean()
    name_data.loc[i, 'Income'] = mean_income
    name_data.loc[i, 'Age'] = mean_age
    name_data.loc[i, 'Spending'] = mean_spending
print(name_data)

# ##Result
# optimal_cluster= 5
# silhouette_score= 0.44597711210019597
#       Income        Age   Spending
# 0  26.200000  24.300000  77.450000
# 1  82.542857  32.742857  82.800000
# 2  82.588235  46.529412  19.294118
# 3  54.512195  27.609756  47.878049
# 4  48.122807  55.070175  42.192982

# Cluster_0: Low Income, Young, High Spender
# Cluster_1: High Income, Middle-Aged, High Spender
# Cluster_2: High Income, Senior, Low Spender
# Cluster_3: Middle Income, Young, Moderate Spender
# Cluster_4: Middle Income, Senior, Moderate Spender



