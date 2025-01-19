import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 加载数据并进行标准化
data = pd.read_csv('market_ds.csv')
data_scaled = (data - data.mean()) / data.std()

# DBSCAN用于检测异常值
dbscan = DBSCAN(eps=0.6, min_samples=5)
labels = dbscan.fit_predict(data_scaled)
data['Cluster'] = labels

# 筛选出异常值并清理数据
outliers = data[data['Cluster'] == -1]
cleaned_data = data[data['Cluster'] != -1].drop('Cluster', axis=1)

# 对清理后的数据进行标准化
scaled_data = (cleaned_data - cleaned_data.mean()) / cleaned_data.std()

# 选择最佳K值（聚类数目）并计算Silhouette得分
silhouette_scores = []
best_score = -1
optimal_cluster = 0
wcss = []
k_values = range(2, 11)

for k in k_values:
    k_means = KMeans(n_clusters=k, random_state=42)
    k_means.fit(scaled_data)
    wcss.append(k_means.inertia_)

    score = silhouette_score(scaled_data, k_means.labels_)
    silhouette_scores.append(score)

    if score > best_score:
        best_score = score
        optimal_cluster = k

print(f'Optimal number of clusters: {optimal_cluster}')
print(f'Best silhouette score: {best_score}')

# 绘制Elbow法图表，显示最佳k值
plt.figure(figsize=(8, 5))
plt.plot(k_values, wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# 使用最佳K值进行K-Means聚类
model = KMeans(n_clusters=optimal_cluster, random_state=42)
model.fit(scaled_data)

# 可视化：收入和支出之间的关系
plt.figure(figsize=(8, 5))
for i in range(optimal_cluster):
    cluster_data = scaled_data.loc[model.labels_ == i]
    plt.scatter(cluster_data['Income'], cluster_data['Spending'], label=f'Cluster {i}')
plt.title('Income vs Spending by Cluster')
plt.xlabel('Income')
plt.ylabel('Spending')
plt.legend()
plt.grid(True)
plt.show()

# 可视化：收入和年龄之间的关系
plt.figure(figsize=(8, 5))
for i in range(optimal_cluster):
    cluster_data = scaled_data.loc[model.labels_ == i]
    plt.scatter(cluster_data['Income'], cluster_data['Age'], label=f'Cluster {i}')
plt.title('Income vs Age by Cluster')
plt.xlabel('Income')
plt.ylabel('Age')
plt.legend()
plt.grid(True)
plt.show()

# 根据每个簇的平均值定义簇名
name_data = pd.DataFrame()
for i in range(optimal_cluster):
    cluster_data = cleaned_data.loc[model.labels_ == i]
    mean_age = cluster_data['Age'].mean()
    mean_income = cluster_data['Income'].mean()
    mean_spending = cluster_data['Spending'].mean()

    name_data.loc[i, 'Income'] = mean_income
    name_data.loc[i, 'Age'] = mean_age
    name_data.loc[i, 'Spending'] = mean_spending

print("Cluster Characteristics:")
print(name_data)