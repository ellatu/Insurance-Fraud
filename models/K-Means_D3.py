import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 假设你的数据集是一个CSV文件
data = pd.read_csv("/Users/ellatu/Desktop/毕业论文/文献数据/数据/d3_清理.csv")

# 查看数据集的前几行
print(data.head())

# 删除包含缺失值的行
data_cleaned = data.dropna()

# 如果有分类变量，进行编码
# data = pd.get_dummies(data, columns=['categorical_column'])

# 假设目标变量是 'fraud' 列
X = data.drop(['FRAUDE'], axis=1)  # 特征

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 初始化K-Means模型
kmeans = KMeans(n_clusters=2, random_state=42)  # 假设分为3个簇

# 训练模型
kmeans.fit(X_scaled)

# 获取聚类标签
cluster_labels = kmeans.labels_

# 将聚类标签添加到原始数据集中
data_cleaned['cluster'] = cluster_labels

# 查看聚类结果
print(data_cleaned.head())

# 计算轮廓系数
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print(f'Silhouette Score: {silhouette_avg:.2f}')

# 尝试不同的K值
inertia_values = []
K_values = range(2, 11)  # 尝试K从2到10
for k in K_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)  # 保存每个K值的惯性值

# 绘制肘部法曲线
plt.plot(K_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# 使用PCA将数据降到2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 可视化聚类结果
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', marker='o')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('D3 Clustering (2D PCA)')
plt.colorbar(label='Cluster')
plt.show()

# 按聚类标签分组，查看每个簇的统计信息
cluster_summary = data_cleaned.groupby('cluster').mean()
print(cluster_summary)

# 查看每个簇中欺诈样本的比例
fraud_by_cluster = data_cleaned.groupby('cluster')['FRAUDE'].mean()
print(fraud_by_cluster)