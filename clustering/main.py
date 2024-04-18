from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Генерация набора данных
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Визуализация сгенерированных точек
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title('Сгенерированные точки')
plt.show()

# Кластеризация с использованием k-ближайших соседей
k = 4  # количество кластеров
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)

# Получение меток кластеров для каждой точки
labels = kmeans.labels_

# Визуализация результатов кластеризации
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

# Визуализация центроидов кластеров
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.title('Результаты кластеризации методом k-ближайших соседей')
plt.show()
