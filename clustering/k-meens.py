import numpy as np
import matplotlib.pyplot as plt

def k_means(X, k, max_iters=100):
    # Инициализация центроидов случайными точками из набора данных
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        # Шаг кластеризации: каждая точка назначается к ближайшему центроиду
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Шаг обновления центроидов: новый центроид - среднее всех точек в кластере
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Если центроиды не изменились, завершаем итерации
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return centroids, labels

# Генерация случайных данных
np.random.seed(42)
X = np.random.rand(100, 2)

# Количество кластеров
k = 3

# Применение алгоритма k-средних
centroids, labels = k_means(X, k)

# Визуализация результатов
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75)
plt.title('Результаты кластеризации методом k-средних')
plt.show()
