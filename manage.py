import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    rand_score,
    adjusted_rand_score
)
from scipy.spatial.distance import cdist

RANDOM_STATE = 42
N_INIT = 10

sns.set_theme(style="whitegrid")
np.random.seed(RANDOM_STATE)

print("===== 1. Описание датасета =====")
print("===== 1.1. Описание датасета =====")

dt = pd.read_csv("Mall_Customers.csv")
dt = dt.rename(columns={
    "Annual Income (k$)": "Annual_Income",
    "Spending Score (1-100)": "Spending_Score",
    "Genre": "Gender"
})

print("Название датасета: Kaggle - Mall Customers")
print(f"Исходный размер: {dt.shape}")
print(dt.head())

print("\nИнформация о датасете")
dt.info()

print("\nКоличество пропусков")
print(dt.isna().sum())

print("\n===== 1.2. Визуализация распределения признаков =====")
print(dt.describe())
print("\nРаспределение мужчин и женщин")
print(dt["Gender"].value_counts())

print("\nГистограммы признаков")
num_cols = ["Age", "Annual_Income", "Spending_Score"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, num_cols):
    sns.histplot(dt[col], kde=False, ax=ax)
    ax.set_title(f"Распределение: {col}")
plt.tight_layout()
plt.show()

print("\n===== 1.3. Оценка информационной значимости признаков =====")
print("Корреляционная матрица числовых признаков")
plt.figure(figsize=(6, 5))
sns.heatmap(dt[num_cols].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Корреляционная матрица числовых признаков")
plt.show()

print("\n===== 1.4. Анализ выбросов =====")
print("Диаграмма ящика с усами")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, num_cols):
    sns.boxplot(y=dt[col], ax=ax)
    ax.set_title(f"Ящик: {col}")
plt.tight_layout()
plt.show()

print("Выбросы:")
outliers_idx = set()
for col in num_cols:
    q1, q3 = dt[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (dt[col] < lower) | (dt[col] > upper)
    outliers_count = mask.sum()

    idx = dt[col][mask].index
    outliers_idx.update(idx)

    print(f" {col}: {outliers_count}")


print("\n===== 2. Предобработка данных =====")
print("===== 2.1. Удаление выбросов =====")
initial_shape = dt.shape[0]
outliers_idx = list(outliers_idx)
dt = dt.drop(index=outliers_idx).reset_index(drop=True)
print(f"Размер до удаления: {initial_shape}")
print(f"Размер после удаления: {dt.shape[0]}")

print("\n===== 2.2. Приведение данных к стандартному виду =====")
features = ["Annual_Income", "Spending_Score"]
X = dt[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Данные приведены к стандартному виду")

print("\n===== 2.3. Матрица диаграмм рассеивания =====")
print("Матрица рассеивания")
sns.pairplot(pd.DataFrame(X_scaled, columns=features))
plt.suptitle("Матрица рассеивания", y=1.02)
plt.show()


print("\n===== 3. Подбор k: метод локтя и силуэтный анализ =====")
inertia, sil_scores = [], []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_scaled)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

print("Диаграмма метода локтя и силуэтного анализа")
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(list(k_range), inertia, "o-")
axes[0].set_title("Метод локтя")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")
axes[1].plot(list(k_range), sil_scores, "o-")
axes[1].set_title("Силуэтный анализ")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Silhouette")
plt.tight_layout()
plt.show()

best_k = list(k_range)[int(np.argmax(sil_scores))]
print(f"Оптимальное k: {best_k}")


print("\n===== 4. K-means =====")
km = KMeans(n_clusters=best_k, n_init=N_INIT, random_state=RANDOM_STATE)
km_labels = km.fit_predict(X_scaled)
centers_km = km.cluster_centers_

print("Размеры кластеров K-means:", np.bincount(km_labels))

sil_km = silhouette_score(X_scaled, km_labels)
db_km = davies_bouldin_score(X_scaled, km_labels)
ch_km = calinski_harabasz_score(X_scaled, km_labels)

intra = []
for k in range(best_k):
    points = X_scaled[km_labels == k]
    center = centers_km[k:k+1]
    dist = cdist(points, center)
    intra.append(np.mean(dist))
inter = cdist(centers_km, centers_km)
inter_off = inter[~np.eye(best_k, dtype=bool)]

print(f"Среднее внутрикластерное расстояние: {np.mean(intra)}")
print(f"Среднее межкластерное расстояние: {np.mean(inter_off)}")


print("\n===== 5. EM-алгоритм (Gaussian Mixture) =====")
gmm = GaussianMixture(n_components=best_k, covariance_type='full',
                      n_init=N_INIT, random_state=RANDOM_STATE)
gmm_labels = gmm.fit_predict(X_scaled)
gmm_means = gmm.means_

print("Размеры кластеров EM:", np.bincount(gmm_labels))

sil_gmm = silhouette_score(X_scaled, gmm_labels)
db_gmm = davies_bouldin_score(X_scaled, gmm_labels)
ch_gmm = calinski_harabasz_score(X_scaled, gmm_labels)


print("\n===== 6. Сравнительная таблица метрик и внешние метрики =====")
print("===== 6.1. Сравнительная таблица метрик качества =====")
metrics_table = pd.DataFrame({
    "Метрика": ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz",
                "Число кластеров"],
    "K-means": [sil_km, db_km, ch_km, best_k],
    "EM (GMM)": [sil_gmm, db_gmm, ch_gmm, best_k]
})
print(metrics_table)

print("\n===== 6.2. Внешние метрики =====")
rand = rand_score(km_labels, gmm_labels)
adjusted_rand = adjusted_rand_score(km_labels, gmm_labels)
print(f"Rand index = {rand}")
print(f"Adjusted Rand = {adjusted_rand}")


print("\n===== 7. Влияние параметров EM (тип ковариации) =====")
results = []
for cov in ['full', 'tied', 'diag', 'spherical']:
    gmm_temp = GaussianMixture(n_components=best_k, covariance_type=cov,
                               n_init=N_INIT, random_state=RANDOM_STATE)
    gmm_temp.fit(X_scaled)
    lbl = gmm_temp.predict(X_scaled)
    s = silhouette_score(X_scaled, lbl)
    results.append((cov, s, gmm_temp.aic(X_scaled), gmm_temp.bic(X_scaled)))

res_dt = pd.DataFrame(results, columns=["covariance_type", "silhouette",
                                        "AIC", "BIC"])
print(res_dt.to_string(index=False))


print("\n===== 8. Визуализация кластеров через PCA =====")
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
print("Объяснённая дисперсия PCA:", pca.explained_variance_ratio_)

centers_km_pca = pca.transform(centers_km)
centers_gmm_pca = pca.transform(gmm_means)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=km_labels,
    cmap="tab10",
    s=40
)
axes[0].scatter(
    centers_km_pca[:, 0],
    centers_km_pca[:, 1],
    marker="X",
    s=200,
    c="black",
    label="Центры"
)
axes[0].set_title("K-means (PCA)")
axes[0].legend()

axes[1].scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=gmm_labels,
    cmap="tab10",
    s=40
)
axes[1].scatter(
    centers_gmm_pca[:, 0],
    centers_gmm_pca[:, 1],
    marker="X",
    s=200,
    c="black",
    label="Средние"
)
axes[1].set_title("EM (GMM) (PCA)")
axes[1].legend()

plt.tight_layout()
plt.show()


print("\n===== 9. Интерпретация центров K-means и EM =====")
print("K-means")
centers_orig_km = scaler.inverse_transform(centers_km)
centers_dt_km = pd.DataFrame(centers_orig_km, columns=features)
centers_dt_km["Размер кластера"] = np.bincount(km_labels)
print("Центры K-means:")
print(centers_dt_km)

print("\nEM-алгоритм (Gaussian Mixture)")
centers_orig_gmm = scaler.inverse_transform(gmm_means)
centers_dt_gmm = pd.DataFrame(centers_orig_gmm, columns=features)
centers_dt_gmm["Размер кластера"] = np.bincount(gmm_labels)
print("Средние значения EM (GMM):")
print(centers_dt_gmm)
