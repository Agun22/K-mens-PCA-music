import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Untuk visualisasi 3D
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


print("--- 1. Memuat Data ---")
df = pd.read_csv('daa.csv')


df_sample = df.sample(n=1000, random_state=42).reset_index(drop=True)


features = ['danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

X = df_sample[features]
print(f"Data shape: {X.shape}")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


print("\n--- 2. Menganalisis Komponen PCA ---")
pca_full = PCA()
pca_full.fit(X_scaled)

explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)


n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
print(f"Komponen yang dibutuhkan untuk 90% variansi: {n_components_90}")


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6, color='blue')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Component')
plt.title('Individual Variance')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-', markersize=6)
plt.axhline(y=0.90, color='r', linestyle='--', label='90% Variance')
plt.xlabel('Number of Components')
plt.title('Cumulative Variance')
plt.legend()
plt.tight_layout()
plt.show()


print("\n--- 3. Melakukan Clustering & Reduksi Dimensi ---")


kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)


sil_score = silhouette_score(X_scaled, cluster_labels)
print(f"Silhouette Score (Kualitas Cluster): {sil_score:.3f}")


pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)


pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)


df_sample['cluster'] = cluster_labels
df_sample['PC1'] = X_pca_2d[:, 0]
df_sample['PC2'] = X_pca_2d[:, 1]


print("\n--- 4. Visualisasi Hasil ---")

fig = plt.figure(figsize=(16, 7))


ax1 = fig.add_subplot(1, 2, 1)
scatter = ax1.scatter(df_sample['PC1'], df_sample['PC2'], 
                     c=df_sample['cluster'], cmap='viridis', alpha=0.6)
ax1.set_xlabel(f'PC1 ({pca_full.explained_variance_ratio_[0]:.1%} var)')
ax1.set_ylabel(f'PC2 ({pca_full.explained_variance_ratio_[1]:.1%} var)')
ax1.set_title('Peta Lagu Spotify (2D PCA)')
plt.colorbar(scatter, ax=ax1, label='Cluster')


ax2 = fig.add_subplot(1, 2, 2, projection='3d')
scatter3d = ax2.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], 
                       c=df_sample['cluster'], cmap='viridis', alpha=0.6)
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_zlabel('PC3')
ax2.set_title('Peta Lagu Spotify (3D PCA)')

plt.tight_layout()
plt.show()


print("\n--- 5. Interpretasi Cluster (Rata-rata Fitur) ---")

cluster_profile = df_sample.groupby('cluster')[features].mean()
print(cluster_profile)


df_sample[['track_name', 'artists', 'cluster', 'PC1', 'PC2']].to_csv('spotify_pca_result.csv', index=False)
print("\n✅ Hasil clustering disimpan ke 'spotify_pca_result.csv'")