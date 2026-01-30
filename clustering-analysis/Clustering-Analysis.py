import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score


# -------------------------
# 1) Load data
# -------------------------

df = pd.read_csv("data/clustering_dataset.csv")
X = df[["attr1", "attr2", "attr3"]].values  
])

print("Data shape:", X.shape)

# Summary stats (to match README)
df_stats = pd.DataFrame(X, columns=["attr1", "attr2", "attr3"]).describe()
print("\nSummary statistics:\n", df_stats)

# -------------------------
# 2) Standardise features
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# 3) Evaluate clustering across k
# -------------------------
k_range = range(2, 11)

results = []

for k in k_range:
    # K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_labels = kmeans.fit_predict(X_scaled)
    km_sil = silhouette_score(X_scaled, km_labels)
    km_db = davies_bouldin_score(X_scaled, km_labels)

    results.append({
        "algorithm": "kmeans",
        "k": k,
        "silhouette": km_sil,
        "davies_bouldin": km_db
    })

    # Agglomerative (Ward)
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    ag_labels = agg.fit_predict(X_scaled)
    ag_sil = silhouette_score(X_scaled, ag_labels)
    ag_db = davies_bouldin_score(X_scaled, ag_labels)

    results.append({
        "algorithm": "agglomerative_ward",
        "k": k,
        "silhouette": ag_sil,
        "davies_bouldin": ag_db
    })

results_df = pd.DataFrame(results)
print("\nResults (first 10 rows):\n", results_df.head(10))

# -------------------------
# 4) Select "best" k (simple rule)
# -------------------------
# Primary: maximize silhouette
# Tie-break: minimize Davies–Bouldin
def choose_best(df_alg: pd.DataFrame) -> pd.Series:
    df_sorted = df_alg.sort_values(
        by=["silhouette", "davies_bouldin"],
        ascending=[False, True]
    )
    return df_sorted.iloc[0]

best_kmeans = choose_best(results_df[results_df["algorithm"] == "kmeans"])
best_agg = choose_best(results_df[results_df["algorithm"] == "agglomerative_ward"])

print("\nBest K-Means configuration:\n", best_kmeans)
print("\nBest Agglomerative configuration:\n", best_agg)

# -------------------------
# 5) Plot metric curves
# -------------------------
plt.figure()
for alg in ["kmeans", "agglomerative_ward"]:
    sub = results_df[results_df["algorithm"] == alg].sort_values("k")
    plt.plot(sub["k"], sub["silhouette"], marker="o", label=f"{alg} silhouette")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs Number of Clusters (k)")
plt.legend()
plt.show()

plt.figure()
for alg in ["kmeans", "agglomerative_ward"]:
    sub = results_df[results_df["algorithm"] == alg].sort_values("k")
    plt.plot(sub["k"], sub["davies_bouldin"], marker="o", label=f"{alg} Davies–Bouldin")
plt.xlabel("k")
plt.ylabel("Davies–Bouldin Index")
plt.title("Davies–Bouldin Index vs Number of Clusters (k)")
plt.legend()
plt.show()

# -------------------------
# 6) Fit best models and visualise clusters (2D projection)
# -------------------------

# Best K-Means
k_best = int(best_kmeans["k"])
kmeans_best = KMeans(n_clusters=k_best, random_state=42, n_init=10)
km_labels_best = kmeans_best.fit_predict(X_scaled)

plt.figure()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=km_labels_best)
plt.xlabel("attr1 (scaled)")
plt.ylabel("attr2 (scaled)")
plt.title(f"K-Means Clusters (k={k_best})")
plt.show()

# Best Agglomerative
k_best_ag = int(best_agg["k"])
agg_best = AgglomerativeClustering(n_clusters=k_best_ag, linkage="ward")
ag_labels_best = agg_best.fit_predict(X_scaled)

plt.figure()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=ag_labels_best)
plt.xlabel("attr1 (scaled)")
plt.ylabel("attr2 (scaled)")
plt.title(f"Agglomerative (Ward) Clusters (k={k_best_ag})")
plt.show()

# -------------------------
# 7)  DBSCAN check 
# -------------------------
dbscan = DBSCAN(eps=0.5, min_samples=5)
db_labels = dbscan.fit_predict(X_scaled)

n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = np.sum(db_labels == -1)

print(f"\nDBSCAN clusters found: {n_clusters}, noise points: {n_noise}")

