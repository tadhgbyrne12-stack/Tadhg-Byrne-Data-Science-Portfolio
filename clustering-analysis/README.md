# Unsupervised Clustering Analysis on Synthetic Numerical Data

## 1. Problem Overview
The objective of this project was to explore the underlying structure of a numerical dataset using unsupervised learning techniques. As the dataset was artificially generated and unlabeled, the goal was not prediction, but to identify and evaluate natural groupings within the data and compare the performance of different clustering algorithms.

---

## 2. Data Overview
- Dataset consisting of three numeric attributes  
- No missing values  
- Artificially generated to support algorithm comparison rather than domain-specific inference  

Because all variables were numeric and scaled similarly, the dataset was well suited to distance-based clustering methods.

---

## 3. Preprocessing
- Summary statistics (mean, standard deviation, min, max) were computed to understand data spread  
- Features were standardised using **StandardScaler** to ensure fair distance calculations  
- Scaled data was used consistently across all clustering experiments  

---

## 4. Clustering Methods
Two unsupervised clustering algorithms were applied and compared:

### K-Means Clustering
- Distance-based algorithm using Euclidean distance  
- Requires pre-specifying the number of clusters (*k*)  
- Produces compact, spherical clusters centred around centroids  

### Agglomerative (Hierarchical) Clustering
- Bottom-up hierarchical clustering using Ward linkage  
- Does not rely on initial centroid placement  
- Can produce less regular cluster shapes, particularly in noisy data  

DBSCAN was explored initially but excluded from final analysis due to limited suitability for this dataset.

---

## 5. Evaluation Metrics
Clustering quality was evaluated using two internal validation metrics:

- **Silhouette Score** – measures cluster cohesion and separation (higher is better)  
- **Davies–Bouldin Index** – measures inter-cluster similarity (lower is better)  

Both metrics were evaluated across a range of cluster values (*k = 2–10*).

---

## 6. Results
### K-Means
- Optimal clustering observed at **k = 6**
- Silhouette score ≈ **0.62**
- Davies–Bouldin index ≈ **0.54**
- Produced compact, well-separated clusters  

### Agglomerative Clustering
- Best-performing configuration at **k = 7**
- Silhouette score ≈ **0.62**
- Davies–Bouldin index ≈ **0.57**
- Clusters showed more overlap but retained meaningful separation  

Overall, both algorithms produced comparable quantitative performance, with K-Means yielding more visually compact clusters and agglomerative clustering offering greater structural flexibility.

---

## 7. Interpretation
- K-Means produced more uniform, spherical clusters due to centroid-based assignment  
- Agglomerative clustering showed increased overlap, reflecting its hierarchical linkage process  
- Neither algorithm clearly dominated the other for this dataset  
- Results aligned with established literature comparing centroid-based and hierarchical clustering approaches  

---

## 8. Limitations
- Dataset was synthetic and may not reflect real-world noise or complexity  
- Cluster quality was assessed using internal metrics only  
- No external ground truth labels were available for validation  

---

## 9. Next Steps
- Apply clustering to real-world datasets  
- Explore density-based clustering (e.g. DBSCAN) on more complex data  
- Investigate dimensionality reduction techniques prior to clustering  

