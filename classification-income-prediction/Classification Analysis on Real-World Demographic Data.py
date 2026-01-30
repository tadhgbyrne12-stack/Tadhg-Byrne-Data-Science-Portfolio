import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------
# 1) Features / target
# -------------------------
y = df["label"]
X = df.drop(columns=["label", "Unnamed: 0"], errors="ignore")

# Identify categorical vs numeric columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns
num_cols = X.select_dtypes(include=[np.number]).columns

# -------------------------
# 2) Train/test split 
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -------------------------
# 3) Preprocessing
# -------------------------
# OneHotEncode categoricals; pass through numeric columns
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

# -------------------------
# 4) Decision Tree pipeline
# -------------------------
dt_model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", DecisionTreeClassifier(max_depth=3, random_state=42))
])

dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, dt_preds))
print(classification_report(y_test, dt_preds))

dt_cm = confusion_matrix(y_test, dt_preds)
print("Decision Tree Confusion Matrix:\n", dt_cm)

# -------------------------
# 5) KNN pipeline (adds scaling)
# -------------------------
knn_model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("scale", StandardScaler(with_mean=False)),  # with_mean=False works with sparse one-hot output
    ("model", KNeighborsClassifier(n_neighbors=11))
])

knn_model.fit(X_train, y_train)
knn_preds = knn_model.predict(X_test)

print("KNN Accuracy:", accuracy_score(y_test, knn_preds))
print(classification_report(y_test, knn_preds))

knn_cm = confusion_matrix(y_test, knn_preds)
print("KNN Confusion Matrix:\n", knn_cm)

# -------------------------
# 6) K sweep for KNN
# -------------------------
k_values = range(1, 21)
knn_accuracies = []

for k in k_values:
    model_k = Pipeline(steps=[
        ("preprocess", preprocess),
        ("scale", StandardScaler(with_mean=False)),
        ("model", KNeighborsClassifier(n_neighbors=k))
    ])
    model_k.fit(X_train, y_train)
    preds_k = model_k.predict(X_test)
    knn_accuracies.append(accuracy_score(y_test, preds_k))

plt.figure()
plt.plot(list(k_values), knn_accuracies, marker="o")
plt.xlabel("Number of Neighbours (k)")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy for Different Values of k")
plt.show()
