import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import Utility
import NeuralNet

# Load data
df = pd.read_csv('iris_extended.csv')

# Encode the soil_type column by -1, 0, 1
df['soil_type'] = df['soil_type'].astype('category').cat.codes - 1

# Extraite features and labels with normalizing
df_columns = df.columns.values.tolist()
features = df_columns[1:]
df[features] = df[features].apply(lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1)
label_col = df_columns[0]
labels = df[label_col].to_numpy()

# We have responses equal to the instances
X = df[features]
y = df[features]

# 1000 instances -- train, 200 instances -- test
X_train, X_val, y_train, y_val = \
    train_test_split(X, y, test_size=200, random_state=42)

# Convert dataframes to numpy arrays
X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
X_val, y_val = X_val.to_numpy(), y_val.to_numpy()
X_all, y_all = X.to_numpy(), y.to_numpy()

# Initialize a neural network
nn = NeuralNet.NeuralNet(hidden_layer_sizes=(16, 8, 16), activation='tahn', learning_rate=0.01, epoch=200, batch_size=1)

# Fit the model
nn.fit(X_train, y_train, X_val, y_val)

with open('neural_net_tp4.txt', 'w') as nn_file:
    print(nn, file=nn_file)


# Find 50% best instances
all_errors = np.square(np.subtract(nn.predict(X_all), y_all)).mean(axis=1) / 2

median = np.median(all_errors)
mask = all_errors < median
X_50 = X_all[mask, :]
labels_50 = labels[mask]

# Make the predictions for best 50%
X_50_compressed = nn.compresse(X_50)


# PCA graphiques
pca = PCA(n_components=2)
unique_labels = np.unique(labels)

# PCA graphique for initial data (all)
X_pca = pca.fit_transform(X_all)
plt.figure(figsize=(12, 8))

for lab in unique_labels:
    mask = labels == lab
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label= f"Espese {lab}",
        alpha=0.7
    )

plt.title('PCA for all data')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid(True)
plt.show()

# PCA graphique for initial data (50% best)
X_50_pca = pca.fit_transform(X_50)
plt.figure(figsize=(12, 8))

for lab in unique_labels:
    mask = labels_50 == lab
    plt.scatter(
        X_50_pca[mask, 0],
        X_50_pca[mask, 1],
        label= f"Espese {lab}",
        alpha=0.7
    )

plt.title('PCA for initial data (50% best for compressing)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid(True)
plt.show()


# PCA graphique for compressed data (50% best)
X_50_comp_pca = pca.fit_transform(X_50_compressed)

plt.figure(figsize=(12, 8))
for lab in unique_labels:
    mask = labels_50 == lab
    plt.scatter(
        X_50_comp_pca[mask, 0],
        X_50_comp_pca[mask, 1],
        label= f"Espese {lab}",
        alpha=0.7
    )

plt.title('PCA for compressed data (50% best for compressing)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid(True)
plt.show()

def mse_instance(X, y):

    return np.mean(np.square(np.subtract(nn.predict(X), y)) / 2, axis=1)


def mse_attribute(X, y):

    return np.mean(np.square(np.subtract(nn.predict(X)), y) / 2, axis=0)


def mse_classe(X, y):
    err_by_label = []
    for lab in unique_labels:
        mask = labels == lab
        X_lab, y_lab = X[mask], y[mask]
        err_by_label.append(np.mean(np.square(np.subtract(nn.predict(X_lab)), y_lab) / 2))

    return err_by_label
