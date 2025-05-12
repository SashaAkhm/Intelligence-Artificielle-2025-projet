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
nn4 = NeuralNet.NeuralNet(hidden_layer_sizes=(16, 4, 16), activation='tahn', learning_rate=0.01, epoch=200, batch_size=5)
nn8 = NeuralNet.NeuralNet(hidden_layer_sizes=(16, 8, 16), activation='tahn', learning_rate=0.01, epoch=200, batch_size=5)
nn12 = NeuralNet.NeuralNet(hidden_layer_sizes=(16, 12, 16), activation='tahn', learning_rate=0.01, epoch=200, batch_size=5)

# Fit the model
nn4.fit(X_train, y_train, X_val, y_val)
nn8.fit(X_train, y_train, X_val, y_val)
nn12.fit(X_train, y_train, X_val, y_val)

# with open('neural_net_tp4.txt', 'w') as nn_file:
#     print(nn, file=nn_file)


# Find 50% best instances
all_errors = np.square(np.subtract(nn.predict(X_all), y_all)).mean(axis=1) / 2

median = np.median(all_errors)
mask = all_errors < median
X_50 = X_all[mask, :]
labels_50 = labels[mask]

# Make the predictions for best 50%
X_50_compressed_to4 = nn4.compresse(X_50)
X_50_compressed_to8 = nn8.compresse(X_50)
X_50_compressed_to12 = nn12.compresse(X_50)


# PCA graphiques for 8-compressing
pca = PCA(n_components=2)
unique_labels = np.unique(labels)

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
X_50_comp_pca = pca.fit_transform(X_50_compressed_to8)

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



# Functions pour l'erreur moyeene et son ecart-type
def er_instance(X, y, nn):
    return None

def er_attribute(X, y, nn):
    return None

def er_classe(X, y, nn):
    return None

def plot_model_comparison(type):
    if type == 'instance':
        error_f = er_instance
    elif type == 'attribute':
        error_f = er_attribute
    elif type == 'classe':
        error_f = er_classe

    # array of errors for 3 models
    errors = [error_f(X_all, y_all, nn4), error_f(X_all, y_all, nn8), error_f(X_all, y_all, nn12)]
    errors_names = [f"Model compressing to {4 * i} features" for i in range(1, 4)]

    # Mean and ecart-type
    means = [np.mean(m) for m in errors]
    stds = [np.std(m) for m in errors]

    # Drawing plot
    plt.figure(figsize=(12, 8))
    plt.errorbar(errors_names, means, yerr=stds, fmt='o', capsize=5, color='black')

    plt.title(f"Comparing of {type}s errors for 3 different models")
    plt.ylabel('Value')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


plot_model_comparison('instance')
plot_model_comparison('attribute')
plot_model_comparison('classe')


