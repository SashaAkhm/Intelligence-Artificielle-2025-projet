import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import NeuralNet

# Load data
df = pd.read_csv('iris_extended.csv')

# Encode the 'soil_type' column into numerical values -1, 0, 1
df['soil_type'] = df['soil_type'].astype('category').cat.codes - 1

# Extract features and labels, applying normalization
df_columns = df.columns.values.tolist()
features = df_columns[1:]
df[features] = df[features].apply(lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1)
label_col = df_columns[0]
labels = df[label_col].to_numpy()
unique_labels = np.unique(labels)

# Instances and targets are the same
X = df[features]
y = df[features]

# Split 1000 instances for training and 200 for testing
X_train, X_val, y_train, y_val = \
    train_test_split(X, y, test_size=200, random_state=42)

# Convert dataframes to numpy arrays
X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
X_val, y_val = X_val.to_numpy(), y_val.to_numpy()
X_all, y_all = X.to_numpy(), y.to_numpy()

# Initialize neural networks
nn4 = NeuralNet.NeuralNet(hidden_layer_sizes=(16, 4, 16), activation='tahn', learning_rate=0.01, epoch=200, batch_size=5)
nn8 = NeuralNet.NeuralNet(hidden_layer_sizes=(16, 8, 16), activation='tahn', learning_rate=0.01, epoch=200, batch_size=5)
nn12 = NeuralNet.NeuralNet(hidden_layer_sizes=(16, 12, 16), activation='tahn', learning_rate=0.01, epoch=200, batch_size=5)

# Fit the model
nn4.fit(X_train, y_train, X_val, y_val)
nn8.fit(X_train, y_train, X_val, y_val)
nn12.fit(X_train, y_train, X_val, y_val)


def get_best_50_percent_instances(nn=nn8):
    """
    Find the 50% of instances with the lowest reconstruction cost

    Parameters:
      nn: model
    Returns:
      X_50: 50% of instances
      X_50_comp: their compressed representations
      labels_50: their class labels, used for visualization iris type
    """

    all_errors = np.square(np.subtract(nn.predict(X_all), y_all)).mean(axis=1) / 2

    median = np.median(all_errors)
    mask_50 = all_errors < median
    X_50 = X_all[mask_50, :]
    labels_50 = labels[mask_50]

    X_50_comp = nn.compresse(X_50)

    return X_50, X_50_comp, labels_50


def plot_PCA(nn=nn8):
    """
    Compares the PCA of the original and compressed data for the top 50% of instances with the lowest reconstruction cost

    Parameters:
      nn: model
    """

    pca = PCA(n_components=2)

    X_50, X_50_comp, labels_50 = get_best_50_percent_instances(nn)

    # PCA of instances (50% best)
    X_50_pca = pca.fit_transform(X_50)
    plt.figure(figsize=(12, 8))

    for lab in unique_labels:
        mask = labels_50 == lab
        plt.scatter(
            X_50_pca[mask, 0],
            X_50_pca[mask, 1],
            label=f"Espèce {lab}",
            alpha=0.7
        )

    plt.title('PCA of instances with the lowest reconstruction error')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True)
    plt.show()


    # PCA of compressed data (50% best)
    X_50_comp_pca = pca.fit_transform(X_50_comp)

    plt.figure(figsize=(12, 8))
    for lab in unique_labels:
        mask = labels_50 == lab
        plt.scatter(
            X_50_comp_pca[mask, 0],
            X_50_comp_pca[mask, 1],
            label=f"Espèce {lab}",
            alpha=0.7
        )

    plt.title('PCA of compressed data with the lowest reconstruction error')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True)
    plt.show()



#### Functions to compute average and std of reconstruction error
"""
Applies to all functions in this section.

Parameters:
  X : batch of instances
  y : corresponding targets
  nn : model
  make_plot : whether to generate a visualization of errors
  
  * plotting is skipped for mse_instance (no meaningful)
Returns:
  mean reconstruction error
"""

def mse_instance(X, y, nn, make_plot = False):

    return np.mean(np.square(np.subtract(nn.predict(X), y)), axis=1)


def mse_attribute(X, y, nn, make_plot = True):

    mse_par_attr = np.square(np.subtract(nn.predict(X), y))
    if make_plot:
        errors = mse_par_attr.T
        errors_names = [f"Atr {i + 1}" for i in range(20)]

        means = np.mean(errors, axis=1)
        stds = np.std(errors, axis=1)

        plt.figure(figsize=(12, 8))
        plt.errorbar(errors_names, means, yerr=stds, fmt='o', capsize=5, color='black')

        plt.title(f"Comparing reconstruction errors for different attributes")
        plt.ylabel('Value')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()


    return np.mean(mse_par_attr, axis=0)


def mse_class(X, y, nn, make_plot = True):
    errors = []
    errors_mean = []
    for lab in unique_labels:
        mask = labels == lab
        X_lab, y_lab = X[mask], y[mask]

        mse_par_class = np.mean(np.square(np.subtract(nn.predict(X_lab), y_lab)), axis = 1)
        errors.append(mse_par_class)
        errors_mean.append(np.mean(mse_par_class))

    if make_plot:
        errors_names = [f"Class {i + 1}" for i in range(3)]

        stds = [np.std(m) for m in errors]

        plt.figure(figsize=(12, 8))
        plt.errorbar(errors_names, errors_mean, yerr=stds, fmt='o', capsize=5, color='black')

        plt.title(f"Comparing reconstruction errors for different classes")
        plt.ylabel('Value')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    return errors_mean




def plot_model_comparison(type):
    """
    Compares reconstruction errors by type across different models

    Parameters:
      type : criterion used for comparison (instance/attribute/class)

    """

    if type == 'instance':
        error_f = mse_instance
    elif type == 'attribute':
        error_f = mse_attribute
    elif type == 'classe':
        error_f = mse_class

    # array of errors for 3 models
    errors = [error_f(X_all, y_all, nn4, make_plot = False), error_f(X_all, y_all, nn8, make_plot = False), error_f(X_all, y_all, nn12, make_plot = False)]
    errors_names = [f"Model with {4 * i} latent features " for i in range(1, 4)]

    # Calculate mean and std of errors for each model
    means = [np.mean(m) for m in errors]
    stds = [np.std(m) for m in errors]


    # Plot the comparison of errors
    plt.figure(figsize=(12, 8))
    plt.errorbar(errors_names, means, yerr=stds, fmt='o', capsize=5, color='black')

    plt.title(f"Comparison of {type}s errors across 3 models")
    plt.ylabel('Error Value')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



# plot PCA visualization for model nn8
plot_PCA(nn8)

# Plot comparisons across attributes for model nn8
mse_attribute(X_all, y_all, nn8)

# Plot comparisons across classes for model nn8
mse_class(X, y, nn8)

# Plot comparisons across 3 models
plot_model_comparison('instance')
plot_model_comparison('attribute')
plot_model_comparison('classe')



def MSE_cost(y_hat, y):
    mse = np.square(np.subtract(y_hat, y)).mean()
    return mse


k = 5 # Number of instances

# Select 5 random instances
instances=[]
indexes = np.random.choice(len(X_all), size=k)
for ind in indexes:
    instances.append(X_all[ind:ind+1])

print("\n\n")
# Calculate error of reconstruction for each instance
for i in range(k):
    inst = instances[i]
    print(f"Error of instance {i + 1}: ", MSE_cost(inst, nn8.predict(inst)))
