import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt


# Define the path to the 8 folders
folder_list = [
    "dataset_actions_250/chew",
    "dataset_actions_250/eye_blink",
    "dataset_actions_250/feet",
    "dataset_actions_250/head",
    "dataset_actions_250/jerk",
    "dataset_actions_250/left_foot",
    "dataset_actions_250/none",
    "dataset_actions_250/right_foot",
]

# List of folders that contain the npy data


# Load all the data from the folders into a single array
all_data = []
for folder in folder_list:
    folder_path = os.path.join(os.getcwd(), folder)
    for filename in os.listdir(folder_path):
        if filename.endswith(".npy"):
            data = np.load(os.path.join(folder_path, filename))
            all_data.append(data)

# Concatenate the data from all folders into a single 2D array
all_data = np.concatenate(all_data)

# Apply PCA to reduce the data to 2 dimensions
pca = PCA(n_components=2)
data_2d = pca.fit_transform(all_data)

# Apply KMeans clustering to the data
kmeans = KMeans(n_clusters=8)
class_labels = kmeans.fit_predict(data_2d)

# Visualize the data using scatter plot
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=class_labels, cmap="viridis")
plt.show()
