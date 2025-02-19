import numpy as np
from sklearn.cluster import Birch
from constants import N_CLUSTERS, BIRCH_THRESHOLD, BIRCH_BRANCHING_FACTOR


def cluster_patches(patches):
    # patches is a list of images (np arrays)
    patch_vectors = [img.reshape(-1) for img in patches]

    if len(patch_vectors) < 2:
        # If we have fewer than 2 patches, all belong to the same cluster
        return np.zeros(len(patch_vectors), dtype=int)

    images_array = np.array(patch_vectors)

    # Perform BIRCH clustering with fixed parameters
    birch = Birch(
        threshold=BIRCH_THRESHOLD,
        branching_factor=BIRCH_BRANCHING_FACTOR,
        n_clusters=N_CLUSTERS,
    )
    cluster_labels = birch.fit_predict(images_array)

    return cluster_labels
