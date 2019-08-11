from typing import Tuple

from numpy import ndarray
from sklearn.cluster import KMeans


def kmeans(samples: ndarray, **options) -> Tuple[ndarray, ndarray]:
    arr: ndarray = KMeans(**options).fit_transform(samples)

    closest_points_to_centroids = arr.argmin(axis=0)
    clustering_labels = arr.argmin(axis=1)

    return clustering_labels, closest_points_to_centroids
