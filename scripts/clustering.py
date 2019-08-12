import pickle
from typing import List, Callable, Any

import numpy
from pandas import DataFrame, read_pickle
from sklearn.cluster import KMeans

from clustering_chat.clustering import kmeans
from clustering_chat.utils import flatten, partition_df


def test_groups(df: DataFrame, test_group_size: int) -> List[DataFrame]:
    videos = df.groupby('video.url')
    windows: List[DataFrame] = flatten([
        partition_df(df_video.sort_values(by='published_at'), test_group_size)
        for _, df_video in videos
    ])

    return windows


def clustering_function(clustering_method: str, **clustering_options) -> Callable[[Any], List[int]]:
    if clustering_method == 'kmeans':
        return KMeans(**clustering_options).fit_predict
    else:
        raise NotImplementedError(clustering_method)


def main(dataset_path: str, output_path: str, test_group_size: int, **clustering_options):
    df_all = read_pickle(dataset_path)
    dfs = test_groups(df_all, test_group_size)

    for df in dfs:
        document_vectors = numpy.stack(df['document_vector'].to_numpy())
        clustering_labels, closest_points_to_centroids = kmeans(document_vectors, **clustering_options)
        df_indices = numpy.arange(len(df.index))
        df['clustering_label'] = clustering_labels
        df['is_representative'] = numpy.isin(df_indices, closest_points_to_centroids)

    with open(output_path, mode='wb') as f:
        pickle.dump(dfs, f)


if __name__ == '__main__':
    main(
        dataset_path='resources/document_vectors.pkl',
        output_path='resources/clustering_results.pkl',
        test_group_size=100,
        n_clusters=10,
        random_state=0,
        n_jobs=4
    )
