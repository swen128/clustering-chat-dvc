import pickle
from typing import List, Callable, Any

import numpy
from pandas import read_pickle
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN

from clustering_chat.utils import partition_df


def clustering_function(clustering_method: str, **clustering_options) -> Callable[[Any], List[int]]:
    if clustering_method == 'kmeans':
        return KMeans(**clustering_options).fit_predict
    else:
        raise NotImplementedError(clustering_method)


def main(dataset_path: str, output_path: str, test_group_size: int, clustering_function):
    df_all = read_pickle(dataset_path)
    df_all.sort_values(by=['video.url', 'published_at'], inplace=True)

    dfs = partition_df(df_all, test_group_size)

    for df in dfs:
        distinct_messages = df.drop_duplicates(subset=['message'])
        document_vectors = numpy.stack(distinct_messages['document_vector'].to_numpy())
        cluster_labels = clustering_function(document_vectors)
        dic = {message: cluster for message, cluster in zip(distinct_messages['message'], cluster_labels)}
        df['clustering_label'] = df['message'].apply(lambda x: dic[x])

    with open(output_path, mode='wb') as f:
        pickle.dump(dfs, f)


if __name__ == '__main__':
    k_means = KMeans(n_clusters=20, random_state=0, n_jobs=4).fit_predict
    spectral = SpectralClustering(n_clusters=20, n_neighbors=5, affinity='nearest_neighbors',
                                  random_state=0, n_jobs=4).fit_predict
    ward, complete, average, single = [
        AgglomerativeClustering(n_clusters=20, linkage=linkage).fit_predict
        for linkage in ['ward', 'complete', 'average', 'single']
    ]

    main(
        dataset_path='resources/document_vectors.pkl',
        output_path='resources/clustering_results.pkl',
        test_group_size=200,
        clustering_function=k_means
    )
