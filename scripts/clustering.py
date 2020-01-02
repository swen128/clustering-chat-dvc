import pickle
from typing import List, Callable, Any

import numpy
from pandas import read_pickle
from sklearn.cluster import KMeans, AgglomerativeClustering

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
        document_vectors = numpy.stack(df['document_vector'].to_numpy())
        df['clustering_label'] = clustering_function(document_vectors)

    with open(output_path, mode='wb') as f:
        pickle.dump(dfs, f)


if __name__ == '__main__':
    # clustering_function = KMeans(n_clusters=20, random_state=0, n_jobs=4).fit_predict
    clustering_function = AgglomerativeClustering(n_clusters=20, linkage='ward').fit_predict

    main(
        dataset_path='resources/document_vectors.pkl',
        output_path='resources/clustering_results.pkl',
        test_group_size=200,
        clustering_function=clustering_function
    )
