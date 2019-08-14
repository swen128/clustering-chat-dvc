import json
import pickle
from statistics import mean
from typing import Dict, Iterable

import numpy
from pandas import DataFrame
from scipy.stats import entropy
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
from toolz import merge


def vocabulary(df: DataFrame) -> Dict[str, int]:
    tokens = numpy.concatenate(df['tokens'].to_numpy())
    vocab, count = numpy.unique(tokens, return_counts=True)

    return {word: n for word, n in zip(vocab, count) if word != '▁'}


def vocabulary_coverage(df: DataFrame) -> Dict[str, float]:
    representatives = df.loc[df['is_representative']]
    all_vocab = vocabulary(df)
    covered_vocab = vocabulary(representatives)

    all_weight = sum(n for word, n in all_vocab.items())
    covered_weight = sum(n for word, n in all_vocab.items() if word in covered_vocab)

    return dict(
        vocabulary_coverage=len(covered_vocab) / len(all_vocab),
        weighted_vocabulary_coverage=covered_weight / all_weight
    )


def entropy_reduction(df: DataFrame) -> Dict[str, float]:
    all_vocab = vocabulary(df)
    clusters_vocabs = [vocabulary(df_cluster) for _, df_cluster in df.groupby('clustering_label')]

    all_entropy = entropy(list(all_vocab.values()))
    clusters_entropies = [entropy(list(vocab.values())) for vocab in clusters_vocabs]

    return dict(entropy_reduction=all_entropy - mean(clusters_entropies))


def clustering_scores(df: DataFrame) -> Dict[str, float]:
    X = numpy.stack(df['document_vector'].values)
    labels = df['clustering_label']

    return dict(
        Calinski_Harabasz=calinski_harabasz_score(X, labels),
        Davies_Bouldin=davies_bouldin_score(X, labels),
        Silhouette=silhouette_score(X, labels, random_state=0)
    )


def all_scores(df: DataFrame) -> Dict[str, float]:
    return merge(
        clustering_scores(df),
        entropy_reduction(df),
        vocabulary_coverage(df)
    )


def aggregate_scores(dfs: Iterable[DataFrame]) -> Dict[str, float]:
    return DataFrame(map(all_scores, dfs)).median().to_dict()


def main(input_path: str, out_path: str):
    with open(input_path, mode='rb') as f:
        dfs = pickle.load(f)

    scores_dic = aggregate_scores(dfs)

    with open(out_path, mode='w') as f:
        json.dump(scores_dic, f)


if __name__ == '__main__':
    main(
        input_path='resources/clustering_results.pkl',
        out_path='resources/eval.json'
    )
