import json
import pickle

import numpy
from pandas import DataFrame
from typing import Dict

from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
from toolz import assoc


def vocabulary(df: DataFrame) -> Dict[str, int]:
    tokens = numpy.concatenate(df['tokens'].to_numpy())
    vocab, count = numpy.unique(tokens, return_counts=True)

    return {word: n for word, n in zip(vocab, count) if word != '▁'}


def vocabulary_coverage(df: DataFrame) -> float:
    representatives = df.loc[df['is_representative']]
    all_vocab = vocabulary(df)
    covered_vocab = vocabulary(representatives)

    all_weight = sum(n for word, n in all_vocab.items())
    covered_weight = sum(n for word, n in all_vocab.items() if word in covered_vocab)

    return covered_weight / all_weight


def clustering_scores(df: DataFrame) -> Dict[str, float]:
    X = numpy.stack(df['document_vector'].values)
    labels = df['clustering_label']

    return dict(
        chs=calinski_harabasz_score(X, labels),
        db=davies_bouldin_score(X, labels),
        silhouette=silhouette_score(X, labels, random_state=0)
    )


def all_scores(df: DataFrame) -> Dict[str, float]:
    scores = clustering_scores(df)
    vocab_cover = vocabulary_coverage(df)
    return assoc(scores, 'vocabulary_coverage', vocab_cover)


def main(input_path: str, out_path: str):
    with open(input_path, mode='rb') as f:
        dfs = pickle.load(f)

    scores_dic = DataFrame(map(all_scores, dfs)).median().to_dict()

    with open(out_path, mode='w') as f:
        json.dump(scores_dic, f)


if __name__ == '__main__':
    main(
        input_path='resources/clustering_results.pkl',
        out_path='resources/eval.json'
    )
