import json
import pickle

import numpy
from pandas import DataFrame
from typing import Dict


def vocabulary(df: DataFrame) -> Dict[str, int]:
    tokens = numpy.concatenate(df['tokens'].to_numpy())
    vocab, count = numpy.unique(tokens, return_counts=True)

    return {word: n for word, n in zip(vocab, count) if word != '▁'}


def vocabulary_coverage(df: DataFrame, random_state: int) -> float:
    n_representatives = df['is_representative'].sum()
    representatives = df.sample(n_representatives, random_state=random_state)
    all_vocab = vocabulary(df)
    covered_vocab = vocabulary(representatives)

    all_weight = sum(n for word, n in all_vocab.items())
    covered_weight = sum(n for word, n in all_vocab.items() if word in covered_vocab)

    return covered_weight / all_weight


def main(input_path: str, out_path: str, random_state: int):
    with open(input_path, mode='rb') as f:
        dfs = pickle.load(f)

    coverages = [vocabulary_coverage(df, random_state) for df in dfs]

    json_results = {
        'vocabulary_coverage_mean': numpy.average(coverages),
        'vocabulary_coverage_median': numpy.median(coverages),
        'vocabulary_coverage_max': max(coverages),
        'vocabulary_coverage_min': min(coverages),
    }

    with open(out_path, mode='w') as f:
        json.dump(json_results, f)


if __name__ == '__main__':
    main(
        input_path='resources/clustering_results.pkl',
        out_path='resources/random_baseline.json',
        random_state=0
    )
