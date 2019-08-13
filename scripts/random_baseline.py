import json
import pickle

from pandas import DataFrame
import random
from scripts.evaluate import aggregate_scores


def random_partition(df: DataFrame):
    n_clusters = df['is_representative'].sum()
    labels = list(range(n_clusters))
    representatives = df.sample(n_clusters)

    df['clustering_label'] = random.choices(labels, k=n_clusters)

    for label, i in zip(labels, representatives.index):
        df[i, 'clustering_label'] = label

    df['is_representative'] = df.index.isin(representatives.index)


def main(input_path: str, out_path: str):
    with open(input_path, mode='rb') as f:
        dfs = pickle.load(f)

    for df in dfs:
        random_partition(df)

    scores_dic = aggregate_scores(dfs)

    with open(out_path, mode='w') as f:
        json.dump(scores_dic, f)


if __name__ == '__main__':
    main(
        input_path='resources/clustering_results.pkl',
        out_path='resources/random_baseline.json'
    )
