import json
import pickle

from pandas import DataFrame
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score


def eval_summary(df: DataFrame) -> dict:
    truth = df['video.url'].values
    clustering_labels = df['clustering_label'].values

    return {
        'NMI': normalized_mutual_info_score(truth, clustering_labels, average_method='arithmetic'),
        'AMI': adjusted_mutual_info_score(truth, clustering_labels, average_method='arithmetic')
    }


def main(input_path: str, out_path: str):
    with open(input_path, mode='rb') as f:
        dfs = pickle.load(f)

    df_results = DataFrame(eval_summary(df) for df in dfs)

    json_results = {
        'NMI': df_results['NMI'].mean(),
        'AMI': df_results['AMI'].mean(),
    }

    with open(out_path, mode='w') as f:
        json.dump(json_results, f)


if __name__ == '__main__':
    main(
        input_path='resources/clustering_results.pkl',
        out_path='resources/eval.json'
    )
