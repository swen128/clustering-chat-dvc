from pathlib import Path

import pandas
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
from scipy.stats import entropy

from clustering_chat.stop_words import remove_stop_words
from scripts.evaluate import entropy_reduction


def cw_ami(df: pandas.DataFrame, average_method: str) -> float:
    def pairs():
        for _, row in df.iterrows():
            for token in row['tokens']:
                yield token, row['clustering_label']

    tokens, labels = zip(*pairs())

    return adjusted_mutual_info_score(labels, tokens, average_method)


def cw_nmi(df: pandas.DataFrame, average_method: str) -> float:
    def pairs():
        for _, row in df.iterrows():
            for token in row['tokens']:
                yield token, row['clustering_label']

    tokens, labels = zip(*pairs())

    return normalized_mutual_info_score(labels, tokens, average_method)


def score(mi, average_method: str, is_stop_word_omitted: bool, csv_path) -> float:
    df = pandas.read_csv(csv_path, delimiter='\t')
    if is_stop_word_omitted:
        tokenize = lambda tokens: remove_stop_words(tokens.split(','))
    else:
        tokenize = lambda tokens: tokens.split(',')
    df['tokens'] = df['tokens'].map(tokenize)

    return mi(df, average_method)


def main_2(input_dir: str):
    data_sets = [
        '../resources/test_entropy_reduction/even_clusters.tsv',
        '../resources/test_entropy_reduction/uneven_clusters.tsv',
        '../resources/test_entropy_reduction/random.tsv'
    ]

    for mi in [cw_nmi, cw_ami]:
        for is_stop_word_omitted in [True, False]:
            print(f'{mi.__name__} {is_stop_word_omitted}')
            for average_method in ['min', 'geometric', 'arithmetic', 'max']:
                scores = [score(mi, average_method, is_stop_word_omitted, csv_path) for csv_path in data_sets]
                print('\t'.join(f'{x:.4f}' for x in scores))

            print('\n')


def main(input_dir: str):
    paths = Path(input_dir).iterdir()
    for csv_path in paths:
        df = pandas.read_csv(csv_path, delimiter='\t')
        df['tokens'] = df['tokens'].map(lambda tokens: tokens.split(','))
        score_for_ground_truth = cw_nmi(df, average_method='geometric')
        df['clustering_label'] = df['random_label']
        score_for_random_label = cw_nmi(df, average_method='geometric')

        print(csv_path)
        print('entropy reduction for ground truth: ', score_for_ground_truth)
        print('entropy reduction for random label: ', score_for_random_label)


if __name__ == '__main__':
    main_2(
        input_dir='../resources/test_entropy_reduction'
    )
