import json
import pickle
from typing import Dict, Iterable, List

from pandas import DataFrame
from sklearn.metrics import adjusted_mutual_info_score

from clustering_chat.mecab import mecab_tag
from clustering_chat.stop_words import is_stop_word


def cw_ami(df: DataFrame) -> Dict[str, float]:
    def pairs():
        for _, row in df.iterrows():
            for token in row['tokens']:
                yield token, row['clustering_label']

    tokens, labels = zip(*pairs())
    score = adjusted_mutual_info_score(labels, tokens, average_method='geometric')

    return dict(cw_ami=score)


def tokenize(sentence: str) -> List[str]:
    tags = mecab_tag(sentence)
    return [word for word, tag in tags if not
    (tag.startswith('助詞') or tag.startswith('助動詞') or tag.startswith('記号') or is_stop_word(word))]


def all_scores(df: DataFrame) -> Dict[str, float]:
    df['tokens'] = df['message'].map(tokenize)

    return cw_ami(df)


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
