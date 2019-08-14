from typing import List, Callable

import pandas
from numpy import ndarray
from pytorch_transformers import BertModel
from clustering_chat.document_embedding import get_transformer
from clustering_chat.tokenizer import load_sentencepiece

DocumentEmbedding = Callable[[List[str]], ndarray]


def main(embedding: DocumentEmbedding, dataset_path: str, tokenizer_path: str) -> pandas.DataFrame:
    df = pandas.read_csv(dataset_path)
    tokenizer = load_sentencepiece(tokenizer_path).EncodeAsPieces

    df['tokens'] = df['message'].fillna('').map(tokenizer)
    df['document_vector'] = df['tokens'].map(embedding)

    return df


if __name__ == '__main__':
    sentencepiece_path = 'resources/BERT_ja/wiki-ja.model'

    bert = get_transformer(
        BertModel,
        transformer_path='resources/BERT_ja',
        sentence_piece_path=sentencepiece_path,
        use_cuda=True
    )
    df = main(
        embedding=bert,
        dataset_path='resources/dev.csv',
        tokenizer_path=sentencepiece_path
    )

    df.to_pickle(path='resources/document_vectors.pkl')
