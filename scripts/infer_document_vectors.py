from typing import List, Callable

import pandas
from numpy import ndarray

from clustering_chat.document_embedding import get_doc2vec
from clustering_chat.tokenizer import load_sentencepiece

DocumentEmbedding = Callable[[List[str]], ndarray]


def main(embedding: DocumentEmbedding, dataset_path: str, tokenizer_path: str) -> pandas.DataFrame:
    df = pandas.read_csv(dataset_path)
    tokenizer = load_sentencepiece(tokenizer_path).EncodeAsPieces

    df['tokens'] = df['message'].fillna('').map(tokenizer)
    df['document_vector'] = df['tokens'].map(embedding)

    return df


if __name__ == '__main__':
    doc2vec = get_doc2vec(
        model_path='resources/doc2vec/model',
        epochs=20
    )

    df = main(
        embedding=doc2vec,
        dataset_path='resources/dev.csv',
        tokenizer_path='resources/tokenizer/sentence_piece.model'
    )

    df.to_pickle(path='resources/document_vectors.pkl')
