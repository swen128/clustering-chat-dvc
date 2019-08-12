import logging
from typing import List

from gensim.models import Doc2Vec

from clustering_chat.doc2vec import Corpus
from clustering_chat.tokenizer import load_sentencepiece

formatter = logging.Formatter(logging.BASIC_FORMAT)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

logger_gensim = logging.getLogger('gensim')
logger_gensim.setLevel(logging.INFO)
logger_gensim.addHandler(stream_handler)


def main(tokenizer_path: str, train_data_path: str, out_path: str, **doc2vec_train_options):
    tokenizer = load_sentencepiece(tokenizer_path)

    def tokenize(sentence: str) -> List[str]:
        return tokenizer.EncodeAsPieces(sentence)[1:]

    corpus = Corpus(train_data_path, preprocess=tokenize)
    doc2vec = Doc2Vec(corpus, **doc2vec_train_options)

    doc2vec.save(out_path)


if __name__ == '__main__':
    main(
        tokenizer_path='resources/tokenizer/sentence_piece.model',
        train_data_path='resources/train.csv',
        out_path='resources/doc2vec_model',
        workers=20,
        vector_size=100
    )
