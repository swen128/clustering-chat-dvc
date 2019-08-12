import logging
from typing import List

from gensim.models import Doc2Vec

formatter = logging.Formatter(logging.BASIC_FORMAT)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

logger_gensim = logging.getLogger('gensim')
logger_gensim.setLevel(logging.INFO)
logger_gensim.addHandler(stream_handler)


def main(train_data_path: str, out_path: str, **doc2vec_train_options):
    doc2vec = Doc2Vec(corpus_file=train_data_path, **doc2vec_train_options)
    doc2vec.save(out_path)


if __name__ == '__main__':
    main(
        train_data_path='resources/doc2vec_corpus.txt',
        out_path='resources/doc2vec_model',
        workers=2,
        vector_size=100
    )
