import logging
from pathlib import Path

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

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Setting `sep_limit=0` enforces it to save all the attributes of the Doc2Vec model into separate files.
    # This ensures the static list of output files.
    doc2vec.save(out_path, sep_limit=0)


if __name__ == '__main__':
    main(
        train_data_path='resources/doc2vec_corpus.txt',
        out_path='resources/doc2vec/model',
        workers=2,
        epochs=20,
        vector_size=100
    )
