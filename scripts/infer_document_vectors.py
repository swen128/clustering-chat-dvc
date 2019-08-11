import pandas
from gensim.models import Doc2Vec
from clustering_chat.tokenizer import load_sentencepiece
from numpy import ndarray


def main(dataset_path: str, doc2vec_path: str, tokenizer_path: str, out_path: str, **doc2vec_infer_options):
    df = pandas.read_csv(dataset_path)
    doc2vec = Doc2Vec.load(doc2vec_path)
    tokenizer = load_sentencepiece(tokenizer_path).EncodeAsPieces

    def infer_vector(sentence: str) -> ndarray:
        return doc2vec.infer_vector(tokenizer(sentence), **doc2vec_infer_options)

    df['document_vector'] = df['message'].fillna('').map(infer_vector)
    df.to_csv(out_path)


if __name__ == '__main__':
    main(
        dataset_path='resources/dev.csv',
        doc2vec_path='resources/doc2vec_model',
        tokenizer_path='resources/tokenizer/sentence_piece.model',
        out_path='resources/document_vectors.csv',
        epochs=20
    )
