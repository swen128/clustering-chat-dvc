from smart_open import open

from clustering_chat.tokenizer import load_sentencepiece


def main(tokenizer_path: str, train_data_path: str, out_path: str):
    tokenizer = load_sentencepiece(tokenizer_path)

    with open(out_path, mode='w', encoding='utf-8') as fp_out:
        with open(train_data_path, encoding='utf-8') as fp_in:
            for line in fp_in:
                tokens = tokenizer.EncodeAsPieces(line)[1:]
                out_line = ' '.join(tokens) + '\n'
                fp_out.write(out_line)


if __name__ == '__main__':
    main(
        tokenizer_path='resources/tokenizer/sentence_piece.model',
        train_data_path='resources/train.txt',
        out_path='resources/doc2vec_corpus.txt'
    )
