from clustering_chat.tokenizer import TokenizerType, SpmType, train_sentencepiece


def main(tokenizer_type: TokenizerType, spm_type: SpmType,
         spm_vocab_size: int, spm_character_coverage: float,
         input_path: str, out_prefix: str):
    if tokenizer_type == TokenizerType.SentencePiece:
        train_sentencepiece(input_path, out_prefix, spm_vocab_size, spm_character_coverage, spm_type)
    else:
        raise NotImplementedError(tokenizer_type)


if __name__ == '__main__':
    main(
        TokenizerType.SentencePiece,
        SpmType.unigram,
        spm_vocab_size=870,
        spm_character_coverage=0.9995,
        input_path='resources/train.csv',
        out_prefix='resources/tokenizer/sentence_piece'
    )
