from enum import Enum
from pathlib import Path

from sentencepiece import SentencePieceTrainer, SentencePieceProcessor


class TokenizerType(Enum):
    SentencePiece = 'SentencePiece'


class SpmType(Enum):
    unigram = 'unigram'
    bpe = 'bpe'
    char = 'char'
    word = 'word'


def train_sentencepiece(input_path: str, model_name: str, vocab_size: int,
                        character_coverage: float, model_type: SpmType):
    """
    Train a SentencePiece model and generate files ``<model_name>.model`` and ``<model_name>.vocab``.

    Args:
        input_path (str): one-sentence-per-line raw corpus file. No need to run tokenizer, normalizer or preprocessor.
                     By default, SentencePiece normalizes the input with Unicode NFKC.
                     You can pass a comma-separated list of files.

        model_name (str): output model name prefix. ``<model_name>.model`` and ``<model_name>.vocab`` are generated.

        vocab_size (int): vocabulary size, e.g., 8000, 16000, or 32000

        character_coverage (float): amount of characters covered by the model, good defaults are:
                                    0.9995 for languages with rich character set like Japanse or Chinese and
                                    1.0 for other languages with small character set.

        model_type (SpmType): model type. Choose from ``unigram``, ``bpe``, ``char``, or ``word``.
                                             The input sentence must be pretokenized when using ``word`` type.
    """
    args = \
        f'--input={input_path} --model_prefix={model_name} --vocab_size={vocab_size}' \
        f'--character_coverage={character_coverage} --model_type={model_type.value}'

    Path(model_name).parent.mkdir(exist_ok=True)
    SentencePieceTrainer.Train(args)


def load_sentencepiece(model_path: str) -> SentencePieceProcessor:
    sp = SentencePieceProcessor()
    sp.Load(model_path)

    return sp
