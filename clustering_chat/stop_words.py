from pathlib import Path
from typing import Iterable, List

stop_words_path = Path(__file__).parent.parent / 'resources/stop_words.txt'
with open(stop_words_path) as f:
    stop_words = [line.strip() for line in f.readlines()]


def is_stop_word(word: str) -> bool:
    return word in stop_words


def remove_stop_words(words: Iterable[str]) -> List[str]:
    return list(filter(lambda word: not is_stop_word(word), words))
