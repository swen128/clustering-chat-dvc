from typing import List, Tuple

import MeCab


def mecab_tag(sentence: str) -> List[Tuple[str, str]]:
    mecab = MeCab.Tagger("-Ochasen")
    result = mecab.parse(sentence)[:- len('\nEOS\n')]

    def iterator():
        for row in result.split('\n'):
            word, _, _, tag, _, _ = row.split('\t')
            yield (word, tag)

    return list(iterator())
