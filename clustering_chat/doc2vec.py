from gensim.models.doc2vec import TaggedDocument
from smart_open import open


class Corpus:
    def __init__(self, filename, preprocess, encoding="utf-8"):
        self.filename = filename
        self.preprocess = preprocess
        self.encoding = encoding

    def __iter__(self):
        with open(self.filename, encoding=self.encoding) as f:
            for i, line in enumerate(f):
                doc = self.preprocess(line)
                yield TaggedDocument(doc, [i])
