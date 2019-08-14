from typing import List

import numpy
import torch
from gensim.models import Doc2Vec
from pytorch_transformers import PreTrainedModel

from clustering_chat.tokenizer import load_sentencepiece


def get_doc2vec(model_path: str, **options):
    doc2vec = Doc2Vec.load(model_path)

    def infer_vector(tokens: List[str]) -> numpy.ndarray:
        return doc2vec.infer_vector(tokens, **options)

    return infer_vector


def get_transformer(model, transformer_path: str, sentence_piece_path: str, use_cuda: bool = False):
    transfomer_model: PreTrainedModel = model.from_pretrained(transformer_path)
    tokenizer = load_sentencepiece(sentence_piece_path)

    def infer_vector(tokens: List[str], pooling_layer: int = -2, pooling_strategy: str = "REDUCE_MEAN"):
        _tokens = ['[CLS]'] + tokens + ['[SEP]']
        ids = list(map(tokenizer.piece_to_id, _tokens))
        tokens_tensor = torch.tensor(ids).reshape(1, -1)

        if use_cuda:
            tokens_tensor = tokens_tensor.to('cuda')
            transfomer_model.to('cuda')

        transfomer_model.eval()
        with torch.no_grad():
            all_encoder_layers, _ = transfomer_model.model(tokens_tensor)

        embedding = all_encoder_layers[pooling_layer].cpu().numpy()[0]
        if pooling_strategy == "REDUCE_MEAN":
            return numpy.mean(embedding, axis=0)
        elif pooling_strategy == "REDUCE_MAX":
            return numpy.max(embedding, axis=0)
        elif pooling_strategy == "REDUCE_MEAN_MAX":
            return numpy.r_[numpy.max(embedding, axis=0), numpy.mean(embedding, axis=0)]
        elif pooling_strategy == "CLS_TOKEN":
            return embedding[0]
        else:
            raise ValueError("specify valid pooling_strategy: {REDUCE_MEAN, REDUCE_MAX, REDUCE_MEAN_MAX, CLS_TOKEN}")

    return infer_vector
