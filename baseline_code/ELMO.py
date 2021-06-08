import warnings
from typing import Dict
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from numpy.core.numeric import tensordot

import torch
from allennlp.data import Token, Vocabulary, TokenIndexer, Tokenizer
from allennlp.data.fields import ListField, TextField
from allennlp.data.token_indexers import (
    SingleIdTokenIndexer,
    TokenCharactersIndexer,
    ELMoTokenCharactersIndexer,
    PretrainedTransformerIndexer,
    PretrainedTransformerMismatchedIndexer,
)
from allennlp.data.tokenizers import (
    CharacterTokenizer,
    PretrainedTransformerTokenizer,
    SpacyTokenizer,
    WhitespaceTokenizer,
)
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import (
    Embedding,
    TokenCharactersEncoder,
    ElmoTokenEmbedder,
    PretrainedTransformerEmbedder,
    PretrainedTransformerMismatchedEmbedder,
)
from allennlp.nn import util as nn_util
import numpy
from torch import tensor

warnings.filterwarnings("ignore")


# It's easiest to get ELMo input by just running the data code.  See the
# exercise above for an explanation of this code.
tokenizer: Tokenizer = WhitespaceTokenizer()
token_indexer: TokenIndexer = ELMoTokenCharactersIndexer()
vocab = Vocabulary()

# We're using a tiny, toy version of ELMo to demonstrate this.
elmo_options_file = (
    "https://allennlp.s3.amazonaws.com/models/elmo/test_fixture/options.json"
)
elmo_weight_file = (
    "https://allennlp.s3.amazonaws.com/models/elmo/test_fixture/lm_weights.hdf5"
)
elmo_embedding = ElmoTokenEmbedder(
    options_file=elmo_options_file, weight_file=elmo_weight_file
)


def get_token_arr(text,sep = "//",sep2 = " "):
    temp = text.split(sep)
    res = [words for segments in temp for words in segments.split(sep2)]
    return [Token(t) for t in res]

def get_token_tensors(token_arr):
    text_field = TextField(token_arr, {"elmo_tokens": token_indexer})
    text_field.index(vocab)
    token_tensor = text_field.as_tensor(text_field.get_padding_lengths())
    tensor_dict = text_field.batch_tensors([token_tensor])
    return tensor_dict["elmo_tokens"]["elmo_tokens"]

def embed_words(text,sep = "//"):
    #text = "This is some text."
    tokens = get_token_arr(text,sep)
    tensor_tensor = get_token_tensors(tokens)
    embedded_tokens = elmo_embedding(tensor_tensor)
    return embedded_tokens

def embed_sentence_with_mean(text,sep = "//"):
    tokens = embed_words(text,sep)
    sentence = torch.mean(tokens[0],dim = 0)
    return sentence

def embed_tensors(tensors):
    return elmo_embedding(tensors)


if __name__ == "__main__":
    q = ["A//b","B//c","C//c"]
    res = []
    for item in q:
        res.append(get_token_tensors(get_token_arr(item)))

    tres = torch.cat(res, dim= 0)
    print(elmo_embedding(tres).shape)
    

