import warnings
from typing import Dict
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder

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

embedder = BasicTextFieldEmbedder(token_embedders={"elmo_tokens": elmo_embedding})


def embed_word(text):
    #text = "This is some text."
    tokens = tokenizer.tokenize(text)
    text_field = TextField(tokens, {"elmo_tokens": token_indexer})
    text_field.index(vocab)
    token_tensor = text_field.as_tensor(text_field.get_padding_lengths())
    tensor_dict = text_field.batch_tensors([token_tensor])
    embedded_tokens = embedder(tensor_dict)
    return embedded_tokens

def embed_sentence(text):
    tokens = embed_word(text)
    sentence = torch.mean(tokens[0],dim = 0)
    return sentence


