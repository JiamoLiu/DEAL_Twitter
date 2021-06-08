import json
from allennlp.data.tokenizers.token_class import Token
import torch
import itertools
from typing import ValuesView
import pandas as pd
import sys
import numpy
from scipy import sparse
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import scipy.sparse
from torch import tensor
import ELMO
import math
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from allennlp.data.token_indexers import (
    SingleIdTokenIndexer,
    TokenCharactersIndexer,
    ELMoTokenCharactersIndexer,
    PretrainedTransformerIndexer,
    PretrainedTransformerMismatchedIndexer,
)


from_link_file = "bidirectional_test.txt"
from_data_file = "test_node_data.json"
adj_file_name = "A_sp.npz"
train_adj_file_name = "ind_train_A.npz"
train_attr_file_name = "ind_train_X.npz"
attr_file_name = "X_sp.npz"
nodes_file_name = "nodes_keep.npy"
ones_zeroes_file = "pv0.10_pt0.00_pn0.10_arrays.npz"
attr_tensor_file = "attr_token_tensor.npz"
train_attr_tensor_file = "train_attr_token_tensor.npz"
attr_text_tensor_file = "attr_text_tensor.pt"
train_attr_text_tensor_file = "train_attr_text_tensor.pt"
number_of_samples = 100
train_range = 0.72
val_range = 0.08


def read_json_as_dict(filename, items_needed):
    with open(filename) as handle:
        jsondict = json.loads(handle.read())
        return dict(itertools.islice(jsondict.items(),items_needed))

def read_txt_pandas(filename,deli = " ",first_n = 100):
    f = pd.read_csv(filename, delimiter = deli,names= ["A","B"]).head(first_n)
    return f

def get_unique_node_ids(pandas_in):
    column_values = pandas_in[["A","B"]].values.ravel()
    unique_values =  pd.unique(column_values)
    #print(unique_values)
    return unique_values

def get_number_of_unique_nodes(pandas_in):
    column_values = pandas_in[["A","B"]].values.ravel()
    unique_values =  pd.unique(column_values)
    return unique_values.shape[0]

def index_id_as_dict(input_array):
    res = {}
    for i in range(len(input_array)):
        res[input_array[i]] = i

    #print(res)
    return res

def get_adj_as_sparse(node_index, pandas_in):    
    input_rows = pandas_in.shape[0]
    unique_nodes_number = get_number_of_unique_nodes(pandas_in)
    row_1s = numpy.zeros(input_rows*2)
    col_1s = numpy.zeros(input_rows*2)
    for index, row in pandas_in.iterrows():
        row_1s[index] = node_index[row["A"]]
        col_1s[index] = node_index[row["B"]]
        
        row_1s[index +input_rows] = node_index[row["B"]]
        col_1s[index + input_rows] = node_index[row["A"]]

    values = numpy.ones(2*input_rows)
    return coo_matrix((values, (row_1s, col_1s)), shape=(unique_nodes_number,unique_nodes_number)).tocsr()
    
def get_node_attr_as_array(sparse_adj, node_index, attr_dict):
    reverse_node_index = {value:key for key, value in node_index.items()}
    res = []
    for i in range(sparse_adj.shape[0]):
        node_attr = attr_dict[str(reverse_node_index[i])]
        res.append(node_attr)
        
    return res

def get_elmo_embedding_as_sparse(text_aray):
    res = []
    counter = 0
    for node_data in text_aray:
        res.append(ELMO.embed_sentence_with_mean(node_data).detach().numpy())
        counter = counter + 1
    return csr_matrix(res)

def save_node_index(node_dict):
    value = list(node_dict.values())
    with open(nodes_file_name, 'wb') as f:
        numpy.save(f, numpy.array(value))

def get_link_ones(pandas_in, node_index):
    res = []
    for index, row in pandas_in.iterrows():
        res.append([node_index[row["A"]],node_index[row["B"]]])
    return numpy.array(res)

def get_linked_nodes_and_links_from_sparse(sparse_adj):
    res = []
    connected = []
    disconnected = []
    for i in range(sparse_adj.shape[0]):
        for j in range(sparse_adj.shape[1]):
            if i == j:
                continue
            if (sparse_adj[i,j] == 1):
                if i not in res:
                    res.append(i)
                if j not in res:
                    res.append(j)

                if ([i,j] not in connected and [j,i] not in connected):
                    connected.append([i,j])
            if (sparse_adj[i,j] == 0):

                if ([i,j] not in disconnected and [j,i] not in disconnected):
                    disconnected.append([i,j])
    
    return numpy.array(sorted(res)), numpy.array(connected), numpy.array(disconnected)


def get_text_tensor(attr_arr):
    res = []
    for attr in attr_arr:
        res.append(ELMO.get_token_tensors([Token(attr)]))
    temp = torch.cat(res,dim =0)
    return torch.cat(res,dim = 0)


def generate_train_val_test_samples(sparse_adj,node_index,node_data):
    number_of_nodes = sparse_adj.shape[0]

    scipy.sparse.save_npz(adj_file_name,sparse_adj)
    attr_arr = get_node_attr_as_array(sparse_adj, node_index, node_data)
    sentence_embed_matrix = get_elmo_embedding_as_sparse(attr_arr)
    scipy.sparse.save_npz(attr_file_name,sentence_embed_matrix)

    attr_tensors = get_node_attr_tensors(attr_arr)
    numpy.savez(attr_tensor_file, attr_tensors)


    torch.save(get_text_tensor(attr_arr),attr_text_tensor_file)

    train_stop = math.floor(number_of_nodes * (train_range + val_range))
    val_start = math.floor(number_of_nodes * (train_range))
    #print(train_stop)
    #print(val_start)
    
    train_adj_matrix = sparse_adj[0:train_stop,0:train_stop]
    train_linked_nodes,train_ones,train_zeroes = get_linked_nodes_and_links_from_sparse(train_adj_matrix)
    
    
    val_adj_matrix = sparse_adj[val_start: train_stop, 0:train_stop]
    linked_nodes,val_ones,val_zeroes = get_linked_nodes_and_links_from_sparse(val_adj_matrix)
    #print(val_adj_matrix)

    test_adj_matrix = sparse_adj[train_stop : sparse_adj.shape[0],:]
    linked_nodes,test_ones,test_zeroes = get_linked_nodes_and_links_from_sparse(test_adj_matrix)

    attr_arr = get_node_attr_as_array(train_adj_matrix, node_index, node_data)
    train_sentence_embed_matrix = get_elmo_embedding_as_sparse(attr_arr)
    #print(train_sentence_embed_matrix.shape)
    numpy.savez(ones_zeroes_file, train_ones,val_ones,val_zeroes,test_ones,test_zeroes)
    scipy.sparse.save_npz(train_adj_file_name,train_adj_matrix)
    scipy.sparse.save_npz(train_attr_file_name,train_sentence_embed_matrix)
    torch.save(get_text_tensor(attr_arr),train_attr_text_tensor_file)
    print("train attr shape:", train_sentence_embed_matrix.shape)

    attr_tensors = get_node_attr_tensors(attr_arr)
    numpy.savez(train_attr_tensor_file, attr_tensors)


    #print(train_pandas)

def get_node_attr_tensors(node_att_arr):
    res = []
    for i in range(len(node_att_arr)):
        tokens = ELMO.get_token_arr(node_att_arr[i])
        token_tensors = ELMO.get_token_tensors(tokens).numpy()
        res.append(token_tensors[0,:,:])
    #print(numpy.array(res).shape)
    return numpy.array(res)






if __name__ == "__main__":
    dict = read_json_as_dict(from_data_file, sys.maxsize)    
    links = read_txt_pandas(from_link_file,first_n= number_of_samples)
    node_ids = get_unique_node_ids(links)
    node_index = index_id_as_dict(node_ids)
    adj_matrix = get_adj_as_sparse(node_index,links)
    save_node_index(node_index)
    generate_train_val_test_samples(adj_matrix, node_index, dict)
    #get_training_adj_attr(links, dict, adj_matrix)
