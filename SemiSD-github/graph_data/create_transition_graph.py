import math

from tqdm import tqdm
import torch
import numpy as np
import scipy.sparse as sp
from scipy import sparse
import pickle
import argparse

'''
create transition graph:
user sequence: i_1 -> i_2 -> i_3 -> i_1 -> i_4 --> i_5 (next-item)
graph: (incoming)
         user i1 i2 i3 i4 mask
    user 1    1  1  1  1   0 
    i1   1    0  0  0  0   0
    i2   1    1  0  0  0   0
    i3   1    0  1  0  0   0
    i4   1    1  0  0  0   0
    mask 0    0  0  0  0   0

graph: (outgoing)
         user i1 i2 i3 i4 mask
    user 1    1  1  1  1   0 
    i1   1    0  1  0  1   0
    i2   1    0  0  1  0   0
    i3   1    1  0  0  0   0
    i4   1    0  0  0  0   0
    mask 0    0  0  0  0   0
'''


class SequenceGraph(object):
    def __init__(self, config, data, dataset_name='demo', is_demo=False):
        self.device = config['device']
        self.data = data
        self.is_demo = is_demo
        self.dataset_name = dataset_name
        self.max_seq_len = config['MAX_ITEM_LIST_LENGTH']
        if is_demo:
            self.seq_data = self._get_data_demo()
            self.graph_data_file = dataset_name + '/seq_graph_data.pkl'
            # self.graph_info_file = dataset_name + '/seq_graph_info.pkl'
        else:
            self.seq_data = self._get_data()
            self.graph_data_file = 'graph_data/' + dataset_name + '/seq_graph_data.pkl'
            # self.graph_info_file = 'graph_data/' + dataset_name + '/seq_graph_info.pkl'

        try:
            print('load graphs')
            with open(self.graph_data_file, 'rb') as f:
                self.graphs = pickle.load(f)
        except:
            print('crate graphs')
            self.graphs = self._get_graphs()
            with open(self.graph_data_file, 'wb') as f:
                pickle.dump(self.graphs, f)

        # with open(self.graph_info_file, 'rb') as f:
        #     self.graphs_info = pickle.load(f)

    def _get_data(self):
        # tr = ([v1], v2)
        # va = ([v1, v2], v3)
        # te = ([v1, v2, v3], v4) = (seq, next-item)
        dataset = self.data.dataset

        data = dataset.inter_feat.interaction
        user_num, item_num = dataset.user_num, dataset.item_num  # item_num=1350 / max_item=1349
        user = data['user_id']  # tensor: seq_num ,  USER_ID = 1:user_num-1
        seq = data['item_id_list']  # tensor: seq_num x seq_len
        seq_len = data['item_length']  # tensor: seq_num
        return user_num-1, item_num, user, seq, seq_len

    def _get_graphs(self):
        print('get base graphs')
        user_num, item_num, users, seqs, seq_lens = self.seq_data
        seqs_list = seqs.tolist()
        seq_graph_list = [None] * user_num
        seq_graph_out_list = [None] * user_num
        seq_graph_in_list = [None] * user_num
        set_list = [list(set(seqs)) for seqs in seqs_list]
        set_len = [len(sets) for sets in set_list]
        max_set_len = max(set_len)

        # padding item set：
        item_set_list = []
        for i in range(len(set_list)):
            item_set_i = set_list[i]
            item_set_len_i = set_len[i]
            item_set_list.append(item_set_i + [0] * (max_set_len - item_set_len_i))

        seqs_item2id_list = [[dict(zip(list(set(seq)), range(1, len(set(seq)) + 1)))] for seq in seqs_list]

        for u_id in tqdm(range(user_num), total=user_num, ncols=100):  # user_num = 0:943  user_u = 1:943 给的user_nbu
            graph_size = self.max_seq_len + 1
            seq_graph = torch.zeros((graph_size, graph_size))  # (user+seq, user+seq)

            # 确定好user和u_id的关系
            user = users[u_id]
            seq = seqs_list[u_id]
            seq_len = seq_lens[u_id]

            seqs_item2id = seqs_item2id_list[u_id]

            seq_graph[0, :] += 1
            seq_graph[:, 0] += 1
            for pos in range(seq_len - 1):
                item_i = seq[pos]
                item_j = seq[pos + 1]
                iid = seqs_item2id[0][item_i]
                jid = seqs_item2id[0][item_j]
                seq_graph[iid, jid] += 1

            seq_graph_in, seq_graph_out = self._get_degree_matrix(seq_graph.numpy())
            seq_graph_out_list[u_id] = seq_graph_out
            seq_graph_in_list[u_id] = seq_graph_in
            seq_graph_list[u_id] = self._dense2sparse(seq_graph.numpy())

        seq_graph_out_tensor = torch.stack(seq_graph_out_list, 0)
        seq_graph_in_list_tensor = torch.stack(seq_graph_in_list, 0)
        seq_graph = torch.stack(seq_graph_list, 0)
        item_set = torch.tensor(item_set_list)
        item_set_len = torch.tensor(set_len)
        return seq_graph, seq_graph_out_tensor, seq_graph_in_list_tensor, item_set, item_set_len

    def _get_data_demo(self):
        user_num = 4
        item_num = 6
        user = torch.tensor([1, 2])
        seq = torch.tensor([[1, 2, 3, 1, 4],
                            [3, 1, 3, 0, 0]])
        seq_len = torch.tensor([5, 3])

        return user_num, item_num, user, seq, seq_len

    def _get_degree_matrix(self, adj_matrix):
        '''
        A = [ 1, 2, 2,
              0, 4, 6,
              1, 0, 0 ]
        in = [ 0.5, 0.0, 0.5,
               0.3,  0.7,  0.0,
               1.0,  0.0,  0.0 ]
        out = [ 0.2, 0.4, 0.4,
                0.0  0.4  0.6,
                1.0  0.0  0.0 ]
        NOTE: E = AE --> E \in R^{n X d}
        '''

        d = np.shape(adj_matrix)[0]
        row_temp = np.sum(adj_matrix, axis=0)
        row = self._bool_numpy(row_temp)
        row = np.reshape(row, (1, d))
        col_temp = np.sum(adj_matrix, axis=1)
        col = self._bool_numpy(col_temp)
        col = np.reshape(col, (d, 1))
        a_out = adj_matrix / col
        a_in = adj_matrix / row
        a_in = a_in.T

        a_in = self._dense2sparse(a_in)
        a_out = self._dense2sparse(a_out)

        # a_out = torch.from_numpy(a_out)
        # a_in = torch.from_numpy(a_in)
        return a_in, a_out

    def _dense2sparse(self, _matrix):
        a_ = sparse.coo_matrix(_matrix)
        v1 = a_.data
        indices = np.vstack((a_.row, a_.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(v1)
        shape = a_.shape
        if torch.cuda.is_available():
            sparse_matrix = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(torch.float32).to(self.device)
        else:
            sparse_matrix = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        return sparse_matrix

    def _bool_numpy(self, numpy_array):
        numpy_array_1 = numpy_array.copy()
        numpy_array_1[numpy_array_1 == 0.] = 1
        return numpy_array_1

