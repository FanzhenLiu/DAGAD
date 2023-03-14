import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from torch_geometric.data import Data

def load_data(dataset, test_ratio = 0.8):

    adj, attrs, label, adj_label = load_anomaly_detection_dataset(dataset)
    adj = adj_label
    adj_t = adj.transpose()
    adj_t = torch.LongTensor(adj_t)
    adj = torch.LongTensor(adj)
    adj_sym = torch.add(adj, adj_t)

    edge_exist = (adj_sym >1).nonzero(as_tuple=True)
    adj_sym[edge_exist] = 1

    attrs = torch.Tensor(attrs)
    adj_label = sp.coo_matrix(adj_sym)
    indices = np.vstack((adj_label.row, adj_label.col))
    adj_label = torch.LongTensor(indices)

    label = torch.LongTensor(label)

    data = Data(x=attrs, edge_index=adj_label, y=label)

    val_size = 0.0
    test_size = test_ratio

    data.train_mask, data.val_mask, data.test_mask, data.train_anm, data.train_norm = split_dataset(data, val_size, test_size)
    return data

def load_anomaly_detection_dataset(dataset, datadir='./data'):
    
    data_mat = sio.loadmat(f'{datadir}/{dataset}.mat')
    adj = data_mat['Network']
    feat = data_mat['Attributes']
    truth = data_mat['Label']
    truth = truth.flatten()

    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_norm = adj_norm.toarray()
    adj = adj + sp.eye(adj.shape[0])
    adj = adj.toarray()
    feat = feat.toarray()
    return adj_norm, feat, truth, adj

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

class GeneralizedCELoss1(nn.Module):

    def __init__(self, q=0.7):
        super(GeneralizedCELoss1, self).__init__()
        self.q = q
             
    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach()**self.q)*self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')

        loss = torch.mean(F.cross_entropy(logits, targets, reduction='none') * loss_weight)
        return loss

def split_dataset(data, val_size, test_size):
    train_size = 1 - val_size - test_size
    train_mask = torch.zeros(data.y.shape).bool()
    val_mask = torch.zeros(data.y.shape).bool()
    test_mask = torch.zeros(data.y.shape).bool()
    train_mask_anm = torch.zeros(data.y.shape).bool()
    train_mask_norm = torch.zeros(data.y.shape).bool()
    anm_list = (data.y).nonzero(as_tuple=True)[0]
    norm_list = (data.y == 0).nonzero(as_tuple=True)[0]

    anm_id_list = torch.Tensor.tolist(anm_list)
    norm_id_list = torch.Tensor.tolist(norm_list)

    num_anm = len(anm_id_list)
    num_norm = len(norm_id_list)

    train_anm_id = random.sample(anm_id_list, int(num_anm * train_size))
    train_norm_id = random.sample(norm_id_list, int(num_norm * train_size))
    anm_id_list = list(set(anm_id_list) - set(train_anm_id))
    norm_id_list = list(set(norm_id_list) - set(train_norm_id))
    val_anm_id = random.sample(anm_id_list, int(num_anm * val_size))
    val_norm_id = random.sample(norm_id_list, int(num_norm * val_size))
    test_anm_id = list(set(anm_id_list) - set(val_anm_id))
    test_norm_id = list(set(norm_id_list) - set(val_norm_id))

    train_mask[train_anm_id] = True
    train_mask[train_norm_id] = True
    val_mask[val_anm_id] = True
    val_mask[val_norm_id] = True
    test_mask[test_anm_id] = True
    test_mask[test_norm_id] = True
    train_mask_anm[train_anm_id] = True
    train_mask_norm[train_norm_id] = True

    return train_mask, val_mask, test_mask, train_mask_anm, train_mask_norm