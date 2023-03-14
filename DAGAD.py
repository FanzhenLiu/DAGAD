from utils import GeneralizedCELoss1
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from tqdm import tqdm

class DAGAD_GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, fcn_dim, num_classes, device):
        super(DAGAD_GCN, self).__init__()
        self.device =device
        self.fcn_dim= fcn_dim
        self.name = 'DAGAD-GCN'

        self.GNN_a_conv1 = GCNConv(input_dim, hidden_dim)
        self.GNN_a_conv2 = GCNConv(hidden_dim, hidden_dim)

        self.GNN_b_conv1 = GCNConv(input_dim, hidden_dim)
        self.GNN_b_conv2 = GCNConv(hidden_dim, hidden_dim)

        self.fc1_a = nn.Linear(hidden_dim*2, fcn_dim)
        self.fc2_a = nn.Linear(fcn_dim, num_classes)
        
        self.fc1_b = nn.Linear(hidden_dim*2, fcn_dim)
        self.fc2_b = nn.Linear(fcn_dim, num_classes)

    def forward(self, data, permute=True):
        h_a = self.GNN_a_conv2(self.GNN_a_conv1(data.x, data.edge_index).relu(), data.edge_index).relu()
        h_b = self.GNN_b_conv2(self.GNN_b_conv1(data.x, data.edge_index).relu(), data.edge_index).relu()

        h_back_a = torch.cat((h_a, h_b.detach()), dim=1)
        h_back_b = torch.cat((h_a.detach(), h_b), dim=1)
        
        h_aug_back_a, h_aug_back_b, data = self.permute_operation(data, h_b, h_a, permute)
        
        h_back_a = F.relu(h_back_a)
        h_back_b = F.relu(h_back_b)
        
        h_aug_back_a = F.relu(h_aug_back_a)
        h_aug_back_b = F.relu(h_aug_back_b)
        
        h_back_a = self.fc1_a(h_back_a)
        h_back_a = h_back_a.relu()
        h_back_b = self.fc1_b(h_back_b)
        h_back_b = h_back_b.relu()
        
        h_aug_back_a = self.fc1_a(h_aug_back_a)
        h_aug_back_a = h_aug_back_a.relu()
        h_aug_back_b = self.fc1_b(h_aug_back_b)
        h_aug_back_b = h_aug_back_b.relu()
        
        pred_org_back_a = F.log_softmax(self.fc2_a(h_back_a), dim=1)
        pred_org_back_b = F.log_softmax(self.fc2_b(h_back_b), dim=1)
        
        pred_aug_back_a = F.log_softmax(self.fc2_a(h_aug_back_a), dim=1)
        pred_aug_bcak_b = F.log_softmax(self.fc2_b(h_aug_back_b), dim=1)
        
        return pred_org_back_a, pred_org_back_b, pred_aug_back_a, pred_aug_bcak_b, data
    
    def permute_operation(self, data, h_b, h_a, permute=True):

        if permute:
            self.indices = np.random.permutation(h_b.shape[0])
        
        indices = self.indices
        h_b_swap = h_b[indices]
        label_swap = data.y[indices]
        data.aug_y = label_swap

        data.aug_train_mask = data.train_mask[indices]
        data.aug_val_mask = data.val_mask[indices]
        data.aug_test_mask = data.test_mask[indices]

        data.aug_train_anm = torch.clone(data.aug_train_mask).detach()
        data.aug_train_norm = torch.clone(data.aug_train_mask).detach()

        temp = data.aug_y == 1
        temp1 = data.aug_train_mask == True
        data.aug_train_anm = torch.logical_and(temp, temp1)

        temp = data.aug_y == 0
        temp1 = data.aug_train_mask == True
        data.aug_train_norm = torch.logical_and(temp, temp1)

        h_aug_back_a = torch.cat((h_a, h_b_swap.detach()), dim=1)
        h_aug_back_b = torch.cat((h_a.detach(), h_b_swap), dim=1)
        
        return h_aug_back_a, h_aug_back_b, data

class DAGAD_GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, fcn_dim, heads_num, num_classes, device):
        super(DAGAD_GAT, self).__init__()

        self.device =device
        self.hid = hidden_dim
        self.fcn_dim=fcn_dim
        self.heads = heads_num
        self.name = 'DAGAD-GAT'

        self.GNN_a_conv1 = GATConv(input_dim, hidden_dim, self.heads)
        self.GNN_a_conv2 = GATConv(hidden_dim * self.heads, hidden_dim, self.heads)

        self.GNN_b_conv1 = GATConv(input_dim, hidden_dim, self.heads)
        self.GNN_b_conv2 = GATConv(hidden_dim*self.heads, hidden_dim, self.heads)
        
        self.fc1_a = nn.Linear(hidden_dim*2*self.heads, fcn_dim)
        self.fc2_a = nn.Linear(fcn_dim, num_classes)
        
        self.fc1_b = nn.Linear(hidden_dim*2*self.heads, fcn_dim)
        self.fc2_b = nn.Linear(fcn_dim, num_classes)

    def forward(self, data, permute=True):
        h_a = self.GNN_a_conv2(self.GNN_a_conv1(data.x, data.edge_index).relu(), data.edge_index).relu()
        h_b = self.GNN_b_conv2(self.GNN_b_conv1(data.x, data.edge_index).relu(), data.edge_index).relu()

        h_back_a = torch.cat((h_a, h_b.detach()), dim=1)
        h_back_b = torch.cat((h_a.detach(), h_b), dim=1)

        h_aug_back_a, h_aug_back_b, data = self.permute_operation(data, h_b, h_a, permute)
        
        h_back_a = F.relu(h_back_a)
        h_back_b = F.relu(h_back_b)
        
        h_aug_back_a = F.relu(h_aug_back_a)
        h_aug_back_b = F.relu(h_aug_back_b)

        h_back_a = self.fc1_a(h_back_a)
        h_back_a = h_back_a.relu()
        h_back_b = self.fc1_b(h_back_b)
        h_back_b = h_back_b.relu()
        
        h_aug_back_a = self.fc1_a(h_aug_back_a)
        h_aug_back_a = h_aug_back_a.relu()
        h_aug_back_b = self.fc1_b(h_aug_back_b)
        h_aug_back_b = h_aug_back_b.relu()
        
        pred_org_back_a = F.log_softmax(self.fc2_a(h_back_a), dim=1)
        pred_org_back_b = F.log_softmax(self.fc2_b(h_back_b), dim=1)
        
        pred_aug_back_a = F.log_softmax(self.fc2_a(h_aug_back_a), dim=1)
        pred_aug_bcak_b = F.log_softmax(self.fc2_b(h_aug_back_b), dim=1)
        
        return pred_org_back_a, pred_org_back_b, pred_aug_back_a, pred_aug_bcak_b, data
    
    def permute_operation(self, data, h_b, h_a, permute=True):

        if permute:
            self.indices = np.random.permutation(h_b.shape[0])
        
        indices = self.indices
        h_b_swap = h_b[indices]
        label_swap = data.y[indices]
        data.aug_y = label_swap

        data.aug_train_mask = data.train_mask[indices]
        data.aug_val_mask = data.val_mask[indices]
        data.aug_test_mask = data.test_mask[indices]

        data.aug_train_anm = torch.clone(data.aug_train_mask).detach()
        data.aug_train_norm = torch.clone(data.aug_train_mask).detach()

        temp = data.aug_y == 1
        temp1 = data.aug_train_mask == True
        data.aug_train_anm = torch.logical_and(temp, temp1)

        temp = data.aug_y == 0
        temp1 = data.aug_train_mask == True
        data.aug_train_norm = torch.logical_and(temp, temp1)

        h_aug_back_a = torch.cat((h_a, h_b_swap.detach()), dim=1)
        h_aug_back_b = torch.cat((h_a.detach(), h_b_swap), dim=1)
        
        return h_aug_back_a, h_aug_back_b, data

def Validation(model_ad, data, epochs, lr, alpha, beta, q, wd):
    labels = data.y
    optimizer_ad = torch.optim.Adam(model_ad.parameters(), lr=lr, weight_decay=wd)
    criterion_gce = GeneralizedCELoss1(q=q)
    criterion = torch.nn.CrossEntropyLoss()

    test_prec_b = []
    test_rec_b = []
    test_f1_b = []
    auc_sc_b = []

    epochs = tqdm(range(epochs))
    model_ad.train()
    for epoch in epochs:

        optimizer_ad.zero_grad()
        if epoch %30 == 0:
            permute = True
        else:
            permute = False

        pred_org_a, pred_org_b, _, pred_aug_bcak_b, data = model_ad(data, permute)
        loss_ce_a = criterion(pred_org_a[data.train_mask], labels[data.train_mask])
        loss_ce_b = criterion(pred_org_b[data.train_mask], labels[data.train_mask])
        loss_ce_weight = loss_ce_b / (loss_ce_b + loss_ce_a + 1e-8)
        loss_ce_anm = criterion(pred_org_a[data.train_anm], labels[data.train_anm])
        loss_ce_norm = criterion(pred_org_a[data.train_norm], labels[data.train_norm])
        loss_ce = loss_ce_weight * (loss_ce_anm + loss_ce_norm)/2

        loss_gce = 0.5 * criterion_gce(pred_org_b[data.train_anm], labels[data.train_anm]) \
                + 0.5 * criterion_gce(pred_org_b[data.train_norm], labels[data.train_norm])
        
        loss_gce_aug = 0.5 * criterion_gce(pred_aug_bcak_b[data.aug_train_anm], data.aug_y[data.aug_train_anm]) \
                + 0.5 * criterion_gce(pred_aug_bcak_b[data.aug_train_norm], data.aug_y[data.aug_train_norm])
        
        loss = alpha * loss_ce + loss_gce + beta * loss_gce_aug
        loss.backward()
        optimizer_ad.step()
        epochs.set_description(f"Epoch: {epoch}")

        with torch.no_grad():
            _, pred_org_b, _, _, data = model_ad(data, permute=False)
            pred_b = pred_org_b.argmax(dim=1)
            test_precision_b = precision_score(labels[data.test_mask].cpu(), pred_b[data.test_mask].cpu(), average='macro')
            test_prec_b.append(test_precision_b)
            test_recall_b = recall_score(labels[data.test_mask].cpu(), pred_b[data.test_mask].cpu(), average='macro')
            test_rec_b.append(test_recall_b)
            test_fscore_b = f1_score(labels[data.test_mask].cpu(), pred_b[data.test_mask].cpu(), average='macro')
            test_f1_b.append(test_fscore_b)
            auc_score_b = roc_auc_score(labels[data.test_mask].cpu(),pred_org_b[data.test_mask][:,1].cpu())
            auc_sc_b.append(auc_score_b)
    return f"F1: {max(test_f1_b):.4f}, Precision: {max(test_prec_b):.4f}, Recall: {max(test_rec_b):.4f}, AUC: {max(auc_sc_b):.4f}"