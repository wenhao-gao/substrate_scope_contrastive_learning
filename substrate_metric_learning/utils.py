import sys
import logging
import numpy as np
import torch
try:
    from features import smiles_to_graph_substrate
except:
    from substrate_metric_learning.features import smiles_to_graph_substrate
from torch_geometric.loader import DataLoader

def setup_logger():
    logger = logging.getLogger("eval_contam")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)

    logger.addHandler(sh)
    return logger

logger = setup_logger()

class Objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

@torch.no_grad()
def test_accuracy(model, loader, device):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        pred, _ = model(data.x, data.edge_index, data.batch, data.atm_idx)
        # pred = pred.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)

@torch.no_grad()
def test_mse(model, loader, device):
    model.eval()
    total_err = 0
    for data in loader:
        data = data.to(device)
        pred, _ = model(data.x, data.edge_index, data.batch, data.atm_idx)
        # pred = pred.argmax(dim=-1)
        total_err += (pred - data.y).pow(2).sum().item()
    return total_err / len(loader.dataset)

@torch.no_grad()
def get_y_pred(model, loader, device):
    model.eval()
    preds_y = []
    for data in loader:
        data = data.to(device)
        pred, _ = model(data.x, data.edge_index, data.batch, data.atm_idx)
        preds_y.append(pred.cpu().numpy())
    return np.concatenate(preds_y, axis=0)

@torch.no_grad()
def get_embedding(model, loader, device):
    model.eval()
    if model.name == 'GIN':
        embeddings = []
        for data in loader:
            data = data.to(device)
            _, emb = model(data.x, data.edge_index, data.batch, data.atm_idx)
            embeddings.append(emb.cpu().numpy())
        return np.concatenate(embeddings, axis=0)
    elif model.name == 'GT':
        embeddings = []
        for data in loader:
            nodes, edges, adj_mat, mask, values, labels, atm_idx, smiles_list, atm_cls = data
            nodes, edges, adj_mat, mask, values, labels, atm_idx, atm_cls = \
                nodes.to(device), edges.to(device), adj_mat.to(device), mask.to(device), values.to(device), labels.to(device), atm_idx.to(device), atm_cls.to(device)
            # Write a loop to iterate through the batch and calculate the embeddings
            # minibatch_size = 12
            # embeddings = torch.zeros((nodes.size(0), config.hidden_channels)).to(device)
            # for i in range(0, nodes.size(0), minibatch_size):
            #     embeddings[i:i+minibatch_size] = net(nodes[i:i+minibatch_size], edges[i:i+minibatch_size], adj_mat[i:i+minibatch_size], mask[i:i+minibatch_size])
            embeddings.append(model(nodes=nodes, edges=edges, adj_mat=adj_mat, mask=mask, atm_idx=atm_idx)[1].cpu().numpy())
            return np.concatenate(embeddings, axis=0)

def check_grad(model):
    """Check if the gradient of a model is explosive"""
    grad_norm_total = 0
    for _, p in model.named_parameters():
        if p.grad is not None:
            grad_norm_total += p.grad.data.norm(2).item()
    return grad_norm_total

@torch.no_grad()
def get_embedding_from_smi(smi_list, c_index_list, model, device):
    assert len(smi_list) == len(c_index_list)
    assert model.pool_method == 'c'
    train_dataset = [smiles_to_graph_substrate(smiles=smi_list[i], s=0, y=0, atm_idx=[c_index_list[i]]) for i in range(len(smi_list))]
    loader = DataLoader(train_dataset, 128, shuffle=False)
    model.eval()
    embeddings = []
    for data in loader:
        data = data.to(device)
        _, emb = model(data.x, data.edge_index, data.batch, data.atm_idx)
        embeddings.append(emb.cpu().detach().numpy())
    return np.concatenate(embeddings, axis=0)