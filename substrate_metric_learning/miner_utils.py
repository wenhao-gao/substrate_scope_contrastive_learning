import numpy as np
import random
import torch
from pytorch_metric_learning.utils.loss_and_miner_utils import get_all_triplets_indices, get_random_triplet_indices
from pytorch_metric_learning.utils import common_functions as c_f
from ipdb import set_trace as st

def convert_to_triplets_shn(indices_tuple, labels, ref_labels=None, t_per_anchor=100, atm_cls=None):
    """
    This returns anchor-positive-negative triplets w/ negatives comprising same halogen atoms (shn)
    regardless of what the input indices_tuple is
    """
    if indices_tuple is None:
        if t_per_anchor == "all":
            return get_all_triplets_indices(labels, ref_labels)
        else:
            return get_random_triplet_indices_shn(
                labels, ref_labels, t_per_anchor=t_per_anchor, atm_cls=atm_cls
            )
    elif len(indices_tuple) == 3:
        return indices_tuple
    else:
        a1, p, a2, n = indices_tuple
        p_idx, n_idx = torch.where(a1.unsqueeze(1) == a2)
        return a1[p_idx], p[p_idx], n[n_idx]
    
def get_random_triplet_indices_shn(labels, ref_labels=None, t_per_anchor=None, weights=None, atm_cls=None):

    a_idx, p_idx, n_idx = [], [], []
    labels_device = labels.device
    ref_labels = labels if ref_labels is None else ref_labels
    unique_labels = torch.unique(labels)

    for label in unique_labels:

        # Get indices of positive samples for this label.
        p_inds = torch.where(ref_labels == label)[0]
        if ref_labels is labels:
            a_inds = p_inds
        else:
            a_inds = torch.where(labels == label)[0]
        n_inds = torch.where(ref_labels != label)[0] # gai yi xia zhe li
        n_a = len(a_inds)
        n_p = len(p_inds)
        min_required_p = 2 if ref_labels is labels else 1
        if (n_p < min_required_p) or (len(n_inds) < 1):
            continue

        k = n_p if t_per_anchor is None else t_per_anchor
        num_triplets = n_a * k
        p_inds_ = p_inds.expand((n_a, n_p))
        # Remove anchors from list of possible positive samples.
        if ref_labels is labels:
            p_inds_ = p_inds_[~torch.eye(n_a).bool()].view((n_a, n_a - 1))
        # Get indices of indices of k random positive samples for each anchor.
        p_ = torch.randint(0, p_inds_.shape[1], (num_triplets,))
        # Get indices of indices of corresponding anchors.
        a_ = torch.arange(n_a).view(-1, 1).repeat(1, k).view(num_triplets)
        p = p_inds_[a_, p_]
        a = a_inds[a_]

        # Get indices of negative samples for this label.
        if weights is not None:
            w = weights[:, n_inds][a]
            non_zero_rows = torch.where(torch.sum(w, dim=1) > 0)[0]
            if len(non_zero_rows) == 0:
                continue
            w = w[non_zero_rows]
            a = a[non_zero_rows]
            p = p[non_zero_rows]
            # Sample the negative indices according to the weights.
            if w.dtype == torch.float16:
                # special case needed due to pytorch cuda bug
                # https://github.com/pytorch/pytorch/issues/19900
                w = w.type(torch.float32)
            n_ = torch.multinomial(w, 1, replacement=True).flatten()
            n = n_inds[n_]
        else:
            # Sample the negative indices uniformly.
            atm_cls_dict = {
                0: list(set((atm_cls == 0).nonzero(as_tuple=True)[0].tolist()) & set(n_inds.tolist())),
                1: list(set((atm_cls == 1).nonzero(as_tuple=True)[0].tolist()) & set(n_inds.tolist())),
                2: list(set((atm_cls == 2).nonzero(as_tuple=True)[0].tolist()) & set(n_inds.tolist())),
                3: list(set((atm_cls == 3).nonzero(as_tuple=True)[0].tolist()) & set(n_inds.tolist()))
            }
            n_ = []
            no_negatives = []
            target_classes = atm_cls[a]
            for i, cls in enumerate(target_classes):
                # allow_negatives = list(set((atm_cls == cls).nonzero(as_tuple=True)[0].tolist()) & set(n_inds.tolist()))
                allow_negatives = atm_cls_dict[cls.item()]
                if len(allow_negatives) == 0:
                    no_negatives.append(i)
                    n_.append(0)
                else:
                    n_.append(random.choice(allow_negatives))
            n = torch.tensor(n_, dtype=torch.int64, device=a.device)
            # n_ = torch.randint(0, len(n_inds), (num_triplets,))
        mask = torch.ones(a.size(0), dtype=torch.bool, device=a.device)
        mask[no_negatives] = False
        a_idx.append(a[mask])
        p_idx.append(p[mask])
        n_idx.append(n[mask])

    if len(a_idx) > 0:
        a_idx = c_f.to_device(torch.cat(a_idx), device=labels_device, dtype=torch.long)
        p_idx = c_f.to_device(torch.cat(p_idx), device=labels_device, dtype=torch.long)
        n_idx = c_f.to_device(torch.cat(n_idx), device=labels_device, dtype=torch.long)
        assert len(a_idx) == len(p_idx) == len(n_idx)
        return a_idx, p_idx, n_idx
    else:
        empty = torch.tensor([], device=labels_device, dtype=torch.long)
        return empty.clone(), empty.clone(), empty.clone()
