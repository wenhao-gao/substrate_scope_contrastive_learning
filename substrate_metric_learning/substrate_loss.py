import torch
import inspect
import time
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.utils.module_with_records_and_reducer import ModuleWithRecordsReducerAndDistance
from pytorch_metric_learning.losses.mixins import EmbeddingRegularizerMixin
from pytorch_metric_learning.distances.base_distance import BaseDistance
from ipdb import set_trace as st
from numba import njit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from miner_utils import convert_to_triplets_shn

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

@njit
def check_if_same_smiles(anchor_idx, positive_idx, negative_idx, smiles_list):
    mask_list = []
    for idx, (i, j, k) in enumerate(zip(anchor_idx, positive_idx, negative_idx)):
        if smiles_list[i] == smiles_list[k] or smiles_list[j] == smiles_list[k]:
            mask_list.append(False)
        else:
            mask_list.append(True)
    return mask_list

class LpDistance_unnormalized(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert not self.is_inverted

    def compute_mat(self, query_emb, ref_emb):
        dtype, device = query_emb.dtype, query_emb.device
        if ref_emb is None:
            ref_emb = query_emb
        if dtype == torch.float16:  # cdist doesn't work for float16
            rows, cols = lmu.meshgrid_from_sizes(query_emb, ref_emb, dim=0)
            output = torch.zeros(rows.size(), dtype=dtype, device=device)
            rows, cols = rows.flatten(), cols.flatten()
            distances = self.pairwise_distance(query_emb[rows], ref_emb[cols])
            output[rows, cols] = distances
            return output
        else:
            return torch.cdist(query_emb, ref_emb, p=self.p)

    def pairwise_distance(self, query_emb, ref_emb):
        return torch.nn.functional.pairwise_distance(query_emb, ref_emb, p=self.p)
    
    def forward(self, query_emb, ref_emb=None):
        self.reset_stats()
        self.check_shapes(query_emb, ref_emb)
        # query_emb_normalized = self.maybe_normalize(query_emb)
        if ref_emb is None:
            ref_emb = query_emb
            # ref_emb_normalized = query_emb_normalized
        else:
            ref_emb_normalized = self.maybe_normalize(ref_emb)
        self.set_default_stats(
            query_emb, ref_emb, query_emb, ref_emb
        )
        mat = self.compute_mat(query_emb, ref_emb)
        if self.power != 1:
            mat = mat**self.power
        assert mat.size() == torch.Size((query_emb.size(0), ref_emb.size(0)))
        return mat

class SubstrateTripletLoss(EmbeddingRegularizerMixin, ModuleWithRecordsReducerAndDistance):
    
    def __init__(
        self,
        device=None,
        margin=1,
        swap=False,
        distance=None,
        smooth_loss=False,
        triplets_per_anchor="all",
        same_halogen_negative=False,
        alpha=1,
        beta=1,
        gamma=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.margin = margin
        self.swap = swap
        self.smooth_loss = smooth_loss
        self.triplets_per_anchor = triplets_per_anchor
        self.same_halogen_negative = same_halogen_negative
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.add_to_recordable_attributes(list_of_names=["margin"], is_stat=False)
        self.l1_dist = LpDistance_unnormalized(p=1)
        self.distance = self.get_default_distance() if distance is None else distance
        print(f"Using {self.distance} for SubstrateTripletLoss")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
    def compute_loss(self, embeddings, values, labels, indices_tuple, ref_emb, ref_labels, smiles_list, atm_cls):
        """
        Calculate the loss using embeddings, labels, and indices_tuple
        Args:
            embeddings: tensor of size (batch_size, embedding_size), on cuda device
            values: tensor of size (batch_size), on cuda device
            labels: tensor of size (batch_size), on cuda device
            indices_tuple: tuple of size 3 for triplets (anchors, positives, negatives)
                            or size 4 for pairs (anchor1, postives, anchor2, negatives)
                            Can also be left as None
            ref_emb: tensor of size (batch_size, embedding_size), on cuda device
            ref_labels: tensor of size (batch_size), on cuda device
            smiles_list: list of SMILES strings
            atm_cls: tensor of size (batch_size), on cuda device, dtype=torch.int64
        Returns: the losses
        """
        c_f.labels_or_indices_tuple_required(labels, indices_tuple)
        if not self.same_halogen_negative:
            indices_tuple = lmu.convert_to_triplets(
                indices_tuple, labels, ref_labels, t_per_anchor=self.triplets_per_anchor
            )
        else:
            indices_tuple = convert_to_triplets_shn(
                indices_tuple, labels, ref_labels, t_per_anchor=self.triplets_per_anchor, atm_cls=atm_cls
            )

        anchor_idx, positive_idx, negative_idx = indices_tuple

        if len(anchor_idx) == 0:
            return self.zero_losses()
        
        # t0 = time.time()
        
        # mask_list_true = []
        # for idx, (i, j, k) in enumerate(zip(anchor_idx, positive_idx, negative_idx)):
        #     if smiles_list[i] == smiles_list[k] or smiles_list[j] == smiles_list[k]:
        #         mask_list_true.append(False)
        #     else:
        #         mask_list_true.append(True)

        mask_list = check_if_same_smiles(anchor_idx.cpu().numpy(), positive_idx.cpu().numpy(), negative_idx.cpu().numpy(), smiles_list)

        # print(f"If calculated correctly: {mask_list == mask_list_true}")
        # print(f"Time to compute mask: {time.time() - t0}")

        # remove the triplets that have the same smiles
        anchor_idx, positive_idx, negative_idx = anchor_idx[mask_list], positive_idx[mask_list], negative_idx[mask_list]
        indices_tuple = (anchor_idx, positive_idx, negative_idx)

        if len(anchor_idx) == 0:
            return self.zero_losses()

        mat = self.distance(embeddings, ref_emb)
        mat_yield = self.l1_dist(values.view(len(values), 1))
        if self.swap:
            mat_yield = 1 - mat_yield
        ap_dists = mat[anchor_idx, positive_idx]
        an_dists = mat[anchor_idx, negative_idx]
        yield_dists = mat_yield[anchor_idx, positive_idx]

        # if self.swap:
        #     pn_dists = mat[positive_idx, negative_idx]
        #     an_dists = self.distance.smallest_dist(an_dists, pn_dists)

        current_margins = self.distance.margin(ap_dists, self.gamma*yield_dists)
        loss_ap = current_margins.pow(2)
        violation = self.margin - an_dists
        if self.smooth_loss:
            loss_an = torch.nn.functional.softplus(violation)
        else:
            loss_an = torch.nn.functional.relu(violation)

        loss = self.alpha * loss_ap + self.beta * loss_an

        return {
            "loss": {"loss": {
                "losses": loss,
                "indices": indices_tuple,
                "reduction_type": "triplet",
            }},
            "loss_ap": {"loss": {
                "losses": loss_ap,
                "indices": indices_tuple,
                "reduction_type": "triplet",
            }},
            "loss_an": {"loss": {
                "losses": loss_an,
                "indices": indices_tuple,
                "reduction_type": "triplet",
            }}
        }

    def get_default_reducer(self):
        return AvgNonZeroReducer()
    
    def forward(
        self, embeddings, values, labels, smiles_list, atm_cls, indices_tuple=None, ref_emb=None, ref_labels=None
    ):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
            indices_tuple: tuple of size 3 for triplets (anchors, positives, negatives)
                            or size 4 for pairs (anchor1, postives, anchor2, negatives)
                            Can also be left as None
        Returns: the loss
        """
        self.reset_stats()
        c_f.check_shapes(embeddings, labels)
        if labels is not None:
            labels = c_f.to_device(labels, embeddings)
        ref_emb, ref_labels = c_f.set_ref_emb(embeddings, labels, ref_emb, ref_labels)
        loss_dict = self.compute_loss(
            embeddings, values, labels, indices_tuple, ref_emb, ref_labels, smiles_list, atm_cls
        )
        self.add_embedding_regularization_to_loss_dict(loss_dict['loss'], embeddings)
        self.add_embedding_regularization_to_loss_dict(loss_dict['loss_ap'], embeddings)
        self.add_embedding_regularization_to_loss_dict(loss_dict['loss_an'], embeddings)
        return self.reducer(loss_dict['loss'], embeddings, labels), \
                self.reducer(loss_dict['loss_ap'], embeddings, labels), \
                self.reducer(loss_dict['loss_an'], embeddings, labels)

    def zero_loss(self):
        return {"losses": 0, "indices": None, "reduction_type": "already_reduced"}

    def zero_losses(self):
        return {loss_name: self.zero_loss() for loss_name in self.sub_loss_names()}

    def _sub_loss_names(self):
        return ["loss"]

    def sub_loss_names(self):
        return self._sub_loss_names() + self.all_regularization_loss_names()

    def all_regularization_loss_names(self):
        reg_names = []
        for base_class in inspect.getmro(self.__class__):
            base_class_name = base_class.__name__
            mixin_keyword = "RegularizerMixin"
            if base_class_name.endswith(mixin_keyword):
                descriptor = base_class_name.replace(mixin_keyword, "").lower()
                if getattr(self, "{}_regularizer".format(descriptor)):
                    reg_names.extend(base_class.regularization_loss_names(self))
        return reg_names
