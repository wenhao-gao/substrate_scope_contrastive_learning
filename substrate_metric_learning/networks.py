import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MLP, GINConv, global_add_pool

from typing import Union

from layers import GraphormerEncoderLayer, CentralityEncoding, SpatialEncoding
from functional import shortest_path_distance, batched_shortest_path_distance

from typing import TypeVar

from einops import rearrange, repeat
from torch import einsum

from rotary_embedding import RotaryEmbedding, apply_rotary_emb
from loader import GTDataLoader
from ipdb import set_trace as st


def local_pick_pool(x, idx, batch):
    mols_start_from = torch.cat([batch.new_zeros(1), batch.bincount()]).cumsum(0)[:-1]
    idx = idx + mols_start_from
    return x[idx]


class Net(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, pool='global'):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.pool_method = pool
        
        if pool == 'global':
            self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                        norm=None, dropout=0.5)
            self.pool = global_add_pool
        elif pool == 'c':
            self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                        norm=None, dropout=0.5)
            self.pool = local_pick_pool
        elif pool == 'cx':
            self.mlp = MLP([hidden_channels*2, hidden_channels, out_channels],
                        norm=None, dropout=0.5)
            self.pool = local_pick_pool

    def forward(self, x, edge_index, batch, atm_idx=None):

        for conv in self.convs:
            x = conv(x, edge_index).relu()

        if atm_idx is not None:
            x = self.pool(x, atm_idx, batch)
        else:
            x = self.pool(x, batch)


        return torch.flatten(self.mlp(x)), x

class Graphormer(nn.Module):
    def __init__(self,
                 num_layers: int,
                 input_node_dim: int,
                 node_dim: int,
                 input_edge_dim: int,
                 edge_dim: int,
                 output_dim: int,
                 n_heads: int,
                 ff_dim: int,
                 max_in_degree: int,
                 max_out_degree: int,
                 max_path_distance: int,
                 out_channels: int,
                 pool='global'):
        """
        :param num_layers: number of Graphormer layers
        :param input_node_dim: input dimension of node features
        :param node_dim: hidden dimensions of node features
        :param input_edge_dim: input dimension of edge features
        :param edge_dim: hidden dimensions of edge features
        :param output_dim: number of output node features
        :param n_heads: number of attention heads
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max in degree of nodes
        :param max_path_distance: max pairwise distance between two nodes
        """
        super().__init__()

        self.num_layers = num_layers
        self.input_node_dim = input_node_dim
        self.node_dim = node_dim
        self.input_edge_dim = input_edge_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.max_path_distance = max_path_distance

        self.node_in_lin = nn.Linear(self.input_node_dim, self.node_dim)
        self.edge_in_lin = nn.Linear(self.input_edge_dim, self.edge_dim)

        self.centrality_encoding = CentralityEncoding(
            max_in_degree=self.max_in_degree,
            max_out_degree=self.max_out_degree,
            node_dim=self.node_dim
        )

        self.spatial_encoding = SpatialEncoding(
            max_path_distance=max_path_distance,
        )

        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(
                node_dim=self.node_dim,
                edge_dim=self.edge_dim,
                n_heads=self.n_heads,
                ff_dim=self.ff_dim,
                max_path_distance=self.max_path_distance) for _ in range(self.num_layers)
        ])

        self.node_out_lin = nn.Linear(self.node_dim, self.output_dim)

        self.pool_method = pool
        
        if pool == 'global':
            self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                        norm=None, dropout=0.5)
            self.pool = global_add_pool
        elif pool == 'c':
            self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                        norm=None, dropout=0.5)
            self.pool = local_pick_pool
        elif pool == 'cx':
            self.mlp = MLP([hidden_channels*2, hidden_channels, out_channels],
                        norm=None, dropout=0.5)
            self.pool = local_pick_pool

    def forward(self, data: Union[Data]) -> torch.Tensor:
        """
        :param data: input graph of batch of graphs
        :return: torch.Tensor, output node embeddings
        """
        x = data.x.float()
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.float()

        if type(data) == Data:
            ptr = None
            node_paths, edge_paths = shortest_path_distance(data)
        else:
            ptr = data.ptr
            node_paths, edge_paths = batched_shortest_path_distance(data)

        x = self.node_in_lin(x)
        edge_attr = self.edge_in_lin(edge_attr)

        x = self.centrality_encoding(x, edge_index)
        b = self.spatial_encoding(x, node_paths)

        for layer in self.layers:
            x = layer(x, edge_attr, b, edge_paths, ptr)

        x = self.node_out_lin(x)

        return torch.flatten(self.mlp(x)), x
    
# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

ModuleList = nn.ModuleList

# normalizations

class PreNorm(nn.Module):
    def __init__(
        self,
        dim,
        fn
    ):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args,**kwargs)

# gated residual

class Residual(nn.Module):
    def forward(self, x, res):
        return x + res

class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, 1, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x, res):
        gate_input = torch.cat((x, res, x - res), dim = -1)
        gate = self.proj(gate_input)
        return x * gate + res * (1 - gate)

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        pos_emb = None,
        dim_head = 64,
        heads = 8,
        edge_dim = None
    ):
        super().__init__()
        edge_dim = default(edge_dim, dim)

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.pos_emb = pos_emb

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_kv = nn.Linear(dim, inner_dim * 2)
        self.edges_to_kv = nn.Linear(edge_dim, inner_dim)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, nodes, edges, mask = None):
        h = self.heads

        q = self.to_q(nodes)
        k, v = self.to_kv(nodes).chunk(2, dim = -1)

        e_kv = self.edges_to_kv(edges)

        q, k, v, e_kv = map(lambda t: rearrange(t, 'b ... (h d) -> (b h) ... d', h = h), (q, k, v, e_kv))

        if exists(self.pos_emb):
            freqs = self.pos_emb(torch.arange(nodes.shape[1], device = nodes.device))
            freqs = rearrange(freqs, 'n d -> () n d')
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)

        ek, ev = e_kv, e_kv

        k, v = map(lambda t: rearrange(t, 'b j d -> b () j d '), (k, v))
        k = k + ek
        v = v + ev

        sim = einsum('b i d, b i j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b i -> b i ()') & rearrange(mask, 'b j -> b () j')
            mask = repeat(mask, 'b i j -> (b h) i j', h = h)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b i j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# optional feedforward

def FeedForward(dim, ff_mult = 4):
    return nn.Sequential(
        nn.Linear(dim, dim * ff_mult),
        nn.GELU(),
        nn.Linear(dim * ff_mult, dim)
    )

# classes

class GraphTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        edge_dim = None,
        heads = 8,
        gated_residual = True,
        with_feedforwards = False,
        norm_edges = False,
        rel_pos_emb = False,
        accept_adjacency_matrix = False,
        out_channels = 1,
        pool = 'global',
    ):
        super().__init__()
        self.layers = ModuleList([])
        edge_dim = default(edge_dim, dim)
        self.norm_edges = nn.LayerNorm(edge_dim) if norm_edges else nn.Identity()

        self.adj_emb = nn.Embedding(2, edge_dim) if accept_adjacency_matrix else None

        pos_emb = RotaryEmbedding(dim_head) if rel_pos_emb else None

        for _ in range(depth):
            self.layers.append(ModuleList([
                ModuleList([
                    PreNorm(dim, Attention(dim, pos_emb = pos_emb, edge_dim = edge_dim, dim_head = dim_head, heads = heads)),
                    GatedResidual(dim)
                ]),
                ModuleList([
                    PreNorm(dim, FeedForward(dim)),
                    GatedResidual(dim)
                ]) if with_feedforwards else None
            ]))

        self.pool_method = pool
        
        if pool == 'global':
            self.mlp = MLP([dim, hidden_channels, out_channels],
                        norm=None, dropout=0.5)
            self.pool = global_add_pool
        elif pool == 'c':
            self.mlp = MLP([dim, hidden_channels, out_channels],
                        norm=None, dropout=0.5)
            self.pool = local_pick_pool
        elif pool == 'cx':
            self.mlp = MLP([dim*2, hidden_channels, out_channels],
                        norm=None, dropout=0.5)
            self.pool = local_pick_pool

    def forward(
        self,
        nodes,
        edges = None,
        adj_mat = None,
        mask = None
    ):
        
        batch, seq, _ = nodes.shape

        if exists(edges):
            edges = self.norm_edges(edges)

        if exists(adj_mat):
            assert adj_mat.shape == (batch, seq, seq)
            assert exists(self.adj_emb), 'accept_adjacency_matrix must be set to True'
            adj_mat = self.adj_emb(adj_mat.long())

        all_edges = default(edges, 0) + default(adj_mat, 0)

        for attn_block, ff_block in self.layers:
            attn, attn_residual = attn_block
            nodes = attn_residual(attn(nodes.float(), all_edges, mask = mask), nodes)

            if exists(ff_block):
                ff, ff_residual = ff_block
                nodes = ff_residual(ff(nodes), nodes)

        if self.pool_method == 'global':
            mask = mask.unsqueeze(-1).expand_as(nodes)
            masked_nodes = nodes * mask
            graphs = masked_nodes.sum(dim=1, keepdim=True)
        elif self.pool_method == 'c':
            raise NotImplementedError
        else:
            raise NotImplementedError

        return torch.flatten(self.mlp(graphs)), nodes, edges
    
import rdkit
from rdkit import Chem, RDLogger
from typing import List, Tuple, Union
from torch_geometric.data import Data

class Featurization_parameters:
    """
    A class holding molecule featurization parameters as attributes.
    """
    def __init__(self) -> None:

        # Atom feature sizes
        self.MAX_ATOMIC_NUM = 100
        self.ATOM_FEATURES = {
            'atomic_num': list(range(self.MAX_ATOMIC_NUM)),
            'degree': [0, 1, 2, 3, 4, 5],
            'formal_charge': [-1, -2, 1, 2, 0],
            'chiral_tag': [0, 1, 2, 3],
            'num_Hs': [0, 1, 2, 3, 4],
            'hybridization': [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2
            ],
        }

        # Distance feature sizes
        self.PATH_DISTANCE_BINS = list(range(10))
        self.THREE_D_DISTANCE_MAX = 20
        self.THREE_D_DISTANCE_STEP = 1
        self.THREE_D_DISTANCE_BINS = list(range(0, self.THREE_D_DISTANCE_MAX + 1, self.THREE_D_DISTANCE_STEP))

        # len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
        self.ATOM_FDIM = sum(len(choices) + 1 for choices in self.ATOM_FEATURES.values()) + 2
        self.EXTRA_ATOM_FDIM = 0
        self.BOND_FDIM = 14
        self.EXTRA_BOND_FDIM = 0
        self.REACTION_MODE = None
        self.EXPLICIT_H = False
        self.REACTION = False
        self.ADDING_H = False
        self.KEEP_ATOM_MAP = False

# Create a global parameter object for reference throughout this module
PARAMS = Featurization_parameters()

def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.
    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    if atom is None:
        features = [0] * PARAMS.ATOM_FDIM
    else:
        features = onek_encoding_unk(atom.GetAtomicNum() - 1, PARAMS.ATOM_FEATURES['atomic_num']) + \
            onek_encoding_unk(atom.GetTotalDegree(), PARAMS.ATOM_FEATURES['degree']) + \
            onek_encoding_unk(atom.GetFormalCharge(), PARAMS.ATOM_FEATURES['formal_charge']) + \
            onek_encoding_unk(int(atom.GetChiralTag()), PARAMS.ATOM_FEATURES['chiral_tag']) + \
            onek_encoding_unk(int(atom.GetTotalNumHs()), PARAMS.ATOM_FEATURES['num_Hs']) + \
            onek_encoding_unk(int(atom.GetHybridization()), PARAMS.ATOM_FEATURES['hybridization']) + \
            [1 if atom.GetIsAromatic() else 0] + \
            [atom.GetMass() * 0.01]  # scaled to about the same range as other features
        if functional_groups is not None:
            features += functional_groups
    return features

def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.
    :param bond: An RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (PARAMS.BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond

def smiles_to_graph(smiles: str, y: float, atm_idx=None, with_hydrogen: bool = False, kekulize: bool = False) -> 'torch.geometric.data.Data':
    """
    Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smiles (str): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
        bonds to single/double bonds. (default: :obj:`False`)
    """
    RDLogger.DisableLog('rdApp.*')

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        mol = Chem.MolFromSmiles('')
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    xs = []
    for atom in mol.GetAtoms():
        xs.append(atom_features(atom))

    x = torch.tensor(xs, dtype=torch.long) #.view(-1, 9)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(bond_features(bond))
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long) # .view(-1, 3)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    if atm_idx is not None:
        return Data(x=x, edge_index=edge_index, edge_attr=torch.squeeze(edge_attr), y=torch.tensor([y], dtype=torch.float32), atm_idx=torch.tensor([atm_idx]))
    else:
        return Data(x=x, edge_index=edge_index, edge_attr=torch.squeeze(edge_attr), y=torch.tensor([y], dtype=torch.float32))
    
def process_graph(databatch):
    nodes = torch.unsqueeze(databatch.x, 0)
    num_nodes = nodes.shape[1]
    adj_mat = torch.zeros(1, num_nodes, num_nodes)
    edge_dim = databatch.edge_attr.shape[1]
    edges = torch.zeros(1, num_nodes, num_nodes, edge_dim)
    mask = torch.ones(1, num_nodes).bool()
    if num_nodes > 1:
        for i in range(databatch.edge_index.shape[1]):
            adj_mat[0, databatch.edge_index[0, i], databatch.edge_index[1, i]] = 1
            edges[0, databatch.edge_index[0, i], databatch.edge_index[1, i], :] = databatch.edge_attr[i]
    return nodes, edges, adj_mat, mask

if __name__ == '__main__':
    from tdc.single_pred import ADME
    from torch.nn import functional as F
    from tqdm import tqdm
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, default='GIN', choices=['GIN', 'Graphormer', 'GraphTransformer'])
    args = argparser.parse_args()

    data = ADME(name = 'Solubility_AqSolDB')
    split = data.get_split(method='random', seed=42, frac=[0.8, 0., 0.2])
    data_train, data_test = split['train'], split['test']

    from torch_geometric.loader import DataLoader

    batch_size = 14
    epochs = 50
    hidden_channels = 64
    num_layers = 4
    lr = 0.0005

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = [smiles_to_graph(smiles=data_train['Drug'][ind], y=data_train['Y'][ind]) for ind in range(len(data_train))]
    train_dataset = [g for g in train_dataset if g.edge_index.shape[1] > 0]
    test_dataset = [smiles_to_graph(smiles=data_test['Drug'][ind], y=data_test['Y'][ind]) for ind in range(len(data_test))]
    test_dataset = [g for g in test_dataset if g.edge_index.shape[1] > 0]

    if args.model == 'GIN':
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size)
        net = Net(in_channels=133, hidden_channels=hidden_channels, out_channels=1, num_layers=num_layers, pool='global').to(device)
    elif args.model == 'Graphormer':
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size)   
        net = Graphormer(
            num_layers=num_layers,
            input_node_dim=133,
            node_dim=hidden_channels,
            input_edge_dim=14,
            edge_dim=hidden_channels,
            output_dim=hidden_channels,
            n_heads=4,
            ff_dim=128,
            max_in_degree=5,
            max_out_degree=5,
            max_path_distance=5,
            out_channels=1,
            pool='global'
        ).to(device)  
    elif args.model == 'GraphTransformer':
        train_loader = GTDataLoader(train_dataset, batch_size, shuffle=True)
        test_loader = GTDataLoader(test_dataset, batch_size)
        net = GraphTransformer(
            dim=133,
            depth=num_layers,
            dim_head=hidden_channels,
            edge_dim=14,
            heads=4,
            gated_residual=True,
            with_feedforwards=True,
            norm_edges=True,
            rel_pos_emb=True,
            accept_adjacency_matrix=True,
            out_channels=1,
            pool='global'
        ).to(device)
    else:
        raise NotImplementedError

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    train_loss_list = []
    val_lost_list = []

    for epcoh in range(1, epochs + 1):
            # Train
            net.train()
            for data_minibatch in tqdm(train_loader):
                    if args.model == 'GIN':
                        data_minibatch = data_minibatch.to(device)
                        optimizer.zero_grad()
                        out = net(data_minibatch.x, data_minibatch.edge_index, data_minibatch.batch)[0]
                        loss = F.l1_loss(out, data_minibatch.y)
                        loss.backward()
                        optimizer.step()
                    elif args.model == 'Graphormer':
                        data_minibatch = data_minibatch.to(device)
                        optimizer.zero_grad()
                        out = net(data_minibatch)[0]
                        loss = F.l1_loss(out, data_minibatch.y)
                        loss.backward()
                        optimizer.step()
                    elif args.model == 'GraphTransformer':
                        nodes, edges, adj_mat, mask, y = data_minibatch
                        nodes, edges, adj_mat, mask, y = nodes.to(device), edges.to(device), adj_mat.to(device), mask.to(device), y.to(device)
                        optimizer.zero_grad()
                        out = net(nodes=nodes, edges=edges, adj_mat=adj_mat, mask=mask)[0]
                        loss = F.l1_loss(out, y)
                        loss.backward()
                        optimizer.step()
                    else:
                        raise NotImplementedError

            # Evaluate on training set
            net.eval()
            total_loss_train = 0
            for data_minibatch in train_loader:
                    if args.model == 'GIN':
                        data_minibatch = data_minibatch.to(device)
                        pred = net(data_minibatch.x, data_minibatch.edge_index, data_minibatch.batch)[0]
                        loss = F.l1_loss(pred, data_minibatch.y)
                        total_loss_train += float(loss) * data_minibatch.num_graphs
                    elif args.model == 'Graphormer':
                        data_minibatch = data_minibatch.to(device)
                        pred = net(data_minibatch)[0]
                        loss = F.l1_loss(pred, data_minibatch.y)
                        total_loss_train += float(loss) * data_minibatch.num_graphs
                    elif args.model == 'GraphTransformer':
                        nodes, edges, adj_mat, mask, y = data_minibatch
                        nodes, edges, adj_mat, mask, y = nodes.to(device), edges.to(device), adj_mat.to(device), mask.to(device), y.to(device)
                        pred = net(nodes=nodes, edges=edges, adj_mat=adj_mat, mask=mask)[0]
                        loss = F.l1_loss(pred, y)
                        total_loss_train += float(loss) * nodes.shape[0]
                    else:
                        raise NotImplementedError

            # Evaluate on test set
            net.eval()
            total_loss_test = 0
            for data_minibatch in test_loader:
                    if args.model == 'GIN':
                        data_minibatch = data_minibatch.to(device)
                        pred = net(data_minibatch.x, data_minibatch.edge_index, data_minibatch.batch)[0]
                        loss = F.l1_loss(pred, data_minibatch.y)
                        total_loss_test += float(loss) * data_minibatch.num_graphs
                    elif args.model == 'Graphormer':
                        data_minibatch = data_minibatch.to(device)
                        pred = net(data_minibatch)[0]
                        loss = F.l1_loss(pred, data_minibatch.y)
                        total_loss_test += float(loss) * data_minibatch.num_graphs
                    elif args.model == 'GraphTransformer':
                        nodes, edges, adj_mat, mask, y = data_minibatch
                        nodes, edges, adj_mat, mask, y = nodes.to(device), edges.to(device), adj_mat.to(device), mask.to(device), y.to(device)
                        pred = net(nodes=nodes, edges=edges, adj_mat=adj_mat, mask=mask)[0]
                        loss = F.l1_loss(pred, y)
                        total_loss_test += float(loss) * nodes.shape[0]
                    else:
                        raise NotImplementedError
            
            total_loss_train /= len(train_dataset)
            total_loss_test /= len(test_dataset)

            train_loss_list.append(total_loss_train)
            val_lost_list.append(total_loss_test)

            print(f'Epoch: {epcoh:03d}, Train MAE: {total_loss_train:.4f}, Test MAE: {total_loss_test:.4f}')
