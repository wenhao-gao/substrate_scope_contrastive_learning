from typing import Any
import torch
import torch_geometric
from typing import List, Tuple, Union
from rdkit import Chem
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from numba import jit
from ipdb import set_trace as st

######################################################################################################################################################
########################### taken from https://github.com/chemprop/chemprop/blob/master/chemprop/features/featurization.py ###########################
######################################################################################################################################################

halogen2index={
    'F': 0,
    'Cl': 1,
    'Br': 2,
    'I': 3
}

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

######################################################################################################################################################

######################################################################################################################################################
###################### taken from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/smiles.html ######################
######################################################################################################################################################

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
    # return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)

######################################################################################################################################################

# def smiles_to_graph_substrate(smiles: str, y: float, substrate_id: int, with_hydrogen: bool = False, kekulize: bool = False) -> 'torch.geometric.data.Data':
#     """
#     Converts a SMILES string to a :class:`torch_geometric.data.Data`
#     instance.

#     Args:
#         smiles (str): The SMILES string.
#         with_hydrogen (bool, optional): If set to :obj:`True`, will store
#             hydrogens in the molecule graph. (default: :obj:`False`)
#         kekulize (bool, optional): If set to :obj:`True`, converts aromatic
#         bonds to single/double bonds. (default: :obj:`False`)
#     """
#     RDLogger.DisableLog('rdApp.*')

#     mol = Chem.MolFromSmiles(smiles)

#     if mol is None:
#         mol = Chem.MolFromSmiles('')
#     if with_hydrogen:
#         mol = Chem.AddHs(mol)
#     if kekulize:
#         Chem.Kekulize(mol)

#     xs = []
#     for atom in mol.GetAtoms():
#         xs.append(atom_features(atom))
    
#     x = torch.tensor(xs, dtype=torch.long) #.view(-1, 9)

#     edge_indices, edge_attrs = [], []
#     for bond in mol.GetBonds():
#         i = bond.GetBeginAtomIdx()
#         j = bond.GetEndAtomIdx()

#         e = []
#         e.append(bond_features(bond))
#         edge_indices += [[i, j], [j, i]]
#         edge_attrs += [e, e]

#     edge_index = torch.tensor(edge_indices)
#     edge_index = edge_index.t().to(torch.long).view(2, -1)
#     edge_attr = torch.tensor(edge_attrs, dtype=torch.long) # .view(-1, 3)

#     if edge_index.numel() > 0:  # Sort indices.
#         perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
#         edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]
    
#     edge_attr = torch.squeeze(edge_attr)

#     return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([y]), smiles=smiles, substrate_id=torch.tensor([substrate_id]))
#     # return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)

def smiles_to_graph_substrate(smiles: str, s: int, y: float, atm_idx=None, with_hydrogen: bool = False, kekulize: bool = False) -> 'torch.geometric.data.Data':
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

    [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    smiles_canonical = Chem.MolToSmiles(mol)

    atm_cls = halogen2index[mol.GetAtomWithIdx(int(atm_idx[1])).GetSymbol()]

    if atm_idx is not None:
        return Data(
            x=x, 
            edge_index=edge_index, 
            edge_attr=torch.squeeze(edge_attr), 
            scope_id=torch.tensor([s]), 
            y=torch.tensor([y], dtype=torch.float32), 
            atm_idx=torch.tensor([atm_idx[0]]),
            atm_cls=torch.tensor([atm_cls]),
            smiles=smiles_canonical
        )
    else:
        return Data(
            x=x, 
            edge_index=edge_index, 
            edge_attr=torch.squeeze(edge_attr), 
            scope_id=torch.tensor([s]), 
            y=torch.tensor([y], dtype=torch.float32)
        )

###### 0: F
###### 1: Cl
###### 2: Br
###### 3: I