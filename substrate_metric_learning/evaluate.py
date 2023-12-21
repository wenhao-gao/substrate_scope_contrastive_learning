import pandas as pd
import numpy as np
from rdkit import Chem
import torch
import os
from features import smiles_to_graph_substrate, smiles_to_graph
from utils import get_embedding

from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

from torch_geometric.loader import DataLoader
torch.autograd.set_detect_anomaly(True)
from ipdb import set_trace as st

test_doyle_smiles = [
    "C(Br)1=CC(C2OC(C=O)=CC=2)=CC=C1",
    "C(Br)1=CC(C2=CC=C(F)C=C2)=CC=C1",
    "C1=C(F)C(OC)=CC(C)=C1Br",
    "C(C#N)1=CC=C(OC)C=C1Br",
    "C1=CC(S(=O)(N2CCOCC2)=O)=CC=C1Br",
    "C1=C(S(=O)(C)=O)C=CC(C)=C1Br",
    "C1=CC(N2CCN(C(=O)OC(C)(C)C)CC2)=C(Cl)C=C1Br",
    "C1=C(C#N)C(F)=CC=C1Br",
    "C1=CC(C(=O)CCC(=O)OC)=CC=C1Br",
    "C1=CC(C2=CNC=N2)=CC=C1Br",
    "C(COC)1=CC=CC(OC)=C1Br",
    "C(Cl)1=CC=C(C(C)(C)C)C=C1Br",
    "C1=C(OC(F)(F)F)C=C(Cl)C=C1Br",
    "C1=CC=C(C(=O)OCC)C(C)=C1Br",
    "C(F)1=CC=C(C(C)(C)C)C=C1Br"
]

test_doyle_yields = [
    0.21, 
    0.47,
    0.28,
    0.95,
    0.82,
    0.95,
    0.0,
    0.79,
    0.78,
    0.28,
    0.0,
    0.55,
    0.65,
    0.7,
    0.72
]

patt_aryl_halide = Chem.MolFromSmarts('c-[Br]')
temp = [Chem.MolFromSmiles(smi).GetSubstructMatch(patt_aryl_halide) for smi in test_doyle_smiles]

data_doyle = pd.DataFrame({
    "smiles": test_doyle_smiles,
    "label": test_doyle_yields,
    "aroma_c_index": [_[0] for _ in temp],
    "halide_index": [_[1] for _ in temp]
})

test_doyle_dataset_cx = [smiles_to_graph(smiles=data_doyle['smiles'][ind], 
                                      y=data_doyle['label'][ind], 
                                      atm_idx=[data_doyle['aroma_c_index'][ind], data_doyle['halide_index'][ind]]) for ind in range(len(data_doyle))]

test_doyle_dataset_c = [smiles_to_graph(smiles=data_doyle['smiles'][ind], 
                                      y=data_doyle['label'][ind], 
                                      atm_idx=[data_doyle['aroma_c_index'][ind]]) for ind in range(len(data_doyle))]


@torch.no_grad()
def test_on_doyle(model, device):
    # try:
    model.eval()
    if model.pool_method == 'c':
        test_doyle_loader = DataLoader(test_doyle_dataset_c, 64, shuffle=False)
    elif model.pool_method == 'cx':
        test_doyle_loader = DataLoader(test_doyle_dataset_cx, 64, shuffle=False)
    else:
        raise ValueError('Invalid pool_method type')
    
    embeddings = get_embedding(model, test_doyle_loader, device)
    y = np.array(data_doyle['label'].to_list())
    mut_info = mutual_info_regression(embeddings, y)
    X = embeddings[:, np.argmax(mut_info)]
    X = X.reshape(-1, 1)
    mut_info = mutual_info_regression(X, y)[0]

    loo = LeaveOneOut()
    ys_pred = []
    ys_true = []

    for i, (train_index, test_index) in enumerate(loo.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        reg = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='l2')
        predictor = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=reg, transformer=StandardScaler()))
        predictor.fit(X_train, y_train)
        y_pred = predictor.predict(X_test)
        if y_pred[0] > 1:
            y_pred[0] = 1
        if y_pred[0] < 0:
            y_pred[0] = 0
        ys_pred.append(y_pred[0])
        ys_true.append(y_test[0])

    r2 = r2_score(ys_true, ys_pred)

    f_test, _ = f_regression(embeddings, y)
    X = embeddings[:, np.argmax(f_test)]
    X = X.reshape(-1, 1)
    f_test = f_regression(X, y)[0][0]

    loo = LeaveOneOut()
    ys_pred = []
    ys_true = []

    for i, (train_index, test_index) in enumerate(loo.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        reg = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='l2')
        predictor = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=reg, transformer=StandardScaler()))
        predictor.fit(X_train, y_train)
        y_pred = predictor.predict(X_test)
        if y_pred[0] > 1:
            y_pred[0] = 1
        if y_pred[0] < 0:
            y_pred[0] = 0
        ys_pred.append(y_pred[0])
        ys_true.append(y_test[0])

    if r2_score(ys_true, ys_pred) > r2:
        r2 = r2_score(ys_true, ys_pred)

    return r2, mut_info, f_test


data_hammett = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/Hammett_xtb.csv"))
test_hammett_dataset_cx = [smiles_to_graph(smiles=data_hammett['SMILES'][ind], 
                                      y=data_hammett['constant'][ind], 
                                      atm_idx=[data_hammett['c_index'][ind], data_hammett['x_index'][ind]]) for ind in range(len(data_hammett))]

test_hammett_dataset_c = [smiles_to_graph(smiles=data_hammett['SMILES'][ind], 
                                      y=data_hammett['constant'][ind], 
                                      atm_idx=[data_hammett['c_index'][ind]]) for ind in range(len(data_hammett))]


@torch.no_grad()
def test_on_hammett(model, device):
    model.eval()
    if model.pool_method == 'c':
        test_loader = DataLoader(test_hammett_dataset_c, 64, shuffle=False)
    elif model.pool_method == 'cx':
        test_loader = DataLoader(test_hammett_dataset_cx, 64, shuffle=False)
    else:
        raise ValueError('Invalid pool_method type')
    
    embeddings = get_embedding(model, test_loader, device)
    y = np.array(data_hammett['constant'].to_list())
    mut_info = mutual_info_regression(embeddings, y)
    X = embeddings[:, np.argmax(mut_info)]
    X = X.reshape(-1, 1)
    mut_info = mutual_info_regression(X, y)[0]

    loo = LeaveOneOut()
    ys_pred = []
    ys_true = []

    for i, (train_index, test_index) in enumerate(loo.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        reg = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='l2')
        predictor = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=reg, transformer=StandardScaler()))
        predictor.fit(X_train, y_train)
        y_pred = predictor.predict(X_test)
        ys_pred.append(y_pred[0])
        ys_true.append(y_test[0])

    r2 = r2_score(ys_true, ys_pred)

    f_test, _ = f_regression(embeddings, y)
    X = embeddings[:, np.argmax(f_test)]
    X = X.reshape(-1, 1)
    f_test = f_regression(X, y)[0][0]

    loo = LeaveOneOut()
    ys_pred = []
    ys_true = []

    for i, (train_index, test_index) in enumerate(loo.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        reg = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='l2')
        predictor = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=reg, transformer=StandardScaler()))
        predictor.fit(X_train, y_train)
        y_pred = predictor.predict(X_test)
        ys_pred.append(y_pred[0])
        ys_true.append(y_test[0])

    if r2_score(ys_true, ys_pred) > r2:
        r2 = r2_score(ys_true, ys_pred)

    return r2, mut_info, f_test

data_autoqchem = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/Hammett_autoqchem.csv"))
test_autoqchem_dataset_cx = [smiles_to_graph(smiles=data_autoqchem['SMILES'][ind], 
                                      y=data_autoqchem['constant'][ind], 
                                      atm_idx=[data_autoqchem['c_index'][ind], data_autoqchem['x_index'][ind]]) for ind in range(len(data_autoqchem))]

test_autoqchem_dataset_c = [smiles_to_graph(smiles=data_autoqchem['SMILES'][ind], 
                                      y=data_autoqchem['constant'][ind], 
                                      atm_idx=[data_autoqchem['c_index'][ind]]) for ind in range(len(data_autoqchem))]

@torch.no_grad()
def test_on_autoqchem_charge(model, device):
    model.eval()
    if model.pool_method == 'c':
        test_loader = DataLoader(test_autoqchem_dataset_c, 64, shuffle=False)
    elif model.pool_method == 'cx':
        test_loader = DataLoader(test_autoqchem_dataset_cx, 64, shuffle=False)
    else:
        raise ValueError('Invalid pool_method type')
    
    embeddings = get_embedding(model, test_loader, device)

    y = np.array(data_autoqchem['local_Mulliken_charge'].to_list())

    mut_info = mutual_info_regression(embeddings, y)
    X = embeddings[:, np.argmax(mut_info)]
    X = X.reshape(-1, 1)
    mut_info = mutual_info_regression(X, y)[0]

    loo = LeaveOneOut()
    ys_pred = []
    ys_true = []

    for i, (train_index, test_index) in enumerate(loo.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        reg = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='l2')
        predictor = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=reg, transformer=StandardScaler()))
        predictor.fit(X_train, y_train)
        y_pred = predictor.predict(X_test)
        ys_pred.append(y_pred[0])
        ys_true.append(y_test[0])

    r2 = r2_score(ys_true, ys_pred)
    f_test, _ = f_regression(embeddings, y)
    X = embeddings[:, np.argmax(f_test)]
    X = X.reshape(-1, 1)
    f_test = f_regression(X, y)[0][0]

    loo = LeaveOneOut()
    ys_pred = []
    ys_true = []

    for i, (train_index, test_index) in enumerate(loo.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        reg = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='l2')
        predictor = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=reg, transformer=StandardScaler()))
        predictor.fit(X_train, y_train)
        y_pred = predictor.predict(X_test)
        ys_pred.append(y_pred[0])
        ys_true.append(y_test[0])

    if r2_score(ys_true, ys_pred) > r2:
        r2 = r2_score(ys_true, ys_pred)

    return r2, mut_info, f_test

@torch.no_grad()
def test_on_autoqchem_nmr(model, device):
    model.eval()
    if model.pool_method == 'c':
        test_loader = DataLoader(test_autoqchem_dataset_c, 64, shuffle=False)
    elif model.pool_method == 'cx':
        test_loader = DataLoader(test_autoqchem_dataset_cx, 64, shuffle=False)
    else:
        raise ValueError('Invalid pool_method type')
    
    embeddings = get_embedding(model, test_loader, device)

    y = np.array(data_autoqchem['local_NMR_shift'].to_list())
    mut_info = mutual_info_regression(embeddings, y)
    X = embeddings[:, np.argmax(mut_info)]
    X = X.reshape(-1, 1)
    mut_info = mutual_info_regression(X, y)[0]

    loo = LeaveOneOut()
    ys_pred = []
    ys_true = []

    for i, (train_index, test_index) in enumerate(loo.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        reg = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='l2')
        predictor = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=reg, transformer=StandardScaler()))
        predictor.fit(X_train, y_train)
        y_pred = predictor.predict(X_test)
        ys_pred.append(y_pred[0])
        ys_true.append(y_test[0])

    r2 = r2_score(ys_true, ys_pred)

    f_test, _ = f_regression(embeddings, y)
    X = embeddings[:, np.argmax(f_test)]
    X = X.reshape(-1, 1)
    f_test = f_regression(X, y)[0][0]

    loo = LeaveOneOut()
    ys_pred = []
    ys_true = []

    for i, (train_index, test_index) in enumerate(loo.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        reg = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='l2')
        predictor = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=reg, transformer=StandardScaler()))
        predictor.fit(X_train, y_train)
        y_pred = predictor.predict(X_test)
        ys_pred.append(y_pred[0])
        ys_true.append(y_test[0])

    if r2_score(ys_true, ys_pred) > r2:
        r2 = r2_score(ys_true, ys_pred)

    return r2, mut_info, f_test



def evaluate(model, device):
    r2_doyle, mut_info_doyle, f_test_doyle = test_on_doyle(model, device)
    r2_hammett, mut_info_hammett, f_test_hammett = test_on_hammett(model, device)
    r2_charge, mut_info_charge, f_test_charge = test_on_autoqchem_charge(model, device)
    r2_nmr, mut_info_nmr, f_test_nmr = test_on_autoqchem_nmr(model, device)
    return r2_doyle, mut_info_doyle, r2_hammett, mut_info_hammett, r2_charge, mut_info_charge, r2_nmr, mut_info_nmr, f_test_doyle, f_test_hammett, f_test_charge, f_test_nmr