import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolHash

from xtb.interface import Calculator
from xtb.utils import get_method

from pyscf import gto
from pyscf.geomopt import geometric_solver
from pyscf import hessian
from pyscf.hessian.thermo import harmonic_analysis, dump_normal_mode, thermo, dump_thermo

from morfeus import SASA
from morfeus import XTB

BASIS_SET = '6311++g**'
au2eV = 27.211386245988 # https://physics.nist.gov/cgi-bin/cuu/Value?hrev
au2kcal_per_mol = 627.5095
au2J_per_mol = 6.02214076 * 4.3597447222071 * 10**(23-18)
au2kJ_per_mol = 6.02214076 * 4.3597447222071 * 10**(20-18)
G_co2 = -185.91354113292104
E_co2 = -185.899136167701

atm2idx = {
    'H': 1,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
    'P': 15,
    'S': 16,
    'Cl': 17,
    'Br': 35,
    'I': 53
}

def smiles2pyscf(smiles):
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
    AllChem.EmbedMultipleConfs(m,numConfs=1,randomSeed=0xf00d,useExpTorsionAnglePrefs=True,\
                            useBasicKnowledge=True)
    str_xyz = Chem.MolToXYZBlock(m,confId=0).split()[1:]
    str_pyscf = ''
    for i in range(len(str_xyz)):
        if (i+1)%4 == 0:
            str_pyscf += str_xyz[i] + '; '
        else:
            str_pyscf += str_xyz[i] + ' '
    return str_pyscf

def xyz2pyscf(xyz):
    str_xyz = xyz.split()[1:]
    str_pyscf = ''
    for i in range(len(str_xyz)):
        if (i+1)%4 == 0:
            str_pyscf += str_xyz[i] + '; '
        else:
            str_pyscf += str_xyz[i] + ' '
    return str_pyscf

def smiles2pyxtb(smiles):
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
    AllChem.EmbedMultipleConfs(m,numConfs=1,randomSeed=0xf00d,useExpTorsionAnglePrefs=True,\
                            useBasicKnowledge=True)
    str_xyz = Chem.MolToXYZBlock(m,confId=0).split()[1:]

    str_pyscf = ''
    for i in range(len(str_xyz)):
        if (i+1)%4 == 0:
            str_pyscf += str_xyz[i] + ';'
        else:
            str_pyscf += str_xyz[i] + ' '

    numbers_pyxtb = []
    positions_pyxtb = []
    for s in str_pyscf.split(';')[:-1]:
        atm_idx = atm2idx[s.split()[0]]
        atm_pos = [float(s.split()[1]), float(s.split()[2]), float(s.split()[3])]
        numbers_pyxtb.append(atm_idx)
        positions_pyxtb.append(atm_pos)
    return np.array(numbers_pyxtb), np.array(positions_pyxtb)

def xyz2pyxtb(str_xyz):
    str_xyz = str_xyz.split()[1:]
    str_pyscf = ''
    for i in range(len(str_xyz)):
        if (i+1)%4 == 0:
            str_pyscf += str_xyz[i] + ';'
        else:
            str_pyscf += str_xyz[i] + ' '

    numbers_pyxtb = []
    positions_pyxtb = []
    for s in str_pyscf.split(';')[:-1]:
        atm_idx = atm2idx[s.split()[0]]
        atm_pos = [float(s.split()[1]), float(s.split()[2]), float(s.split()[3])]
        numbers_pyxtb.append(atm_idx)
        positions_pyxtb.append(atm_pos)
    return np.array(numbers_pyxtb), np.array(positions_pyxtb)

def calc_pc_desc(mol):
    try:
        smiles, c_index, x_index = mol
        method='GFN2-xTB'
        charge=0
        numbers, positions = smiles2pyxtb(smiles)
        # calc = Calculator(get_method(method), numbers, positions, charge=charge)
        # calc.set_max_iterations(50)
        # res = calc.singlepoint()
        # e = res.get_energy()
        # charge_c = res.get_charges()[c_index]
        # bond_order_c_x = res.get_bond_orders()[c_index][x_index]

        sasa = SASA(numbers, positions)
        sasa_c = sasa.atom_areas[int(c_index+1)]

        xtb = XTB(numbers, positions)
        e_homo = xtb.get_homo()
        e_lumo = xtb.get_lumo()
        ip = xtb.get_ip()
        ea = xtb.get_ea()
        electrophilicity = xtb.get_global_descriptor("electrophilicity", corrected=True)
        nucleophilicity = xtb.get_global_descriptor("nucleophilicity", corrected=True)
        electrofugality = xtb.get_global_descriptor("electrofugality", corrected=True)
        nucleofugality = xtb.get_global_descriptor("nucleofugality", corrected=True)

        charge_c = xtb.get_charges()[int(c_index+1)]
        bond_order_c_x = xtb.get_bond_order(int(c_index+1), int(x_index+1))
        fukui_electro = xtb.get_fukui("electrophilicity")[int(c_index+1)]
        fukui_nucleo = xtb.get_fukui("nucleophilicity")[int(c_index+1)]
        fukui_radical = xtb.get_fukui("radical")[int(c_index+1)]
        fukui_dual = xtb.get_fukui("dual")[int(c_index+1)]
        fukui_electro_local = xtb.get_fukui("local_electrophilicity")[int(c_index+1)]
        fukui_nucleo_local = xtb.get_fukui("local_nucleophilicity")[int(c_index+1)]

        return sasa_c, e_homo, e_lumo, ip, ea, electrophilicity, nucleophilicity, \
            electrofugality, nucleofugality, charge_c, bond_order_c_x, fukui_electro, \
                fukui_nucleo, fukui_radical, fukui_dual, fukui_electro_local, fukui_nucleo_local, True
    except:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False
    
def get_xtb_energy(smiles, method='GFN2-xTB', charge=0):
    try:
        numbers, positions = smiles2pyxtb(smiles)
        calc = Calculator(get_method(method), numbers, positions, charge=charge)
        res = calc.singlepoint()
        return res.get_energy(), True
    except:
        return 0, False
    
def get_xtb_energy_list(smiles_list, method='GFN2-xTB', charge=0):
    es, masks = [], []
    for i in range(len(smiles_list)):
        e, mask = get_xtb_energy(smiles_list[i], method, charge)
        es.append(e)
        masks.append(mask)
    return es, masks

def calc_single_point(mol):
    if mol.spin == 0:
        mf = mol.RKS(xc='b3lyp')
    else:
        mf = mol.UKS(xc='b3lyp')
    energy = mf.kernel()
    return mf, energy

def calc_single_point_soluted(mol):
    if mol.spin == 0:
        mf = mol.RKS(xc='b3lyp').DDCOSMO()
    else:
        mf = mol.UKS(xc='b3lyp').DDCOSMO()
    # mf.with_solvent.eps = 78.3553  # water
    mf.with_solvent.eps = 46.826   # DMSO
    # mf.with_solvent.eps = 32.613   # methanol
    energy = mf.kernel()
    return mf, energy

def optimize_geometry(mol):
    if mol.spin == 0:
        mf = mol.RKS(xc='b3lyp')
    else:
        mf = mol.UKS(xc='b3lyp')
    conv_params = {
        'convergence_energy': 1e-6,  # Eh
        'convergence_grms': 3e-4,    # Eh/Bohr
        'convergence_gmax': 4.5e-4,  # Eh/Bohr
        'convergence_drms': 1.2e-3,  # Angstrom
        'convergence_dmax': 1.8e-3,  # Angstrom
    }
    return geometric_solver.optimize(mf, maxsteps=100, **conv_params)

def calc_thermo_HF(mol):
    mf, _ = calc_single_point_soluted(mol)
    if mol.spin == 0:
        hess = hessian.RHF(mf).kernel()
    else:
        hess = hessian.UHF(mf).kernel()
    results = harmonic_analysis(mol, hess)
    dump_normal_mode(mol, results)
    results = thermo(mf, results['freq_au'], 298.15, 101325)
    dump_thermo(mol, results)
    return results

def calc_thermo(mol):
    mf, _ = calc_single_point_soluted(mol)
    if mol.spin == 0:
        hess = hessian.rks.Hessian(mf).kernel()
    else:
        hess = hessian.uks.Hessian(mf).kernel()
    results = harmonic_analysis(mol, hess)
    dump_normal_mode(mol, results)
    results = thermo(mf, results['freq_au'], 298.15, 101325)
    dump_thermo(mol, results)
    return results

def get_single_point_energy(smiles):
    original_net_charge = int(rdMolHash.MolHash(Chem.MolFromSmiles(smiles), rdMolHash.HashFunction.NetCharge))

    mol = gto.M(atom=smiles2pyscf(smiles), basis='321g', charge=original_net_charge-1, spin=1)
    if mol.spin == 0:
        mf = mol.RKS(xc='b3lyp')
    else:
        mf = mol.UKS(xc='b3lyp')
    energy = mf.kernel()
    return energy


