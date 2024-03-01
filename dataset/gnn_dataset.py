from rdkit import Chem
from rdkit import RDLogger   
RDLogger.DisableLog('rdApp.*')  
import torch
from torch_geometric.data import Data as TorchGeometricData
from torch_geometric.data import  Dataset
from torch_geometric.loader import  DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import StandardScaler
import csv
import numpy as np
import random

#データの読み込み
def read_smiles(data_name):
    smiles_list = []
    if data_name == 'QM9':
        data_path = 'data/QM9/qm9.csv'
        homo_list = []
        lumo_list = []
        with open(data_path) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                smiles_list.append(row['smiles'])
                homo_list.append(row['homo'])
                lumo_list.append(row['lumo'])
            labels = {'homo':homo_list, 'lumo': lumo_list}
    else:
        if data_name == 'Ames':
            data_path = 'data/Ames/Ames_data.csv'
            target = 'Activity'
        elif data_name == 'Ames_test':
            data_path = 'data/Ames/Ames_test_data.csv'
            target = 'Activity'
        elif data_name == 'Sol':
            data_path = 'data/solubility/solubility.csv'
            target = 'sol_class'
        elif data_name == 'Sol_test':
            data_path = 'data/solubility/solubility_test.csv'
            target = 'sol_class'
        elif data_name == 'Sol_rgr':
            data_path = 'data/solubility/solubility.csv'
            target = 'sol'
        elif data_name == 'Sol_rgr_test':
            data_path = 'data/solubility/solubility_test.csv'
            target = 'sol'
        labels = []
        with open(data_path) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                smiles = row['smiles']
                mol = Chem.MolFromSmiles(smiles)
                if mol == None: continue #'NoData'などmolオブジェクトに変換できない分子を含むrxnを取り除く
                smiles_list.append(smiles)
                if isinstance(row[target], str):
                    if i == 0:
                        print('target label was str!')
                    if data_name == 'Sol_rgr' or data_name == 'Sol_rgr_test':
                        labels.append(float(row[target])) 
                    else:
                        labels.append(int(row[target])) #AmesのActivityはstrで出力されているため。
                else:
                    if i == 0:
                        print('lable data type is {}'.format(type(row[target])))
                    labels.append(row[target])
    
        assert len(smiles_list) == len(labels) 
    return smiles_list, labels

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(
            "input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_atom_features(atom, en_list=None, explicit_H=False, use_sybyl=False, use_electronegativity=False,
                  use_gasteiger=False, degree_dim=17):
    if use_sybyl:
        atom_type = ordkit._sybyl_atom_type(atom)
        atom_list = ['C.ar', 'C.cat', 'C.1', 'C.2', 'C.3', 'N.ar', 'N.am', 'N.pl3', 'N.1', 'N.2', 'N.3', 'N.4', 'O.co2',
                     'O.2', 'O.3', 'S.O', 'S.o2', 'S.2', 'S.3', 'F', 'Si', 'P', 'P3', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                     'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                     'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
    else:
        atom_type = atom.GetSymbol()
        atom_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V',
                     'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
                     'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
    results = one_of_k_encoding_unk(atom_type, atom_list) + \
        one_of_k_encoding(atom.GetDegree(), list(range(degree_dim))) + \
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
        [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
        one_of_k_encoding_unk(atom.GetHybridization(),
                              [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                               Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                               Chem.rdchem.HybridizationType.SP3D2]) + \
        [atom.GetIsAromatic()]

    if use_electronegativity:
        results = results + [en_list[atom.GetAtomicNum() - 1]]
    if use_gasteiger:
        gasteiger = atom.GetDoubleProp('_GasteigerCharge')
        if np.isnan(gasteiger) or np.isinf(gasteiger):
            gasteiger = 0  # because the mean is 0
        results = results + [gasteiger]

    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    return np.array(results, dtype=np.float32)

def get_bond_features(bond):
    results=one_of_k_encoding_unk(bond.GetBondType(),[Chem.rdchem.BondType.SINGLE,
                         Chem.rdchem.BondType.DOUBLE,
                         Chem.rdchem.BondType.TRIPLE,
                         Chem.rdchem.BondType.AROMATIC])
    return np.array(results, dtype=np.float32)

def get_edge_features(mol):
    edge_list= []
    num_bond_features=0
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_features = get_bond_features(bond)
        num_bond_features=len(bond_features)
        edge_list += [([i, j],bond_features), ([j, i],bond_features)]
    return edge_list, num_bond_features

def get_rxn_edge_features(mol,before_atom_num):
    edge_list= []
    num_bond_features=0
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_features = get_bond_features(bond)
        num_bond_features=len(bond_features)
        edge_list += [([i+before_atom_num, j+before_atom_num],bond_features), 
                      ([j+before_atom_num, i+before_atom_num],bond_features)] #atom番号の調整
    return edge_list, num_bond_features

def mol2geodata(mol,y):
    smile = Chem.MolToSmiles(mol)
    atom_features =[get_atom_features(atom) for atom in mol.GetAtoms()]
    num_atom_features=len(atom_features[0])
    atom_features = np.array(atom_features)
    atom_features = torch.FloatTensor(atom_features).view(-1, len(atom_features[0]))

    edge_list,num_bond_features = get_edge_features(mol)
    edge_list=sorted(edge_list)
    
    edge_indices=[e for e,v in edge_list]
    edge_attributes=[v for e,v in edge_list]
    edge_indices = torch.tensor(edge_indices)
    edge_indices = edge_indices.t().to(torch.long).view(2, -1)
    edge_attributes = np.array(edge_attributes)
    edge_attributes = torch.FloatTensor(edge_attributes)
    #print(num_atom_features,num_bond_features)
    return TorchGeometricData(x=atom_features, edge_index=edge_indices, edge_attr=edge_attributes, num_atom_features=num_atom_features,num_bond_features=num_bond_features,smiles=smile, y=y) 

def mol2geodata_for_QM9(mol,h_y,l_y):
    smile = Chem.MolToSmiles(mol)
    atom_features =[get_atom_features(atom) for atom in mol.GetAtoms()]
    atom_features = np.array(atom_features)
    num_atom_features=len(atom_features[0])
    atom_features = torch.FloatTensor(atom_features).view(-1, len(atom_features[0]))

    edge_list,num_bond_features = get_edge_features(mol)
    edge_list=sorted(edge_list)
    
    edge_indices=[e for e,v in edge_list]
    edge_attributes=[v for e,v in edge_list]
    edge_indices = torch.tensor(edge_indices)
    edge_indices = edge_indices.t().to(torch.long).view(2, -1)
    edge_attributes = np.array(edge_attributes)
    edge_attributes = torch.FloatTensor(edge_attributes)
    #print(num_atom_features,num_bond_features)
    return TorchGeometricData(x=atom_features, edge_index=edge_indices, edge_attr=edge_attributes, num_atom_features=num_atom_features,num_bond_features=num_bond_features,smiles=smile, h_y=h_y, l_y=l_y)

class GNN_Dataset(Dataset):
    def __init__(self, data_name):
        super(GNN_Dataset, self).__init__()
        self.data_name = data_name
        self.smiles_list, self.labels = read_smiles(data_name)
    
    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_list[index])
        mol = Chem.AddHs(mol)
        if self.data_name == 'QM9':
            h_y = float(self.labels['homo'][index])
            l_y = float(self.labels['lumo'][index])
            h_y = torch.tensor(h_y, dtype=torch.float)
            l_y = torch.tensor(l_y, dtype=torch.float)
            data = mol2geodata_for_QM9(mol,h_y,l_y)
        else:
            y = torch.tensor(self.labels[index], dtype=torch.float)
            data = mol2geodata(mol,y)
        #data.validate(raise_on_error=True)
        return data
    
    def __len__(self):
        return len(self.smiles_list)
    
    def get(self, index):
        return self.__getitem__(index)
    def len(self):
        return self.__len__()
    
class GNN_DatasetWrapper(object):
    
    def __init__(self, 
        batch_size, num_workers, valid_size, test_size, data_name, splitting, random_seed=None):
        super(object, self).__init__()
        self.data_name = data_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.splitting = splitting
        self.valid_size = valid_size
        self.test_size = test_size
        self.random_seed = random_seed
        assert splitting in ['random', 'scaffold']
    
    def get_data_loaders(self):
        train_dataset = GNN_Dataset(data_name=self.data_name)
        print('dataset:', train_dataset)
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset, random_seed=self.random_seed)
        return train_loader, valid_loader, test_loader
    
    def get_train_validation_data_loaders(self, train_dataset, random_seed=None):
        if random_seed != None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        if self.splitting == 'random':
            # obtain training indices that will be used for validation
            num_train = len(train_dataset)
            indices = list(range(num_train))
            np.random.shuffle(indices)
            
            split = int(np.floor(self.valid_size * num_train))
            split2 = int(np.floor(self.test_size * num_train))
            valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]
        
        elif self.splitting == 'scaffold':
            train_idx, valid_idx, test_idx = scaffold_split(train_dataset, self.valid_size, self.test_size)
        
        print('number of data; train:{}, valid:{}, test:{}'.format(len(train_idx), len(valid_idx), len(test_idx)))
        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        print('sampling done')
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        valid_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        test_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=test_sampler,
            num_workers=self.num_workers, drop_last=False
        )

        return train_loader, valid_loader, test_loader