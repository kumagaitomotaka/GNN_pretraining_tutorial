# ライブラリ確認
import numpy as np
import pandas as pd
import csv
import os
import torch
from torch import nn
from dataset.gnn_dataset import GNN_DatasetWrapper
from sklearn.metrics import r2_score
import pytorch_lightning as pl
import torch.multiprocessing as mp
from pytorch_lightning.callbacks import ModelCheckpoint
from models.PL_BasicGNN_models import PL_BasicGNNs
from models.PL_topK_model import PL_TopKmodel
from models.PL_set2set_model import PL_Set2Setmodel
from torch_geometric.nn.model_hub import PyGModelHubMixin
from callbacks.pl_callbacks import CSVLogger
from torch_geometric.utils import scatter #デバック

# Define your class with the mixin:
class PL_Basic_GNN(PL_BasicGNNs, PyGModelHubMixin):
    def __init__(self,model_name, dataset_name, model_kwargs):
        PL_BasicGNNs.__init__(self,**model_kwargs)
        PyGModelHubMixin.__init__(self, model_name,
            dataset_name, model_kwargs)
class PL_TopK_GNN(PL_TopKmodel, PyGModelHubMixin):
    def __init__(self,model_name, dataset_name, model_kwargs):
        PL_TopKmodel.__init__(self,**model_kwargs)
        PyGModelHubMixin.__init__(self, model_name,
            dataset_name, model_kwargs)
class PL_Set2Set_GNN(PL_Set2Setmodel, PyGModelHubMixin):
    def __init__(self,model_name, dataset_name, model_kwargs):
        PL_Set2Setmodel.__init__(self,**model_kwargs)
        PyGModelHubMixin.__init__(self, model_name,
            dataset_name, model_kwargs)
        
def main():
    batch_size = 512
    num_workers = 18                # dataloader number of workers
    valid_size = 0.1               # ratio of validation data
    test_size = 0.1                # ratio of test data
    splitting = 'random'          # data splitting (i.e., random/scaffold)
    data_name = 'QM9'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name= 'GraphSAGE'
    repo_id = "kumatomo/BasicGraphSAGE" # your repo id: kumatomo/TopK_GNN, kumatomo/set2set_GNN, kumatomo/BasicGCN, kumatomo/BasicGIN, kumatomo/BasicGraphSAGE
    task = 'regression'
    model_type = 'pretrain'
    finetune_dim = 0
    
    print('{} model start with {}'.format(model_type, model_name))
    
    dataset = GNN_DatasetWrapper(batch_size=batch_size, 
                                 num_workers=num_workers, 
                                 valid_size=valid_size, 
                                 test_size=test_size, 
                                 data_name=data_name, 
                                 splitting=splitting)
    train_loader, valid_loader, test_loader = dataset.get_data_loaders()
    

    #nomalizer
    labels_h = []
    labels_l = []
    for d in train_loader:
        labels_h.extend(d.h_y)
        labels_l.extend(d.l_y)
    mean_value_h = np.mean(labels_h)
    mean_value_l = np.mean(labels_l)
    std_value_h = np.std(labels_h)
    std_value_l = np.std(labels_l)
    print('homo_normalizing...mean: {}, std: {}, shape: {}'.format(mean_value_h,
                                                                  std_value_h, 
                                                                  len(labels_h)))
    print('lumo_normalizing...mean: {}, std: {}, shape: {}'.format(mean_value_l,
                                                                  std_value_l, 
                                                                  len(labels_l)))
    
    # 保存するCSVファイルのパスとフィールド名を指定
    data_dir = 'Hug_PL_data'
    os.makedirs(data_dir, exist_ok=True)
    log_dir = os.path.join(data_dir, 'QM9_{}'.format(model_name))
    os.makedirs(log_dir, exist_ok=True)
    log_file = 'logs.csv'
    log_file_path= os.path.join(log_dir, log_file)
    csv_logger = CSVLogger(file_path=log_file_path, fieldnames=['epoch', 'train_loss', 'val_loss'])
    
    
    #modelを保存するmodelcheckpointの作成
    state_save_dir ='pl_pretrain_ckpt/'
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',dirpath=state_save_dir, filename=model_name + '_pretrain_model-{epoch:02d}', save_top_k=1, mode='min')# 最良のモデル1つだけ保存
    #get num_feature
    check_iter = iter(train_loader)
    check_data = next(check_iter)
    num_atom_features = int(check_data[0].num_atom_features)
    num_bond_features = int(check_data[0].num_bond_features)
    
    if model_name in ['GCN', 'GIN', 'GraphSAGE']:
        pl_model = PL_Basic_GNN(model_name=model_name, dataset_name='QM9',model_kwargs=dict(model_name=model_name, task=task, model_type=model_type, in_channels=num_atom_features, finetune_dim=finetune_dim))
    elif model_name == 'TopK':
        pl_model = PL_TopK_GNN(model_name=model_name, dataset_name='QM9',model_kwargs=dict(task=task, model_type=model_type, num_atom_features=num_atom_features, finetune_dim=finetune_dim))
    elif model_name == 'set2set':
        pl_model = PL_Set2Set_GNN(model_name=model_name, dataset_name='QM9',model_kwargs=dict(task=task, model_type=model_type, num_atom_features=num_atom_features, num_bond_features=num_bond_features, finetune_dim=finetune_dim))
        
    #平均と分散の追加
    pl_model.norm = True
    pl_model.mean_h = mean_value_h
    pl_model.mean_l = mean_value_l
    pl_model.std_h = std_value_h
    pl_model.std_l = std_value_l
    trainer = pl.Trainer(callbacks=[csv_logger,checkpoint_callback], max_epochs=200, log_every_n_steps=1, devices=1, num_nodes=1)
    trainer.fit(pl_model, train_loader, valid_loader)

    # Push to the HuggingFace hub:
    my_token = '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'
    print('Save this file: {}'.format(state_save_dir))
    pl_model.save_pretrained(
        state_save_dir, #wightの保存先
        push_to_hub=True,
        repo_id=repo_id,
        token=my_token
     )
    print('Pushing to the HuggingFace was Done!')
    
    train_loss = pl_model.train_loss
    val_loss = pl_model.val_loss
    trainer.test(ckpt_path='best', dataloaders=test_loader) 
    #print(trainer.callback_metrics)
    preds, labels = pl_model.test_step_outputs
    print('h_r2: ', r2_score(labels[0], preds[0]))
    print('l_r2: ', r2_score(labels[1], preds[1]))
    print('R2: ', r2_score(labels, preds))
    p_l_file = os.path.join(log_dir, 'PandL')
    np.savez(p_l_file, pred=preds,labels=labels)

if __name__ == "__main__":
    mp.set_sharing_strategy('file_system')
    
    main()

