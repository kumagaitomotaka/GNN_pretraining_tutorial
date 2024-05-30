import numpy as np
import os
from sklearn.metrics import r2_score, accuracy_score
import torch
import torch.multiprocessing as mp
from torch_geometric.nn.model_hub import PyGModelHubMixin
import pytorch_lightning as pl
#自作モジュール
from models.PL_BasicGNN_models import PL_BasicGNNs
from models.PL_topK_model import PL_TopKmodel
from models.PL_set2set_model import PL_Set2Setmodel
from dataset.gnn_dataset import GNN_DatasetWrapper
from pytorch_lightning.callbacks import ModelCheckpoint
from callbacks.pl_callbacks import CSVLogger

repo_id = "kumatomo/TopK_GNN" # 自身で事前学習モデルを作成した場合は、モデルを保存したrepo_idに変更してください。

# Define your class with the mixin:
class PL_Basic_GNN(PL_BasicGNNs, PyGModelHubMixin):
    def __init__(self, dataset_name, model_name, model_kwargs):
        #正常に読み込まれないため再定義
        model_kwargs['finetune_dim'] = 2 
        model_kwargs['task'] = 'classification'
        model_kwargs['model_type'] = 'finetune'
        model_kwargs['model_name'] = model_name #変更しないでください。
        model_kwargs['in_channels'] = 81 #変更しないでください。
        PL_BasicGNNs.__init__(self,**model_kwargs)
        PyGModelHubMixin.__init__(self, model_name, dataset_name, model_kwargs)
class PL_TopK_GNN(PL_TopKmodel, PyGModelHubMixin):
    def __init__(self, dataset_name, model_name, model_kwargs):
        #正常に読み込まれないため再定義
        model_kwargs['finetune_dim'] = 2
        model_kwargs['task'] = 'classification'
        model_kwargs['model_type'] = 'finetune'
        model_kwargs['num_atom_features'] = 81 #変更しないでください。
        PL_TopKmodel.__init__(self,**model_kwargs)
        PyGModelHubMixin.__init__(self, model_name, dataset_name, model_kwargs)
class PL_Set2Set_GNN(PL_Set2Setmodel, PyGModelHubMixin):
    def __init__(self, dataset_name, model_name, model_kwargs):
        #正常に読み込まれないため再定義
        model_kwargs['finetune_dim'] = 2
        model_kwargs['task'] = 'classification'
        model_kwargs['model_type'] = 'finetune'
        model_kwargs['num_atom_features'] = 81 #変更しないでください。
        model_kwargs['num_bond_features'] = 4 #変更しないでください。
        PL_Set2Setmodel.__init__(self,**model_kwargs)
        PyGModelHubMixin.__init__(self, model_name, dataset_name, model_kwargs)
        
def main():
    epochs = 10            #number of train epoch
    batch_size = 128
    num_workers = 10                # dataloader number of workers
    valid_size = 0.1               # ratio of validation data
    test_size = 0.1                # ratio of test data
    splitting = 'random'          # data splitting (i.e., random/scaffold)
    random_seed = None
    data_name = 'Ames'
    finetune_dim = 2
    model_name = 'TopK'
    task = 'classification'
    model_type = 'finetune'
    
    print('{} model start with {}, {} predicting'.format(model_type, model_name, data_name))
    
    #dataset
    dataset = GNN_DatasetWrapper(batch_size, num_workers, valid_size, test_size, data_name, splitting, random_seed=random_seed)
    train_loader, valid_loader, test_loader = dataset.get_data_loaders()

    #nomalizer
    norm = False
    if task == 'regression':
        labels = []
        for d in train_loader:
            labels.extend(d.y.numpy())
        mean_value = np.mean(labels)
        std_value = np.std(labels)
        print('normalizing...mean: {}, std: {}, shape: {}'.format(mean_value,std_value, len(labels)))
        norm = True
    else:
        normalizer = None
        print('No normalizing.')
    
    # 保存するCSVファイルのパスとフィールド名を指定
    data_dir = 'Hug_PL_data_finetune'
    os.makedirs(data_dir, exist_ok=True)
    log_dir = os.path.join(data_dir, 'QM9_'+model_name+ '_' +data_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = 'logs.csv'
    log_file_path= os.path.join(log_dir, log_file)
    csv_logger = CSVLogger(file_path=log_file_path, fieldnames=['epoch', 'train_loss', 'val_loss'])
    
    #modelを保存するmodelcheckpointの作成
    state_save_dir ='pl_finetune_ckpt/'
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',dirpath=state_save_dir, filename=data_name + '_' + model_name +'_model-{epoch:02d}', save_top_k=1, mode='min')# 最良のモデル1つだけ保存
        
    #get num_feature
    check_iter = iter(train_loader)
    check_data = next(check_iter)
    num_atom_features = int(check_data[0].num_atom_features)
    num_bond_features = int(check_data[0].num_bond_features)
    
    if model_name in['GCN','GIN','GraphSAGE']:
        pl_gnn = PL_Basic_GNN(model_name=model_name, dataset_name='QM9',model_kwargs=dict(model_name=model_name, task=task, model_type=model_type, in_channels=num_atom_features, finetune_dim=finetune_dim))
    elif model_name == 'TopK':
        pl_gnn = PL_TopK_GNN(model_name=model_name, dataset_name='QM9',model_kwargs=dict(task=task, model_type=model_type, num_atom_features=num_atom_features, finetune_dim=finetune_dim))
    elif model_name == 'set2set':
        pl_gnn = PL_Set2Set_GNN(model_name=model_name, dataset_name='QM9',model_kwargs=dict(task=task, model_type=model_type, num_atom_features=num_atom_features, num_bond_features=num_bond_features, finetune_dim=finetune_dim))
    else:
        raise ValueError('{} model was not supported'.format(model_name))
    #pretraine済みモデルの読み込み
    pl_model = pl_gnn.from_pretrained(repo_id, dataset_name='QM9', model_name=model_name)
    #平均と分散の追加
    if norm:
        pl_model.norm = True
        pl_model.mean = mean_value
        pl_model.std = std_value
    else:
        pl_model.norm = False
    pl_model.train()
    trainer = pl.Trainer(callbacks=[csv_logger, checkpoint_callback], max_epochs=epochs, log_every_n_steps=1, devices=1)#devices=1を指定しないとregressionの際にRuntimeErrorがでる
    trainer.fit(pl_model, train_loader, valid_loader)
    train_loss = pl_model.train_loss
    val_loss = pl_model.val_loss
    trainer.test(ckpt_path='best', dataloaders=test_loader) 
    #print(trainer.callback_metrics)
    preds, labels = pl_model.test_step_outputs
    if task == 'classification':
        predicted = np.argmax(preds, axis=1)
        print('Accuracy', accuracy_score(labels, predicted))
    elif task == 'regression':
        print('R2: ', r2_score(labels, preds))
    p_l_file = os.path.join(log_dir, 'PandL')
    np.savez(p_l_file, pred=preds,labels=labels)

if __name__ == "__main__":
    mp.set_sharing_strategy('file_system')
    
    main()

