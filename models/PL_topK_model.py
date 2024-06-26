import torch
import torch.nn.functional as F
from torch.nn import Sequential, ReLU, GRU
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
import pytorch_lightning as pl
from torchmetrics.regression import R2Score
#import utils_for_models as utils
import models.utils_for_models as utils

#https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_topk_pool.py
class PL_TopKmodel(pl.LightningModule):
    def __init__(self, task, model_type, num_atom_features, finetune_dim):
        super(PL_TopKmodel, self).__init__()
        
        num_layer = 2
        dim = 128
        
        self.task = task
        self.model_type = model_type
        self.finetune_dim = finetune_dim
        self.train_step_loss = []
        self.validation_step_loss = []
        self.train_loss = []
        self.val_loss = []
        if self.model_type == 'pretrain':
            self.test_step_preds = [[],[]]
            self.test_step_labels = [[],[]]
        elif self.model_type == 'finetune':
            self.test_step_preds = []
            self.test_step_labels = []
        
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
    
        for i in range(num_layer):
            if i == 0:
                self.convs.append(GraphConv(num_atom_features, dim))
            else:
                self.convs.append(GraphConv(dim, dim))
            self.pools.append(TopKPooling(dim, ratio=0.8))
            
        self.lin1 = torch.nn.Linear(dim*2, dim)
        self.lin2 = torch.nn.Linear(dim, dim//2)
        self.lin3 = torch.nn.Linear(dim//2, 2)
        
        if self.finetune_dim != 0:
            print('Fintune model!')
            self.f_lin = torch.nn.Linear(dim//2, self.finetune_dim)
            

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = []
        for conv, pool in zip(self.convs, self.pools):
            x = F.relu(conv(x, edge_index))
            x, edge_index, _, batch, _, _ = pool(x, edge_index, None, batch)
            cat_x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            xs.append(cat_x)
        xs = torch.stack(xs, dim=0)
        z = torch.sum(xs, dim=0)

        z = F.relu(self.lin1(z))
        z = F.dropout(z, p=0.5, training=self.training)
        z = F.relu(self.lin2(z))
        if self.finetune_dim != 0:
            out = self.f_lin(z)
        else:
            out = self.lin3(z)
            
        return out
    
    def on_train_batch_start(self, batch, batch_idx):
        # ラベルを正規化
        if self.norm:
            if self.model_type == 'pretrain':
                batch.h_y = (batch.h_y - self.mean_h) / self.std_h
                batch.l_y = (batch.l_y - self.mean_l) / self.std_l
            elif self.model_type == 'finetune':
                batch.y = (batch.y - self.mean) / self.std
        
    def on_validation_batch_start(self, batch, batch_idx):
        # ラベルを正規化
        if self.norm:
            if self.model_type == 'pretrain':
                batch.h_y = (batch.h_y - self.mean_h) / self.std_h
                batch.l_y = (batch.l_y - self.mean_l) / self.std_l
            elif self.model_type == 'finetune':
                batch.y = (batch.y - self.mean) / self.std
    
    def on_test_batch_start(self, batch, batch_idx):
        # ラベルを正規化
        if self.norm:
            if self.model_type == 'pretrain':
                batch.h_y = (batch.h_y - self.mean_h) / self.std_h
                batch.l_y = (batch.l_y - self.mean_l) / self.std_l
            elif self.model_type == 'finetune':
                batch.y = (batch.y - self.mean) / self.std

    def on_test_batch_end(self, outputs, batch, batch_idx):
        # ラベルと予測値のデノルム
        if self.norm:
            if self.model_type == 'pretrain':
                denorm_pred_h = (outputs['pred_h']* self.std_h) + self.mean_h
                denorm_pred_l = (outputs['pred_l']* self.std_l) + self.mean_l
                denorm_label_h = (batch.h_y* self.std_h) + self.mean_h
                denorm_label_l = (batch.l_y* self.std_l) + self.mean_l
                self.test_step_preds[0].extend(denorm_pred_h)
                self.test_step_preds[1].extend(denorm_pred_l)
                self.test_step_labels[0].extend(denorm_label_h)
                self.test_step_labels[1].extend(denorm_label_l)
            elif self.model_type == 'finetune':
                denorm_pred = (outputs['pred']* self.std) + self.mean
                denorm_label = (batch.y* self.std) + self.mean
                self.test_step_preds.extend(denorm_pred)
                self.test_step_labels.extend(denorm_label)
            
        else:
            if self.model_type == 'pretrain':
                self.test_step_preds[0].extend(outputs.pred_h)
                self.test_step_preds[1].extend(outputs.pred_l)
                self.test_step_labels[0].extend(batch.h_y)
                self.test_step_labels[1].extend(batch.l_y)
            elif self.model_type == 'finetune':
                self.test_step_preds.extend(outputs['pred'])
                self.test_step_labels.extend(batch.y)
            
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        batch_size = len(batch)
        pred = self.forward(batch)
        loss, train_loss = utils.loss_function(pred, batch, task=self.task, model_type=self.model_type)
        self.log('train_loss', train_loss.item(), on_epoch=True, on_step=True, batch_size=batch_size) #https://qiita.com/sakagami_notebook/items/e92970c657f78f04fc3e
        self.train_step_loss.append(train_loss)      
        results = loss
        
        return results
    
    def validation_step(self, batch, batch_idx):
        batch_size = len(batch)
        pred = self.forward(batch)
        loss, valid_loss = utils.loss_function(pred, batch, task=self.task, model_type=self.model_type)
        self.log('val_loss', valid_loss.item(), on_step=False, on_epoch=True, batch_size=batch_size)
        self.validation_step_loss.append(valid_loss)
        results = loss
        
        return results

    def on_train_epoch_end(self):
        epoch_loss = torch.stack(self.train_step_loss).mean()
        results = {'train_epoch_loss': epoch_loss}
        print('\ntrain epoch loss: {}'.format(epoch_loss.item()))
        self.train_loss.append(epoch_loss.item())
        
        return results
    

    def on_validation_epoch_end(self):
        epoch_loss = torch.stack(self.validation_step_loss).mean()
        results = {'validation_epoch_loss': epoch_loss}
        print('\nvalidation epoch loss: {}'.format(epoch_loss.item()))
        self.val_loss.append(epoch_loss.item())
        
        return results
    
    def test_step(self, batch, batch_idx):
        batch_size = len(batch)
        pred = self.forward(batch)
        loss, test_loss = utils.loss_function(pred, batch, task=self.task, model_type=self.model_type)
        self.log('test_loss', test_loss.item(), on_step=False, on_epoch=True, batch_size=batch_size)
        if self.model_type == 'pretrain':
            results = {'test_loss': test_loss, 'pred_h': pred[:,0], 'pred_l': pred[:,1]}
        elif self.model_type == 'finetune':
            results = {'test_loss': test_loss, 'pred': pred}
        
        return results
    
    def on_test_epoch_end(self):
        if self.model_type == 'pretrain':
            test_end_pred_h = torch.tensor(self.test_step_preds[0])
            test_end_pred_l = torch.tensor(self.test_step_preds[1])
            test_end_label_h = torch.tensor(self.test_step_labels[0])
            test_end_label_l = torch.tensor(self.test_step_labels[1])
            print('\nend_test_pred: {}'.format(len(test_end_pred_h)))
            self.test_step_outputs = ([test_end_pred_h.cpu().detach().numpy(), test_end_pred_l.cpu().detach().numpy()], [test_end_label_h.cpu().detach().numpy(), test_end_label_l.cpu().detach().numpy()])
            # R^2値を計算
            r2_h = utils.r_squared(test_end_label_h, test_end_pred_h)
            r2_l = utils.r_squared(test_end_label_l, test_end_pred_l)
            r2_all = utils.r_squared(torch.cat((test_end_label_h,test_end_label_l), dim=0), torch.cat((test_end_pred_h,test_end_pred_l), dim=0))
            results = {'R2': r2_all}
            self.log('test_end_homo r2:', r2_h.item())
            self.log('test_end_lumo r2:', r2_l.item())
            self.log('test_end_R2:', r2_all.item())
        elif self.model_type == 'finetune':
            test_end_pred = torch.stack(self.test_step_preds)
            test_end_label = torch.stack(self.test_step_labels)
            print('\nend_test_pred: {}'.format(len(test_end_pred)))
            # 結果の評価
            if self.task == 'regression':
                r2 = utils.r_squared(test_end_label, test_end_pred)
                results = {'R2': r2}
                self.log('test_end_R2:', r2.item())
            elif self.task == 'classification':
                acc = utils.accuracy(test_end_label, test_end_pred)
                results = {'Accuracy': acc}
                self.log('test_end_Acc:', acc)
            self.test_step_outputs = (test_end_pred.cpu().detach().numpy(),test_end_label.cpu().detach().numpy())
        
        return results
        

    
    def configure_optimizers(self):
        #https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-5),
            "monitor": 'val_loss'},
                }
    
        


