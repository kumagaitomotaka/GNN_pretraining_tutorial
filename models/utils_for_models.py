import torch

def r_squared(y_true, y_pred):
    y_mean = torch.mean(y_true)
    tss = torch.sum((y_true - y_mean)**2)
    rss = torch.sum((y_true - y_pred)**2)
    r2 = 1 - (rss / tss)
    return r2
    
def accuracy(y_true, y_pred):
    # モデルの予測結果からクラスを選択
    _, predicted = torch.max(y_pred, 1)

    # 正確な予測の割合を計算
    correct = (predicted == y_true).sum().item()
    total = y_true.size(0)
    acc = correct / total

    return acc

def loss_function(pred, batch, task, model_type):
    if model_type == 'pretrain':
        criterion = torch.nn.MSELoss()
        h_loss = criterion(pred[:,0], batch.h_y)
        l_loss = criterion(pred[:,1], batch.l_y)
        loss = (h_loss + l_loss)/2
        step_loss = loss * batch.h_y.size(0)
    elif model_type == 'finetune':
        label = batch.y
        if task == 'classification':
            criterion = torch.nn.CrossEntropyLoss()
            label = label.to(torch.int64)
            loss = criterion(pred, label.flatten())
        elif task == 'regression':
            criterion = torch.nn.MSELoss()
            loss = criterion(pred, label.unsqueeze(1))
        step_loss = loss * label.size(0)
    
    return loss, step_loss