![CUDA 11.8](https://img.shields.io/badge/cuda-11.8-blue.svg)
![Python 3.8](https://img.shields.io/badge/python-3.8-yellow.svg)
![PyTorch 2.1.0](https://img.shields.io/badge/pytorch-2.1.0-red.svg)
![PyG 2.4.0](https://img.shields.io/badge/PyG-2.4.0-orange.svg)
# グラフニューラルネットワークを使った有機化合物の事前学習と物性予測
有機化合物データセットであるQM9データセットを用いてモデルの事前学習を行い、有機化合物の物性を予測するモデルを構築しました。
まずQM9データセットを用いてHOMO・LUMOの予測を行い、事前学習モデルを作成します。作成したモデルはHugging Faceに保存します（事前学習）。
次に、Hugging Faceに保存されている事前学習モデルを読み込み、ファインチューニングおよび物性予測を行います(ファインチューニング)。
事前学習に用いたQM9のデータは[こちら](https://github.com/yuyangw/MolCLR, 'https://github.com/yuyangw/MolCLR')からダウンロードしました。
ファインチューニング用のデータは[Ames試験データ](https://pubs.acs.org/doi/abs/10.1021/ci900161g)と[水溶性に関するデータ](https://github.com/rdkit/rdkit/tree/master/Docs/Book/data)を用いました。
ファインチューニングについてはGoogle colablatoryでも公開しています。
内容の詳細についてはのQiita記事も参考にしてください。

・[Google colablatory](https://colab.research.google.com/drive/1rUaXXIKZaG6C9NTSwlUGTwcEBSB-Q_dK?usp=sharing)
・[Qiita記事](https://qiita.com/KUMAtomo/items/086fdd3cb21ece82dcb5)

## インストール
```
#環境の構築
$ conda create -n py38 python=3.8
$ conda activate py38

#必要な物のインストール
$ conda install -c "nvidia/label/cuda-11.8.0" cuda-toolki
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
$ conda install -c conda-forge rdkit
$ pip install torch_geometric
$ pip install --upgrade huggingface_hub
$ pip install pytorch_lightning
```
## 使い方
### データの準備
QM9データセットおよびAmes試験データ、水溶性に関するデータはすべて前処理済みのものをcsv形式で[data](https://github.com/kumagaitomotaka/Pretrain_models/tree/main/data)にまとめています。
### 事前学習
事前学習を行うためには以下のコードを実行してください。
```
$ python PL_GNN_to_Hug.py
```
#### ※注意
PL_GNN_to_Hug.py内20,21行目に自身のHugging Faceのアクセストークン及び自身のrepo_idを入力してから実行してください。
```
20 my_token = '*************************************' #自身のトークンを入力してください。
21 repo_id = "kumatomo/TopK_GNN" #自身で作成したレポジトリのrepo_idに変更してください。
```
### ファインチューニング
ファインチューニングを行う場合は以下のコードを実行してください。
```
$ python PL_GNN_from_Hug.py
```
#### ※注意1
ファインチューニングを行うモデルに合わせてPL_GNN_from_Hug.py内のrepo_idを適切に変更してください。
```
repo_id = "kumatomo/TopK_GNN" # 自身で事前学習モデルを作成した場合は、モデルを保存したrepo_idに変更してください。
```
modelはGCN、GIN、TopK_GCN、set2set_NMP、GraphSAGEの5種類が用意してあります。
それらを用いる際にはrepo_idにそれぞれ'kumatomo/BasicGCN'、'kumatomo/BasicGIN'、'kumatomo/TopK_GNN'、'kumatomo/set2set_GNN'、'kumatomo/BasicGraphSAGE'を入力してください。
#### ※注意2
ファインチューニングを行うデータセットに合わせてPL_GNN_from_Hug.py内のdata_nameやmodel_kwagsなどを適切に変更してから実行してください。
(Ames: finetune_dim=2, task=classification, sol: finetune_dim=3, task=classification, sol_rgr: finetune_dim=1, task=regression)
```
model_kwargs['finetune_dim'] = 2 
model_kwargs['task'] = 'classification'
model_kwargs['model_type'] = 'finetune'
・
・
・
data_name = 'Ames'
finetune_dim = 2
model_name = 'TopK'
task = 'classification'
model_type = 'finetune'
```
modelはGCN、GIN、TopK_GCN、set2set_NMP、GraphSAGEの5種類が用意してあります。
それらを用いる際にはmodel_nameにそれぞれ'GCN'、'GIN'、'TopK'、'set2set'、'GraphSAGE'を入力してください。