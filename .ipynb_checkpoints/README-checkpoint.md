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
ファインチューニングについては[Google colablatory](https://colab.research.google.com/drive/1rUaXXIKZaG6C9NTSwlUGTwcEBSB-Q_dK?usp=sharing)でも公開しています。
[こちら]()のQiita記事も参考にしてください。

## インストール
```
#環境の構築
$ conda create -n py38 python=3.8
$ conda activate py38

#必要な物のインストール
$ conda install -c "nvidia/label/cuda-11.8.0" cuda-toolki
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
$ conda install pyg -c pyg
$ rdkit: conda install -c conda-forge rdkit
$ pip install --upgrade huggingface_hub
$ pip install pytorch_lightning
```
## 使い方
### データの準備
QM9データセットおよびAmes試験データ、水溶性に関するデータはすべて必要な形にまとめられcsv形式で[data](https://github.com/kumagaitomotaka/Pretrain_models/tree/main/data)にまとめられています。
必要に応じてダウンロードして使用してください。
### 事前学習
事前学習を行うためには以下のコードを実行してください。
```

```
#### ※注意点

### ファインチューニング
ファインチューニングを行う場合は以下のコードを実行してください。
```

```