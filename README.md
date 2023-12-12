# Pytorch CycleGAN with Hydra and MLflow

このリポジトリは、[CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)をもとに、ハイパーパラメータ探索を行いやすくするために、[Hydra](https://github.com/facebookresearch/hydra)と[MLflow](https://github.com/mlflow/mlflow)を組み込んだものです。
オリジナルのコマンドライン引数の代わりに、訓練時にはconf/train.yaml、推論時にはconf/test.yamlを編集し、それぞれ`pythom train.py`あるいは`python test.py`で実行します。Hydraのsweep機能を使用する際には、コマンドラインスイッチ`-m`を使用します、

This repository is based on [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and integrates [Hydra] (https://github.com/facebookresearch/hydra) and [MLflow](https://github.com/mlflow/mlflow) to facilitate easier hyperparameter exploration.

Instead of using the original command-line arguments, edit conf/train.yaml for training and conf/test.yaml for inference, and then run them with python train.py or python test.py, respectively. When using the sweep function of Hydra, use the command-line switch -m.