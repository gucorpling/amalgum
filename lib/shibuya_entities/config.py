from __future__ import division

from training.utils import Optimizer


class Config:
    def __init__(self) -> None:
        self.root_path: str = "."

        # for data loader
        self.data_set: str = "sample"
        self.batch_size: int = 32
        self.if_shuffle: bool = True

        # override when loading data
        self.voc_size: int = None
        self.label_size: int = None

        # for bert
        self.bert_model: str = 'bert-large-cased'

        # for lstm
        self.hidden_size: int = 200
        self.layers: int = 2
        self.lstm_dropout: float = 0.20

        # for training
        # originally 500 epochs
        # self.epoch: int = 500

        self.epoch: int = 200

        self.if_gpu: bool = True
        self.opt: Optimizer = Optimizer.AdaBound
        self.lr: float = 0.001 if self.opt != Optimizer.SGD else 0.1
        self.final_lr: float = 0.1 if self.opt == Optimizer.AdaBound else None
        self.l2: float = 0.
        self.check_every: int = 1
        self.clip_norm: int = 5

        # for early stop
        self.lr_patience: int = 5 if self.opt != Optimizer.SGD else 3

        self.data_path: str = self.root_path + "/data/{}".format(self.data_set)
        self.train_data_path: str = self.data_path + "_train.pkl"
        self.dev_data_path: str = self.data_path + "_dev.pkl"
        self.test_data_path: str = self.data_path + "_test.pkl"
        self.config_data_path: str = self.data_path + "_config.pkl"
        self.model_root_path: str = self.root_path + "/dumps"
        self.model_path: str = self.model_root_path + "/{}_model".format(self.data_set)

    def __repr__(self) -> str:
        return str(vars(self))


config: Config = Config()
