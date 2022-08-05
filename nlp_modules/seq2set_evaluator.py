import argparse
import math
import os
import numpy as np
import torch
from torch.nn import DataParallel
from torch.optim import Optimizer
import transformers
from torch.utils.data import DataLoader
from transformers import BertConfig
from transformers import BertTokenizer

from ssn import models
from ssn import sampling
from ssn import util
from ssn.entities import Dataset
from ssn.evaluator import Evaluator
from ssn.input_reader import JsonInputReader, BaseInputReader
from ssn.trainer import BaseTrainer

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class SSNTrainer(BaseTrainer):
    """ Entity recognition training and evaluation """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        # byte-pair encoding
        self._tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)

    def eval(self, input_data, input_reader_cls, types_path: str):
        args = self.args
        dataset_label = 'test'

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer, wordvec_filename=args.wordvec_path)
        input_reader.read(input_data)

        # create model
        model_class = models.get_model(self.args.model_type)

        config = BertConfig.from_pretrained(self.args.model_path, cache_dir=self.args.cache_path)
        util.check_version(config, model_class, self.args.model_path)
        embed = torch.from_numpy(input_reader.embedding_weight).float()
        model = model_class.from_pretrained(self.args.model_path,
                                            config=config,
                                            embed=embed,
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            entity_types=input_reader.entity_type_count,
                                            prop_drop=self.args.prop_drop,
                                            freeze_transformer=self.args.freeze_transformer,
                                            num_decoder_layers=self.args.decoder_layers,
                                            lstm_layers=self.args.lstm_layers,
                                            lstm_drop=self.args.lstm_drop,
                                            pos_size=self.args.pos_size,
                                            char_lstm_layers=self.args.char_lstm_layers,
                                            char_lstm_drop=self.args.char_lstm_drop,
                                            char_size=self.args.char_size,
                                            use_glove=self.args.use_glove,
                                            use_pos=self.args.use_pos,
                                            use_char_lstm=self.args.use_char_lstm,
                                            pool_type=self.args.pool_type,
                                            reduce_dim=self.args.reduce_dim,
                                            bert_before_lstm=self.args.bert_before_lstm,
                                            num_query=self.args.num_query)

        model.to(self._device)

        # evaluate
        pred_data = self._eval(model, input_reader.get_dataset(dataset_label), input_reader, confidence=self.args.confidence)

        return pred_data

    def _eval(self, model: torch.nn.Module, dataset: Dataset, input_reader: JsonInputReader,
              epoch: int = 0, updates_epoch: int = 0, iteration: int = 0, confidence: float = 0.5):

        if isinstance(model, DataParallel):
            # currently no multi GPU support during evaluation
            model = model.module

        # create evaluator
        evaluator = Evaluator(dataset, input_reader, self._tokenizer, self.args.no_overlapping,
                              self.args.example_count, epoch, dataset.label)

        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self.args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=self.args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        with torch.no_grad():
            model.eval()

            # iterate batches
            for batch in data_loader:
                # move batch to selected device
                batch = util.to_device(batch, self._device)

                # run model (forward pass)
                entity_clf, entity_bdy = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                                               token_masks_bool=batch['token_masks_bool'],
                                               token_masks=batch['token_masks'],
                                               pos_encoding=batch['pos_encoding'],
                                               wordvec_encoding=batch['wordvec_encoding'],
                                               char_encoding=batch['char_encoding'],
                                               token_masks_char=batch['token_masks_char'],
                                               char_count=batch['char_count'], evaluate=True)

                evaluator.eval_batch(entity_clf, entity_bdy, confidence)

        preds = evaluator.get_predictions()
        return preds

    def _log_eval(self, ner_prec_micro: float, ner_rec_micro: float, ner_f1_micro: float,
                  ner_prec_macro: float, ner_rec_macro: float, ner_f1_macro: float,
                  epoch: int, iteration: int, global_iteration: int, label: str):

        # log to tensorboard
        self._log_tensorboard(label, 'eval/ner_prec_micro', ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_micro', ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_micro', ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_prec_macro', ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_macro', ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_macro', ner_f1_macro, global_iteration)

        # log to csv
        self._log_csv(label, 'eval', ner_prec_micro, ner_rec_micro, ner_f1_micro,
                      ner_prec_macro, ner_rec_macro, ner_f1_macro,
                      epoch, iteration, global_iteration)

    def _log_datasets(self, input_reader):
        self._logger.info("Entity type count: %s" % input_reader.entity_type_count)

        self._logger.info("Entities:")
        for e in input_reader.entity_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        for k, d in input_reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.document_count)
            self._logger.info("Entity count: %s" % d.entity_count)

        self._logger.info("Context size: %s" % input_reader.context_size)

    def _init_eval_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'eval': ['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro',
                                                 'ner_prec_macro', 'ner_rec_macro', 'ner_f1_macro',
                                                 'epoch', 'iteration', 'global_iteration']})
