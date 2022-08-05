from enum import unique

from transformers.utils.dummy_pt_objects import PretrainedBartModel
from .entities import Token
import json
import os
import sys
import warnings
from typing import List, Tuple, Dict

import torch
from sklearn.metrics import precision_recall_fscore_support as prfs
from transformers import BertTokenizer

from ssn import util
from ssn.entities import Document, Dataset, EntityType
from ssn.input_reader import JsonInputReader
import jinja2

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class Evaluator:
    def __init__(self, dataset: Dataset, input_reader: JsonInputReader, text_encoder: BertTokenizer, no_overlapping: bool,
                 example_count: int, epoch: int, dataset_label: str):
        self._text_encoder = text_encoder
        self._input_reader = input_reader
        self._dataset = dataset
        self._no_overlapping = no_overlapping

        self._epoch = epoch
        self._dataset_label = dataset_label

        self._example_count = example_count

        # entities
        self._gt_entities = []  # ground truth
        self._pred_entities = []  # prediction

        self._pseudo_entity_type = EntityType('Entity', 1, 'Entity', 'Entity')  # for span only evaluation
        self._convert_gt(self._dataset.documents)


    def eval_batch(self, batch_entity_clf: torch.tensor, batch_entity_bdy:torch.tensor, confidence: float):
        batch_size = batch_entity_clf.shape[0]

        batch_entity_clf = batch_entity_clf.softmax(dim=-1)
        batch_entity_types = batch_entity_clf.argmax(dim=-1) 
        batch_entity_scores = batch_entity_clf.max(dim=-1)[0] 
        batch_entity_mask = batch_entity_scores > confidence

        entity_left = batch_entity_bdy[0].argmax(dim=-1)
        entity_right = batch_entity_bdy[1].argmax(dim=-1) + 1
        batch_entity_spans = torch.stack([entity_left, entity_right], dim=-1)

        batch_entity_mask = batch_entity_mask * (batch_entity_spans[:,:,0] < batch_entity_spans[:,:,1]) * (batch_entity_types != 0)

        for i in range(batch_size):
            entity_mask = batch_entity_mask[i]

            entity_types = batch_entity_types[i][entity_mask]
            entity_scores = batch_entity_scores[i][entity_mask]
            entity_spans = batch_entity_spans[i][entity_mask]

            sample_pred_entities = self._convert_pred_entities(entity_types, entity_spans, entity_scores)
            sample_pred_entities = sorted(sample_pred_entities, key=lambda x:x[3], reverse=True)
            sample_pred_entities = self._remove_duplicate(sample_pred_entities)
            
            self._pred_entities.append(sample_pred_entities)

    def compute_scores(self):
        gt, pred = self._convert_by_setting(self._gt_entities, self._pred_entities, include_entity_types=True)
        ner_eval = self._score(gt, pred, print_results=False)

        return ner_eval

    def get_predictions(self):
        predictions = []

        for i, doc in enumerate(self._dataset.documents):
            tokens = doc.tokens
            pred_entities = self._pred_entities[i]

            # convert entities
            converted_entities = []
            for entity in pred_entities:
                entity_span = entity[:2]
                span_tokens = util.get_span_tokens(tokens, entity_span)
                entity_type = entity[2].identifier
                converted_entity = dict(type=entity_type, start=span_tokens[0].index, end=span_tokens[-1].index + 1)
                converted_entities.append(converted_entity)
            converted_entities = sorted(converted_entities, key=lambda e: e['start'])

            doc_predictions = dict(tokens=[t.phrase for t in tokens], entities=converted_entities,)
            predictions.append(doc_predictions)

        return predictions

    def _convert_gt(self, docs: List[Document]):
        for doc in docs:
            gt_entities = doc.entities
            # convert ground truth relations and entities for precision/recall/f1 evaluation
            sample_gt_entities = []
            for entity in gt_entities:
                try:
                    sample_gt_entities.append(entity.as_tuple_token())
                except:
                    sys.stderr.write(str(entity))
                    sys.stderr.write('\n')
                    sys.stderr.write(entity.as_tuple_token())
                    AssertionError('-- Error --')
                
#             sample_gt_entities = [entity.as_tuple_token() for entity in gt_entities]

            if self._no_overlapping:
                sample_gt_entities = self._remove_overlapping(sample_gt_entities)

            self._gt_entities.append(sample_gt_entities)

    def _convert_pred_entities(self, pred_types: torch.tensor, pred_spans: torch.tensor, pred_scores: torch.tensor):
        
        converted_preds = []

        for i in range(pred_types.shape[0]):
            label_idx = pred_types[i].item()
            entity_type = self._input_reader.get_entity_type(label_idx)

            start, end = pred_spans[i].tolist()
            cls_score = pred_scores[i].item()

            converted_pred = (start, end, entity_type, cls_score)
            converted_preds.append(converted_pred)
        return converted_preds

    def _remove_duplicate(self, entities):

        non_duplicate_entities = []
        
        for i, can_entity in enumerate(entities):
            find = False
            for j, entity in enumerate(non_duplicate_entities):
                if can_entity[0] == entity[0] and can_entity[1] == entity[1]:
                    find = True
            if not find:
                non_duplicate_entities.append(can_entity)
        return non_duplicate_entities

    def _remove_overlapping(self, entities):
        non_overlapping_entities = []
        for entity in entities:
            if not self._is_overlapping(entity, entities):
                non_overlapping_entities.append(entity)

        return non_overlapping_entities

    def _is_overlapping(self, e1, entities):
        for e2 in entities:
            if self._check_overlap(e1, e2):
                return True

        return False

    def _check_overlap(self, e1, e2):
        if e1 == e2 or e1[1] <= e2[0] or e2[1] <= e1[0]:
            return False
        else:
            return True

    def _convert_by_setting(self, gt: List[List[Tuple]], pred: List[List[Tuple]],
                            include_entity_types: bool = True, include_score: bool = False):
        assert len(gt) == len(pred)

        # either include or remove entity types based on setting
        def convert(t):
            if not include_entity_types:
                # remove entity type and score for evaluation
                if type(t[0]) == int:  # entity
                    c = [t[0], t[1], self._pseudo_entity_type]
                else:  # relation
                    c = [(t[0][0], t[0][1], self._pseudo_entity_type),
                         (t[1][0], t[1][1], self._pseudo_entity_type), t[2]]
            else:
                c = list(t[:3])

            if include_score and len(t) > 3:
                # include prediction scores
                c.append(t[3])

            return tuple(c)

        converted_gt, converted_pred = [], []

        for sample_gt, sample_pred in zip(gt, pred):
            converted_gt.append([convert(t) for t in sample_gt])
            converted_pred.append([convert(t) for t in sample_pred])

        return converted_gt, converted_pred

    def _score(self, gt: List[List[Tuple]], pred: List[List[Tuple]], print_results: bool = False):
        assert len(gt) == len(pred)

        gt_flat = []
        pred_flat = []
        types = set()

        for (sample_gt, sample_pred) in zip(gt, pred):
            union = set()
            union.update(sample_gt)
            union.update(sample_pred)

            for s in union:
                if s in sample_gt:
                    t = s[2]
                    gt_flat.append(t.index)
                    types.add(t)
                else:
                    gt_flat.append(0)

                if s in sample_pred:
                    t = s[2]
                    pred_flat.append(t.index)
                    types.add(t)
                else:
                    pred_flat.append(0)
        metrics = self._compute_metrics(gt_flat, pred_flat, types, print_results)
        return metrics

    def _compute_metrics(self, gt_all, pred_all, types, print_results: bool = False):
        labels = [t.index for t in types]
        per_type = prfs(gt_all, pred_all, labels=labels, average=None)
        micro = prfs(gt_all, pred_all, labels=labels, average='micro')[:-1]
        macro = prfs(gt_all, pred_all, labels=labels, average='macro')[:-1]
        total_support = sum(per_type[-1])

        return [m * 100 for m in micro + macro]

    def _get_row(self, data, label):
        row = [label]
        for i in range(len(data) - 1):
            row.append("%.2f" % (data[i] * 100))
        row.append(data[3])
        return tuple(row)

    def _convert_example(self, doc: Document, gt: List[Tuple], pred: List[Tuple],
                         include_entity_types: bool, to_html):
        # encoding = doc.encoding
        tokens = doc.tokens

        gt, pred = self._convert_by_setting([gt], [pred], include_entity_types=include_entity_types, include_score=True)
        gt, pred = gt[0], pred[0]

        # get micro precision/recall/f1 scores
        if gt or pred:
            pred_s = [p[:3] for p in pred]  # remove score
            precision, recall, f1 = self._score([gt], [pred_s])[:3]
        else:
            # corner case: no ground truth and no predictions
            precision, recall, f1 = [100] * 3

        cls_scores = [p[3] for p in pred]
        pred = [p[:3] for p in pred]
        union = set(gt + pred)

        # true positives
        tp = []
        # false negatives
        fn = []
        # false positives
        fp = []

        for s in union:
            type_verbose = s[2].verbose_name

            if s in gt:
                if s in pred:
                    cls_score = cls_scores[pred.index(s)]
                    tp.append((to_html(s, tokens), type_verbose, cls_score))
                else:
                    fn.append((to_html(s, tokens), type_verbose, -1))
            else:
                cls_score = cls_scores[pred.index(s)]
                fp.append((to_html(s, tokens), type_verbose, cls_score))

        tp = sorted(tp, key=lambda p: p[2], reverse=True)
        fp = sorted(fp, key=lambda p: p[2], reverse=True)

        phrases = []
        for token in tokens:
            phrases.append(token.phrase)
        text = " ".join(phrases)
        

        # text = self._prettify(self._text_encoder.decode(encoding))
        text = self._prettify(text)
        return dict(text=text, tp=tp, fn=fn, fp=fp, precision=precision, recall=recall, f1=f1, length=len(doc.tokens))


    def _entity_to_html(self, entity: Tuple, tokens: List[Token]):
        start, end = entity[:2]
        entity_type = entity[2].verbose_name

        tag_start = ' <span class="entity">'
        tag_start += '<span class="type">%s</span>' % entity_type

        # ctx_before = self._text_encoder.decode(encoding[:start])
        # e1 = self._text_encoder.decode(encoding[start:end])
        # ctx_after = self._text_encoder.decode(encoding[end:])

        ctx_before = ""
        ctx_after = ""
        e1 = ""
        for i in range(start):
            ctx_before += tokens[i].phrase
            if i!=start-1:
                ctx_before += " "
        for i in range(end, len(tokens)):
            ctx_after += tokens[i].phrase
            if i!=(len(tokens)-1):
                ctx_after += " "
        for i in range(start, end):
            e1 += tokens[i].phrase
            if i!=end-1:
                e1 += " "

        html = ctx_before + tag_start + e1 + '</span> ' + ctx_after
        html = self._prettify(html)

        return html


    def _prettify(self, text: str):
        text = text.replace('_start_', '').replace('_classify_', '').replace('<unk>', '').replace('⁇', '')
        text = text.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
        return text

    def _store_examples(self, examples: List[Dict], file_path: str, template: str):
        template_path = os.path.join(SCRIPT_PATH, 'templates', template)

        # read template
        with open(os.path.join(SCRIPT_PATH, template_path)) as f:
            template = jinja2.Template(f.read())

        # write to disc
        template.stream(examples=examples).dump(file_path)
