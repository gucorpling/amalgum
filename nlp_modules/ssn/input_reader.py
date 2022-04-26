import json
import io
import sys
from abc import abstractmethod, ABC
from collections import OrderedDict
from typing import Iterable, List
from transformers import BertTokenizer
import string
import os
import numpy as np
from collections import Counter
import spacy
from ssn.entities import Dataset, EntityType, RelationType, Entity, Document

nlp = spacy.load("en_core_web_sm")


class BaseInputReader(ABC):
    def __init__(self, types_path: str, tokenizer: BertTokenizer):
#         sys.stdout.write(f'\n{types_path}\n')
#         sys.stdout.write(os.getcwd())
#         f = io.open(types_path)
        types = json.load(open(types_path), object_pairs_hook=OrderedDict)  # entity + relation types

        self._entity_types = OrderedDict()
        self._idx2entity_type = OrderedDict()
        self._relation_types = OrderedDict()
        self._idx2relation_type = OrderedDict()

        # entities
        # add 'None' entity type
        none_entity_type = EntityType('None', 0, 'None', 'No Entity')
        self._entity_types['None'] = none_entity_type
        self._idx2entity_type[0] = none_entity_type

        # specified entity types
        for i, (key, v) in enumerate(types['entities'].items()):
            entity_type = EntityType(key, i + 1, v['short'], v['verbose'])
            self._entity_types[key] = entity_type
            self._idx2entity_type[i + 1] = entity_type

        # relations
        # add 'None' relation type
        none_relation_type = RelationType('None', 0, 'None', 'No Relation')
        self._relation_types['None'] = none_relation_type
        self._idx2relation_type[0] = none_relation_type

        # specified relation types
        for i, (key, v) in enumerate(types['relations'].items()):
            relation_type = RelationType(key, i + 1, v['short'], v['verbose'], v['symmetric'])
            self._relation_types[key] = relation_type
            self._idx2relation_type[i + 1] = relation_type

        self._datasets = dict()

        self._tokenizer = tokenizer

        self._vocabulary_size = tokenizer.vocab_size
        self._context_size = -1

    @abstractmethod
    def read(self, datasets):
        pass

    def get_dataset(self, label) -> Dataset:
        return self._datasets[label]

    def get_entity_type(self, idx) -> EntityType:
        entity = self._idx2entity_type[idx]
        return entity

    def get_relation_type(self, idx) -> RelationType:
        relation = self._idx2relation_type[idx]
        return relation

    def _calc_context_size(self, datasets: Iterable[Dataset]):
        sizes = []

        for dataset in datasets:
            for doc in dataset.documents:
                sizes.append(len(doc.encoding))

        context_size = max(sizes)
        return context_size

    def _log(self, text):
        if self._logger is not None:
            self._logger.info(text)

    @property
    def datasets(self):
        return self._datasets

    @property
    def entity_types(self):
        return self._entity_types

    @property
    def relation_types(self):
        return self._relation_types

    @property
    def relation_type_count(self):
        return len(self._relation_types)

    @property
    def entity_type_count(self):
        return len(self._entity_types)

    @property
    def vocabulary_size(self):
        return self._vocabulary_size

    @property
    def context_size(self):
        return self._context_size

    def __str__(self):
        string = ""
        for dataset in self._datasets.values():
            string += "Dataset: %s\n" % dataset
            string += str(dataset)

        return string

    def __repr__(self):
        return self.__str__()


class JsonInputReader(BaseInputReader):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, build_vocab=False, wordvec_filename=None):
        super().__init__(types_path, tokenizer)
        vec_size = wordvec_filename.split(".")[-2] # str: 100d
        self.build_vocab = False
        self.word2inx = json.load(open(os.path.dirname(types_path)+"/vocab.json", "r"))
        self.embedding_weight = np.load(os.path.dirname(types_path)+f"/vocab_embed_{vec_size}.npy")

        self.vec_size = vec_size
        self.wordvec_filename = wordvec_filename
        self.POS_MAP = ["<UNK>"]
        for k, v in json.load(open(types_path.replace("types", "pos"))).items():
            if v > 15:
                self.POS_MAP.append(k)

    def load_wordvec(self, filename):
        self.embedding_weight = np.random.rand(len(self.word2inx),len(next(iter(self.word2vec.values()))))
        for word, inx in self.word2inx.items():
            if word in self.word2vec:
                self.embedding_weight[inx,:] = self.word2vec[word]

    def read(self, input_data):
        dataset_label = 'test'
        dataset = Dataset('dataset_label', self._relation_types, self._entity_types)
        self._parse_dataset(input_data, dataset)
        self._datasets[dataset_label] = dataset
        self._context_size = self._calc_context_size(self._datasets.values())

    def _build_vocab(self, documents, min_freq = 1):
        self.word2vec = {}
        with open(self.wordvec_filename, "r") as f:
            if "glove" not in self.wordvec_filename:
                f.readline()
            for line in f:
                fields = line.strip().split(' ')
                self.word2vec[fields[0]] = list(float(x) for x in fields[1:])
        counter = Counter()
        for doc in documents:
            try:
                counter.update(list(map(lambda x: x.lower(), doc['tokens'])))
            except:
                with open('error.txt', 'w', encoding='utf8') as f:
                    f.write(str(doc))
                sys.stderr.write(str(type(doc)))
                sys.stderr.write(str(doc[0]))
                counter.update(list(map(lambda x: x.lower(), doc['tokens'])))
        for k, v in counter.items():
            if v >= min_freq and k in self.word2vec:
                self.word2inx[k] = len(self.word2inx)

    def _parse_dataset(self, documents, dataset):
        for document in documents:
            self._parse_document(document, dataset)

    def _parse_document(self, doc, dataset) -> Document:
        jtokens = doc['tokens']
        jentities = doc['entities']
        jpos = doc['pos']
        ltokens = doc["ltokens"]
        rtokens = doc["rtokens"]

        # parse tokens
        doc_tokens, doc_encoding, char_encoding = self._parse_tokens(jtokens, ltokens, rtokens, jpos, dataset)
        # parse entity mentions
        entities = self._parse_entities(jentities, doc_tokens, dataset)
        # create document
        document = dataset.create_document(doc_tokens, entities, doc_encoding, char_encoding)

        return document

    def _parse_tokens(self, jtokens, ltokens, rtokens, jpos, dataset):
        doc_tokens = []
        char_vocab = ['<PAD>'] + list(string.printable) + ['<EOT>', '<UNK>']
        poss = [self.POS_MAP.index(pos) if pos in self.POS_MAP else self.POS_MAP.index("<UNK>") for pos in jpos]

        # full document encoding including special tokens ([CLS] and [SEP]) and byte-pair encodings of original tokens
        doc_encoding = [self._tokenizer.convert_tokens_to_ids('[CLS]')]
        char_encoding = []

        # parse tokens
        for token_phrase in ltokens:
            token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
            doc_encoding += token_encoding

        for i, token_phrase in enumerate(jtokens):
            token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
            token_encoding_char = []
            for c in token_phrase:
                if c in char_vocab:
                    token_encoding_char.append(char_vocab.index(c))
                else:
                    token_encoding_char.append(char_vocab.index("<UNK>"))
            
            span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))
            char_start, char_end = (len(char_encoding), len(char_encoding) + len(token_encoding_char))

            # try:
            if token_phrase.lower() in  self.word2inx:
                inx = self.word2inx[token_phrase.lower()]
            else:
                inx = self.word2inx["<unk>"]
            token = dataset.create_token(i, span_start, span_end, token_phrase, poss[i], inx, char_start, char_end)

            doc_tokens.append(token)
            doc_encoding += token_encoding
            token_encoding_char += [char_vocab.index('<EOT>')]
            char_encoding.append(token_encoding_char)
        
        for token_phrase in rtokens:
            token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
            doc_encoding += token_encoding

        doc_encoding += [self._tokenizer.convert_tokens_to_ids('[SEP]')]

        return doc_tokens, doc_encoding, char_encoding

    def _parse_entities(self, jentities, doc_tokens, dataset) -> List[Entity]:
        entities = []

        for entity_idx, jentity in enumerate(jentities):
            entity_type = self._entity_types[jentity['type']]
            start, end = jentity['start'], jentity['end']

            # create entity mention
            tokens = doc_tokens[start:end]
            phrase = " ".join([t.phrase for t in tokens])
            entity = dataset.create_entity(entity_type, tokens, phrase)
            entities.append(entity)

        return entities

