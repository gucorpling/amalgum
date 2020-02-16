import stanfordnlp
import torch

from nlp_modules.base import NLPModule, PipelineDep


class StanfordNLPParser(NLPModule):
    requires = (PipelineDep.TOKENIZE, PipelineDep.POS_TAG)
    provides = (PipelineDep.PARSE,)

    def __init__(self, config):
        use_gpu = config.get("use_gpu", None)
        if use_gpu:
            torch.cuda.init()

        # default english pipeline
        config = {
            "lang": "en",
            "treebank": "en_gum",
            "processors": "tokenize,pos,lemma,depparse",
            "tokenize_pretokenized": True,
            "pos_batch_size": 500,  # 1000,
            "depparse_batch_size": 500
            # We could possibly mix and match models here, but it is probably a bad idea
            #  'pos_model_path': 'en_ewt_models/en_ewt_tagger.pt',
            #  'pos_pretrain_path': 'en_ewt_models/en_ewt.pretrain.pt',
            #  'lemma_model_path': 'en_ewt_lemmatizer/en_ewt_lemmatizer.pt',
            #  'depparse_model_path': 'en_ewt_lemmatizer/en_ewt_parser.pt',
            #  'depparse_pretrain_path': 'en_ewt_lemmatizer/en_ewt.pretrain.pt'
        }
        self.snlp = stanfordnlp.Pipeline(config, use_gpu=use_gpu)

    def test_dependencies(self):
        # This downloads the English models for the neural pipeline
        stanfordnlp.download("en", confirm_if_exists=False)
        # This adds gum models
        stanfordnlp.download("en_gum", confirm_if_exists=False)

    def parse(self, tokenized):
        # StanfordNLP expects a list of sentences, each a list of token strings, in order to run in pre-tokenized mode
        sent_list = [s.strip().split() for s in tokenized.strip().split("\n")]
        torch.cuda.empty_cache()

        doc = self.snlp(sent_list)

        return doc.conll_file.conll_as_string()
