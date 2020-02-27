"""
xrenner_coreferencer.py

Resolves entity types and coreference from conllu parses.
Optionally consults external sequence labels in ACE format for entity types.
"""

import stanfordnlp
import torch

from nlp_modules.base import NLPModule, PipelineDep
from xrenner import Xrenner

class XrennerCoref(NLPModule):
    # Note this module requires PARSE to contain s_type comments like:
    # s_type = decl
    # The CRF_ENTITIES file comes from a secondary nested sequency labeler and is consulted by the entity recognizer
    requires = (PipelineDep.PARSE, PipelineDep.ACE_ENTITIES)
    provides = (PipelineDep.TSV_OUT,)

    def __init__(self, rule_based=False, no_rnn=False, use_oracle=True):
        """

        :param rule_based: whether to turn off machine learning coref classification (much faster, less accurate)
        :param no_rnn: whether to turn off internal neural sequence labeler for classifying xrenner entity span
        :param use_oracle: whether to consult input from an external sequence labeler IFF it has matching spans
        """
        self.rule_base = rule_based
        self.no_rnn = no_rnn
        self.use_oracle = use_oracle
        self.test_dependencies()
        self.xrenner = Xrenner(model="eng", override="GUM", rule_based=rule_based, no_seq=no_rnn)

        # Make sure neural sequence labeler model is installed and download if not
        self.xrenner.check_model()

    def test_dependencies(self):
        import six
        import depedit
        import xrenner
        if not self.rule_base:
            import xgboost
        if not self.no_rnn:
            import flair

    def resolve_entities_coref(self, doc_dict):
        self.xrenner.set_doc_name(doc_dict["filename"])
        if self.use_oracle:
            self.xrenner.lex.read_oracle(doc_dict["ace"], as_text=True)

        tsv_output = self.xrenner.analyze(doc_dict["dep"], "webannotsv")

        return {"tsv": tsv_output}

    def run(self, input_dir, output_dir):


        # Identify a function that takes data and returns output at the document level
        processing_function = self.resolve_entities_coref

        # use process_files, inherited from NLPModule, to apply this function to all docs
        self.process_files_multiformat(
            input_dir, output_dir, processing_function
        )

