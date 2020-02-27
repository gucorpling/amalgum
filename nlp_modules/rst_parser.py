import os
import glob
import sys
import gzip
from nlp_modules.base import NLPModule, PipelineDep
from lib.dplp_plusplus.src.docreader import DocReader
from lib.dplp_plusplus.src.make_merge import Sequencer, merge
from lib.dplp_plusplus.src.evalparser import evalparser

PY3 = sys.version_info[0] > 2

if PY3:
    from pickle import load
else:
    from cPickle import load


class rstParser(NLPModule):

    requires = (PipelineDep.S_TYPE, PipelineDep.EDUS, PipelineDep.PARSE)
    provides = (PipelineDep.RST_OUT,)

    def __init__(self, config):
        self.LIB_DIR = config["LIB_DIR"]
        self.test_dependencies()
        self.parse = Sequencer()
        self.dr = DocReader()

    def test_dependencies(self):
        import flair            # version==0.4.4
        import torch            # version==1.2.0; torchvision==0.4.0; tqdm==4.43.0
        # Check flair and torch version
        if flair.__version__ != '0.4.4':
            raise Exception("flair version 1.4.0 must be used with rstParser")
        if torch.__version__ != '1.4.0':
            raise Exception("Torch version 1.4.0 must be used with rstParser")

        # Check the pretrain model
        model_dir = os.path.join(self.LIB_DIR, "dplp_plusplus", "models")

    def parse(self, doc_dict):

        # construct .merge file that contains all the features
        fmerge = merge(doc_dict["edu"], doc_dict["xml"], doc_dict["dep"], doc_dict["filename"], seq=self.parse, as_text=True)

        # Use brown clsuters
        with gzip.open("resources/bc3200.pickle.gz") as fin:
            print('Load Brown clusters for creating features ...')
            bcvocab = load(fin)

        rst_out = evalparser(fmerge, report=False, bcvocab=bcvocab,  draw=False, withdp=False,
                   fdpvocab="data/resources/word-dict.pickle.gz", fprojmat="data/resources/projmat.pickle.gz")

        return {"rst": rst_out}

    def run(self, input_dir, output_dir):

        processing_function = self.parse

        # use process_files, inherited from NLPModule, to apply this function to all docs
        self.process_files_multiformat(input_dir, processing_function, output_dir, multithreaded=False)
