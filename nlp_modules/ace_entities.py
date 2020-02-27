import os, io, re

from glob import glob
from nlp_modules.base import NLPModule, PipelineDep
from lib.shibuya_entities.ShibuyaEntities import ShibuyaEntities


class AceEntities(NLPModule):
    requires = (PipelineDep.PARSE)
    provides = (PipelineDep.CRF_ENTITIES)

    def __init__(self, config, serialnumber="200226_153935"):
        self.LIB_DIR = config["LIB_DIR"]
        self.acedir = os.join(self.LIB_DIR, 'shibuya_entities', 'data', 'amalgum')
        self.dumpsdir = os.join(self.LIB_DIR, 'shibuya_entities', 'dumps')
        self.serialnumber = serialnumber


    def parse(self, inputstr):
        
        assert '\n\n' in inputstr and '\t' in inputstr
        acegoldstr = self.shibuyaentities.conllu2acegold(inputstr)
        
        with io.open(self.acedir + 'amalgum.test', 'w', encoding='utf8') as ftest:
            ftest.write(acegoldstr)
        
        # Pickle test data
        self.shibuyaentities.gen_data(dataset="amalgum")
        
        # predicts and outputs subtoks
        outputstr, _ = self.shibuyaentities.predict(dataset="amalgum", serialnumber=self.serialnumber)
        
        # convert subtoks to toks
        outputstr = self.shibuyaentities.subtok2tok(outputstr, acegoldstr)
        
        return {"ace": outputstr}


    def run(self, input_dir, output_dir):
        self.shibuyaentities = ShibuyaEntities()


        # Identify a function that takes data and returns output at the document level
        processing_function = self.parse

        # use process_files, inherited from NLPModule, to apply this function to all docs
        # self.process_files(
        #     input_dir, output_dir, processing_function, multithreaded=False
        # )
        
        self.process_files_multiformat(input_dir, processing_function, output_dir, multithreaded=False)
