import os, io, re

from glob import glob
from nlp_modules.base import NLPModule, PipelineDep
from lib.shibuya_entities.ShibuyaEntities import ShibuyaEntities


class AceEntities(NLPModule):
    requires = (PipelineDep.S_SPLIT,)
    provides = (PipelineDep.CRF_ENTITIES)

    def __init__(self, config, serialnumber="200226_153935"):
        self.LIB_DIR = config["LIB_DIR"]
        self.acedir = os.join(self.LIB_DIR, 'shibuya_entities', 'data', 'amalgum')
        self.dumpsdir = os.join(self.LIB_DIR, 'shibuya_entities', 'dumps')
        self.serialnumber = serialnumber


    def parse(self, sent_data):
        # sentence data should be a string in the form of "I love dogs.\nThey are cute.\nYeah!"
        sent_data = sent_data.replace('\r','\n').strip()
        assert '\n\n' not in sent_data
        
        # save single file in ACE format to acedir
        sent_data = sent_data.replace('\n', '\n0,1 person\n\n') + '\n0,1 person\n\n'
        
        with io.open(self.acedir + 'amalgum.test', 'w', encoding='utf8') as ftest:
            ftest.write(sent_data)
        
        # Pickle test data
        self.shibuyaentities.gen_data(dataset="amalgum")
        
        # predicts and outputs subtoks
        outputstr, _ = self.shibuyaentities.predict(dataset="amalgum", serialnumber=self.serialnumber)
        
        # convert subtoks to toks
        outputstr = self.shibuyaentities.subtok2tok(outputstr, sent_data)
        
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

