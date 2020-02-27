import os
import re

from glob import glob
from nlp_modules.base import NLPModule, PipelineDep
import conllu
from collections import OrderedDict


def is_sgml_tag(line):
    return line.startswith("<") and line.endswith(">")


def tokens2conllu(tokens):
    tokens = [
        OrderedDict(
            (k, v)
            for k, v in zip(
                conllu.parser.DEFAULT_FIELDS,
                [i + 1, token]
                + ["_" for i in range(len(conllu.parser.DEFAULT_FIELDS) - 1)],
            )
        )
        for i, token in enumerate(tokens)
    ]
    tl = conllu.TokenList(tokens)
    return tl


class GumdropSplitter(NLPModule):
    requires = (PipelineDep.TOKENIZE,)
    provides = (PipelineDep.S_SPLIT, PipelineDep.S_TYPE)

    def __init__(self, config):
        self.LIB_DIR = config["LIB_DIR"]
        self.test_dependencies()

    def test_dependencies(self):
        import tensorflow
        import keras
        import rfpimp
        import xgboost
        import nltk
        import pandas
        import scipy
        import torch
        import hyperopt
        import wget

        if tensorflow.version.VERSION[0] != "1":
            raise Exception("TensorFlow version 1.x must be used with GumdropSplitter")
        model_dir = os.path.join(self.LIB_DIR, "gumdrop", "lib", "udpipe")
        if len(glob(os.path.join(model_dir, "english-*.udpipe"))) == 0:
            raise Exception(
                "No English udpipe model found. Please download the an English udpipe model from "
                "from https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3131"
                f" and place it in {model_dir}/."
            )

    def split(self, context):
        xml_data = context["xml"]
        genre = re.findall(r'type="(.*?)"', xml_data.split("\n")[0])
        assert len(genre) == 1
        genre = genre[0]
        # don't feed the sentencer our pos and lemma predictions, if we have them
        no_pos_lemma = re.sub(r"([^\n\t]*?)\t[^\n\t]*?\t[^\n\t]*?\n", r"\1\n", xml_data)
        split_indices = self.sentencer.predict(
            no_pos_lemma, as_text=True, plain=True, genre=genre
        )

        # for xml
        counter = 0
        splitted = []
        opened_sent = False
        para = True

        # for conllu
        conllu_sentences = []
        tokens = []

        for line in xml_data.strip().split("\n"):
            if not is_sgml_tag(line):
                # Token
                if split_indices[counter] == 1 or para:
                    if opened_sent:
                        rev_counter = len(splitted) - 1
                        while is_sgml_tag(splitted[rev_counter]):
                            rev_counter -= 1
                        splitted.insert(rev_counter + 1, "</s>")
                        conllu_sentences.append(tokens2conllu(tokens))
                    splitted.append("<s>")
                    tokens = []
                    opened_sent = True
                    para = False
                counter += 1
            elif (
                "<p>" in line or "<head>" in line or "<caption>" in line
            ):  # New block, force sentence split
                para = True
            splitted.append(line)

        splitted = "\n".join(splitted)
        if opened_sent:
            if splitted.endswith("</text>"):
                splitted = splitted.replace("</text>", "</s>\n</text>")
            else:
                splitted += "\n</s>"

        return {
            "xml": splitted,
            "dep": "\n".join(tl.serialize() for tl in conllu_sentences),
        }

    def run(self, input_dir, output_dir):
        from lib.gumdrop.EnsembleSentencer import EnsembleSentencer

        self.sentencer = EnsembleSentencer(
            lang="eng", model="eng.rst.gum", genre_pat="_([^_]+)_"
        )

        # Identify a function that takes data and returns output at the document level
        processing_function = self.split

        # use process_files, inherited from NLPModule, to apply this function to all docs
        self.process_files_multiformat(
            input_dir, output_dir, processing_function, multithreaded=False
        )
