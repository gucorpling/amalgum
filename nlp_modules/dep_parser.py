import io, os, re, sys
from glob import glob
import subprocess as sp
from nlp_modules.base import NLPModule, PipelineDep


def replace_xpos(doc, doc_with_our_xpos):
    for i, sent in enumerate(doc.sentences):
        our_sent = doc_with_our_xpos.sentences[i]
        for j, word in enumerate(sent.words):
            our_word = our_sent.words[j]
            if word.xpos != our_word.xpos:
                word.xpos = our_word.xpos


class DepWithPosParser(NLPModule):
    requires = PipelineDep.POS_TAG
    provides = (PipelineDep.PARSE,)

    def __init__(self, model="gum"):
        self.LIB_DIR = config["LIB_DIR"]
        self.model = model

    def test_dependencies(self):
        import stanfordnlp
        from stanfordnlp.utils.conll import CoNLL

        self.model_dir = os.path.join(self.LIB_DIR, "dep_parsing", "models")
        if len(glob(os.path.join(self.model_dir, "en_*.pt"))) == 0:
            # TODO: upload pretrained GUM models
            raise NLPDependencyException(
                "No pre-trained GUM stanfordnlp models. Please download the pretrained GUM models"
                f"from xxx and place it in {self.model_dir}/"
            )

    def predict_with_pos(self, conllu_data: str):
        # before pos replacements
        config1 = {
            "lang": "en",
            "processors": "tokenize,pos,lemma",
            "pos_model_path": self.model_dir + os.sep + f"en_{self.model}_tagger.pt",
            "lemma_model_path": self.model_dir
            + os.sep
            + f"en_{self.model}_lemmatizer.pt",
            "pos_pretrain_path": self.model_dir
            + os.sep
            + f"en_{self.model}.pretrain.pt",
            "tokenize_pretokenized": True,
        }

        # after pos replacements
        config2 = {
            "lang": "en",
            "processors": "depparse",
            "depparse_model_path": self.model_dir
            + os.sep
            + f"en_{self.model}_parser.pt",
            "depparse_pretrain_path": self.model_dir
            + os.sep
            + f"en_{self.model}.pretrain.pt",
            "tokenize_pretokenized": True,
            "depparse_pretagged": True,
        }

        self.nlp1 = stanfordnlp.Pipeline(**config1)
        self.nlp2 = stanfordnlp.Pipeline(**config2)

        doc = CoNLL.conll2dict(input_str=conllu_data)

        # get just the text
        sents = []
        for sent in doc:
            words = []
            for word in sent:
                words.append(word["text"])
            sents.append(" ".join(words))

        # put it through first part of the pipeline
        doc = self.nlp1("\n".join(sents))

        # overwrite snlp's xpos with our xpos
        doc_with_our_xpos = stanfordnlp.Document(
            CoNLL.conll2dict(input_str=conll_string)
        )
        replace_xpos(doc, doc_with_our_xpos)

        parsed = self.nlp2(doc)
        return parsed

    def run(self, input_dir, output_dir):
        # Identify a function that takes data and returns output at the document level
        processing_function = self.predict_with_pos

        # use process_files, inherited from NLPModule, to apply this function to all docs
        self.process_files(input_dir, output_dir, processing_function, file_type="dep")


if __name__ == "__main__":
    x = open("temp.conllu").read()
    input_dir = x
    output_dir = 1
    module = DepWithPosParser()
    module.test_dependencies()
    module.predict_with_pos(x)
