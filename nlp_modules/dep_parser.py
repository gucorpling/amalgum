import io, os, sys
from glob import glob
import stanfordnlp
from lib.dep_parsing.conll import CoNLL
from nlp_modules.base import NLPModule, PipelineDep, NLPDependencyException


def replace_xpos(doc, doc_with_our_xpos):
    for i, sent in enumerate(doc.sentences):
        our_sent = doc_with_our_xpos[i]
        for j, word in enumerate(sent.words):
            if word.xpos != our_sent[j]["xpos"]:
                word.xpos = our_sent[j]["xpos"]


def conllu2xml(conllu, xml):
    xml_lines = xml.split("\n")
    s_count = -1
    tok_count = -1
    for i, line in enumerate(xml_lines):
        line = line.strip()
        if line:
            # match xml sent with conllu sent
            if line.startswith("<s "):
                s_count += 1
                tok_count = 0
                continue
            if line.startswith("<"):
                continue

            sentence = conllu.sentences[s_count]
            conllu_line = sentence.words[tok_count]
            xml_lines[i] = (
                conllu_line.text
                + "\t"
                + conllu_line.xpos
                + "\t"
                + conllu_line.lemma if conllu_line.lemma is not None else conllu_line.text
                + "\t"
                + conllu_line.dependency_relation
            )
            tok_count += 1

    return "\n".join(xml_lines)


class DepParser(NLPModule):
    requires = (PipelineDep.S_SPLIT, PipelineDep.POS_TAG)
    provides = (PipelineDep.PARSE,)

    def __init__(self, config, model="gum"):
        self.use_gpu = config.get("use_gpu", False)
        self.LIB_DIR = config["LIB_DIR"]
        self.model_dir = os.path.join(self.LIB_DIR, "dep_parsing", "models")
        self.model = model
        # before pos replacements
        config1 = {
            "lang": "en",
            "treebank": "en_gum",
            "use_gpu": self.use_gpu,
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
            "treebank": "en_gum",
            "use_gpu": self.use_gpu,
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

    def test_dependencies(self):
        if not os.path.isdir(os.getcwd() + os.sep + "stanfordnlp"):
            raise NLPDependencyException("Download stanfordnlp from https://github.com/stanfordnlp/stanfordnlp.git")

        if len(glob(os.path.join(self.LIB_DIR, "dep_parsing", "*.py"))) == 0:
            raise NLPDependencyException("No stanfordnlp dependencies. Please download the files from"
                                         "https://drive.google.com/open?id=1MAWXSUDCYZSmVcoFkDGFt0ARlkK5vq00. "
                                         "Put core.py and depparse_processor.py under stanfordnlp/pipeline/ "
                                         "and overwrite the two scripts.")

        if len(glob(os.path.join(self.model_dir, "en_*.pt"))) == 0:
            raise NLPDependencyException(
                "No pre-trained GUM stanfordnlp models. Please download the pretrained GUM models"
                "from https://drive.google.com/open?id=1s5DRHHGqpnlCQ6UK95GbAxexXIh6mFzr"
                f" and place it in {self.model_dir}/"
            )

    def predict_with_pos(self, doc_dict):
        conllu_data = doc_dict["dep"]
        xml_data = doc_dict["xml"]

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
        doc_with_our_xpos = CoNLL.conll2dict(input_str=conllu_data)
        replace_xpos(doc, doc_with_our_xpos)

        parsed = self.nlp2(doc)
        deped = parsed.conll_file.conll_as_string().strip()

        xmled = conllu2xml(parsed, xml_data)

        return {"dep": deped, "xml": xmled}

    def run(self, input_dir, output_dir):
        # Identify a function that takes data and returns output at the document level
        processing_function = self.predict_with_pos

        # use process_files, inherited from NLPModule, to apply this function to all docs
        self.process_files_multiformat(input_dir, output_dir, processing_function)


def test_main():
    test_xml = """<text id="GUM_academic_art" author="Claire Bailey-Ross, Andrew Beresford, Daniel Smith, Claire Warwick" dateCollected="2017-09-13" dateCreated="2017-08-08" dateModified="2017-09-13" shortTitle="art" sourceURL="https://dh2017.adho.org/abstracts/333/333.pdf" speakerCount="0" speakerList="none" title="Aesthetic Appreciation and Spanish Art: Insights from Eye-Tracking" type="academic">
<head>
<hi rend="bold blue">
<s>
Aesthetic	JJ	aesthetic
Appreciation	NN	appreciation
and	CC	and
Spanish	JJ	Spanish
Art	NN	art
:	:	:
</s>
<s>
Insights	NNS	insight
from	IN	from
Eye-Tracking	NN	eye-tracking
</s>
</hi>
</head>"""

    test_conll = """1	Aesthetic	_	_	JJ	_	_	_	_	_
2	Appreciation	_	_	NN	_	_	_	_	_
3	and	_	_	CC	_	_	_	_	_
4	Spanish	_	_	JJ	_	_	_	_	_
5	Art	_	_	NN	_	_	_	_	_
6	:	_	_	:	_	_	_	_	_

1	Insights	_	_	NNS	_	_	_	_	_
2	from	_	_	IN	_	_	_	_	_
3	Eye-Tracking	_	_	NN	_	_	_	_	_
"""
    module = DepParser({'LIB_DIR': 'lib'})
    module.test_dependencies()
    res = module.predict_with_pos({"xml": test_xml, "dep": test_conll})
    print(res["xml"])
    print(res["dep"])


if __name__ == "__main__":
    test_main()
