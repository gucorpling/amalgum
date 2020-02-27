import io, os, re, sys
from glob import glob
import subprocess as sp
from nlp_modules.base import NLPModule, PipelineDep


def fix_conllu(pos, tokens):
    lines = tokens.split("\n")
    pos = pos.split("\n")
    assert len(pos) == len(lines)

    for i, line in enumerate(lines):
        if line.strip() != "":
            # if in conllu file
            line = line.strip().split["\t"]
            line[4] = pos[i].strip().split["\t"][1]
            lines[i] = "\t".join(line)

    out_string = "\n".join(lines)
    return out_string


def fix_upos(word):
    """If we replaced the xpos, we should try to fix the upos.
    Follows https://github.com/amir-zeldes/gum/blob/dev/_build/utils/upos.ini
    """

    def xpos_sub(xpos_pattern, upos):
        nonlocal word
        if re.match("^" + xpos_pattern + "$", word.xpos):
            word.upos = upos

    xpos_sub(r"JJ[RS]?", "ADJ")
    xpos_sub(r"WRB", "SCONJ")
    xpos_sub(r"UH", "INTJ")
    xpos_sub(r"CC", "CCONJ")
    xpos_sub(r"CD", "NUM")
    xpos_sub(r"NNS?", "NOUN")
    xpos_sub(r"NNPS?", "PROPN")
    xpos_sub(r"V.*", "VERB")
    xpos_sub(r"FW|LS", "X")
    xpos_sub(r"MD", "AUX")
    xpos_sub(r"SENT", "PUNCT")
    xpos_sub(r"POS", "PART")
    xpos_sub(r"\$", "SYM")
    xpos_sub(r"-[RL][SR]B-", "PUNCT")
    if word.text == "%":
        word.upos = "SYM"


def replace_xpos(doc, doc_with_our_xpos):
    for i, sent in enumerate(doc.sentences):
        our_sent = doc_with_our_xpos.sentences[i]
        for j, word in enumerate(sent.words):
            our_word = our_sent.words[j]
            if word.xpos != our_word.xpos:
                word.xpos = our_word.xpos
                fix_upos(word)


class DepWithPosParser(NLPModule):
    requires = (PipelineDep.S_SPLIT, PipelineDep.POS_TAG)
    provides = (PipelineDep.PARSE,)

    def __init__(self, model="gum"):
        self.LIB_DIR = config["LIB_DIR"]
        self.model = model

    def test_dependencies(self):
        import stanfordnlp
        from stanfordnlp.utils.conll import CoNLL

        self.model_dir = os.path.join(self.LIB_DIR, "parse-dependencies", "models")
        if len(glob(os.path.join(self.model_dir, "en_*.pt"))) == 0:
            # TODO: upload pretrained GUM models
            raise NLPDependencyException(
                "No pre-trained GUM stanfordnlp models. Please download the pretrained GUM models"
                f"from xxx and place it in {self.model_dir}/"
            )

    def predict_with_pos(self, doc_dict):
        conll_string = fix_conllu(doc_dict["pos"], doc_dict["dep"])
        doc = CoNLL.conll2dict(input_str=conll_string)

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
        # before pos replacements
        config1 = {
            "lang": "en",
            "processors": "tokenize,pos,lemma",
            "pos_model_path": self.model_dir + f"en_{self.model}_tagger.pt",
            "lemma_model_path": self.model_dir + f"en_{self.model}_lemmatizer.pt",
            "tokenize_pretokenized": True,
        }

        # after pos replacements
        config2 = {
            "lang": "en",
            "processors": "depparse",
            "depparse_model_path": self.model_dir + f"en_{self.model}_parser.pt",
            "tokenize_pretokenized": True,
            "depparse_pretagged": True,
        }

        if self.model == "gum":
            config1["pos_pretrain_path"] = (
                self.model_dir + f"en_{self.model}.pretrain.pt"
            )
            config2["depparse_pretrain_path"] = (
                self.model_dir + f"en_{self.model}.pretrain.pt"
            )
        self.nlp1 = stanfordnlp.Pipeline(**config1)
        self.nlp2 = stanfordnlp.Pipeline(**config2)

        # Identify a function that takes data and returns output at the document level
        processing_function = self.predict_with_pos

        # use process_files, inherited from NLPModule, to apply this function to all docs
        self.process_files_multiformat(input_dir, output_dir, processing_function)
