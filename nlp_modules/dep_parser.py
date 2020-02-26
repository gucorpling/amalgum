import io, os, re, sys
import stanfordnlp
from stanfordnlp.utils.conll import CoNLL
from glob import glob
import subprocess as sp
from nlp_modules.base import NLPModule, PipelineDep


def fix_conllu(filepath):
    sentences = []
    token_list = None
    with open(filepath, "r") as f:
        s = f.read()
    for line in s.split("\n"):
        if line.strip() == "" and token_list is not None:
            sentences.append(token_list)
            token_list = None
        elif line.strip() != "":
            if token_list is None:
                token_list = []
            tok, xpos = line.strip().split("\t")
            token_list.append((tok, xpos))

    outstring = ""
    for sentence in sentences:
        for i, (tok, xpos) in enumerate(sentence):
            outstring += f"{i+1}\t{tok}\t_\t_\t{xpos}\t_\t_\t_\t_\t_\n"
        outstring += "\n"

    with open(filepath.replace("tagged", "tagged_fixed"), "w") as f:
        f.write(outstring)

    return outstring


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
    requires = PipelineDep.POS_TAG
    provides = PipelineDep.PARSE

    def __init__(self, model):
        """
		the argument "model" is required to indicate which model (gum or ewt) to use
		"""
        self.model = model

    def test_dependencies(self):
        if not os.path.exists("parse-dependencies"):
            raise NLPDependencyException("The folder 'parse-dependencies' is missing.")

    def process_amalgum(self, nlp1, nlp2, filepath):
        # process file first
        conll_string = fix_conllu(filepath)
        sys.stdout.write("Reading from " + filepath + "...")
        doc = CoNLL.conll2dict(input_str=conll_string)

        # get just the text
        sents = []
        for sent in doc:
            words = []
            for word in sent:
                words.append(word["text"])
            sents.append(" ".join(words))

        # put it through first part of the pipeline
        doc = nlp1("\n".join(sents))

        # overwrite snlp's xpos with our xpos
        doc_with_our_xpos = stanfordnlp.Document(
            CoNLL.conll2dict(input_str=conll_string)
        )
        replace_xpos(doc, doc_with_our_xpos)

        processed = nlp2(doc)
        d = processed.to_dict()

        # write into a conllu file under output_dir
        CoNLL.dict2conll(d, self.output_dir + os.sep + filepath.split("/")[-1])
        sys.stdout.write(
            f"Wrote predictions to {self.output_dir}/" + filepath.split("/")[-1]
        )

        return processed

    def concat(self, f_dir, out_path):
        conll_out = ""
        for filename in sorted(os.listdir(f_dir)):
            with open(f_dir + os.sep + filename, encoding="utf-8") as f:
                lines = f.read()
                conll_out += lines
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(conll_out)

    def eval_amalgum(self, config1, config2):
        os.makedirs("parse-dependencies" + os.sep + "tagged_fixed", exist_ok=True)
        os.makedirs("parse-dependencies" + os.sep + "conllu", exist_ok=True)
        nlp1 = stanfordnlp.Pipeline(**config1)
        nlp2 = stanfordnlp.Pipeline(**config2)

        for filepath in glob(self.input_dir + os.sep + "conllu/*.conllu"):
            self.process_amalgum(nlp1, nlp2, filepath)

        # concatenate files for further evaluation
        self.concat("parse-dependencies" + os.sep + "conllu", "en_gumby-ud.pred.conllu")
        self.concat("parse-dependencies" + os.sep + "gold", "en_gumby-ud.gold.conllu")
        sys.stdout.write("GUMBY score using '" + self.model + "':")

        # TODO: stanfordnlp filepath
        p = sp.Popen(
            [
                "python",
                os.sep.join(
                    [
                        "parse-dependencies",
                        "stanfordnlp",
                        "stanfordnlp",
                        "utils",
                        "conll18_ud_eval.py",
                    ]
                ),
                "parse-dependencies" + os.sep + "en_gumby-ud.gold.conllu",
                "parse-dependencies" + os.sep + "en_gumby-ud.pred.conllu",
                "--verbose",
            ]
        )
        p.communicate()
        p.wait()

    def process_gum(self, nlp1, nlp2, filepath):
        with open(filepath, "r") as f:
            conll_string = f.read()
        doc = CoNLL.conll2dict(input_str=conll_string)

        # get just the text
        sents = []
        for sent in doc:
            words = []
            for word in sent:
                words.append(word["text"])
            sents.append(" ".join(words))

        # put it through first part of the pipeline
        doc = nlp1("\n".join(sents))

        # overwrite snlp's xpos with our xpos
        doc_with_our_xpos = stanfordnlp.Document(
            CoNLL.conll2dict(input_str=conll_string)
        )
        replace_xpos(doc, doc_with_our_xpos)

        processed = nlp2(doc)
        d = processed.to_dict()

        # warning: do not pass the predicted results to the next module
        CoNLL.dict2conll(
            d,
            "parse-dependencies"
            + os.sep
            + "gum_predicted"
            + os.sep
            + filepath.split("/")[-1],
        )
        sys.stdout.write(
            "Wrote predictions to parse-dependencies/gum_predicted/"
            + filepath.split("/")[-1]
        )

        return processed

    def eval_gum(self, config1, config2):
        os.makedirs("parse-dependencies" + os.sep + "tagged_fixed", exist_ok=True)
        os.makedirs("parse-dependencies" + os.sep + "gum_predicted", exist_ok=True)
        nlp1 = stanfordnlp.Pipeline(**config1)
        nlp2 = stanfordnlp.Pipeline(**config2)
        self.process_gum(nlp1, nlp2, "en_gum-ud-test.conllu")
        sys.stdout.write("GUM score using '" + self.model + "':")
        p = sp.Popen(
            [
                "python",
                os.sep.join(
                    [
                        "parse-dependencies",
                        "stanfordnlp",
                        "stanfordnlp",
                        "utils",
                        "conll18_ud_eval.py",
                    ]
                ),
                "parse-dependencies" + os.sep + "en_gum-ud-test.conllu",
                "parse-dependencies"
                + os.sep
                + "gum_predicted"
                + os.sep
                + "en_gum-ud-test.conllu",
                "--verbose",
            ]
        )
        p.communicate()
        p.wait()

    def run(self, input_dir, output_dir):
        model_dir = "parse-dependencies" + os.sep + "models"
        self.input_dir = input_dir
        self.output_dir = output_dir

        # before pos replacements
        config1 = {
            "lang": "en",
            "processors": "tokenize,pos,lemma",
            "pos_model_path": model_dir + f"en_{self.model}_tagger.pt",
            "lemma_model_path": model_dir + f"en_{self.model}_lemmatizer.pt",
            "tokenize_pretokenized": True,
        }

        # after pos replacements
        config2 = {
            "lang": "en",
            "processors": "depparse",
            "depparse_model_path": model_dir + f"en_{self.model}_parser.pt",
            "tokenize_pretokenized": True,
            "depparse_pretagged": True,
        }
        if self.model == "gum":
            config1["pos_pretrain_path"] = model_dir + f"en_{self.model}.pretrain.pt"
            config2["depparse_pretrain_path"] = (
                model_dir + f"en_{self.model}.pretrain.pt"
            )

        self.eval_amalgum(config1, config2)
        self.eval_gum(config1, config2)

        sys.stdout.write("Completed: Module stanfordnlp parser")
