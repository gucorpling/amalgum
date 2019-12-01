"""
NOTE: you need to use the dev branch of stanfordnlp to run this
To do that:
    git clone https://github.com/stanfordnlp/stanfordnlp.git
    cd stanfordnlp
    git checkout dev
    pip install -e .
"""
import os
import sys
import re
from argparse import ArgumentParser

if not os.path.exists("./stanfordnlp"):
    print(
        """
    NOTE: you need to use the dev branch of stanfordnlp to run this.
    To do that:
        git clone https://github.com/stanfordnlp/stanfordnlp.git
        cd stanfordnlp
        git checkout dev
        pip install -e .
    """
    )
    sys.exit(1)
import stanfordnlp
from stanfordnlp.utils.conll import CoNLL
from glob import glob
import subprocess as sp


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


def process_gumby(nlp1, nlp2, filepath):
    conll_string = fix_conllu(filepath)
    print("Reading from " + filepath + "...")
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
    doc_with_our_xpos = stanfordnlp.Document(CoNLL.conll2dict(input_str=conll_string))
    replace_xpos(doc, doc_with_our_xpos)

    processed = nlp2(doc)
    d = processed.to_dict()
    CoNLL.dict2conll(d, "predicted/" + filepath.split("/")[-1])
    print("Wrote predictions to predicted/" + filepath.split("/")[-1])

    return processed


def concat(f_dir, out_path):
    conll_out = ""
    for filename in sorted(os.listdir(f_dir)):
        with open(f_dir + os.sep + filename, encoding="utf-8") as f:
            lines = f.read()
            conll_out += lines
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(conll_out)


def eval_gumby(config1, config2, model):
    os.makedirs("tagged_fixed", exist_ok=True)
    os.makedirs("predicted", exist_ok=True)
    nlp1 = stanfordnlp.Pipeline(**config1)
    nlp2 = stanfordnlp.Pipeline(**config2)
    for filepath in glob("tagged/*.conllu"):
        process_gumby(nlp1, nlp2, filepath)
    concat("predicted", "en_gumby-ud.pred.conllu")
    concat("gold", "en_gumby-ud.gold.conllu")
    print("GUMBY score using '" + model + "':")
    p = sp.Popen(
        [
            "python",
            os.sep.join(["stanfordnlp", "stanfordnlp", "utils", "conll18_ud_eval.py"]),
            "en_gumby-ud.gold.conllu",
            "en_gumby-ud.pred.conllu",
            "--verbose",
        ]
    )
    p.communicate()
    p.wait()


def process_gum(nlp1, nlp2, filepath):
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
    doc_with_our_xpos = stanfordnlp.Document(CoNLL.conll2dict(input_str=conll_string))
    replace_xpos(doc, doc_with_our_xpos)

    processed = nlp2(doc)
    d = processed.to_dict()
    CoNLL.dict2conll(d, "predicted/" + filepath.split("/")[-1])
    print("Wrote predictions to predicted/" + filepath.split("/")[-1])

    return processed


def eval_gum(config1, config2, model):
    os.makedirs("tagged_fixed", exist_ok=True)
    os.makedirs("predicted", exist_ok=True)
    nlp1 = stanfordnlp.Pipeline(**config1)
    nlp2 = stanfordnlp.Pipeline(**config2)
    process_gum(nlp1, nlp2, "en_gum-ud-test.conllu")
    print("GUM score using '" + model + "':")
    p = sp.Popen(
        [
            "python",
            os.sep.join(["stanfordnlp", "stanfordnlp", "utils", "conll18_ud_eval.py"]),
            "en_gum-ud-test.conllu",
            "predicted/en_gum-ud-test.conllu",
            "--verbose",
        ]
    )
    p.communicate()
    p.wait()


def download_models():
    os.makedirs("./models/", exist_ok=True)
    # stanfordnlp.download("en_gum", resource_dir="./models", confirm_if_exists=True)
    # stanfordnlp.download("en_ewt", resource_dir="./models", confirm_if_exists=True)


if __name__ == "__main__":
    download_models()
    ap = ArgumentParser()
    ap.add_argument("model_dirs", nargs="+")
    args = ap.parse_args()
    args.model_dirs = [
        dir if dir.endswith(os.sep) else dir + os.sep for dir in args.model_dirs
    ]

    for dir in args.model_dirs:
        corpus_name = "ewt" if "ewt" in dir else "gum"

        # before pos replacements
        config1 = {
            "lang": "en",
            "processors": "tokenize,pos,lemma",
            "pos_model_path": dir + f"en_{corpus_name}_tagger.pt",
            "lemma_model_path": dir + f"en_{corpus_name}_lemmatizer.pt",
            "tokenize_pretokenized": True,
        }
        # after pos replacements
        config2 = {
            "lang": "en",
            "processors": "depparse",
            "depparse_model_path": dir + f"en_{corpus_name}_parser.pt",
            "tokenize_pretokenized": True,
            "depparse_pretagged": True,
        }
        if corpus_name == "gum":
            config1["pos_pretrain_path"] = dir + f"en_{corpus_name}.pretrain.pt"
            config2["depparse_pretrain_path"] = dir + f"en_{corpus_name}.pretrain.pt"

        eval_gumby(config1, config2, dir)
        eval_gum(config1, config2, dir)
