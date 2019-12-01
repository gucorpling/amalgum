"""
NOTE: you need to use the dev branch of stanfordnlp to run this
To do that:
    git clone https://github.com/stanfordnlp/stanfordnlp.git
    cd stanfordnlp
    git checkout dev
    pip install -e .
"""
from glob import glob
import stanfordnlp
from stanfordnlp.utils.conll import CoNLL
import os
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


def process(nlp1, nlp2, filepath):
    conll_string = fix_conllu(filepath)
    print("Reading from " + filepath + "...")
    doc = CoNLL.conll2dict(input_str=conll_string)

    # get rid of xpos
    sents = []
    for sent in doc:
        words = []
        for word in sent:
            words.append(word["text"])
        sents.append(" ".join(words))

    # put it through first part of the pipeline
    doc = nlp1("\n".join(sents))

    # overwrite snlp's xpos with our xpos
    # doc_with_our_xpos = stanfordnlp.Document(CoNLL.conll2dict(input_str=conll_string))
    # for i, sent in enumerate(doc.sentences):
    #    our_sent = doc_with_our_xpos.sentences[i]
    #    for j, word in enumerate(sent.words):
    #        our_word = our_sent.words[j]
    #        word.xpos = our_word.xpos

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


def eval_gumby(config1, config2, corpus):
    os.makedirs("tagged_fixed", exist_ok=True)
    os.makedirs("predicted", exist_ok=True)
    nlp1 = stanfordnlp.Pipeline(**config1)
    nlp2 = stanfordnlp.Pipeline(**config2)
    for filepath in glob("tagged/*.conllu"):
        process(nlp1, nlp2, filepath)
    concat("predicted", "en_gumby-ud.pred.conllu")
    concat("gold", "en_gumby-ud.gold.conllu")
    print("GUMBY score using " + corpus + ":")
    p = sp.Popen(
        [
            "python",
            os.sep.join(["stanfordnlp", "stanfordnlp", "utils", "conll18_ud_eval.py"]),
            "en_gumby-ud.gold.conllu",
            "en_gumby-ud.pred.conllu",
        ]
    )
    p.communicate()
    p.wait()


def download_models():
    os.makedirs("./models/", exist_ok=True)
    stanfordnlp.download("en_gum", resource_dir="./models", confirm_if_exists=True)
    stanfordnlp.download("en_ewt", resource_dir="./models", confirm_if_exists=True)


if __name__ == "__main__":
    download_models()
    gum_config1 = {
        "lang": "en",
        "tokenize_pretokenized": True,
        "processors": "tokenize,pos,lemma",
        "pos_model_path": "./models/en_gum_models/en_gum_tagger.pt",
        "pos_pretrain_path": "./models/en_gum_models/en_gum.pretrain.pt",
        "lemma_model_path": "./models/en_gum_models/en_gum_lemmatizer.pt",
    }
    gum_config2 = {
        "lang": "en",
        "processors": "depparse",
        "tokenize_pretokenized": True,
        "depparse_model_path": "./models/en_gum_models/en_gum_parser.pt",
        "depparse_pretrain_path": "./models/en_gum_models/en_gum.pretrain.pt",
        "depparse_pretagged": True,
    }

    ewt_config1 = {
        "lang": "en",
        "tokenize_pretokenized": True,
        "processors": "tokenize,pos,lemma",
        "pos_model_path": "./models/en_ewt_models/en_ewt_tagger.pt",
        "lemma_model_path": "./models/en_ewt_models/en_ewt_lemmatizer.pt",
    }
    ewt_config2 = {
        "lang": "en",
        "processors": "depparse",
        "tokenize_pretokenized": True,
        "depparse_model_path": "./models/en_ewt_models/en_ewt_parser.pt",
        "depparse_pretagged": True,
    }

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
        import sys

        sys.exit(1)

    eval_gumby(gum_config1, gum_config2, "gum")
    eval_gumby(ewt_config1, ewt_config2, "ewt")
