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


def process(nlp, filepath):
    conll_string = fix_conllu(filepath)
    doc = stanfordnlp.Document(CoNLL.conll2dict(input_str=conll_string))

    processed = nlp(doc)
    CoNLL.dict2conll(processed.to_dict(), "predicted/" + filepath.split("/")[-1])

    return processed


def concat(f_dir, out_path):
    conll_out = ""
    for filename in os.listdir(f_dir):
        with open("predicted" + os.sep + filename, encoding="utf-8") as f:
            lines = f.read()
            conll_out += lines
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(conll_out)


def main(config):
    os.makedirs("tagged_fixed", exist_ok=True)
    os.makedirs("predicted", exist_ok=True)
    nlp = stanfordnlp.Pipeline(**config)
    for filepath in glob("tagged/*.conllu"):
        process(nlp, filepath)
    concat("predicted", "en_gumby-ud.pred.conllu")


if __name__ == "__main__":
    config = {
        "processors": "depparse",
        "lang": "en",
        "treebank": "en_gum",
        # 'treebank': 'en_ewt',
        "tokenize_pretokenized": True,
        "pos_batch_size": 1000,
        # 'pos_model_path': './stanfordnlp/saved_models/pos/en_gum_tagger.pt',
        # 'pos_pretrain_path': './stanfordnlp/saved_models/depparse/en_gum.pretrain.pt',
        # 'lemma_model_path': './stanfordnlp/saved_models/lemma/en_gum_lemmatizer.pt',
        # 'depparse_model_path': './stanfordnlp/saved_models/depparse/en_gum_parser.pt',
        # 'depparse_pretrain_path': './stanfordnlp/saved_models/depparse/en_gum.pretrain.pt',
        # 'pos_model_path': './stanfordnlp/en_ewt_models/en_ewt_tagger.pt',
        # 'pos_pretrain_path': './stanfordnlp/en_ewt_models/en_ewt.pretrain.pt',
        # 'lemma_model_path': './stanfordnlp/en_ewt_models/en_ewt_lemmatizer.pt',
        "depparse_model_path": "./stanfordnlp/en_ewt_models/en_ewt_parser.pt",
        "depparse_pretrain_path": "./stanfordnlp/en_ewt_models/en_ewt.pretrain.pt",
        "depparse_pretagged": True,
    }

    main(config)
