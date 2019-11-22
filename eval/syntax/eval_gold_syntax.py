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


def main():
    os.makedirs("tagged_fixed", exist_ok=True)
    os.makedirs("predicted", exist_ok=True)
    nlp = stanfordnlp.Pipeline(
        **{"processors": "depparse", "lang": "en", "depparse_pretagged": True}
    )
    for filepath in glob("tagged/*.conllu"):
        process(nlp, filepath)


if __name__ == "__main__":
    main()
