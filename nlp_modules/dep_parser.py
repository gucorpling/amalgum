import io, os, sys, re
from glob import glob
import stanza
from stanza.models.common.doc import Document
from lib.dep_parsing.conll import CoNLL
from nlp_modules.base import NLPModule, PipelineDep, NLPDependencyException
import torch, flair
from depedit import DepEdit
from flair.data import Sentence
from flair.models import SequenceTagger
from udapi.core.document import Document as UdapiDocument
from udapi.block.ud.fixpunct import FixPunct

#flair.device = torch.device('cpu')  # Uncomment to use CPU for flair

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
nlp_modules_dir = script_dir + ".." + os.sep + "nlp_modules" + os.sep
parser_dep_dir = nlp_modules_dir + "parser-dependencies" + os.sep
gum_root = "GUM" + os.sep

depedit1 = DepEdit(config_file=parser_dep_dir+"postprocess_parser.ini")
depedit2 = DepEdit(config_file=parser_dep_dir+"upos.ini")
depedit3 = DepEdit(config_file=parser_dep_dir+"eng_morph_enhance_no_stype.ini")

vocab = set(io.open(parser_dep_dir + "eng_vocab.tab",encoding="utf8").read().strip().split("\n"))

def fix_punct(conllu_string):
    conllu_string = re.sub(r"\t'\t([^\t\n]+\tPART\tPOS)", r'\t&udapi_apos;\t\1', conllu_string, flags=re.MULTILINE)
    conllu_string = re.sub(r'\t"\t([^\t\n]+\t[^\t\n]+\t[^\t\n]+\t[^\t\n]+\t[^\t\n]+\t[^p])', r'\t&udapi_quot;\t\1', conllu_string, flags=re.MULTILINE)
    doc = UdapiDocument()
    doc.from_conllu_string(conllu_string)
    fixpunct_block = FixPunct()
    fixpunct_block.process_document(doc)
    output_string = doc.to_conllu_string()
    output_string = output_string.replace('&udapi_apos;', "'").replace('&udapi_quot;', '"')
    output_string = re.sub(r'# sent_id = [0-9]+\n', r'', output_string)  # remove udapi sent_id
    return output_string


def fix_lemma(word, pos, lemma):
    non_lemmas = {"them":"they", "me":"I", "him":"he", "n't":"not",'vlogg':"vlog","whom":"who","worshippe":"worship"}
    non_lemma_combos = {("PRP", "her"): "she", ("MD", "wo"): "will", ("PRP", "us"):"we", ("DT", "an"):"a",
                        ("POS","be"):"'s", ("POS","have"):"'s"}
    non_cap_lemmas = ["There", "How", "Why", "Where", "When"]
    num_lemmas = {"two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9",
                  "ten":"10","eleven":"11","dozen":"12", "thirteen":"13", "fourteen":"14", "fifteen":"15",
                  "sixteen":"16", "seventeen":"17", "eighteen":"18", "nineteen":"19", "twenty":"20",
                  "thirty":"30", "forty":"40", "fourty":"40", "fifty":"50", "sixty":"60", "seventy":"70",
                  "eighty":"80", "ninety":"90", "hundred":"100", "thousand":"1000", "million":"1000000",
                  "billion":"1000000000","trillion":"1000000000000"}
    false_non_e = {"pleas","tun"}  # e.g. pleas is a word, but not the lemma of pleased, tun != tune(d)

    if lemma in non_cap_lemmas:
        lemma = lemma.lower()

    if (pos,lemma) in non_lemma_combos:
        lemma = non_lemma_combos[(pos,lemma)]

    if lemma in non_lemmas:
        lemma = non_lemmas[lemma]

    if pos == "NN" and word.endswith("ing"):
        if not lemma.endswith("ing"):
            lemma = word

    if pos == "VBG" and word.endswith("ing") and not word.endswith("inging"):
        if lemma.endswith("ing"):
            lemma = word.replace("ing","")
            if lemma not in vocab:
                if lemma + "e" in vocab:
                    lemma = lemma + "e"
        elif lemma not in vocab and lemma + "e" in vocab:
            lemma += "e"
        elif lemma not in vocab and lemma.endswith("e"):
            if lemma[:-1] in vocab:
                lemma = lemma[:-1]

    if pos in ["VBN","VBD"] and word.endswith("ed") and not word.endswith("eded"):
        if lemma.endswith("ed"):
            lemma = word.replace("ed","")
            if lemma not in vocab or lemma in false_non_e:
                if lemma + "e" in vocab:
                    lemma = lemma + "e"
        elif lemma.endswith("e"):
            if lemma not in vocab and lemma[:-1] in vocab:
                lemma = lemma[:-1]
        elif lemma not in vocab:
            if lemma + "e" in vocab:
                lemma += "e"

    if pos == "JJ" and word.endswith("ed"):
        if not lemma.endswith("ed"):
            lemma = word

    if pos =="CD" and word.lower() in num_lemmas:
        lemma = num_lemmas[word.lower()]

    if pos == "NNS" and word.endswith("sses"):  # witnesses:witnesse -> witness
        if lemma.endswith("sse"):
            if lemma not in vocab and lemma[:-1] in vocab:
                lemma = lemma[:-1]

    lemma = lemma.replace('"',"''")  # standard quotes, and also good for ethercalc pastability

    return lemma


def postprocess_lemmas(conllu):
    output = []
    for line in conllu.split("\n"):
        if "\t" in line:
            fields = line.split("\t")
            fields[2] = fix_lemma(fields[1], fields[4], fields[2])
            line = "\t".join(fields)
        output.append(line)
    return "\n".join(output)


def replace_xpos(doc, doc_with_our_xpos):
    for i, sent in enumerate(doc):
        our_sent = doc_with_our_xpos[i]
        for j, word in enumerate(sent):
            word[4] = our_sent[j]["xpos"]
            word[3] = our_sent[j]["xpos"]


def add_second_deps(doc, ewt_parse):
    output = []
    parse2 = [l.split("\t") for l in ewt_parse.split("\n")]
    for l, line in enumerate(doc.split("\n")):
        if "\t" in line:
            fields = line.split("\t")
            fields[8] = parse2[l][6] + ":" + parse2[l][7]
            line = "\t".join(fields)
        output.append(line)
    return "\n".join(output)

def conllu2xml(conllu, xml):
    def escape_xml(data):
        data = data.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        return data

    xml_lines = xml.split("\n")
    toknum = 0
    words = [l.split("\t") for l in conllu.split("\n") if "\t" in l]
    words = [[escape_xml(w[1]),w[4],escape_xml(w[2])] for w in words if not "-" in w[0]]
    for i, line in enumerate(xml_lines):
        line = line.strip()
        if line:
            if line.startswith("<s ") or line.startswith("<s>"):
                continue
            if line.startswith("<") and line.endswith(">"):
                continue
            xml_lines[i] = "\t".join(words[toknum])
            toknum += 1

    return "\n".join(xml_lines)


def diaparse(parser, conllu):
    sents = conllu.strip().split("\n\n")
    parser_input = []

    for sent in sents:
        words = [(l.split("\t")[0],l.split("\t")[1]) for l in sent.split("\n") if "\t" in l]
        words = [w[1] for w in words if "." not in w[0] and "-" not in w[0]]
        parser_input.append(words)

    dataset = parser.predict(parser_input, prob=True)

    out_parses = []
    for sent in dataset.sentences:
        out_parses.append(str(sent).strip())

    return "\n\n".join(out_parses) + "\n\n"


def add_sequence_tagger_preds(tagger, doc):
    sents = [s.split("\n") for s in doc.strip().split("\n\n")]
    flair_sents = []
    for sent in sents:
        words = [l.split("\t")[1] for l in sent if "\t" in l]
        tags = [l.split("\t")[4] for l in sent if "\t" in l]
        flair_sents.append(Sentence(" ".join(words), use_tokenizer=lambda x: x.split(" ")))
        for i, word in enumerate(flair_sents[-1]):
            word.add_tag("xpos", tags[i])
    tagger.predict(flair_sents)
    pred_deprels = []
    for s in flair_sents:
        for t in s.tokens:
            pred_deprels.append(t.labels[1].value)
    output = []
    toknum = 0
    for line in doc.split("\n"):
        if "\t" in line:
            fields = line.split("\t")
            fields[5] = pred_deprels[toknum]
            toknum += 1
            line = "\t".join(fields)
        output.append(line)
    return "\n".join(output)


class DepParser(NLPModule):
    requires = (PipelineDep.S_SPLIT, PipelineDep.POS_TAG)
    provides = (PipelineDep.PARSE,)

    def __init__(self, config, model="gum"):
        self.use_gpu = config.get("use_gpu", False)
        self.LIB_DIR = config["LIB_DIR"]
        self.model_dir = parser_dep_dir
        self.model = model

        self.download_models()

        # before pos replacements
        config1 = {
            "lang": "en",
            "treebank": "en_gum",
            "use_gpu": self.use_gpu,
            "processors": "tokenize,pos,lemma",
            "pos_model_path": self.model_dir + f"en_{self.model}_tagger.pt",
            "lemma_model_path": self.model_dir + f"en_{self.model}_lemmatizer.pt",
            "pos_pretrain_path": self.model_dir + f"en_{self.model}.pretrain.pt",
            "tokenize_pretokenized": True,
        }

        # after pos replacements
        config2 = {
            "lang": "en",
            #"treebank": "en_gum",
            "package":"gum",
            "use_gpu": self.use_gpu,
            "processors": "lemma",
            #"depparse_model_path": self.model_dir + f"en_{self.model}_parser.pt",
            #"depparse_pretrain_path": self.model_dir + f"en_{self.model}.pretrain.pt",
            "lemma_model_path": self.model_dir + f"en_{self.model}_lemmatizer.pt",
            "tokenize_pretokenized": True,
            "depparse_pretagged": True,
            "lemma_pretagged":True
        }

        from diaparser.parsers import Parser
        parser = Parser.load(self.model_dir + "en_gum.electra-base"+os.sep+"model")
        self.nlp1 = parser
        ewt_parser = Parser.load(self.model_dir + "en_ewt.distilbert"+os.sep+"model")
        self.aux_parser = ewt_parser
        self.nlp2 = stanza.Pipeline(**config2)
        self.sequence_tagger = SequenceTagger.load(self.model_dir + "best-model_deprel_xpos_gum7.pt")

    # Separating the model download from the other dependencies
    def download_models(self):
        if len(glob(os.path.join(self.model_dir, "en_*.pt"))) < 5:
            models = [
                "en_gum.pretrain.pt",
                "en_gum_lemmatizer.pt",
                "en_gum_parser.pt",
                "en_gum_tagger.pt",
                "en_gum_tokenizer.pt",
            ]
            for model in models:
                if not os.path.exists(parser_dep_dir + model):
                    self.download_file(model, parser_dep_dir, subfolder="dep")

    def test_dependencies(self):
        if len(glob(os.path.join(self.model_dir, "en_*.pt"))) < 5:
            raise NLPDependencyException(
                "No pre-trained GUM stanza models. Please download the pretrained GUM models from corpling"
                f" and place them in {self.model_dir}/"
            )

    def predict_with_pos(self, doc_dict):
        # fix potential memory leak
        torch.cuda.empty_cache()

        conllu_data = doc_dict["dep"]
        xml_data = doc_dict["xml"]

        conllu_data = re.sub(r'\n[0-9]+-[^\n]+\n','\n',conllu_data)  # Remove any super tokens in input, we'll add them at the end

        # First parse - just get best deprel and heads
        diaparsed = diaparse(self.nlp1, conllu_data)
        doc = CoNLL.load_conll(io.StringIO(diaparsed))

        # overwrite xpos with our ensemble xpos
        doc_with_our_xpos = CoNLL.conll2dict(input_str=conllu_data)
        replace_xpos(doc, doc_with_our_xpos)

        doc = [["\t".join(l) for l in sent] for sent in doc]
        doc = "\n\n".join(["\n".join(sent) for sent in doc])

        # Second parse - postprocess based on:
        # 1. auxiliary parser predictions trained on EWT for PP attachment disambiguation
        ewt_parse = diaparse(self.aux_parser, conllu_data)
        doc = add_second_deps(doc, ewt_parse)
        # 2. sequence tagger deprel predictions using high quality POS tags and embeddings
        doc = add_sequence_tagger_preds(self.sequence_tagger, doc)
        # 3. postprocessing rules to adjudicate these predictions in a harmonized way
        doc = depedit1.run_depedit(doc)

        # Add upos
        uposed = depedit2.run_depedit(doc)
        uposed = [[l.split("\t") for l in s.split("\n")] for s in uposed.strip().split("\n\n")]
        dicts = CoNLL.convert_conll(uposed)

        # Now add lemmas using Stanza based on pretagged predicted upos (converted from our predicted xpos)
        for sent in dicts:
            for tok in sent:
                tok["id"] = int(tok["id"])
        doc = Document(dicts)
        lemmatized  = self.nlp2(doc)
        output = []
        for sent in lemmatized.sentences:
            for tok in sent.tokens:
                word = tok.words[0]
                row = [str(word.id),word.text,word.lemma,word.upos,word.xpos,'_',str(word.head),word.deprel,"_","_"]
                output.append("\t".join(row))
            output.append("")
        lemmatized = "\n".join(output)

        # Postprocess implausible lemmas (VBG ending in -ed, VBN ending in -ing, etc., incorrect -e restoration...)
        lemmatized = postprocess_lemmas(lemmatized)

        # Fix punctuation
        lemmatized = fix_punct(lemmatized)

        if "<text id=" in xml_data:
            docname = re.search(r'<text id="([^"]+)"',xml_data).group(1)
            morphed_and_enhanced = depedit3.run_depedit(lemmatized,sent_id=True,sent_text=True,docname=docname,filename=docname)
        else:
            morphed_and_enhanced = depedit3.run_depedit(lemmatized,sent_text=True)

        if xml_data != "":
            xmled = conllu2xml(morphed_and_enhanced, xml_data)
        else:
            xmled = ""

        return {"dep": morphed_and_enhanced, "xml": xmled}

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
    test_xml = io.open(gum_root + os.sep.join(["_build","target","xml","GUM_speech_austria.xml"])).read()
    test_conll = io.open(gum_root + os.sep.join(["_build","target","dep","not-to-release","GUM_speech_austria.conllu"])).read()

    module = DepParser({"LIB_DIR": "lib"})
    res = module.predict_with_pos({"xml": test_xml, "dep": test_conll})
    print(res["xml"])
    print(res["dep"])


if __name__ == "__main__":

    #test_main()

    ud_test = ["GUM_interview_libertarian", "GUM_interview_hill",
               "GUM_news_nasa", "GUM_news_sensitive",
               "GUM_voyage_oakland", "GUM_voyage_vavau",
               "GUM_whow_mice", "GUM_whow_cactus",
               "GUM_fiction_falling", "GUM_fiction_teeth",
               "GUM_bio_jespersen", "GUM_bio_dvorak",
               "GUM_academic_eegimaa", "GUM_academic_discrimination",
               # "GUM_reddit_escape", "GUM_reddit_monsters",
               "GUM_speech_austria", "GUM_textbook_chemistry",
               "GUM_vlog_studying", "GUM_conversation_retirement"]
    files = glob(gum_root + os.sep.join(["_build","target","dep","not-to-release","*.conllu"]))
    files = [f for f in files if os.path.basename(f).replace(".conllu","") in ud_test]

    all_gold = []
    all_pred = []
    module = DepParser({"LIB_DIR": "lib"})
    for i, file_ in enumerate(files):
        conllu = io.open(file_,encoding="utf8").read()
        lines = [l.split("\t") for l in conllu.split("\n") if "\t" in l]
        gold = [[l[6],l[7]] for l in lines if "-" not in l[0] and "." not in l[0]]
        gold = [[g[0],g[1]] if g[1] != "punct" else ["0","punct"] for g in gold]
        all_gold += gold

        res = module.predict_with_pos({"xml": "", "dep": conllu})
        preds = res["dep"]
        lines = [l.split("\t") for l in preds.split("\n") if "\t" in l]
        pred = [[l[6],l[7]] for l in lines if "-" not in l[0] and "." not in l[0]]
        pred = [[p[0],p[1]] if p[1] != "punct" else ["0","punct"] for p in pred]
        all_pred += pred

    from sklearn.metrics import classification_report, accuracy_score
    gold_labs = [g[1] for g in all_gold]
    gold_attach = [g[0] for g in all_gold]
    gold_lab_attach = [str(g) for g in all_gold]
    pred_labs = [g[1] for g in all_pred]
    pred_attach = [g[0] for g in all_pred]
    pred_lab_attach = [str(g) for g in all_pred]
    print("LAS:" + str(accuracy_score(gold_lab_attach,pred_lab_attach)))
    print("UAS:" + str(accuracy_score(gold_attach,pred_attach)))
    print(classification_report(gold_labs,pred_labs))

