from glob import glob
from flair.data import Sentence, Token
from flair.models import SequenceTagger
import flair
import conllu
import stanfordnlp
import pickle
import numpy as np
import xgboost as xgb
import shutil
import os, io
from nlp_modules.base import NLPModule, PipelineDep

from tqdm import tqdm

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep

class PoSTagger(NLPModule):
    requires = (PipelineDep.TOKENIZE, PipelineDep.S_SPLIT)
    provides = (PipelineDep.POS_TAG,)

    def __init__(self, global_config):
        # self.LIB_DIR = config["LIB_DIR"]
        self.test_dependencies()
        self.stanford_ewt = stanfordnlp.Pipeline(
            processors="tokenize,pos",
            models_dir="nlp_modules/pos-dependencies/",
            tokenize_pretokenized=True,
            treebank="en_ewt",
            use_gpu=global_config.get("use_gpu", False),
            pos_batch_size=1000,
        )

        config = {
            "processors": "tokenize,pos",
            "tokenize_pretokenized": True,
            "pos_model_path": "nlp_modules/pos-dependencies/en_gum_tagger.pt",
            "pos_pretrain_path": "nlp_modules/pos-dependencies/en_gum.pretrain.pt",
            "pos_batch_size": 1000,
            "use_gpu": global_config.get("use_gpu", False),
            "treebank": "en_gum",
        }
        self.stanford_gum = stanfordnlp.Pipeline(**config)
        self.flair_onto = SequenceTagger.load("nlp_modules/pos-dependencies/en-pos-ontonotes-v0.4.pt")
        self.flair_gum = SequenceTagger.load("nlp_modules/pos-dependencies/gum-pos-flair.pt")

    def test_dependencies(self):
        files = ["en_gum.pretrain.pt","en-pos-ontonotes-v0.4.pt","gum-pos-flair.pt","en_ewt.pretrain.pt","en_ewt_parser.pt",
                 "en_ewt_tagger.pt","en_gum_tagger.pt","en_ewt_lemmatizer.pt"]
        pos_dependencies_dir = script_dir + "pos-dependencies" + os.sep
        for file_ in files:
            suffix = ""
            if file_.startswith("en_ewt"):
                suffix = 'en_ewt_models' + os.sep
            if not os.path.exists(pos_dependencies_dir + suffix + file_):
                self.download_file(file_, pos_dependencies_dir + suffix, subfolder="pos")

    def get_stanford_predictions(self, model, data_path):
        if model == 'ewt':
            nlp = self.stanford_ewt
        else:
            nlp = self.stanford_gum

        data = []
        with io.open(data_path, encoding="utf8") as file:
            data.append([])
            for line in file:
                if line.startswith("# newdoc id"):
                    continue
                if line.startswith("#") or line.startswith("\n"):
                    continue
                else:
                    sp = line.split("\t")
                    if sp[0] == "1":
                        data[-1].append([])
                    data[-1][-1].append(sp[1])

        f = []
        for tokenized_text in data:
            doc = nlp(tokenized_text)
            for sent in doc.sentences:
                for word in sent.words:
                    f.append(word.xpos)
        return f

    def get_flair_predictions(self, model_type, data_path, fileName):
        if model_type == "onto":
            model = self.flair_onto   
        else:
            model = self.flair_gum

        sentences = []
        with io.open(data_path, 'r', encoding="utf8") as f:
            for token_list in conllu.parse(f.read()):
                sentence = Sentence()
                for token in token_list:
                    sentence.add_token(Token(token['form']))
                sentences.append(sentence)

        output = []
        
        preds = model.predict(sentences)
        if preds is None:  # Newer versions of flair have void predict method, use modified Sentence list
            preds = sentences

        for sentence in preds:
            for token in sentence:
                if str(flair.__version__).startswith("0.4"):
                    output.append(token.tags['pos'].value)
                else:
                    output.append(token.labels[0].value)
        return output

    def get_ensemble_predictions(self, test_x):
        test_encoded = []
        with open("nlp_modules/pos-dependencies/all-encodings.pickle.dat", "rb") as f:
            le = pickle.load(f)
        with open("nlp_modules/pos-dependencies/y-encodings.pickle.dat", "rb") as f:
            le2 = pickle.load(f)

        test_x = np.column_stack((k for k in test_x))
        for k in test_x:
            test_encoded.append(le.transform(k))

        dtest = xgb.DMatrix(np.array(test_encoded))
        with open("nlp_modules/pos-dependencies/xg-model.pickle.dat", "rb") as f:
            loaded_model = pickle.load(f)

        predictions = loaded_model.predict(dtest)
        predictions = [int(x) for x in predictions]
        predictions = le2.inverse_transform(predictions)
        for i in range(len(predictions)):
            if predictions[i] == "-LSB-":
                predictions[i] = "-LRB-"
            if predictions[i] == "-RSB-":
                predictions[i] = "-RRB-"

        return predictions

    def run(self, input_dir, output_dir):
        # Identify a function that takes data and returns output at the document level
        # processing_function = self.tokenize

        # use process_files, inherited from NLPModule, to apply this function to all docs
        file_type = "dep"
        os.makedirs(os.path.join(output_dir, file_type), exist_ok=True)
        sorted_filepaths = sorted(glob(os.path.join(input_dir, file_type, "*")))
        if not os.path.exists("pos_tmp"):
            os.mkdir("pos_tmp")
        for filepath in tqdm(sorted_filepaths):
            filename = filepath.split(os.sep)[-1]

            print(f"POS tagging {filepath}...")
            stanford_ewt_t = self.get_stanford_predictions("ewt", filepath)
            stanford_gum_t = self.get_stanford_predictions("gum", filepath)
            flair_onto_t = self.get_flair_predictions("onto", filepath, filename)
            flair_gum_t = self.get_flair_predictions("gum", filepath, filename)
            results = self.get_ensemble_predictions(
                np.array([stanford_ewt_t, stanford_gum_t, flair_onto_t, flair_gum_t])
            )

            indx = 0

            with io.open(filepath, encoding="utf8") as inp:
                output = io.open(os.path.join(output_dir, file_type, filename), "w", encoding="utf8", newline="\n")
                for line in inp:
                    sp = line.split("\t")
                    if sp[0].isdigit():
                        sp[4] = results[indx]
                        output.write("\t".join(sp))
                        indx += 1
                    else:
                        output.write(line)

                output.close()
        try:
            shutil.rmtree("pos_tmp")
        except:
            pass
        return
