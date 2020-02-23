from glob import glob
from flair.models import SequenceTagger
import stanfordnlp
import pickle
import numpy as np
import xgboost as xgb
import shutil
import os
from nlp_modules.base import NLPModule, PipelineDep

from tqdm import tqdm


class PoSTagger(NLPModule):
    requires = ()
    provides = (PipelineDep.TOKENIZE,)

    def __init__(self):
        # self.LIB_DIR = config["LIB_DIR"]
        pass

    def test_dependencies(self):
        if not os.path.exists("pos-dependencies"):
            raise NLPDependencyException(
                "Could not locate folder `pos-dependencies`. Please download from github: gucorpling/amalgum/nlp_modules"
            )
            sys.exit(1)

    def get_stanford_predictions(self, model, data_path):

        output_path = "pos_tmp/stanford_" + model + "_predictions.txt"

        if model == "ewt":
            nlp = stanfordnlp.Pipeline(
                processors="tokenize,pos",
                models_dir="pos-dependencies/stanfordnlp_models/",
                tokenize_pretokenized=True,
                treebank="en_ewt",
                use_gpu=True,
                pos_batch_size=1000,
            )
        else:
            config = {
                "processors": "tokenize,pos",
                "tokenize_pretokenized": True,
                "pos_model_path": "pos-dependencies/saved_models/pos/en_gum_tagger.pt",
                "pos_pretrain_path": "pos-dependencies/saved_models/pos/en_gum.pretrain.pt",
                "pos_batch_size": 1000,
                "treebank": "en_gum",
            }
            nlp = stanfordnlp.Pipeline(**config)
        data = []
        with open(data_path) as file:
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

        f = open(output_path, "w")
        for tokenized_text in data:
            doc = nlp(tokenized_text)
            for sent in doc.sentences:
                for word in sent.words:
                    f.write(word.xpos + "\n")
        f.close()

    def get_flair_predictions(self, model_type, data_path):
        f = open("pos_tmp/flair_" + model_type + "_reformat.txt", "w")
        count = 0
        notFirst = False

        with open(data_path) as file:
            for line in file:
                if line.startswith("#") or line.startswith("\n"):
                    continue
                else:
                    sp = line.split("\t")
                    if sp[0] == "1" and notFirst:
                        f.write("\n")
                    f.write(sp[1] + "\t" + sp[4] + "\n")
                    count += 1
                    notFirst = True
            f.write("\n")

        f.close()

        # load the model you trained
        if model_type == "onto":
            model = SequenceTagger.load("pos")
        else:
            model = SequenceTagger.load("pos-dependencies/gum-flair/final-model.pt")
        sentences = []
        with open("pos_tmp/flair_" + model_type + "_reformat.txt") as f:
            s = ""
            for line in f:
                if line == "\n" and len(s) > 0:
                    sentences.append(s)
                    s = ""
                else:
                    s += line.split("\t")[0] + " "
        sents = [(len(s.split()), i, s) for i, s in enumerate(sentences)]

        sents.sort(key=lambda x: x[0], reverse=True)
        sentences = [s[2] for s in sents]

        preds = model.predict(sentences)

        # sort back
        sents = [tuple(list(sents[i]) + [s]) for i, s in enumerate(preds)]
        sents.sort(key=lambda x: x[1])
        sents = [s[3] for s in sents]
        output = open("pos_tmp/flair_" + model_type + "_predictions.txt", "w")
        for s in sents:
            for tok in s.tokens:
                output.write(tok.tags["pos"].value + "\n")

    def get_model_predictions(self, path):
        preds = []
        with open(path) as f:
            for line in f:
                if line.startswith("\n"):
                    continue
                else:
                    line = line.strip("\n")
                    line = line.split("\t")
                    preds.append([line[0]])
        return preds

    def get_ensemble_predictions(self, test_x):
        test_encoded = []
        le = pickle.load(open("pos-dependencies/all-encodings.pickle.dat", "rb"))
        le2 = pickle.load(open("pos-dependencies/y-encodings.pickle.dat", "rb"))

        test_x = np.column_stack((k for k in test_x))
        for k in test_x:
            test_encoded.append(le.transform(k))

        dtest = xgb.DMatrix(np.array(test_encoded))
        loaded_model = pickle.load(open("pos-dependencies/xg-model.pickle.dat", "rb"))

        predictions = loaded_model.predict(dtest)
        predictions = [int(x) for x in predictions]
        predictions = le2.inverse_transform(predictions)
        for i in range(len(predictions)):
            if predictions[i] == "-LSB-":
                predictions[i] = "-LRB-"
            if predictions[i] == "-RSB-":
                predictions[i] = "-RRB-"
        pickle.dump(predictions, open("pos_tmp/preds.pickle.dat", "wb"))

        return

    def predict(self):
        stanford_gum_test = self.get_model_predictions(
            "pos_tmp/stanford_gum_predictions.txt"
        )

        stanford_ewt_test = self.get_model_predictions(
            "pos_tmp/stanford_ewt_predictions.txt"
        )

        flair_gum_test = self.get_model_predictions("pos_tmp/flair_gum_predictions.txt")

        flair_onto_test = self.get_model_predictions(
            "pos_tmp/flair_onto_predictions.txt"
        )

        self.get_ensemble_predictions(
            np.array(
                [stanford_ewt_test, stanford_gum_test, flair_onto_test, flair_gum_test]
            )
        )

    def run(self, input_dir, output_dir):
        # Identify a function that takes data and returns output at the document level
        # processing_function = self.tokenize

        # use process_files, inherited from NLPModule, to apply this function to all docs
        file_type = "conllu"
        os.makedirs(os.path.join(output_dir, file_type), exist_ok=True)
        sorted_filepaths = sorted(glob(os.path.join(input_dir, file_type, "*")))
        for filepath in tqdm(sorted_filepaths):
            filename = filepath.split(os.sep)[-1]
            try:
                shutil.rmtree("pos_tmp")
            except:
                pass
            os.mkdir("pos_tmp")

            stanfordnlp.download("en", "pos-dependencies/stanfordnlp_models/")

            self.get_stanford_predictions("ewt", filepath)
            self.get_stanford_predictions("gum", filepath)
            self.get_flair_predictions("onto", filepath)
            self.get_flair_predictions("gum", filepath)
            self.predict()
            results = pickle.load(open("pos_tmp/preds.pickle.dat", "rb"))
            indx = 0

            with open(filepath) as inp:
                output = open(os.path.join(output_dir, file_type, filename), "w")
                for line in inp:
                    if line.startswith("#"):
                        continue
                    elif line.startswith("\n"):
                        output.write("\n")
                    else:
                        sp = line.split("\t")
                        output.write(sp[1] + "\t" + results[indx] + "\n")
                        indx += 1
                output.close()
        try:
            shutil.rmtree("pos_tmp")
        except:
            pass
        return
