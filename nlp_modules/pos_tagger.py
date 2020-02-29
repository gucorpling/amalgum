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
    requires = (PipelineDep.TOKENIZE, PipelineDep.S_SPLIT)
    provides = (PipelineDep.POS_TAG,)

    def __init__(self, opts):
        # self.LIB_DIR = config["LIB_DIR"]
        self.stanford_ewt = nlp = stanfordnlp.Pipeline(
            processors="tokenize,pos",
            models_dir="nlp_modules/pos-dependencies/stanfordnlp_models/",
            tokenize_pretokenized=True,
            treebank="en_ewt",
            use_gpu=True,
            pos_batch_size=1000,
        )

        config = {
            "processors": "tokenize,pos",
            "tokenize_pretokenized": True,
            "pos_model_path": "nlp_modules/pos-dependencies/saved_models/pos/en_gum_tagger.pt",
            "pos_pretrain_path": "nlp_modules/pos-dependencies/saved_models/pos/en_gum.pretrain.pt",
            "pos_batch_size": 1000,
            "treebank": "en_gum",
        }
        self.stanford_gum = stanfordnlp.Pipeline(**config)
        self.flair_onto = SequenceTagger.load("pos")
        self.flair_gum = SequenceTagger.load("nlp_modules/pos-dependencies/gum-flair/final-model.pt")

    def test_dependencies(self):
        stanfordnlp.download("en", "nlp_modules/pos-dependencies/stanfordnlp_models/")
        if not os.path.exists("nlp_modules/pos-dependencies"):
            raise NLPDependencyException(
                "Could not locate folder `pos-dependencies`. It has to be in the same directory as this script. Please download from https://drive.google.com/file/d/1P5yRDKuBx1hDgOmZU1tNYyt6oZciRx_u/view?usp=sharing"
            )
            sys.exit(1)

    def get_stanford_predictions(self, model, data_path):
        if model == 'ewt':
            nlp = self.stanford_ewt
        else:
            nlp = self.stanford_gum

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

        f = open("pos_tmp/flair_" + fileName + "_" + model_type + "_reformat.txt", "w")
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

        sentences = []
        with open(
            "pos_tmp/flair_" + fileName + "_" + model_type + "_reformat.txt"
        ) as f:
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
        output = []
        for s in sents:
            for tok in s.tokens:
                output.append(tok.tags["pos"].value)
        os.remove("pos_tmp/flair_" + fileName + "_" + model_type + "_reformat.txt")
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

            stanford_ewt_t = self.get_stanford_predictions("ewt", filepath)
            stanford_gum_t = self.get_stanford_predictions("gum", filepath)
            flair_onto_t = self.get_flair_predictions("onto", filepath, filename)
            flair_gum_t = self.get_flair_predictions("gum", filepath, filename)
            results = self.get_ensemble_predictions(
                np.array([stanford_ewt_t, stanford_gum_t, flair_onto_t, flair_gum_t])
            )

            indx = 0

            with open(filepath) as inp:
                output = open(os.path.join(output_dir, file_type, filename), "w")
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
