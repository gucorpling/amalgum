from glob import glob
from flair.data import Sentence, Token
from flair.models import SequenceTagger
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier
from argparse import ArgumentParser
import flair
import stanza
import pickle
import numpy as np
import pandas as pd
import shutil
import os, io, sys
from nlp_modules.base import NLPModule, PipelineDep
try:
    from .configuration import GUM_ROOT
except ImportError:
    from configuration import GUM_ROOT

from tqdm import tqdm

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep

import torch

torch.backends.cudnn.enabled = False

if not GUM_ROOT.endswith(os.sep):
    GUM_ROOT += os.sep


class PoSTagger(NLPModule):
    requires = (PipelineDep.TOKENIZE, PipelineDep.S_SPLIT)
    provides = (PipelineDep.POS_TAG,)

    def __init__(self, global_config, is_main=False, tag="xpos"):
        # self.LIB_DIR = config["LIB_DIR"]
        self.test_dependencies()
        self.stanford_ewt = stanza.Pipeline(
            lang="en",package="ewt",
            processors="tokenize,pos",
            #models_dir="nlp_modules/pos-dependencies/",
            tokenize_pretokenized=True,
            #treebank="en_ewt",
            use_gpu=global_config.get("use_gpu", False), #False
            #pos_batch_size=1000,
        )

        config = {
            "processors": "tokenize,pos",
            "tokenize_pretokenized": True,
            "pos_model_path": script_dir+"pos-dependencies/en_gum_tagger.pt",
            "pos_pretrain_path": script_dir+"pos-dependencies/en_gum.pretrain.pt",
            #"pos_batch_size": 1000,
            "use_gpu": global_config.get("use_gpu", False), #False,
            "lang": "en",
            "package":"gum"
        }
        self.stanford_gum = stanza.Pipeline(**config)
        if is_main:
            self.prefix = ""
        else:
            self.prefix = "nlp_modules/"
        #self.flair_onto = SequenceTagger.load(self.prefix+"pos-dependencies/en-pos-ontonotes-v0.4.pt")
        #self.flair_gum = SequenceTagger.load(self.prefix+"pos-dependencies/gum-pos-flair.pt")
        if tag == "xpos":
            self.flair_onto = SequenceTagger.load("pos")
            self.flair_gum = SequenceTagger.load(self.prefix+"pos-dependencies/flair_tagger/best-model_xpos_gum.pt")
        else:
            self.flair_onto = SequenceTagger.load(self.prefix + "pos-dependencies/flair_tagger/best-model_deprel_ewt.pt")
            self.flair_gum = SequenceTagger.load(self.prefix + "pos-dependencies/flair_tagger/best-model_deprel_gum.pt")
        self.flair_ner = SequenceTagger.load("ner")
        with open(self.prefix + "pos-dependencies/xg.pickle.dat", "rb") as f:
            self.clf = pickle.load(f)
        with open(self.prefix + "pos-dependencies/all-encodings.pickle.dat", "rb") as f:
            self.multicol_dict = pickle.load(f)
        with open(self.prefix + "pos-dependencies/y-encodings.pickle.dat", "rb") as f:
            self.label_encoder = pickle.load(f)

        self.features = ["stan_ewt", "stan_gum", "flair_onto_tag",
                   "flair_onto_score",
                   "flair_gum_tag", "flair_gum_score", "flair_ner_tag", "flair_ner_score",
                   "first_char",
                   "last_char",
                   "word_shape_score",
                   "length_score"
                   ]

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

    def get_stanford_predictions(self, model, input_conllu, colnum=4):
        if model == 'ewt':
            nlp = self.stanford_ewt
        else:
            nlp = self.stanford_gum

        data = []

        data.append([])
        for line in input_conllu.strip().split("\n"):
            if line.startswith("# newdoc id"):
                continue
            if line.startswith("#") or line.startswith("\n"):
                continue
            elif "\t" in line:
                sp = line.split("\t")
                if sp[0] == "1":
                    data[-1].append([])
                if "." not in sp[0] and "-" not in sp[0]:
                    data[-1][-1].append(sp[1])

        f = []
        for tokenized_text in data:
            doc = nlp(tokenized_text)
            for sent in doc.sentences:
                for word in sent.words:
                    if colnum == 4:
                        f.append(word.xpos)
                    else:
                        f.append(word.dependency_relation)
        return f

    def get_flair_predictions(self, model_type, input_conllu, with_score=False):
        if model_type == "onto":
            model = self.flair_onto
        elif model_type == "ner":
            model = self.flair_ner
        else:
            model = self.flair_gum

        sentences = []
        conll_sents = input_conllu.strip().split("\n\n")

        for sent in conll_sents:
            token_list = [l.split("\t") for l in sent.split("\n") if "\t" in l]
            token_list = [t[1] for t in token_list if "." not in t[0] and "-" not in t[0]]
            sentence = Sentence()
            for token in token_list:
                sentence.add_token(token)
            sentences.append(sentence)

        output = []
        scores = []

        preds = model.predict(sentences,all_tag_prob=with_score)
        if preds is None:  # Newer versions of flair have void predict method, use modified Sentence list
            preds = sentences

        for sentence in preds:
            for token in sentence:
                if str(flair.__version__).startswith("0.4"):
                    output.append(token.tags['pos'].value)
                else:
                    output.append(token.labels[0].value)
                if with_score:
                    scores.append(token.labels[0].score)
        if with_score:
            return (output, scores)
        else:
            return [output]

    def get_ensemble_predictions(self, test_x):
        test_encoded = []
        with open(self.prefix + "pos-dependencies/all-encodings.pickle.dat", "rb") as f:
            le = pickle.load(f)
        with open(self.prefix + "pos-dependencies/y-encodings.pickle.dat", "rb") as f:
            le2 = pickle.load(f)

        test_x = np.column_stack((k for k in test_x))
        for k in test_x:
            test_encoded.append(le.transform(k))

        dtest = xgb.DMatrix(np.array(test_encoded))
        with open(self.prefix + "pos-dependencies/xg-model.pickle.dat", "rb") as f:
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

    @staticmethod
    def postprocess(preds, toks):
        output = []
        for i, pred in enumerate(preds):
            if len(toks[i])>2:
                if toks[i][0].isdigit() and toks[i][-2:] in ["th","rd","st","nd"]:
                    pred = "JJ"
            if i == 0 and toks[i][-1] == "." and pred == "CD":
                pred = "LS"  # CD ending in . at first token is LS
            pred = pred.replace("LSB","LRB").replace("RSB","RRB")
            output.append(pred)
        return output

    def ensemble_pred(self, input_conll):

        toks = [l.split("\t") for l in input_conll.split("\n") if "\t" in l]
        toks = [tok[1] for tok in toks if "-" not in tok[0]]
        df = self.get_features(input_conll, self.features, toks)
        test_X = self.multicol_transform(df, np.array([h for h in self.features if "score" not in h]),
                                         self.multicol_dict["all_encoders_"])
        ngram_test_X = self.ngram(test_X, self.features)
        predictions = self.clf.predict(ngram_test_X)
        predictions = [int(x) for x in predictions]
        pred_labels = self.label_encoder.inverse_transform(predictions)
        pred_labels = self.postprocess(pred_labels,toks)

        return pred_labels

    @staticmethod
    def multicol_fit_transform(dframe, columns):
        """
        Transforms a pandas dataframe's categorical columns into pseudo-ordinal numerical columns and saves the mapping

        :param dframe: pandas dataframe
        :param columns: list of column names with categorical values to be pseudo-ordinalized
        :return: the transformed dataframe and the saved mappings as a dictionary of encoders and labels
        """

        if isinstance(columns, list):
            columns = np.array(columns)
        else:
            columns = columns

        encoder_dict = {}
        # columns are provided, iterate through and get `classes_` ndarray to hold LabelEncoder().classes_
        # for each column; should match the shape of specified `columns`
        all_classes_ = np.ndarray(shape=columns.shape, dtype=object)
        all_encoders_ = np.ndarray(shape=columns.shape, dtype=object)
        all_labels_ = np.ndarray(shape=columns.shape, dtype=object)
        for idx, column in enumerate(columns):
            # instantiate LabelEncoder
            le = LabelEncoder()
            # fit and transform labels in the column
            dframe.loc[:, column] = le.fit_transform(dframe.loc[:, column].values)
            encoder_dict[column] = le
            # append the `classes_` to our ndarray container
            all_classes_[idx] = (column, np.array(le.classes_.tolist(), dtype=object))
            all_encoders_[idx] = le
            all_labels_[idx] = le

        multicol_dict = {"encoder_dict": encoder_dict, "all_classes_": all_classes_, "all_encoders_": all_encoders_,
                         "columns": columns}
        return dframe, multicol_dict

    @staticmethod
    def multicol_transform(dframe, columns, all_encoders_):
        """
        Transforms a pandas dataframe's categorical columns into pseudo-ordinal numerical columns based on existing mapping
        :param dframe: a pandas dataframe
        :param columns: list of column names to be transformed
        :param all_encoders_: same length list of sklearn encoders, each mapping categorical feature values to numbers
        :return: transformed numerical dataframe
        """
        for idx, column in enumerate(columns):
            copied = list(dframe.loc[:, column])
            if "NN" in all_encoders_[idx].classes_:
                default = "NN"
            else:
                default = "_"
            dframe.loc[:, column] = dframe.loc[:, column].map(lambda v: default if v not in all_encoders_[idx].classes_ else v)
            try:
                dframe.loc[:, column] = all_encoders_[idx].transform(dframe.loc[:, column].values)
            except:
                a=3
        return dframe

    @staticmethod
    def ngram(unigram_df, headers):
        prev = unigram_df.copy()
        first_row = prev.loc[0]
        prev = prev.drop([0, 0])
        prev = prev.append(first_row)
        prev.columns = ["prev_" + h for h in headers]
        for col in prev.columns:
            if "score" not in col:
                prev[col] = prev[col].astype(int)
        return pd.concat([unigram_df, prev.set_index(unigram_df.index)], axis=1)

    @staticmethod
    def get_word_features(words):
        first_letters = []
        last_letters = []
        word_shapes = []
        lens = []
        for word in words:
            first = word[0].lower()
            first = first if first in "abcdefghijklmnopqrstuvwxyz" else "_"
            first_letters.append(first)
            last = word[-1].lower()
            last = last if last in "abcdefghijklmnopqrstuvwxyz" else "_"
            last_letters.append(last)
            if word.isnumeric():
                shape = 1
            elif not word.isalpha():
                shape = 2
            elif word.istitle():
                shape = 3
            elif word.isupper():
                shape = 4
            else:
                shape = 5
            word_shapes.append(shape)
            lens.append(len(word))

        return first_letters, last_letters, word_shapes, lens

    def get_features(self, conllu, headers, toks):
        base_preds = []
        base_preds.append(self.get_stanford_predictions("ewt", conllu))  # Stanza trained on EWT (stanza pretrained)
        base_preds.append(self.get_stanford_predictions("gum", conllu))  # Stanza trained on GUM (re-trained)
        for p in self.get_flair_predictions("onto", conllu, with_score=True):
            base_preds.append(p)  # Flair tagger xpos trained on OntoNotes (flair pre-trained model), preds + scores
        for p in self.get_flair_predictions("gum", conllu, with_score=True):
            base_preds.append(p)  # Flair tagger xpos trained on GUM, preds + scores
        for p in self.get_flair_predictions("ner", conllu, with_score=True):
            base_preds.append(p)  # Flair tagger ner, preds + scores

        first_letters, last_letters, word_shapes, lens = self.get_word_features(toks)

        base_preds.append(first_letters)
        base_preds.append(last_letters)
        base_preds.append(word_shapes)
        base_preds.append(lens)

        in_dict = {h: base_preds[i] for i, h in enumerate(headers)}
        df = pd.DataFrame(in_dict, columns=headers)
        return df

    def train_ensemble(self, train_conllu_file, test_conllu_file=None, use_cache=False, colnum=4, conf=False, retrain=True):
        """
        :param train_conllu: training data as conllu format string; make sure not to use data that base learners have seen!
        :param test_conllu: optional conllu format string; used to score ensemble after training
        :return: None
        """
        if os.path.exists(self.prefix + "pos-dependencies/train_X.tab") and use_cache:
            sys.stderr.write("o Reading cached test data\n")
            df = pd.read_csv(self.prefix + "pos-dependencies/train_X.tab",sep="\t",quoting=3)
        elif use_cache:
            use_cache = False
            sys.stderr.write("! Cached ensemble training matrix not found, computing from training data\n")

        train_conllu = io.open(train_conllu_file,encoding="utf8").read()
        # Ensure rare LS tag and item that will be tagged ADD and NFP by EWT are present in training data
        oov_toks = [["1","1.","1.","X","LS","_","2","dep","_","_"]]
        oov_toks.append(["2","kim@abc.com","kim@abc.com","PROPN","NNP","_","0","root","_","_"])
        oov_toks.append(["3",":)",":)","SYM","NFP","_","2","discourse","_","_"])
        oov_toks.append([""])
        oov_toks.append(["1", "Me", "I", "PRON", "PRP", "Case=Acc|Number=Sing|Person=1|PronType=Prs", "4", "dislocated", "_", "_"])
        oov_toks.append(["2", ",", ",", "PUNCT", ",", "_", "1", "punct", "_", "_"])
        oov_toks.append(["3", "I", "I", "PRON", "PRP", "Case=Nom|Number=Sing|Person=1|PronType=Prs", "4", "nsubj", "_", "_"])
        oov_toks.append(["4", "like", "like", "VERB", "VBP", "Mood=Ind|Tense=Pres|VerbForm=Fin", "0", "root", "_", "_"])
        oov_toks.append(["5", "Kim", "Kim", "PROPN", "NNP", "Number=Sing", "4", "obj", "_", "_"])
        oov_toks.append(["6", ".", ".", "PUNCT", ".", "_", "4", "punct", "_", "_"])
        oov_toks.append([""])
        oov_toks.append(["1", "That", "that", "SCONJ", "IN", "_", "3", "mark", "_", "_"])
        oov_toks.append(["2", "Jo", "Jo", "PROPN", "NNP", "Number=Sing", "3", "nsubj", "_", "_"])
        oov_toks.append(["3", "came", "come", "VERB", "VBD", "Mood=Ind|Tense=Past|VerbForm=Fin", "5", "csubj:pass", "_", "_"])
        oov_toks.append(["4", "was", "be", "AUX", "VBD", "Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin", "5", "aux:pass","_", "_"])
        oov_toks.append(["5", "believed", "believe", "VERB", "VBN", "Tense=Past|VerbForm=Part", "0", "root", "_", "_"])
        oov_toks.append(["6", "by", "by", "ADP", "IN", "_", "8", "case", "_", "_"])
        oov_toks.append(["7", "no", "no", "DET", "DT", "Polarity=Neg", "8", "det", "_", "_"])
        oov_toks.append(["8", "one", "one", "NUM", "CD", "NumType=Card", "5", "obl", "_", "_"])
        oov_toks.append(["9", ".", ".", "PUNCT", ".", "_", "5", "punct", "_", "_"])
        oov_toks.append([""])
        oov_toks.append(["1", "Y-", "you", "PRON", "PRP", "_", "2", "reparandum", "_", "_"])
        oov_toks.append(["2", "you", "you", "PRON", "PRP", "Number=Sing", "4", "nsubj", "_", "_"])
        oov_toks.append(["3", "are", "be", "VERB", "VBP", "Mood=Ind|Tense=Pres|VerbForm=Fin", "4", "reparandum", "_", "_"])
        oov_toks.append(["4", "are", "be", "VERB", "VBP", "Mood=Ind|Tense=Pres|VerbForm=Fin", "0", "root", "_", "_"])
        oov_toks = ["\t".join(tok) for tok in oov_toks]
        train_conllu = "\n".join(oov_toks) + "\n\n" + train_conllu
        test_conllu = None
        if test_conllu_file is not None:
            test_conllu = io.open(test_conllu_file,encoding="utf8").read()

        # Format true labels
        toks = [l.split("\t") for l in train_conllu.splitlines() if "\t" in l]
        labels = [l[colnum] for l in toks if "." not in l[0] and "-" not in l[0]]
        y_encoder = LabelEncoder()
        train_y = y_encoder.fit_transform(labels)

        sys.stderr.write("o Collecting base learner training predictions\n")
        headers = self.features

        if not use_cache:
            words = [l[1] for l in toks if "." not in l[0] and "-" not in l[0]]
            df = self.get_features(train_conllu, headers, words)

        df.to_csv(self.prefix + "pos-dependencies/train_X.tab", sep="\t", quoting=None, index=False)

        train_X, multicol_dict = self.multicol_fit_transform(df,np.array([h for h in headers if "score" not in h]))

        ngram_train_X = self.ngram(train_X, headers)
        all_headers = ngram_train_X.columns

        param = {
            'max_depth': 3,  # the maximum depth of each tree
            'eta': 0.05,  # the training step for each iteration
            "n_estimators": 100,
            "min_child_weight": 1,
            'colsample_bytree': 0.7,
            'gamma': 0,
            'subsample': 0.8}  # the number of classes that exist in this datset
        if retrain:
            sys.stderr.write("o Training ensemble\n")
            clf = XGBClassifier(random_state=42,silent=True,n_jobs=6,**param)
            clf.fit(ngram_train_X,train_y)

            pickle.dump(multicol_dict, open(self.prefix+"pos-dependencies/all-encodings.pickle.dat", "wb"))
            pickle.dump(y_encoder, open(self.prefix+"pos-dependencies/y-encodings.pickle.dat", "wb"))
            pickle.dump(clf, open(self.prefix+"pos-dependencies/xg.pickle.dat", "wb"))
        else:
            clf = pickle.load(open(self.prefix+"pos-dependencies/xg.pickle.dat","rb"))
        self.clf = clf

        if test_conllu is not None:
            toks = [l.split("\t") for l in test_conllu.splitlines() if "\t" in l]
            labels = [l[colnum] for l in toks if "." not in l[0] and "-" not in l[0]]
            test_y = y_encoder.transform(labels)

            # loaded_model = pickle.load(open("xg-gum.pickle.dat", "rb"))
            if os.path.exists(self.prefix + "pos-dependencies/test_X.tab") and use_cache:
                sys.stderr.write("o Reading cached test data\n")
                df = pd.read_csv(self.prefix + "pos-dependencies/test_X.tab", sep="\t", quoting=3)
            else:
                if use_cache:
                    sys.stderr.write("! Cached ensemble test matrix not found, computing from test data\n")
                sys.stderr.write("o Collecting base learner test predictions\n")
                words = [l[1] for l in toks if "." not in l[0] and "-" not in l[0]]
                df = self.get_features(test_conllu, headers, words)

            df.to_csv(self.prefix + "pos-dependencies/test_X.tab", sep="\t", quoting=None,index=False)

            test_X = self.multicol_transform(df,np.array([h for h in headers if "score" not in h]),multicol_dict["all_encoders_"])
            ngram_test_X = self.ngram(test_X,headers)

            predictions = self.clf.predict(ngram_test_X)
            predictions = [int(x) for x in predictions]
            pred_labels = y_encoder.inverse_transform(predictions)

            words = [l[1] for l in toks if "." not in l[0] and "-" not in l[0]]

            pred_labels = self.postprocess(pred_labels,words)
            predictions = y_encoder.transform(pred_labels)

            diff = ["\t".join([words[i],labels[i],pred_labels[i],str(labels[i]==pred_labels[i])]) for i in range(len(words))]
            with io.open("diff_ensemble.tab",'w',encoding="utf8",newline="\n") as d:
                d.write("\n".join(diff))

            print('ens acc:%1.4f' % accuracy_score(test_y, predictions))
            importances = sorted([(imp,all_headers[i]) for i, imp in enumerate(clf.feature_importances_)],reverse=True)
            importances = [imp[1]+"\t"+str(imp[0]) for imp in importances]
            print("\n".join(importances))

            if conf:
                sn_conf(labels,pred_labels)

    def run(self, input_dir=None, output_dir=None, input_conll=None, colnum=4):
        file_type = "dep"
        if input_conll is None:
            os.makedirs(os.path.join(output_dir, file_type), exist_ok=True)
            sorted_filepaths = sorted(glob(os.path.join(input_dir, file_type, "*")))
            if not os.path.exists("pos_tmp"):
                os.mkdir("pos_tmp")
        else:
            sorted_filepaths = [input_conll]
        for filepath in tqdm(sorted_filepaths):
            if input_conll is None:
                filename = filepath.split(os.sep)[-1]
                print(f"POS tagging {filepath}...")
                conllu = io.open(filepath, encoding="utf8").read()
                output_file = io.open(os.path.join(output_dir, file_type, filename), "w", encoding="utf8", newline="\n")
            else:
                conllu = filepath
            indx = 0
            results = self.ensemble_pred(conllu)
            output = []
            for line in conllu.split("\n"):
                sp = line.split("\t")
                if sp[0].isdigit():
                    sp[colnum] = results[indx]
                    output.append("\t".join(sp).strip())
                    indx += 1
                else:
                    output.append(line.strip())

            output = "\n".join(output).strip() + "\n\n"
            if input_conll is None:
                output_file.write(output)
                output_file.close()
                try:
                    shutil.rmtree("pos_tmp")
                except:
                    pass
            else:
                return output
        return

def sn_conf(golds, preds):
    import seaborn as sn
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    confmat = confusion_matrix(golds, preds)

    class_names = sorted(list(set(golds).union(set(preds))))
    class_names.sort()

    sn.set(font_scale=0.5)
    np.fill_diagonal(confmat, 0)
    df_cm = pd.DataFrame(confmat, index=class_names, columns=class_names)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.show()


if __name__ == "__main__":

    p = ArgumentParser()
    p.add_argument("-m","--mode",choices=["train","predict"],default="predict")
    p.add_argument("-c","--cached",action="store_true",help="Whether to use cached base estimator train and test matrices")
    p.add_argument("-t","--tag",choices=["xpos","deprel"],default="xpos",help="Whether to use cached base estimator train and test matrices")
    p.add_argument("--confmat",action="store_true",help="Whether to plot a confusion matrix of test data after training")
    p.add_argument("-p","--pickled",action="store_true",help="Evaluate pickled classifier (don't retrain)")

    opts = p.parse_args()

    if opts.mode == "train":
        tagger = PoSTagger({}, is_main=True, tag=opts.tag)
        tagcol = 4 if opts.tag != "deprel" else 7
        retrain = False if opts.pickled else True
        tagger.train_ensemble(#GUM_ROOT + os.sep.join(["_build","target","dep"]) + os.sep + "en_gum-ud-dev.conllu",
                              "C:\\Uni\\Corpora\\GUM\\autogum\\autogum.git\\en_gum-ud-dev-plus-supplemental.conllu",
                              GUM_ROOT + os.sep.join(["_build","target","dep"]) + os.sep + "en_gum-ud-test.conllu",
                              use_cache=opts.cached,colnum=tagcol, conf=opts.confmat, retrain=retrain)
    else:
        tagger = PoSTagger({},is_main=True)
        conll_in = io.open(GUM_ROOT + os.sep.join(["_build","target","dep","not-to-release"]) + os.sep + "GUM_academic_exposure.conllu").read()
        conll_in = io.open("C:\\Uni\\Corpora\\GUM\\autogum\\autogum.git\\target\\02_FlairSentSplitter\\dep\\AMALGUM_reddit_projecting.conllu").read()
        conll_tagged = tagger.run(input_conll=conll_in)
        with io.open("pred.conllu",'w',encoding="utf8",newline="\n") as f:
            f.write(conll_tagged)

        true_tags = [l.split("\t")[4] for l in conll_in.splitlines() if "\t" in l]
        pred_tags = [l.split("\t")[4] for l in conll_tagged.splitlines() if "\t" in l]

        score = len([t for i,t in enumerate(true_tags) if pred_tags[i] == t])/len(true_tags)
        print(score)
