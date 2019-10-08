from keras.models import load_model
from gensim.models import KeyedVectors
import io, os, sys, re, requests, time, argparse, gc, wget
from datetime import timedelta
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
script_dir = os.path.dirname(os.path.realpath(__file__))
lib = os.path.abspath(script_dir + os.sep + "..")
sys.path.append(lib)
# from conll_reader import read_conll, tt_tag, udpipe_tag, shuffle_cut_conllu

model_dir = os.path.abspath(script_dir + os.sep + ".." + os.sep + ".." + os.sep + "models" + os.sep)
vec_dir = os.path.abspath(script_dir + os.sep + ".." + os.sep + ".." + os.sep + "vec" + os.sep)


def read_conll(features, infile, mode="sent", genre_pat=None, as_text=False):
    if as_text:
        lines = infile.split("\n")
    else:
        lines = io.open(infile,encoding="utf8").readlines()
    docname = infile
    output = []  # List to hold dicts of each observation's features
    cache = []  # List to hold current sentence tokens before adding complete sentence features for output
    toks = []  # Plain list of token forms
    firsts = set([])  # Attested first characters of words
    lasts = set([])  # Attested last characters of words
    vocab = defaultdict(int)  # Attested token vocabulary counts
    sent_start = True
    tok_id = 0  # Track token ID within document
    doc_id = 0  # Track document ID
    genre = "_"
    for line in lines:
        if "\t" in line:
            fields = line.split("\t")
            word_id, word, lemma, pos, cpos, feats, head, deprel = fields[0:-2]
            if mode=="seg":
                if "BeginSeg=Yes" in fields[-1]:
                    label = 1
                else:
                    label = 0
            elif mode == "sent":
                if sent_start:
                    label = 1
                else:
                    label = 0
            else:
                raise ValueError("read_conll mode must be one of: seg|sent\n")
            feats = fields[-2].split("|")
            vocab[word] += 1
            # head_dist = int(fields[0]) - int(head)
            toks.append(word)
            firsts.add(word[0])
            lasts.add(word[-1])
            tent_dict = {"word_id":word_id, "word":word, "lemma":lemma, "pos":pos, "cpos":cpos, "deprel":deprel,
                           "docname":docname,"tok_len":len(word),"label":label,"first":word[0],"last":word[-1],
                           "tok_id": tok_id,"genre":genre, "doc_id":doc_id}
            if len(features) <= 1:
                cache.append(tent_dict)
            else:
                cache.append({k:tent_dict[k] for k in features})
            tok_id += 1
            sent_start = False
        elif "# newdoc id = " in line:
            doc_id += 1
            tok_id = 1
        elif len(line.strip())==0:
            sent_start = True
            if len(cache)>0:
                if mode == "seg":  # Don't add s_len in sentencer learning mode
                    for tok in cache:
                        tok["s_len"] = len(cache)
                output += cache
                cache = []

    # Flush last sentence if no final newline
    if len(cache)>0:
        if mode == "seg":  # Don't add s_len in sentencer learning mode
            for tok in cache:
                tok["s_len"] = len(cache)
        output += cache
    # output = pd.DataFrame(output)
    return output


def n_gram(fc, n):
    assert n%2 == 1
    half_window= n//2
    n_gram_cols = []
    label = []
    word_list = []
    count = 0
    count_doc = 0
    for word_dict in fc:
        word = word_dict['word']
        label.append(word_dict['label'])
        # if count == 0:
        #     word_list += ['<s>']*half_window
        #     count_doc += 1
        if count_doc != word_dict['doc_id']:
            if count != 0:
                word_list += ['</s>']*half_window
            word_list += ['<s>']*half_window
            count_doc += 1
        word_list.append(word)
        if count == len(fc)-1:
            word_list += ['</s>']*half_window
        count += 1
    sent_l = ['<s>', '</s>']
    for i in range(len(word_list)-1):
        if word_list[i] not in sent_l:
            n_gram_cols.append(word_list[i-half_window: i+half_window+1])
    return n_gram_cols, label


def loadvec(word_embed_path):
    #sys.stdout.write('##### Load word embeddings...\n')
    # word_embed = KeyedVectors.load(word_embed_path)
    word_embed = {}
    f = open(word_embed_path, encoding='utf-8').readlines()
    for line in f:
        parts = line.strip().split()
        word = parts[0]
        vec = [x for x in parts[1:]]
        word_embed[word] = vec
    return word_embed


def unique_embed(f_cols, word_embed):
    word_unique = []
    for x in f_cols:
        word = x['word']
        if word not in word_unique:
            word_unique.append(word)
    embed_vector = {w:word_embed[w] if w in word_embed else [0.0001]*300 for w in word_unique}
    embed_vector['<s>'] = [0.7]*300
    embed_vector['</s>'] = [0.05]*300
    return embed_vector


def mergeWord2Vec(word_grams, uni_embed):
    fec_vec = []
    for words in word_grams:
        word_vec = []
        for word in words:
            word_vec += uni_embed[word]
        fec_vec.append(word_vec)
    return np.asarray(fec_vec, dtype=np.float32)


def convert_params():
    space = {
        'num_layers':['one', 'two'],
        'units2': [16, 32, 48, 64],
        'dropout2': 0,
        'units1': [32, 64, 96, 128],
        'dropout1': 0,
        'epoch': [5, 10, 15, 20],
        'batch_size': [128, 256, 384, 512],
        'optimizer': ['adadelta','adam','rmsprop', 'sgd'],
        'gram': [5, 7, 9]
        }

    model_name = "DNNSentencer"
    file_dir = script_dir + os.sep + "params" + os.sep + model_name + '_best_params.tab'
    # file_name = '%s_best_params.tab' % model_name
    if os.path.isfile(file_dir):
        os.remove(file_dir)

    f = open("results_dnn_sent.txt", encoding="utf-8").readlines()
    for line in f:
        if len(line) > 20:
            line = line.split('{')
            corpus = re.search(r"'([\W\w]+?)'", line[1]).group(1)
            choices = line[-1].strip().strip('}').split(',')
            params = {}
            for choice in choices:
                choice = choice.split(':')
                assert len(choice) == 2
                param = re.search(r"'([\W\w]+?)'", choice[0]).group(1)
                value = choice[-1].strip()

                if type(space[param]) != list:
                    params[param] = value
                else:
                    params[param] = space[param][int(value)]

            f_params = open(file_dir, 'a', encoding='utf8')
            for k,v in params.items():
                f_params.write('%s\t%s\t%s\t%s\n' % (corpus, model_name, k, v))
            f_params.close()


def get_best_params(corpus, model_name):
    infile = script_dir + os.sep + "params" + os.sep + model_name + "_best_params.tab"
    lines = io.open(infile).readlines()
    params = ['num_layers', 'units2', 'dropout2', 'units1', 'dropout1', 'epoch', 'batch_size', 'optimizer', 'gram']
    str_feats = ['one', 'two', 'adadelta', 'adam', 'rmsprop', 'sgd']
    space = {}

    for line in lines:
        if "\t" in line:
            corp, clf_name, param, val = line.split("\t")
            val = val.strip()
            if corp == corpus:
                if param in params:
                    if val in str_feats:
                        space[param] = val
                    elif "0." in val:
                        space[param] = float(val)
                    else:
                        space[param] = int(val)

    return space


class DNNSentencer:
    def __init__(self,lang="eng",model="eng.rst.gum"):
        self.lang = lang
        self.name = "DNNSentencer"
        self.corpus = model
        self.model_path = model_dir + os.sep + model + "_dnn_sent" + ".hd5"
        if lang == "zho":
            vec = "cc.zho.300.vec_trim.vec"
        elif lang == "eng":
            vec = "glove.6B.300d_trim.vec"
        else:
            vec = "wiki.**lang**.vec_trim.vec".replace("**lang**",lang)
        self.vec_path = vec_dir + os.sep + vec
        self.space = get_best_params(self.corpus, self.name)


    def process_data(self,path,as_text=True):
        MB = 1024*1024
        word_embed = loadvec(self.vec_path)
        features = ['word', 'label', 'doc_id']
        output = read_conll(features, path, mode="sent", genre_pat=None, as_text=as_text)
        uni_embed = unique_embed(output, word_embed)
        #sys.stdout.write('##### Loaded dataset word embeddings.\n')
        n_gram_cols, label = n_gram(output, self.space['gram'])
        fc_vec = mergeWord2Vec(n_gram_cols, uni_embed)
        #sys.stdout.write("fc_vec %d MB\n" % (sys.getsizeof(fc_vec)/MB))

        X = fc_vec
        y = label

        return X, y


    def keras_model(self):
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Activation
        from keras import metrics

        np.random.seed(11)

        # create model
        model = Sequential()
        model.add(Dense(self.space['units1'], input_dim=300*self.space['gram']))
        model.add(Activation('relu'))
        model.add(Dropout(self.space['dropout1']))
        if self.space['num_layers'] == 'two':
            model.add(Dense(self.space['units2']))
            model.add(Activation('relu'))
            model.add(Dropout(self.space['dropout2']))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy',
                      optimizer=self.space['optimizer'],
                      metrics=['accuracy'])
        return model


    def get_multitrain_preds(self, X, y, multifolds):
        all_preds = []
        all_probas = []
        X_folds = np.array_split(X, multifolds)
        y_folds = np.array_split(y, multifolds)
        for i in range(multifolds):
            model = self.keras_model()
            X_train = np.vstack(tuple([X_folds[j] for j in range(multifolds) if j != i]))
            y_train = np.concatenate(tuple([y_folds[j] for j in range(multifolds) if j != i]))
            X_heldout = X_folds[i]
            sys.stdout.write("##### Training on fold " + str(i + 1) + " of " + str(multifolds) + "\n")
            model.fit(X_train, y_train, epochs=self.space['epoch'], batch_size=self.space['batch_size'], verbose=2)
            probas = model.predict(X_heldout)
            preds = [str(int(p > 0.5)) for p in probas]
            probas = [str(p[0]) for p in probas]
            all_preds += preds
            all_probas += probas

        pairs = list(zip(all_preds, all_probas))
        pairs = ["\t".join(pair) for pair in pairs]

        return "\n".join(pairs)


    def train(self,x_train,y_train,multitrain=False):
        model = self.keras_model()

        if multitrain:
            multitrain_preds = self.get_multitrain_preds(x_train, y_train, 5)
            multitrain_preds = "\n".join(multitrain_preds.strip().split("\n"))
            with io.open(script_dir + os.sep + "multitrain" + os.sep + self.name + '_' + self.corpus,'w',newline="\n") as f:
                sys.stdout.write("##### Serializing multitraining predictions\n")
                f.write(multitrain_preds)

        # Fit the model
        model.fit(x_train, y_train, epochs=self.space['epoch'], batch_size=self.space['batch_size'], verbose=2)
        model.save(self.model_path)


    def predict(self,test_path,as_text=True):
        # predict the model
        model = load_model(self.model_path)
        X_test, y_test = self.process_data(test_path,as_text=as_text)

        probas = model.predict(X_test)
        preds = [int(p > 0.5) for p in probas]
        probas = [p[0] for p in probas]

        # give dev F1 score
        if not as_text:
            print(classification_report(y_test, preds, digits=6))
            print(confusion_matrix(y_test, preds))

        return zip(preds, probas)


    def predict_cached(self,test_data):
        infile = script_dir + os.sep + "multitrain" + os.sep + self.name + '_' + self.corpus
        if os.path.exists(infile):
            pairs = io.open(infile).read().split("\n")
        else:
            sys.stdout.write("##### No multitrain file at: " + infile + "\n")
            sys.stdout.write("##### Falling back to live prediction for DNNSentencer\n")
            return self.predict(test_data,as_text=True)
        preds = [(int(pr.split()[0]), float(pr.split()[1])) for pr in pairs if "\t" in pr]
        return preds


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description='Input parameters')
    parser.add_argument('--corpus', '-c', action='store', dest='corpus', default="spa.rst.sctb", help='corpus name')
    parser.add_argument('--mode', '-m', action='store', dest='mode', default="train", choices=["train", "predict"], help='Please specify train or predict mode')
    parser.add_argument("-d", "--data_dir", default=os.path.normpath('../../../data'), help="Path to shared task data folder")
    parser.add_argument('--multitrain', action='store_true', help='whether to perform multitraining')

    args = parser.parse_args()

    start_time = time.time()

    data_folder = args.data_dir
    convert_params()    # train "build_dnn.py" without baseline first, then the function will read the optimized params from "results_dnn_sent.txt"
    if data_folder is None:
        data_folder = os.path.normpath(r'./../../../sharedtask2019/data/')
    corpora = args.corpus

    if corpora == "all":
        corpora = os.listdir(data_folder)
        corpora = [c for c in corpora if os.path.isdir(os.path.join(data_folder, c))]
    else:
        corpora = [corpora]

    for corpusname in corpora:
        if "." in corpusname:
            lang = corpusname.split(".")[0]
        else:
            lang = "eng"

        sys.stdout.write('##### Dataset from [%s]...\n' % corpusname)

        sentencer = DNNSentencer(lang=lang,model=corpusname)

        # Train model
        if args.mode == "train":
            train_path = data_folder + os.sep + corpusname + os.sep + corpusname + "_train.conll"
            X_train, y_train = sentencer.process_data(train_path,as_text=False)
            sentencer.train(X_train,y_train,multitrain=args.multitrain)

        # Now evaluate model
        dev_path = data_folder + os.sep + corpusname + os.sep + corpusname + "_dev.conll"
        predictions, probas = zip(*sentencer.predict(dev_path, as_text=False))

        elapsed = time.time() - start_time
        sys.stdout.write(str(timedelta(seconds=elapsed)) + "\n\n")
