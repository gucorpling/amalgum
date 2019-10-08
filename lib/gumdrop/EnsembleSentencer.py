#!/usr/bin/python
# -*- coding: utf-8 -*-

import io, os, sys, re, copy
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from glob import glob
from argparse import ArgumentParser

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

script_dir = os.path.dirname(os.path.realpath(__file__))
lib = os.path.abspath(script_dir + os.sep + "lib")

sys.path.append(script_dir + os.sep)

from .lib.sentencers.nltk_sent_wrapper import NLTKSentencer
from .lib.sentencers.udpipe_sent_wrapper import UDPipeSentencer
from .lib.sentencers.lr_sent_wrapper import LRSentencer
from .lib.sentencers.RuleBasedSentencer import RuleBasedSplitter
from .lib.sentencers.dnn_sent_wrapper import DNNSentencer
from .lib.conll_reader import read_conll, udpipe_tag, tt_tag, shuffle_cut_conllu
from .lib.tune import report_correlations, report_theils_u, hyper_optimize, get_best_params, get_best_score

#from lib.udpipe.run_udpipe import udpipe
from .lib.exec import exec_via_temp
#, ... some other estimators


class EnsembleSentencer:

	def __init__(self,lang="eng",model="eng.rst.gum",genre_pat="^(..)"):
		self.name = "EnsembleSentencer"
		self.lang = lang
		lang_map = {"deu":"german","eng":"english","spa":"spanish","fra":"french","nld":"dutch","rus":"russian","eus":"basque","por":"portuguese","zho":"chinese", "tur":"turkish"}
		self.long_lang = lang_map[lang] if lang in lang_map else lang
		self.estimators = []
		self.genre_pat = genre_pat
		try:
			self.udpipe_model = glob(os.path.abspath(os.path.join(lib,"udpipe",self.long_lang+"*.udpipe")))[0]
		except:
			sys.stderr.write("! Model not found for language " + self.long_lang + "*.udpipe in " + os.path.abspath(os.path.join([lib,"udpipe",self.long_lang+"*.udpipe"]))+"\n")
			sys.exit(0)
		self.udpipe_path = os.path.abspath(os.path.join(lib,"udpipe")) + os.sep
		self.corpus = model
		self.estimators.append(DNNSentencer(lang=lang,model=model))
		self.estimators.append(NLTKSentencer(lang=lang))
		self.estimators.append(UDPipeSentencer(lang=lang))
		self.estimators.append(LRSentencer(lang=lang,model=model))
		self.estimators.append(RuleBasedSplitter(lang=lang))

	def train(self,training_file,rare_thresh=100,clf_params=None,model_path=None,chosen_feats=None,tune_mode=None,size=None,as_text=False,multitrain=True,chosen_clf=None):
		"""
		Train the EnsembleSentencer. Note that the underlying estimators are assumed to be pretrained already.

		:param training_file: File in DISRPT shared task .conll format
		:param model_path: Path to dump pickled model to
		:param rare_thresh: Rank of rarest word to include (rarer items are replace with POS)
		:param genre_pat: Regex pattern with capturing group to extract genre from document names
		:param as_text: Boolean, whether the input is a string, rather than a file name to read
		:return:
		"""

		if tune_mode is not None and size is None and tune_mode != "hyperopt":
			size = 5000
			sys.stderr.write("o No sample size set - setting size to 5000\n")

		if not as_text:
			train = io.open(training_file,encoding="utf8").read().strip().replace("\r","") + "\n"
		else:
			train = training_file

		if size is not None:
			train = shuffle_cut_conllu(train,size)
		#tagged = udpipe_tag(train,self.udpipe_model)
		tagged = tt_tag(train,self.lang,preserve_sent=True)

		if model_path is None:  # Try default model location
			model_path = script_dir + os.sep + "models" + os.sep + self.corpus + "_ensemble_sent.pkl"

		if clf_params is None:
			# Default classifier parameters
			#clf_params = {"n_estimators":125,"min_samples_leaf":1, "max_depth":15, "max_features":None, "n_jobs":4, "random_state":42, "oob_score":True, "bootstrap":True}
			clf_params = {"n_estimators":100,"min_samples_leaf":1, "min_samples_split":5, "max_depth":10, "max_features":None, "n_jobs":4, "random_state":42, "oob_score":True, "bootstrap":True}

		if chosen_clf is None:
			chosen_clf = RandomForestClassifier(n_jobs=4,oob_score=True, bootstrap=True)
			chosen_clf.set_params(**clf_params)

		cat_labels = ["word","first","last","genre","pos","cpos"]
		num_labels = ["tok_len","tok_id"]

		train_feats, vocab, toks, firsts, lasts = read_conll(tagged,genre_pat=self.genre_pat,mode="sent",as_text=True,char_bytes=self.lang=="zho")
		gold_feats, _, _, _, _ = read_conll(train,mode="sent",as_text=True)
		gold_feats = [{"wid":0}] + gold_feats + [{"wid":0}]  # Add dummies to gold

		# Ensure that "_" is in the possible values of first/last for OOV chars at test time
		oov_item = train_feats[-1]
		oov_item["first"] = "_"
		oov_item["last"] = "_"
		oov_item["lemma"] = "_"
		oov_item["word"] = "_"
		oov_item["pos"] = "_"
		oov_item["cpos"] = "_"
		oov_item["genre"] = "_"
		train_feats.append(oov_item)
		train_feats = [oov_item] + train_feats
		toks.append("_")
		toks = ["_"] + toks

		vocab = Counter(vocab)
		top_n_words = vocab.most_common(rare_thresh)
		top_n_words, _ = zip(*top_n_words)

		headers = sorted(list(train_feats[0].keys()))
		data = []

		preds = {}

		for e in self.estimators:
			if multitrain and e.name in ["LRSentencer","DNNSentencer"]:
				pred = e.predict_cached(tagged)
			else:
				pred = e.predict(tagged)
			_, preds[e.name + "_prob"] = [list(x) for x in zip(*pred)]
			preds[e.name + "_prob"] = [0.0] + preds[e.name + "_prob"] + [0.0]  # Add dummy wrap for items -1 and +1
			headers.append(e.name + "_prob")
			num_labels.append(e.name + "_prob")

		for i, item in enumerate(train_feats):
			if item["word"] not in top_n_words:
				item["word"] = item["pos"]
			for e in self.estimators:
				item[e.name + "_prob"] = preds[e.name + "_prob"][i]

			feats = []
			for k in headers:
				feats.append(item[k])

			data.append(feats)

		data, headers, cat_labels, num_labels = self.n_gram(data, headers, cat_labels, num_labels)
		# No need for n_gram feats for the following:
		if "NLTKSentencer_prob_min1" in num_labels:
			num_labels.remove("NLTKSentencer_prob_min1")
			num_labels.remove("NLTKSentencer_prob_pls1")
		if "UDPipeSentencer_prob_min1" in num_labels:
			num_labels.remove("UDPipeSentencer_prob_min1")
			num_labels.remove("UDPipeSentencer_prob_pls1")
		if "LRSentencer_prob_min1" in num_labels:
			num_labels.remove("LRSentencer_prob_min1")
			num_labels.remove("LRSentencer_prob_pls1")
		if "RuleBasedSplitter_prob_min1" in num_labels:
			num_labels.remove("RuleBasedSplitter_prob_min1")
			num_labels.remove("RuleBasedSplitter_prob_pls1")
		if "DNNSentencer_prob_min1" in num_labels:
			num_labels.remove("DNNSentencer_prob_min1")
			num_labels.remove("DNNSentencer_prob_pls1")
		if "tok_id_min1" in num_labels:
			num_labels.remove("tok_id_min1")
			num_labels.remove("tok_id_pls1")
		if "genre_min1" in cat_labels:
			cat_labels.remove("genre_min1")
			cat_labels.remove("genre_pls1")

		# Use specific feature subset
		if chosen_feats is not None:
			new_cat = []
			new_num = []
			for feat in chosen_feats:
				if feat in cat_labels:
					new_cat.append(feat)
				elif feat in num_labels:
					new_num.append(feat)
			cat_labels = new_cat
			num_labels = new_num

		data = pd.DataFrame(data, columns=headers)
		data_encoded, multicol_dict = self.multicol_fit_transform(data, pd.Index(cat_labels))

		data_x = data_encoded[cat_labels+num_labels].values
		data_y = [int(t['wid'] == 1) for t in gold_feats]

		sys.stderr.write("o Learning...\n")

		if tune_mode is not None:
			# Randomize samples for training
			data_x = data_encoded[cat_labels+num_labels+["label"]].sample(frac=1,random_state=42)
			data_y = np.where(data_x['label'] == "_", 0, 1)
			data_x = data_x[cat_labels+num_labels]

			# Reserve 10% for validation
			val_x = data_x[int(len(data_y)/9):]
			val_y = data_y[int(len(data_y)/9):]
			data_x = data_x[:int(len(data_y)/9)]
			data_y = data_y[:int(len(data_y)/9)]

		if tune_mode == "importances":
			sys.stderr.write("o Measuring correlation of categorical variables\n")
			theil_implications = report_theils_u(val_x,cat_labels)
			for (var1, var2) in theil_implications:
				if var1 in cat_labels and var2 in cat_labels:
					drop_var = var2
					u = theil_implications[(var1, var2)]
					sys.stderr.write("o Removed feature " + drop_var + " due to Theil's U " + str(u)[:6] + " of " + var1 + "->" + var2 + "\n")
					cat_labels.remove(drop_var)

			sys.stderr.write("o Measuring correlation of numerical variables\n")
			cor_mat = report_correlations(val_x[num_labels],thresh=0.95)
			for (var1, var2) in cor_mat:
				if var1 in num_labels and var2 in num_labels:
					drop_var = var2
					corr_level = cor_mat[(var1, var2)]
					sys.stderr.write("o Removed feature " + drop_var + " due to correlation " + str(corr_level) + " of " + var1 + ":" + var2 + "\n")
					num_labels.remove(drop_var)

			return cat_labels, num_labels

		if tune_mode in ["paramwise","full"]:
			best_params = {}
			# Tune individual params separately for speed, or do complete grid search if building final model
			params_list = [{"n_estimators":[100,125,150]},
						   {'max_depth': [10,15,20,None]},
						   {"min_samples_split": [5, 10, 15]},
						   {"min_samples_leaf":[1,2,3]},
						   {"max_features":[None,"sqrt","log2"]}]
			if tune_mode == "full":
				# Flatten dictionary if doing full CV
				params_list = [{k: v for d in params_list for k, v in d.items()}]
			for params in params_list:
				base_params = copy.deepcopy(clf_params)  # Copy default params
				for p in params:
					if p in base_params:  # Ensure base_params don't conflict with grid search params
						base_params.pop(p)
				grid = GridSearchCV(RandomForestClassifier(**base_params),params,cv=3,n_jobs=4,error_score="raise",refit=False)
				grid.fit(data_x,data_y)
				for param in params:
					best_params[param] = grid.best_params_[param]
			with io.open("best_params.tab",'a',encoding="utf8") as bp:
				corpus = os.path.basename(training_file).split("_")[0]
				best_clf = RandomForestClassifier(**best_params)
				clf_name = best_clf.__class__.__name__
				for k, v in best_params.items():
					bp.write("\t".join([corpus, clf_name, k, str(v)]))
				bp.write("\n")
			return best_clf, best_params
		elif tune_mode == "hyperopt":
			from hyperopt import hp
			from hyperopt.pyll.base import scope
			space = {
				'n_estimators': scope.int(hp.quniform('n_estimators', 50, 150, 10)),
				'max_depth': scope.int(hp.quniform('max_depth', 5, 30, 1)),
				'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
				'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 10, 1)),
				'max_features': hp.choice('max_features', ["sqrt", None, 0.5, 0.7, 0.9]),
				'clf': hp.choice('clf', ["rf","et","gbm"])
			}
			#space = {
			#	'n_estimators': scope.int(hp.quniform('n_estimators', 50, 150, 10)),
			#	'max_depth': scope.int(hp.quniform('max_depth', 3, 30, 1)),
			#	'eta': scope.float(hp.quniform('eta', 0.01, 0.2, 0.01)),
			#	'gamma': scope.float(hp.quniform('gamma', 0.01, 0.2, 0.01)),
			#	'colsample_bytree': hp.choice('colsample_bytree', [0.4,0.5,0.6,0.7,1.0]),
			#	'subsample': hp.choice('subsample', [0.5,0.6,0.7,0.8,1.0]),
			#	'clf': hp.choice('clf', ["xgb"])
			#}

			best_clf, best_params = hyper_optimize(data_x,data_y,cat_labels=cat_labels,space=space,max_evals=50)
			return best_clf, best_params
		else:
			clf = chosen_clf
			clf.set_params(**clf_params)
			if clf.__class__.__name__ in ["RandomForestClassifier","ExtraTreesClassifier","XGBClassifier"]:
				clf.set_params(**{"n_jobs":3,"random_state":42,"oob_score":True,"bootstrap":True})
			else:
				clf.set_params(**{"random_state":42})
			clf.fit(data_x,data_y)

		feature_names = cat_labels + num_labels

		zipped = zip(feature_names, clf.feature_importances_)
		sorted_zip = sorted(zipped, key=lambda x: x[1], reverse=True)
		sys.stderr.write("o Feature importances:\n\n")
		for name, importance in sorted_zip:
			sys.stderr.write(name + "=" + str(importance) + "\n")

		if hasattr(clf, "oob_score_"):
			sys.stderr.write("\no OOB score: " + str(clf.oob_score_)+"\n")

		sys.stderr.write("\no Serializing model...\n")

		joblib.dump((clf, num_labels, cat_labels, multicol_dict, top_n_words, firsts, lasts), model_path, compress=3)

	def plain2disrpt(self, text, genre="voyage"):
		"""
		Turns vertical TT SGML file with one token or tag per line to just tokens in DISRPT style conll format
		:param text:
		:return: conll
		"""

		lines = text.strip().split("\n")
		output = []
		counter = 1
		for line in lines:
			if line.startswith("<") and line.endswith(">"):
				continue
			output.append("\t".join([str(counter),line] + ["_"]*8))
			counter += 1
		output = ["# newdoc id = GUM_"+genre+"_document"] + output
		return "\n".join(output)

	def predict(self, infile, model_path=None, eval_gold=False, as_text=False, plain=False, genre="voyage"):
		"""
		Predict sentence splits using an existing model

		:param infile: File in DISRPT shared task *.tok or *.conll format (sentence breaks will be ignored in .conll)
		:param model: Pickled model file, default: models/sent_model.pkl
		:param eval_gold: Whether to score the prediction; only applicable if using a gold .conll file as input
		:param as_text: Boolean, whether the input is a string, rather than a file name to read
		:return: tokenwise binary prediction vector if eval_gold is False, otherwise prints evaluation metrics and diff to gold
		"""

		if model_path is None:  # Try default model location
			model_path = script_dir + os.sep + "models" + os.sep + self.corpus + "_ensemble_sent.pkl"

		clf, num_labels, cat_labels, multicol_dict, vocab, firsts, lasts = joblib.load(model_path)

		if as_text:
			conllu = infile
		else:
			conllu = io.open(infile,encoding="utf8").read()

		if plain:
			conllu = self.plain2disrpt(conllu,genre=genre)

		#tagged = udpipe_tag(conllu,self.udpipe_model)
		tagged = tt_tag(conllu,self.lang)

		train_feats, _, toks, _, _ = read_conll(tagged,genre_pat=self.genre_pat,mode="sent",as_text=True,char_bytes=self.lang=="zho")
		headers = sorted(list(train_feats[0].keys()))

		data = []

		preds = {}
		for e in self.estimators:
			pred = e.predict(tagged)
			_, preds[e.name + "_prob"] = [list(x) for x in zip(*pred)]
			headers.append(e.name + "_prob")

		genre_warning = False
		for i, item in enumerate(train_feats):
			item["first"] = item["word"][0] if item["word"][0] in firsts else "_"
			item["last"] = item["word"][-1] if item["word"][-1] in lasts else "_"
			if "genre" in cat_labels:
				if item["genre"] not in multicol_dict["encoder_dict"]["genre"].classes_:  # New genre not in training data
					if not genre_warning:
						sys.stderr.write("! WARN: Genre not in training data: " + item["genre"] + "; suppressing further warnings\n")
						genre_warning = True
					item["genre"] = "_"
			if "pos" in cat_labels:
				if item["pos"] not in multicol_dict["encoder_dict"]["pos"].classes_:
					item["pos"] = "_"
			if "cpos" in cat_labels:
				if item["cpos"] not in multicol_dict["encoder_dict"]["cpos"].classes_:
					item["cpos"] = "_"
			if item["word"] not in vocab and "word" in multicol_dict["encoder_dict"]:
				if item["pos"] in multicol_dict["encoder_dict"]["word"].classes_:
					item["word"] = item["pos"]
				else:
					item["word"] = "_"
			for e in self.estimators:
				item[e.name + "_prob"] = preds[e.name + "_prob"][i]

			feats = []
			for k in headers:
				feats.append(item[k])

			data.append(feats)

		data, headers, _, _ = self.n_gram(data,headers,[],[])

		data = pd.DataFrame(data, columns=headers)
		data_encoded = self.multicol_transform(data,columns=multicol_dict["columns"],all_encoders_=multicol_dict["all_encoders_"])

		data_x = data_encoded[cat_labels+num_labels].values
		pred = clf.predict(data_x)

		# Ensure first token in document is always a sentence break
		for i, x in enumerate(data_encoded["tok_id"].values):
			if x == 1:
				pred[i] = 1

		if eval_gold:
			gold_feats, _,_,_,_ = read_conll(conllu,genre_pat=self.genre_pat,mode="sent",as_text=True)
			gold = [int(t['wid'] == 1) for t in gold_feats]
			conf_mat = confusion_matrix(gold, pred)
			sys.stderr.write(str(conf_mat) + "\n")
			true_positive = conf_mat[1][1]
			false_positive = conf_mat[0][1]
			false_negative = conf_mat[1][0]
			prec = true_positive / (true_positive + false_positive)
			rec = true_positive / (true_positive + false_negative)
			f1 = 2*prec*rec/(prec+rec)
			sys.stderr.write("P: " + str(prec) + "\n")
			sys.stderr.write("R: " + str(rec) + "\n")
			sys.stderr.write("F1: " + str(f1) + "\n")
			with io.open("diff.tab",'w',encoding="utf8") as f:
				for i in range(len(gold)):
					f.write("\t".join([toks[i],str(gold[i]),str(pred[i])])+"\n")
			return conf_mat, prec, rec, f1
		else:
			return pred

	def optimize(self, train, rare_thresh=100, size=5000, tune_mode="paramwise", cached_params=False,as_text=False):

		# Estimate useful features on a random sample of |size| instances
		selected_cat, selected_num = self.train(train,model_path=None,rare_thresh=100,as_text=False,size=size,tune_mode="importances")
		selected_feats = selected_cat + selected_num
		sys.stderr.write("o Chose "+str(len(selected_feats))+" features: " + ",".join(selected_feats)+"\n")
		sys.stderr.write("o Tuning hyperparameters\n\n")

		# Optimize hyperparameters via grid search or hyperopt
		if cached_params:
			best_clf, best_params, _ = get_best_params(self.corpus, self.name)
			sys.stderr.write("\no Using cached best hyperparameters\n")
		else:
			best_clf, best_params = self.train(train,rare_thresh=rare_thresh,tune_mode=tune_mode,size=size,as_text=as_text)
			sys.stderr.write("\no Found best hyperparameters\n")
		for key, val in best_params.items():
			sys.stderr.write(key + "\t" + str(val) + "\n")
		sys.stderr.write("\n")

		return best_clf, selected_feats, best_params

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

		multicol_dict = {"encoder_dict":encoder_dict, "all_classes_":all_classes_,"all_encoders_":all_encoders_,"columns": columns}
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
			dframe.loc[:, column] = all_encoders_[idx].transform(dframe.loc[:, column].values)
		return dframe


	@staticmethod
	def n_gram(data, headers, cat_labels, num_labels):
		"""
		Turns unigram feature list into list of tri-skipgram features by adding features of adjacent tokens

		:param data: List of observations, each an ordered list of feature values
		:param headers: List of all feature names in the data
		:param cat_labels: List of categorical features to be used in model
		:param num_labels: List of numerical features to be used in the model
		:return: Modified data, headers and label lists including adjacent token properties
		"""
		n_grammed = []

		for i, tok in enumerate(data):
			if i == 0:
				n_grammed.append(data[-1]+tok+data[i+1])
			elif i == len(data) - 1:
				n_grammed.append(data[i-1]+tok+data[0])
			else:
				n_grammed.append(data[i-1]+tok+data[i+1])

		n_grammed_headers = [header + "_min1" for header in headers] + headers + [header + "_pls1" for header in headers]
		n_grammed_cat_labels = [lab + "_min1" for lab in cat_labels] + cat_labels + [lab + "_pls1" for lab in cat_labels]
		n_grammed_num_labels = [lab + "_min1" for lab in num_labels] + num_labels + [lab + "_pls1" for lab in num_labels]

		return n_grammed, n_grammed_headers, n_grammed_cat_labels, n_grammed_num_labels


if __name__ == "__main__":

	p = ArgumentParser()
	p.add_argument("-c","--corpus",default="spa.rst.sctb")
	p.add_argument("-d","--data_dir",default=os.path.normpath("../data"),help="Path to shared task data folder")
	p.add_argument("-s","--size",type=int,default=220000,help="Maximum sample size to train on")
	p.add_argument("-t","--tune_mode",default="paramwise",choices=["paramwise","full","hyperopt"])
	p.add_argument("-b","--best_params",action="store_true",help="Load best parameters from file")
	p.add_argument("--mode",choices=["train","test","train-test","optimize-train-test"],default="train-test")
	p.add_argument("--eval_test",action="store_true",help="Evaluate on test, not dev")
	opts = p.parse_args()

	specific_corpus = opts.corpus
	data_dir = opts.data_dir

	corpora = os.listdir(data_dir)
	if specific_corpus == "all":
		corpora = [c for c in corpora if os.path.isdir(os.path.join(data_dir, c))]
	else:
		corpora = [c for c in corpora if os.path.isdir(os.path.join(data_dir, c)) and c== specific_corpus]

	for corpus in corpora:

		# Set corpus and file information
		genre_pat = "^(..)"  # By default 2 first chars of docname identify genre
		train = os.path.join(data_dir,corpus, corpus + "_train.conll")
		dev = os.path.join(data_dir, corpus, corpus + "_dev.conll")
		test = os.path.join(data_dir, corpus, corpus + "_test.conll")
		model_path = "models" + os.sep + corpus + "_sent.pkl"

		# Run automatic POS tagging with UDPipe
		if "." in corpus:
			lang = corpus.split(".")[0]
		else:
			lang = "eng"

		sys.stderr.write("\no Corpus "+corpus+"\n")

		# Predict sentence splits
		e = EnsembleSentencer(lang=lang,model=corpus)

		# Special genre patterns
		if "gum" in corpus:
			e.genre_pat = "GUM_(.+)_.*"

		# For large corpora we do not perform multitraining, just use a subset
		large_corpora = ["eng.pdtb.pdtb","eng.rst.rstdt","rus.rst.rrt","tur.pdtb.tdb"]
		size = opts.size if corpus in large_corpora else None

		best_params = None
		if "optimize" in opts.mode:
			best_clf, vars, best_params = e.optimize(train,size=size,tune_mode=opts.tune_mode,cached_params=opts.best_params)
			if "best_score" in best_params:
				best_params.pop("best_score")
			# Now train on whole training set with those variables
			sys.stderr.write("\no Training best configuration\n")
			e.train(train,rare_thresh=200,clf_params=best_params,as_text=False,chosen_clf=best_clf,chosen_feats=vars,size=220000)
		elif "train" in opts.mode:
			tune_mode=None if opts.tune_mode != "hyperopt" else "hyperopt"
			feats = None
			params = None
			best_clf = None
			if opts.best_params:
				best_clf, params, feats = get_best_params(corpus, "EnsembleSentencer")
				if len(feats) == 0:
					feats = None
			e.train(train,chosen_feats=feats,as_text=False,tune_mode=tune_mode,clf_params=params,chosen_clf=best_clf,size=220000)
		if "test" in opts.mode:
			if opts.eval_test:
				conf_mat, prec, rec, f1 = e.predict(test,eval_gold=True,as_text=False)
			else:
				conf_mat, prec, rec, f1 = e.predict(dev,eval_gold=True,as_text=False)
				if best_params is not None and "optimize" in opts.mode:  # For optimization check if this is a new best score
					prev_best_score = get_best_score(corpus,"EnsembleSentencer")
					if f1 > prev_best_score:
						sys.stderr.write("o New best F1: " + str(f1) + "\n")
						with io.open(lib + os.sep + "sentencers" + os.sep + "params" + os.sep + "EnsembleSentencer_best_params.tab",'a',encoding="utf8") as bp:
							for k, v in best_params.items():
								bp.write("\t".join([corpus, best_clf.__class__.__name__, k, str(v)])+"\n")
							bp.write("\t".join([corpus, best_clf.__class__.__name__, "features", ",".join(vars)])+"\n")
							bp.write("\t".join([corpus, best_clf.__class__.__name__, "best_score", str(f1)])+"\n\n")





