import io, re, sys, os
import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp
from glob import glob
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep

np.random.seed(42)


TEST_DOCS = ["GUM_academic_discrimination","GUM_academic_eegimaa","GUM_bio_dvorak","GUM_bio_jespersen","GUM_fiction_falling","GUM_fiction_teeth","GUM_interview_hill","GUM_interview_libertarian","GUM_interview_mcguire","GUM_news_expo","GUM_news_nasa","GUM_news_sensitive","GUM_voyage_oakland","GUM_voyage_thailand","GUM_voyage_vavau","GUM_whow_cactus","GUM_whow_cupcakes","GUM_whow_mice"]
DEV_DOCS = ["GUM_academic_exposure","GUM_academic_librarians","GUM_bio_byron","GUM_bio_emperor","GUM_fiction_beast","GUM_fiction_lunre","GUM_interview_cyclone","GUM_interview_gaming","GUM_interview_peres","GUM_news_defector","GUM_news_homeopathic","GUM_news_iodine","GUM_voyage_athens","GUM_voyage_coron","GUM_voyage_isfahan","GUM_whow_joke","GUM_whow_overalls","GUM_whow_skittles"]

class STypeClassifier:

	def __init__(self):
		# Names of raw features, before encoding/vectorizing
		self.feat_names = ["root_pos", "words", "postags", "funcs", "parents","wordfuncs","hasverb","hassubj","len",
						   "case","root_modal","do_support","root_combo_pos"]
		self.model_loaded = False
		try:
			self.clf, self.transformer = joblib.load(script_dir + "stype.pkl")
			self.model_loaded = True
		except:
			sys.stderr.write("Could not find model stype.pkl\nDo you need to train STypeClassifier?\n")


	@staticmethod
	def get_column_names_from_ColumnTransformer(column_transformer):
		col_name = []
		for transformer_in_columns in column_transformer.transformers_[:-1]:#the last transformer is ColumnTransformer's 'remainder'
			raw_col_name = transformer_in_columns[2]
			if isinstance(transformer_in_columns[1],Pipeline):
				transformer = transformer_in_columns[1].steps[-1][1]
			else:
				transformer = transformer_in_columns[1]
			try:
				names = transformer.get_feature_names()
			except AttributeError: # if no 'get_feature_names' function, use raw column name
				names = raw_col_name
			if isinstance(names,np.ndarray): # eg.
				col_name += names.tolist()
			elif isinstance(names,list):
				col_name += names
			elif isinstance(names,str):
				col_name.append(names)
		return col_name


	@staticmethod
	def featurize_sentence(conll_sent):
		conll_lines = conll_sent.split("\n")
		word_list = []
		xpos_list = []
		deprel_list = []
		rel_pair_list = []
		head_list = []
		root_pos = ""
		word_func_list = []
		for line in conll_lines:
			if re.match(r'^[0-9]+\t',line) is not None:
				toknum, word, lemma, upos, xpos, morph, head, deprel, _, _ = line.split("\t")
				deprel = re.sub(r':.+','',deprel)
				word_list.append(word)
				xpos_list.append(xpos)
				deprel_list.append(deprel)
				head_list.append(head)
				word_func_list.append(word + "_" + deprel)
				if head == "0":
					root_pos = xpos

		# Propagate root to conj or parataxis child of root
		changes = {}
		root_combo_pos = "_"
		for i, deprel in enumerate(deprel_list):
			if deprel != "root":
				parent_rel = deprel_list[int(head_list[i])-1]
				if parent_rel =="root":
					if deprel in ["conj","parataxis"]:
						changes[i] = "root"
						root_combo_pos = root_pos + "_" + xpos_list[i]
		for i in changes:
			deprel_list[i] = changes[i]

		# Now look for root auxiliaries
		changes = {}
		do_support = "0"
		for i, deprel in enumerate(deprel_list):
			if deprel != "root":
				parent_rel = deprel_list[int(head_list[i])-1]
				if parent_rel =="root":
					if deprel in ["aux","aux:pass"]:
						changes[i] = "rt" + deprel
						if word_list[i].lower() in ["do","did"]:
							do_support = "1"
		for i in changes:
			deprel_list[i] = changes[i]

		for i, deprel in enumerate(deprel_list):
			if deprel != "root":
				parent_rel = deprel_list[int(head_list[i])-1]
				rel_pair = deprel + "_" + parent_rel
			else:
				rel_pair = "root"
			rel_pair_list.append(rel_pair)

		verb = "1" if any([p.startswith("V") for p in xpos_list]) else "0"
		subj = "1" if any(["subj" in f for f in deprel_list]) else "0"

		case = "0"
		if all([w.isupper() for w in word_list]):
			case = "3"
		elif all([w.istitle() for w in word_list]):
			case = "2"
		else:
			title_prop = len([w for w in word_list if w.istitle()])/len(word_list)
			if title_prop > 0.5:
				case = "1"

		root_modal = "0"
		for i, rel in enumerate(deprel_list):
			if rel == "rtaux":
				if xpos_list[i] == "MD":
					root_modal = "1"

		word_list = " ".join(["#"] + word_list + ["#"])
		xpos_list = " ".join(["#"] + xpos_list + ["#"])
		deprel_list = " ".join(["#"] + deprel_list + ["#"])
		rel_pair_list = " ".join(["#"] + rel_pair_list + ["#"])
		word_func_list = " ".join(["#"] + word_func_list + ["#"])

		length = str(word_list.count(" ") - 1)

		return [root_pos, word_list, xpos_list, deprel_list, rel_pair_list, word_func_list, verb, subj, length, case, root_modal, do_support, root_combo_pos]

	def make_transformer(self, dataframe):
		"""Feature engineering pipeline"""

		white_funcs = ["nsubj","obj","csubj","nsubj:pass","csubj:pass","xcomp","ccomp","expl","advcl","acl","acl:relcl","cop",
					  "mark","appos","parataxis","cc","conj","vocative"]
		white_words = ["! #","!","# !","# 'll","# (","# )","# .","# :","# ;","# ?","# and","# are","# as","# be","# being","# can","# cf.","# could","# did","# do","# had","# have","# how","# i","# if","# is","# it","# ll","# look","# may","# might","# note","# of","# or","# say","# see","# shall","# should","# so","# some","# that","# then","# there","# this","# to","# was","# what","# where","# who","# whom","# why","# will","# you","# your","'ll","( #","(",") #",")",", so",". #",".",": #",":","; #",";","? #","?","and","are #","are","as","be #","be","being #","being","can #","can","cf.","could #","could","did #","did","do #","do","had #","had","have #","have","how #","how","i #","i","if","is #","is","it #","it","ll","look #","look","may #","may","might #","might","note","of #","of","or","say #","say","see","shall #","shall","should #","should","so #","so","some #","some","that","then #","then","there #","there","this","to #","to","was #","was","what #","what","where #","where","who #","who","whom #","whom","why #","why","will #","will","you #","you","your #","your","been",
					  "n't","should n't","could n't","wo n't","wo","must","must n't","have to","not","got to","gotta","need to","need n't",
					  "which","# which"]

		pos_tags = set()
		dataframe['postags'].str.split().apply(pos_tags.update)
		if "#" in pos_tags:
			pos_tags.remove("#")
		root_combo_tags = ['_']
		for tag1 in pos_tags:
			for tag2 in pos_tags:
				root_combo_tags.append(tag1 + "_" + tag2)
		root_combo_tags = [root_combo_tags]
		pos_tags = [list(pos_tags)]

		func_vectorizer = CountVectorizer(vocabulary=white_funcs,token_pattern='[^\s]+')
		word_func_vectorizer = CountVectorizer(max_features=300,token_pattern='[^\s]+')#,vocabulary=allowed_vocab)
		func_pair_vectorizer = CountVectorizer(max_features=200,token_pattern='[^\s]+')#,vocabulary=allowed_vocab)
		pos_vectorizer = CountVectorizer(max_features=200,token_pattern='[^\s]+',ngram_range=(1,2))#,vocabulary=allowed_vocab)
		word_vectorizer = CountVectorizer(max_features=500,token_pattern='[^\s]+',ngram_range=(1,2),vocabulary=white_words)#,vocabulary=allowed_vocab)#,vocabulary=white_words)
		root_pos_encoder = OrdinalEncoder(categories=pos_tags)
		root_combo_pos_encoder = OrdinalEncoder(categories=root_combo_tags)

		column_trans = make_column_transformer(
			(func_vectorizer, 'funcs'),
			(word_func_vectorizer, 'wordfuncs'),
			(func_pair_vectorizer, 'parents'),
			(pos_vectorizer, 'postags'),
			(word_vectorizer, 'words'),
			(root_pos_encoder, ['root_pos']),
			(root_combo_pos_encoder, ['root_combo_pos']),
			('passthrough',['hasverb','hassubj','len','case','root_modal','do_support'])
			)

		return column_trans


	def predict(self,conllu_string):
		"""Predict a list of sentence types for the conllu string"""
		sents = conllu_string.strip().split("\n\n")
		featlist = [self.featurize_sentence(sent) for sent in sents]
		df = pd.DataFrame(featlist,columns=self.feat_names)
		X = self.transformer.transform(df)
		preds = self.clf.predict(X)
		return preds

	def predict_from_dir(self, conllu_dir, extension="conllu"):
		"""Take directory of conllu files and return list of lists, each containing strings predicting the stype
		of each sentence in a file"""

		if not conllu_dir.endswith(os.sep):
			conllu_dir += os.sep
		files = glob(conllu_dir + "*." + extension)
		preds = []
		for file_ in files:
			conll_string = io.open(file_,encoding="utf8").read()
			preds.append(self.predict(conll_string))
		return preds

	def train(self, root_dir, test_docs=None, dev_docs=None, write_table=False, train_with_dev=False):
		"""
		Train the classifier. The root_dir should contain two sub directories dep/ and xml/ with conllu parses and
		XML files indicating sentence types. Document names and token counts must match in the two directories.

		xml/docname.xml in TreeTagger/CWB format, with sentence type tags like <s type="decl">
		dep/docname.conllu with conllu dependency files

		:param root_dir: directory with dep/ and xml/
		:param test_docs: list of document names (without extension) used as a test set
		:param dev_docs: list of document names (without extension) used as a dev set
		:param write_table: whether to dump the featurized data table
		:param train_with_dev: adds dev set to training data
		:return: None
		"""

		if dev_docs is None:
			dev_docs = DEV_DOCS
		if test_docs is None:
			test_docs = TEST_DOCS

		if not root_dir.endswith(os.sep):
			root_dir += os.sep
		xml_dir = root_dir + "xml" + os.sep
		dep_dir = root_dir + "dep" + os.sep

		stypes = defaultdict(lambda : defaultdict(str))

		# Read gold s_types
		for file_ in glob(xml_dir +"*.xml"):
			lines = io.open(file_,encoding="utf8").readlines()
			docname = os.path.basename(file_).replace(".xml","")
			counter = 0
			s_type = None
			for line in lines:
				line = line.strip()
				if len(line) == 0:
					continue
				if "s type=" in line:
					s_type = re.search(r'type="([^"]+)"',line).group(1)
				elif not (line.startswith("<") and line.endswith(">")):
					stypes[docname][counter] = s_type
					counter += 1

		# Read conll
		headers = self.feat_names + ["partition","doc","label"]

		facts = []
		for file_ in glob(dep_dir + "*.conllu"):
			docname = os.path.basename(file_).replace(".conllu","")
			partition = "train"
			if docname in test_docs:
				partition = "test"
			elif docname in dev_docs:
				partition = "dev"

			sents = io.open(file_,encoding="utf8").read().strip().split("\n\n")
			counter = 0
			for sent in sents:
				slen = len(re.findall(r'^([0-9]+)\t',sent,flags=re.MULTILINE))
				feats = self.featurize_sentence(sent)
				stype = stypes[docname][counter]
				entry = feats + [partition, docname, stype]
				facts.append(entry)
				counter += slen

		df = pd.DataFrame(facts,columns=headers)
		train = df[df["partition"]=="train"]
		dev = df[df["partition"]=="dev"]
		test = df[df["partition"]=="test"]

		# Transform while dropping unused columns to prevent error at test time when they're missing
		self.transformer = self.make_transformer(df.drop(["doc","partition","label"],axis=1))
		self.transformer.fit(train.drop(["doc","partition","label"],axis=1))
		X = self.transformer.transform(train.drop(["doc","partition","label"],axis=1))
		X_dev = self.transformer.transform(dev.drop(["doc","partition","label"],axis=1))
		X_test = self.transformer.transform(test.drop(["doc","partition","label"],axis=1))

		xg = XGBClassifier(random_state=42, n_jobs=3,colsample_bytree=0.8,max_depth=11,n_estimators=100,gamma=0.1)

		sys.stderr.write("o Training...\n")
		xg.fit(X,train["label"])
		preds = xg.predict(X_dev)

		sys.stderr.write("o Performance on dev:\n")
		sys.stderr.write(str(classification_report(dev["label"],preds)))

		if train_with_dev:
			sys.stderr.write("\no Retraining on train+dev\n")
			X_devtrain = self.transformer.transform(pd.concat([train,dev]).drop(["doc","partition","label"],axis=1))
			y_devtrain = pd.concat([train["label"],dev["label"]])
			xg.fit(X_devtrain,y_devtrain)
			preds = xg.predict(X_test)
			sys.stderr.write("o Performance on test:\n")
			sys.stderr.write(classification_report(test["label"],preds))

		joblib.dump((xg,self.transformer),script_dir+"stype.pkl")

		if write_table:
			rows = ["\t".join([row for row in facts])]
			with io.open("stypes_data.tab",'w',encoding="utf8",newline="\n") as f:
				f.write("\n".join(rows) + "\n")


if __name__ == "__main__":
	from argparse import ArgumentParser, RawTextHelpFormatter

	usage = "Training:\n  python stype_classifier.py -td gum/_build/src/\n"
	usage += "Predicting:\n  python stype_classifier.py conll_files_dir/\n"

	p = ArgumentParser(epilog=usage, formatter_class=RawTextHelpFormatter)
	p.add_argument("-t","--train",action="store_true")
	p.add_argument("-d","--devtrain",action="store_true",help="train with dev")
	p.add_argument("-w","--write",action="store_true",help="write training data table")
	p.add_argument("target_path",default=None,help="directory of conll files or single conllu file for tagging, \
					or directory with xml/ and dep/ sub-directories if training")

	opts = p.parse_args()

	target_path = opts.target_path

	stp = STypeClassifier()
	if opts.train:
		stp.train(target_path,train_with_dev=opts.devtrain,write_table=opts.write)
	else:
		stp = STypeClassifier()
		if os.path.isfile(target_path):
			preds = stp.predict(io.open(target_path,encoding="utf8").read())
		else:
			preds = stp.predict_from_dir(target_path + os.sep + "dep" + os.sep)
		print(preds)
