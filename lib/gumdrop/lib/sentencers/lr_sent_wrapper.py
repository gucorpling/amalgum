import re, sys, os, operator, argparse, pickle, io, time, gc
from scipy.sparse import csr_matrix
import pandas as pd
from collections import Counter
from glob import glob
from sklearn.linear_model import LogisticRegressionCV
from datetime import timedelta
script_dir = os.path.dirname(os.path.realpath(__file__))
lib = os.path.abspath(script_dir + os.sep + "..")
sys.path.append(lib)
from conll_reader import read_conll, tt_tag, udpipe_tag, shuffle_cut_conllu, get_multitrain_preds
from random import seed, shuffle

seed(42)

# Allow package level imports in module
model_dir = os.path.abspath(script_dir + os.sep + ".." + os.sep + ".." + os.sep + "models" + os.sep)

# categorical feature to multiple one-hot vectors
def cattodummy(cat_vars, data):
	for var in cat_vars:
		cat_list = pd.get_dummies(data[var], prefix=var, drop_first=True)
		data1 = data.join(cat_list)
		data = data1
	data.drop(cat_vars, axis=1, inplace=True)
	return data


# categorical feature to one label mapped vector
def cattolabel(cat_vars, data):
	from sklearn.preprocessing import LabelEncoder
	lb_make = LabelEncoder()
	for var in cat_vars:
		data[var+'label'] = lb_make.fit_transform(data[var])
	data.drop(cat_vars, axis=1, inplace=True)
	return data


def copyforprevnext(data, prevnexttype):

	data1 = data.drop(['gold_seg'], axis=1)

	col_names = data1.columns.values.tolist()

	# renaming column headers
	new_names = []
	for name in col_names:
	#	data1.rename(columns={name: prevnexttype+'_'+name}, inplace=True) # @Logan - this is slow!
		new_names.append(prevnexttype+'_'+name)  # Just make new names and reassign to df.columns

	data1.columns = new_names

	nrows = data1.shape[0]

	d2 = data1.copy()

	if prevnexttype == 'prev':
		d2 = pd.concat([d2.loc[nrows-1:nrows-1, :], d2.loc[0:nrows-2, :]], axis=0).reset_index(drop=True)
	elif prevnexttype == 'prevprev':
		d2 = pd.concat([d2.loc[nrows-2:nrows-1, :], d2.loc[0:nrows-3, :]], axis=0).reset_index(drop=True)
	elif prevnexttype == 'prevprevprev':
		d2 = pd.concat([d2.loc[nrows - 3:nrows - 1, :], d2.loc[0:nrows - 4, :]], axis=0).reset_index(drop=True)

	elif prevnexttype == 'next':
		d2 = pd.concat([d2.loc[1:nrows-1, :], d2.loc[0:0, :]], axis=0).reset_index(drop=True)
	elif prevnexttype == 'nextnext':
		d2 = pd.concat([d2.loc[2:nrows-1, :], d2.loc[0:1, :]], axis=0).reset_index(drop=True)
	elif prevnexttype == 'nextnextnext':
		d2 = pd.concat([d2.loc[3:nrows-1, :], d2.loc[0:2, :]], axis=0).reset_index(drop=True)

	else:
		print("x wrong prev or next type.")

	# print(d2.head(10))

	return d2


def gettokfreq(infile,as_text=False):
	toks = []
	if as_text:
		f_lines = infile.split("\n")
	else:
		with io.open(infile, 'r', encoding='utf8') as f_in:
			f_lines = f_in.readlines()
	for line in f_lines:
		if '\t' in line:
			toks.append(line.split('\t')[1])
	return Counter(toks)


class LRSentencer:

	def __init__(self,lang="eng",model="eng.rst.gum",windowsize=5,use_words=False):
		self.lang = lang
		self.name = "LRSentencer"
		self.corpus = model
		lang_map = {"deu":"german","eng":"english","spa":"spanish","fra":"french","nld":"dutch","rus":"russian","eus":"basque","por":"portuguese","zho":"chinese","tur":"turkish"}
		self.long_lang = lang_map[lang] if lang in lang_map else lang
		try:
			self.udpipe_model = glob(os.path.abspath(os.path.join(lib,"udpipe",self.long_lang+"*.udpipe")))[0]
			self.udpipe_path = os.path.abspath(os.path.join(lib,"udpipe")) + os.sep
		except:
			pass
			#sys.stderr.write("! Model not found for language " + self.long_lang + "*.udpipe in " + os.path.abspath(os.path.join([lib,"udpipe",self.long_lang+"*.udpipe"]))+"\n")
			#sys.exit(0)
		self.model_path = model_dir + os.sep + model + "_lr_sent" + ".pkl"
		self.windowsize=windowsize
		self.use_words = use_words
		self.verbose = False

	def train(self,train_path,as_text=False,standardization=False,cut=True,multitrain=False):

		sys.stderr.write("o Reading training data...\n")

		if multitrain:
			df_train, todrop_train = self.read_conll_sentbreak(train_path, neighborwindowsize=self.windowsize,as_text=as_text,cut=False,multitrain=multitrain)
		else:
			df_train, todrop_train = self.read_conll_sentbreak(train_path, neighborwindowsize=self.windowsize,as_text=as_text,cut=cut)
		cols2keep = [col for col in df_train.columns if col not in todrop_train]
		X_train = df_train[cols2keep]
		Y_train = df_train['gold_seg']

		df_train = None
		predictors_train = list(X_train)

		# standardization of vectors
		if standardization:
			from sklearn import preprocessing
			std_scale = preprocessing.StandardScaler().fit(X_train)
			X_train = std_scale.transform(X_train)

		gc.collect()  # Free up memory for csr_matrix conversion

		X_train = X_train[sorted(X_train.columns)]
		#X_train = csr_matrix(X_train)

		logmodel = LogisticRegressionCV(cv=3,n_jobs=3,penalty='l1',solver="liblinear",random_state=42)
		if multitrain:
			if X_train.shape[0] <= 95000:
				multitrain_preds = get_multitrain_preds(logmodel,X_train,Y_train,5)
				multitrain_preds = "\n".join(multitrain_preds.strip().split("\n"))
				with io.open(script_dir + os.sep + "multitrain" + os.sep + self.name + '_' + self.corpus,'w',newline="\n") as f:
					sys.stderr.write("o Serializing multitraining predictions\n")
					f.write(multitrain_preds)
			else:
				sys.stderr.write('o Skipping multitrain\n')
		# Fit complete dataset
		logmodel.fit(X_train, Y_train)
		logmodel.sparsify()

		if multitrain and X_train.shape[0] > 95000:
			preds, probas = zip(*self.predict(train_path,as_text=False))
			with io.open(script_dir + os.sep + "multitrain" + os.sep + self.name + '_' + self.corpus,'w',newline="\n") as f:
				sys.stderr.write("o Serializing predictions from partial model\n")
				outlines = [str(preds[i]) + "\t" + str(probas[i]) for i in range(len(probas))]
				outlines = "\n".join(outlines)
				f.write(outlines+"\n")

		pickle_objects = (logmodel, predictors_train)
		pickle.dump(pickle_objects, open(self.model_path, 'wb'))

	def predict_cached(self,test_data):
		infile = script_dir + os.sep + "multitrain" + os.sep + self.name + '_' + self.corpus
		if os.path.exists(infile):
			pairs = io.open(infile).read().split("\n")
		else:
			sys.stderr.write("o No multitrain file at: " + infile + "\n")
			sys.stderr.write("o Falling back to live prediction for LRSentencer\n")
			return self.predict(test_data)
		preds = [(int(pr.split()[0]), float(pr.split()[1])) for pr in pairs if "\t" in pr]
		return preds

	def predict(self,test_path,as_text=True,standardization=False,do_tag=False):

		if self.verbose:
			sys.stderr.write("o Reading test data...\n")
		X_test, todrop_test = self.read_conll_sentbreak(test_path, neighborwindowsize=self.windowsize,as_text=as_text,cut=False,do_tag=do_tag)

		X_test = X_test.drop(todrop_test, axis=1)
		predictors_test = list(X_test)

		loaded_model, predictors_train = pickle.load(open(self.model_path, 'rb'))
		loaded_model.set_params(**{"n_jobs":3})

		diff_predictors = list(set(predictors_train) - set(predictors_test))
		for c in diff_predictors:
			X_test[c] = 0

		revdiff_predictors = list(set(list(predictors_test)) - set(predictors_train))
		X_test = X_test.drop(revdiff_predictors, axis=1)

		X_test = X_test.to_sparse(fill_value=0)

		# standardization of vectors
		# @logan: note that this won't work at predict time unless you preserve the scale from training
		#if standardization:
		#	from sklearn import preprocessing
		#	std_scale = preprocessing.StandardScaler().fit(X_train)
		#	X_train = std_scale.transform(X_train)
		#	X_dev = std_scale.transform(X_dev)

		X_test = X_test[sorted(X_test.columns)]

		gc.collect()

		probas = loaded_model.predict_proba(X_test)
		probas = [p[1] for p in probas]
		preds = [int(p>0.5) for p in probas]


		# # Verify we are returning as many predictions as we received input tokens
		# print(logmodel.predict(tokens))
		# assert len(tokens) == len(output)
		return zip(preds,probas)

	def read_conll_sentbreak(self, infile, neighborwindowsize=5,as_text=True,cut=True,do_tag=True,multitrain=False):
		global TRAIN_LIMIT
		# read data from conll_reader
		tok_counter = gettokfreq(infile,as_text=as_text)
		total_toks = sum(tok_counter.values())

		vowels = "AEIOUaeiouéèàáíìúùòóаэыуояеёюиⲁⲉⲓⲟⲩⲱⲏ"

		# collect top 100 freq tokens
		if self.use_words:
			top100tuple = sorted(tok_counter.items(), key=operator.itemgetter(1), reverse=True)[:500]
			top100keys = [x[0] for x in top100tuple]

		conll_entries = []
		if as_text:
			conllu_in = infile
		else:
			conllu_in = io.open(infile, 'r', encoding='utf8').read()

		# Reduce data if too large
		if not cut and conllu_in.count("\n") > 100000 and multitrain:
			sys.stderr.write("o Data too large; forcing cut and turning off multitraining\n")
			cut = True
			TRAIN_LIMIT = 100000
		if cut:
			conllu_in = shuffle_cut_conllu(conllu_in,limit=TRAIN_LIMIT)

		lines = conllu_in.split("\n")

		if do_tag:
			tagged = tt_tag(conllu_in,self.lang)
			#tagged = udpipe_tag(conllu_in,self.udpipe_model)
		else:
			tagged = conllu_in

		labels = []
		for line in lines:
			if "\t" in line:
				fields = line.split("\t")
				if "-" in fields[0]:
					continue
				if fields[0] == "1":
					labels.append(1)
				else:
					labels.append(0)

		f_lines = tagged.split("\n")

		isnewdoc = 0
		tok_num = 0
		for lnum, line in enumerate(f_lines):

			if '# newdoc' in f_lines[lnum]:
				isnewdoc = 1
			if '\t' in line:
				cols = line.split('\t')

				if labels[tok_num] == 1:
					gold_seg = 1
					# whether it is newdoc
				else:
					gold_seg = 0
				tok_num += 1
				if self.use_words:
					if cols[1] in top100keys:
						tokiftop100 = cols[1]
					else:
						tokiftop100 = 'OOV'

				firstletter = str(cols[1][0].encode("utf8")[0]) if self.lang == "zho" else cols[1][0]
				firstisupper = int (firstletter.upper() == firstletter)
				firstisconsonant = len(re.findall('[^'+vowels+']', firstletter))
				firstisvowel = len(re.findall('['+vowels+']', firstletter))
				firstisdigit = len(re.findall('[0-9]', firstletter))
				firstisspecial = len(re.findall('[^A-Za-z0-9Ⲁ-Ⲱⲁ-ⲱ]', firstletter))

				lastletter = str(cols[1][-1].encode("utf8")[-1]) if self.lang == "zho" else cols[1][-1]
				lastisupper = int(lastletter.upper() == lastletter)
				lastisconsonant = len(re.findall('[^'+vowels+']', lastletter))
				lastisvowel = len(re.findall('['+vowels+']', lastletter))
				lastisdigit = len(re.findall('[0-9]', lastletter))
				lastisspecial = len(re.findall('[^A-Za-z0-9Ⲁ-Ⲱⲁ-ⲱ]', lastletter))


				tokenlength = len(cols[1])
				numconsonants = len(re.findall('[^'+vowels+']', cols[1]))
				numvowels = len(re.findall('['+vowels+']', cols[1]))
				numdigits = len(re.findall('[0-9]', cols[1]))
				numspecials = len(re.findall('[^A-Za-z0-9Ⲁ-Ⲱⲁ-ⲱ]', cols[1]))

				if cols[1] in tok_counter.keys():
					tokenfreq = tok_counter[cols[1]]
				else:
					tokenfreq = 0
				tokenratio = tokenfreq / total_toks

				if not self.use_words:
					conll_entries.append(
						(gold_seg, cols[1], cols[3], cols[4], tokenlength, tokenfreq, tokenratio, isnewdoc,
						 numconsonants, numvowels, numdigits, numspecials,
						 firstletter, firstisupper, firstisconsonant, firstisvowel, firstisdigit, firstisspecial,
						 lastletter, lastisupper, lastisconsonant, lastisvowel, lastisdigit, lastisspecial))
				else:
					conll_entries.append(
						(gold_seg, cols[1], tokiftop100, cols[3], cols[4], tokenlength, tokenfreq, tokenratio, isnewdoc,
						 numconsonants, numvowels, numdigits, numspecials,
						 firstletter, firstisupper, firstisconsonant, firstisvowel, firstisdigit, firstisspecial,
						 lastletter, lastisupper, lastisconsonant, lastisvowel, lastisdigit, lastisspecial))
				if isnewdoc == 1:
					isnewdoc = 0  # reset newdoc after one token

		# pos = CCONJ (col 3)  cpos = CC (col 4)
		if not self.use_words:
			df = pd.DataFrame(conll_entries,
							  columns=['gold_seg', 'token', 'gold_pos', 'gold_cpos', 'tokenlength',
									   'tokenfreq', 'tokenratio', 'isnewdoc',
									   'numconsonants', 'numvowels', 'numdigits', 'numspecials',
									   'firstletter', 'firstisupper', 'firstisconsonant', 'firstisvowel', 'firstisdigit',
									   'firstisspecial',
									   'lastletter', 'lastisupper', 'lastisconsonant', 'lastisvowel', 'lastisdigit',
									   'lastisspecial'])

		else:
			df = pd.DataFrame(conll_entries, columns=['gold_seg', 'token','tokiftop100', 'gold_pos', 'gold_cpos', 'tokenlength', 'tokenfreq', 'tokenratio', 'isnewdoc',
													  'numconsonants', 'numvowels', 'numdigits', 'numspecials',
													  'firstletter', 'firstisupper', 'firstisconsonant', 'firstisvowel', 'firstisdigit', 'firstisspecial',
													  'lastletter', 'lastisupper', 'lastisconsonant', 'lastisvowel', 'lastisdigit', 'lastisspecial'])

		# dummy-lize categorial features which creates binary sub features
		categorial_vars = ['gold_pos', 'gold_cpos',
						   'firstletter',
						   'lastletter']

		# Dummy multi vectors cattodummy is much better
		df = cattodummy(categorial_vars, df)
		if self.use_words:
			df = cattolabel(['tokiftop100'], df)

		df = df.to_sparse()

		# incorporate prevprev, prev, next, nextnext tokens into account
		if neighborwindowsize == 7:
			df = pd.concat([copyforprevnext(df, 'prevprevprev'), copyforprevnext(df, 'prevprev'), copyforprevnext(df, 'prev'), df, copyforprevnext(df, 'next'), copyforprevnext(df, 'nextnext'), copyforprevnext(df, 'nextnextnext')], axis=1)
			dropcolumns = ['gold_seg', 'token', 'next_token', 'nextnext_token','nextnextnext_token','prev_token','prevprev_token','prevprevprev_token']
		elif neighborwindowsize == 5:
			df = pd.concat([copyforprevnext(df, 'prevprev'), copyforprevnext(df, 'prev'), df, copyforprevnext(df, 'next'), copyforprevnext(df, 'nextnext')], axis=1)
			dropcolumns = ['gold_seg', 'token', 'next_token', 'nextnext_token','prev_token','prevprev_token']
		elif neighborwindowsize == 3:
			df = pd.concat([copyforprevnext(df, 'prev'), df, copyforprevnext(df, 'next')], axis=1)
			dropcolumns = ['gold_seg', 'token', 'next_token', 'prev_token']
		elif neighborwindowsize == 1:
			dropcolumns = ['gold_seg', 'token']

		return df, dropcolumns


if __name__ == "__main__":

	# Argument parser
	parser = argparse.ArgumentParser(description='Input parameters')
	parser.add_argument('--corpus', '-c', action='store', dest='corpus', default="spa.rst.sctb", help='corpus name')
	parser.add_argument('--mode', '-m', action='store', dest='mode', default="train", choices=["train","predict"],help='Please specify train or predict mode')
	parser.add_argument('--windowsize', '-w', action='store', dest='windowsize', default=5, type=int, choices=[1,3,5,7], help='Please specify windowsize which has to be an odd number, i.e. 1, 3, 5, 7 (defaulted to 5).')
	parser.add_argument('--limit', '-l', action='store', default=5000, type=int, help='Subset size of training data to use')
	parser.add_argument("-d","--data_dir",default=os.path.normpath('../../../data'),help="Path to shared task data folder")
	parser.add_argument("-v","--verbose",action="store_true",help="Output verbose messages")
	parser.add_argument('--top100tokensincluded', '-t', action='store_true', dest='top100tokensincluded', help='whether to include top100tokens or not')
	parser.add_argument('--standardization', '-s', action='store_true', dest='standardization', help='whether to standardize features')
	parser.add_argument('--multitrain', action='store_true', help='whether to perform multitraining')

	args = parser.parse_args()

	start_time = time.time()

	data_folder = args.data_dir
	if data_folder is None:
		data_folder = os.path.normpath(r'./../../../sharedtask2019/data/')
	corpora = args.corpus

	if corpora == "all":
		corpora = os.listdir(data_folder)
		corpora = [c for c in corpora if os.path.isdir(os.path.join(data_folder, c))]
	else:
		corpora = [corpora]

	# Set a global limit to training size document in lines
	# We set a very conservative, small training size to avoid over-reliance of the ensemble, which also
	# uses the same training data to assess the usefulness of this estimator
	TRAIN_LIMIT = args.limit

	for corpusname in corpora:
		if "." in corpusname:
			lang = corpusname.split(".")[0]
		else:
			lang = "eng"

		# Run test
		sentencer = LRSentencer(lang=lang,model=corpusname,windowsize=args.windowsize,use_words=args.top100tokensincluded)
		if args.verbose:
			sentencer.verbose = True

		if sentencer.verbose:
			sys.stderr.write("o Processing corpus "+corpusname+"\n")

		tokens = ['Introduction', 'Research', 'has', 'shown', 'examples', '.', 'But', 'we', 'need', 'more', '.']

		if args.mode == "train":
			# When running from CLI, we always train (predict mode is done on imported class)
			sentencer.train(data_folder + os.sep+ corpusname + os.sep + corpusname + "_train.conll",as_text=False,standardization=args.standardization,multitrain=args.multitrain)

		# Now evaluate model
		predictions, probas = zip(*sentencer.predict(data_folder + os.sep+ corpusname + os.sep +corpusname + "_dev.conll",
													 as_text=False,standardization=args.standardization,do_tag=True))

		# Get gold labels for comparison
		conllu_in = io.open(data_folder + os.sep+ corpusname + os.sep +corpusname + "_dev.conll", 'r', encoding='utf8').read()
		labels = []
		for line in conllu_in.split("\n"):
			if "\t" in line:
				fields = line.split("\t")
				if "-" in fields[0]:
					continue
				if fields[0] == "1":
					labels.append(1)
				else:
					labels.append(0)

		# give dev F1 score
		from sklearn.metrics import classification_report, confusion_matrix
		print(classification_report(labels, predictions, digits=6))
		print(confusion_matrix(labels, predictions))

		with io.open("diff.conll",'w',encoding="utf8") as f:
			toknum=0
			for line in conllu_in.split("\n"):
				if "\t" in line:
					fields = line.split("\t")
					if "-" in fields[0]:
						continue
					f.write("\t".join([fields[1], fields[4], str(labels[toknum]), str(predictions[toknum])])+"\n")
					toknum+=1

		elapsed = time.time() - start_time
		sys.stderr.write(str(timedelta(seconds=elapsed)) + "\n\n")

