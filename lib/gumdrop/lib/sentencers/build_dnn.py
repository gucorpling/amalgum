from gensim.models import KeyedVectors
import io, os, sys, re, requests, time, argparse
import wget
from collections import defaultdict, OrderedDict
import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperas.distributions import choice, uniform
from hyperas import optim
from sklearn.metrics import classification_report, confusion_matrix


def read_conll(features, infile,mode="seg",genre_pat=None,as_text=False):
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
					label = "BeginSeg=Yes"
				elif "Seg=B-Conn" in fields[-1]:
					label = "Seg=B-Conn"
				elif "Seg=I-Conn" in fields[-1]:
					label = "Seg=I-Conn"
				else:
					label = "_"
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
	for word_dict in fc:
		word = word_dict['word']
		label.append(word_dict['label'])
		if count == 0:
			word_list += ['<s>']*half_window
		word_list.append(word)
		if count == len(fc)-1:
			word_list += ['</s>']*half_window
		count += 1
	sent_l = ['<s>', '</s>']
	for i in range(len(word_list)-1):
		if word_list[i] not in sent_l:
			n_gram_cols.append(word_list[i-half_window: i+half_window+1])
	return n_gram_cols, label


def loadvec(lang):
	path = '../../vec'
	if not os.path.isdir(path):
		os.mkdir(path)
	# word_embed_path = path + '/%s.vec' % lang

	if lang == "zho":
		vec = "cc.zho.300.vec_trim.vec"
	elif lang == "eng":
		vec = "glove.6B.300d_trim.vec"
	else:
		vec = "wiki.**lang**.vec_trim.vec".replace("**lang**",lang)
	word_embed_path = path + os.sep + vec

	sys.stdout.write('##### Load word embeddings...\n')
	word_embed = {}
	f = open(word_embed_path, encoding='utf-8').readlines()
	for line in f:
		parts = line.strip().split()
		word = parts[0]
		vec = [x for x in parts[1:]]
		word_embed[word] = vec
	return word_embed


def unique_embed(f_cols, word_embed):
	word_unique = ['<s>', '</s>']
	for x in f_cols:
		word = x['word']
		if word not in word_unique:
			word_unique.append(word)
	embed_vector = {w:word_embed[w] if w in word_embed else [0.0001]*300 for w in word_unique}
	return embed_vector


def mergeWord2Vec(word_grams, uni_embed):
	fec_vec = []
	for words in word_grams:
		word_vec = []
		for word in words:
			word_vec += uni_embed[word]
		fec_vec.append(word_vec)
	return np.asarray(fec_vec, dtype=np.float32)


def get_multitrain_preds(model,X,y,multifolds,space):
	all_preds = []
	all_probas = []
	X_folds = np.array_split(X, multifolds)
	y_folds = np.array_split(y, multifolds)
	for i in range(multifolds):
		X_train = np.vstack(tuple([X_folds[j] for j in range(multifolds) if j!=i]))
		y_train = np.concatenate(tuple([y_folds[j] for j in range(multifolds) if j!=i]))
		X_heldout = X_folds[i]
		sys.stdout.write("##### Training on fold " + str(i+1) +" of " + str(multifolds) + "\n")
		model.fit(X_train, y_train, epochs=space['epoch'], batch_size=space['batch_size'], verbose=2)
		probas = model.predict(X_heldout)
		preds = [str(int(p > 0.5)) for p in probas]
		probas = [str(p[0]) for p in probas]
		all_preds += preds
		all_probas += probas

	pairs = list(zip(all_preds,all_probas))
	pairs = ["\t".join(pair) for pair in pairs]

	return "\n".join(pairs)


def keras_train(space):
	MB = 1024*1024
	# dnn_name = "DNNSentencer"

	sys.stdout.write('##### Loaded trainning set word embeddings.\n')
	train_n_gram, train_label = n_gram(t_output, space['gram'])
	train_fc_vec = mergeWord2Vec(train_n_gram, t_uni_embed)
	sys.stdout.write("train_fc_vec %d MB\n" % (sys.getsizeof(train_fc_vec)/MB))

	sys.stdout.write('##### Loaded eval set word embeddings.\n')
	eval_n_gram, eval_label = n_gram(e_output, space['gram'])
	eval_fc_vec= mergeWord2Vec(eval_n_gram, e_uni_embed)


	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Activation
	from keras import metrics

	np.random.seed(11)

	# create model
	model = Sequential()
	model.add(Dense(space['units1'], input_dim=300*space['gram']))
	model.add(Activation(space['activation']))
	model.add(Dropout(space['dropout1']))
	if space['choice']['num_layers'] == 'two':
		model.add(Dense(space['choice']['units2']))
		model.add(Activation(space['activation']))
		model.add(Dropout(space['choice']['dropout2']))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy',
				  optimizer=space['optimizer'],
				  metrics=['accuracy'])

	# Fit the model
	# if multitrain:
		# multitrain_preds = get_multitrain_preds(model, train_fc_vec, train_label, 5, space)
	model.fit(train_fc_vec, train_label, epochs=space['epoch'], batch_size=space['batch_size'], verbose=2)
	predications = list(model.predict(eval_fc_vec))
	pred_labels = [1 if x > 0.5 else 0 for x in predications]

	x = classification_report(list(eval_label), pred_labels, digits=6)
	sys.stdout.write(x)
	f_score = float(x.split()[12])

	try:
		with open("metric/%s_metric.txt" % language_framework, encoding='utf-8') as f:
			x = f.readlines()
			tent_result = float(x[0].strip().split('\t')[0])
	except FileNotFoundError:
		tent_result = f_score
	if f_score >= tent_result:
		model.save('../../models/%s_dnn_sent.hd5' % language_framework)
		with open("metric/%s_metric.txt" % language_framework, "w") as f:
			f.write('%s\t%s' % (str(f_score), space))
		# if multitrain:
		# 	multitrain_preds = "\n".join(multitrain_preds.strip().split("\n"))
		# 	with io.open(script_dir + os.sep + "multitrain" + os.sep + dnn_name + '_' + model,'w',newline="\n") as f:
		# 		sys.stdout.write("##### Serializing multitraining predictions\n")
		# 		f.write(multitrain_preds)

	return {'loss': -f_score, 'status': STATUS_OK, 'model': model}


def main(lang,model,features,max_evals,baseline,train_data_path,eval_data_path):
	# Store the best result and hyper parameters for each language
	results = {}

	start_time = time.time()

	global language_framework
	language_framework = model

	word_embed = loadvec(lang)

	# Training dataset
	train_output = read_conll(features, train_data_path, mode="sent", genre_pat=None, as_text=False)
	train_uni_embed = unique_embed(train_output, word_embed)

	# Evaluation dataset
	eval_output = read_conll(features, eval_data_path, mode="sent", genre_pat=None, as_text=False)
	eval_uni_embed = unique_embed(eval_output, word_embed)

	global t_output
	t_output = train_output
	global t_uni_embed
	t_uni_embed = train_uni_embed
	global e_output
	e_output = eval_output
	global e_uni_embed
	e_uni_embed = eval_uni_embed

	sys.stdout.write('##### Start training...\n')

	# Hyper parameters
	if baseline:
		space = {
			'choice': {'num_layers':'one'},
			'units1': 96,
			'dropout1': .40,
			'epoch' :  5,
			'batch_size' : 128,
			'optimizer': 'adam',
			'activation': 'relu',
			'gram': 5
			}
		result = keras_train(space)['loss'] * (-1)
		results[model] = {'f_score': result, 'params': space}
	else:
		space = {
			'choice': hp.choice('num_layers',
			[{'num_layers':'one'},
			{'num_layers':'two',
			 'units2': hp.choice('units2', [16, 32, 48, 64]),
			 'dropout2': hp.uniform('dropout2', .25,.75)}
			]),

		'units1': hp.choice('units1', [32, 64, 96, 128]),
		'dropout1': hp.uniform('dropout1', .25,.75),

		'epoch':  hp.choice('epoch', [5, 10, 15, 20]),
		'batch_size': hp.choice('batch_size', [128, 256, 384, 512]),
		'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop', 'sgd']),
		'activation': 'relu',
		'gram': hp.choice('gram', [5, 7, 9])
		}

		# keras_train(x_train, y_train, x_eval, y_eval, dir_path)
		trials = Trials()
		best = fmin(fn=keras_train,
					space=space,
					algo=tpe.suggest,
					max_evals=max_evals,
					trials=trials
					)
		sys.stdout.write(str(best) + '\n')
		sys.stdout.write(str(trials.best_trial))
		results[model] = {'best_params': best}
	process_time = time.time() - start_time
	sys.stdout.write('Runtime: %d:%d:%d\n\n\n' % (process_time//3600, (process_time%3600)//60, (process_time%3600)%60))
	return results


if __name__ == "__main__":
	# play features:
	# sample features {"word_id", "word", "lemma", "pos", "cpos", "deprel", "docname", "tok_len", "label", "first", "last",
	# 				   "tok_id", "genre", "doc_id"}
	parser = argparse.ArgumentParser(description='Input parameters')
	parser.add_argument('--corpus', '-c', action='store', dest='corpus', default="spa.rst.sctb", help='corpus name')
	parser.add_argument('--baseline', action='store_true', help='whether to perform baseline')
	# parser.add_argument('--multitrain', action='store_true', help='whether to perform multitraining')
	parser.add_argument("-d","--data_dir",default=os.path.normpath('../../../data'),help="Path to shared task data folder")
	parser.add_argument('--max', action='store', default=30, help='maximum evaluation rounds')

	args = parser.parse_args()

	data_folder = args.data_dir
	if data_folder is None:
		data_folder = os.path.normpath(r'./../../../sharedtask2019/data/')
	corpora = args.corpus
	max_evals = args.max
	baseline = args.baseline
	features = ['word', 'label']
	# global multitrain
	# multitrain = args.multitrain

	results = {}

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

		train_path = data_folder + os.sep+ corpusname + os.sep + corpusname + "_train.conll"
		dev_path = data_folder + os.sep+ corpusname + os.sep +corpusname + "_dev.conll"

		results[corpusname] = main(lang, corpusname, features, max_evals, baseline, train_path, dev_path)

	f = open('results_dnn_sent.txt', 'w', encoding='utf8')
	for k,v in results.items():
		f.write('%s\n%s\n\n' % (k, v))
	f.close()
	sys.stdout.write('##### Results written.')
