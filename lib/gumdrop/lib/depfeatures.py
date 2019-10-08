import io, os, sys
from glob import glob
import numpy as np
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import re, io, os, sys, time
from datetime import timedelta
from argparse import ArgumentParser
script_dir = os.path.dirname(os.path.realpath(__file__))
lib = os.path.abspath(script_dir + os.sep + "..")
sys.path.append(lib)

class DepFeatures:

	def __init__(self,model="eng.rst.gum",genre_pat="^(..)"):
		self.name = "DepFeatures"
		self.model = model
		self.ext = "bmes"  # use bio for bio encoding
		self.genre_pat = genre_pat

	# @profile
	def extract_depfeatures(self, train_feats):
		clausal_deprels = {'csubj', 'ccomp', 'xcomp', 'advcl', 'acl', 'acl:relcl', 'list', 'parataxis', 'appos', 'conj', 'nmod:prep'}

		# <class 'dict'>: {'word': 'creada', 'lemma': 'creado', 'pos': 'VERB', 'cpos': '_', 'head': '17', 'head_dist': 2, 'deprel': 'acl', 'docname': 'BMCS_ESP1-GS', 'case': 'l', 'tok_len': 6, 'label': '_', 'first': 'c', 'last': 'a', 'tok_id': 92, 'genre': '_', 'wid': 19, 'quote': 0, 'morph': 'Part', 'heading_first': '_', 'heading_last': '_', 's_num': 2, 's_len': 47, 's_id': 2, 'sent_doc_percentile': 0.6666666666666666}
		# 'word': 'creada' , 'head': '17',  'wid': 19, 'head_dist': 2, 'deprel': 'acl','docname': 'BMCS_ESP1-GS', 's_len': 47, 's_id': 2, 'tok_id': 92, ('s_num': 2, )
		# [s_id, wid, line_id ,head, deprel, word]
		feats_tups = [tuple([x['s_id'],x['wid'],x['line_id'],int(x['head']),x['deprel'],x['word']]) for x in train_feats]
		num_s = sorted(list(set([x[0] for x in feats_tups])))
		feats_parents = defaultdict(list)

		all_feats_s = defaultdict(list)

		sorted_feats_s = sorted([x for x in feats_tups], key=lambda x: x[1])

		# Do only one pass through data to break up into sentences
		for x in sorted_feats_s:
			s_num = x[0]
			all_feats_s[s_num].append(x)

		# looping through sentences
		for s in num_s:
			#feats_s = sorted([x for x in  feats_tups if x[0]==s], key=lambda x: x[1])
			feats_s = all_feats_s[s]
			feats_parents.clear()

			# looping through tokens in a sentence
			wid2lineid = {}
			for t in feats_s:

				# finding all (grand)parents (grand-heads)
				wid = t[1]
				head = t[3]
				wid2lineid[wid]=t[2]
				while head != 0:
					head_t = [x for x in feats_s if x[1]==head][0]
					feats_parents[t].append(head_t)
					head = head_t[3]


			for id_t, t in enumerate(feats_s):
				parentrels = [x[4] for x in feats_parents[t]]
				parentcls = []
				for r in parentrels:
					for dr in clausal_deprels:
						if r.startswith(dr):
							parentcls.append(r)
							continue
				train_feats[wid2lineid[id_t + 1]]['parentclauses'] = "|".join(parentcls)


			# loop through clausal_deprels (non-conj & conj) and create BIO list for sentence tokens
			dr_d = defaultdict(list)

			## finding all tokens in a sentence who or whose parents has a deprel (dr) -- non-conj
			for id_t, t in enumerate(feats_s):
				t_gen = [t] + feats_parents[t]

				# all including conj
				for dr in clausal_deprels:
					in_t_gen = [x for x in t_gen if x[4].startswith(dr)]
					if len(in_t_gen)>0:
						dr_d[(in_t_gen[0][1], in_t_gen[0][4])].append(t)



			#  sort dictionary values
			dr_dl = defaultdict(list)
			for k,v in dr_d.items():
				if v!=[]:
					dr_dl[k+(len(v),)] = sorted(list(set([x[1] for x in v])))
					# sorted(v, key=lambda x: x[1])

			# collect all BIEO features, for conj and non-conj separately
			feats_l = [[] for x in range(len(feats_s))]
			feats_conjl = [[] for x in range(len(feats_s))]
			for i in range(len(feats_s)):
				for k,v in dr_dl.items():
					# for non-conj
					if not k[1].startswith('conj'):
						if not i+1 in v:
							feats_l[i].append('_')
						elif v[0]==i+1:
							feats_l[i].append(('B'+ k[1], v[0], v[-1]))
						elif v[-1]==i+1:
							feats_l[i].append(('E'+ k[1], v[0], v[-1]))
						else:
							feats_l[i].append(('I'+ k[1], v[0], v[-1]))

					# for conj
					else:
						if not i+1 in v:
							feats_conjl[i].append('_')
						elif v[0]==i+1:
							feats_conjl[i].append(('B'+ k[1], v[0], v[-1]))
						elif v[-1]==i+1:
							feats_conjl[i].append(('E'+ k[1], v[0], v[-1]))
						else:
							feats_conjl[i].append(('I'+ k[1], v[0], v[-1]))


			# Prioritize Bsmall > Blarge > Elarge > Esmall > Ismall > Ilarge > _
			# non-conj
			for id_l, l in enumerate(feats_l):
				Bsub = sorted([x for x in l if x[0].startswith('B')], key=lambda x: x[2]-x[1])
				Esub = sorted([x for x in l if x[0].startswith('E')], key=lambda x: x[2]-x[1], reverse=True)
				Isub = sorted([x for x in l if x[0].startswith('I')], key=lambda x: x[2]-x[1])
				if Bsub!=[]:
					feats_l[id_l]=Bsub[0][0]
				elif Esub!=[]:
					feats_l[id_l] = Esub[0][0]
				elif Isub!=[]:
					feats_l[id_l] = Isub[0][0]
				else:
					feats_l[id_l] = '_'

			# remove sub-deprel after :, e.g. csubj:pass -> csubj (however, acl:relcl stays as acl:relcl)
			feats_l = [re.sub(r':[^r].*$', '', x) if x != "nmod:prep" else x for x in feats_l]

			# add non-conj to train_feats
			for id_l, l in enumerate(feats_l):
				train_feats[wid2lineid[id_l+1]]['depchunk'] = l


			# conj
			for id_l, l in enumerate(feats_conjl):
				Bsub = sorted([x for x in l if x[0].startswith('B')], key=lambda x: x[2]-x[1])
				Esub = sorted([x for x in l if x[0].startswith('E')], key=lambda x: x[2]-x[1], reverse=True)
				Isub = sorted([x for x in l if x[0].startswith('I')], key=lambda x: x[2]-x[1])
				if Bsub!=[]:
					feats_conjl[id_l]=Bsub[0][0]
				elif Esub!=[]:
					feats_conjl[id_l] = Esub[0][0]
				elif Isub!=[]:
					feats_conjl[id_l] = Isub[0][0]
				else:
					feats_conjl[id_l] = '_'

			# add conj to train_feats
			for id_l, l in enumerate(feats_conjl):
				train_feats[wid2lineid[id_l+1]]['conj'] = l

			#sys.stderr.write('\r Adding deprel BIEO features to train_feats %s ### o Sentence %d' %(corpus, s))

		return train_feats


if __name__ == "__main__":
	start_time = time.time()
	corpus_start_time = start_time
	table_out = []

	log = io.open("depfeatures.log", 'w', encoding="utf8")

	p = ArgumentParser()
	p.add_argument("-c","--corpus",default="spa.rst.sctb",help="Corpus to train on")
	p.add_argument("-d","--data_dir",default=os.path.normpath("../../data"),help="Path to shared task data folder")
	p.add_argument("--mode",action="store",default="train-test",choices=["train-test","test"])

	opts = p.parse_args()

	corpora = opts.corpus

	data_dir = opts.data_dir

	if corpora == "all":
		corpora = os.listdir(data_dir )
		corpora = [c for c in corpora if (os.path.isdir(os.path.join(data_dir, c)) and "pdtb" not in c)]  # No PDTB
	else:
		corpora = [corpora]

	for corpus in corpora:

		corpus_start_time = time.time()

		train = glob(opts.data_dir + os.sep + corpus + os.sep + corpus + "_train.conll")[0]
		dev = glob(opts.data_dir + os.sep + corpus + os.sep + corpus + "_dev.conll")[0]
		test = glob(opts.data_dir + os.sep + corpus + os.sep + corpus + "_test.conll")[0]

		feats = DepFeatures()

		from conll_reader import read_conll
		outputs = read_conll(dev)
		train = outputs[0]

		# Extract features from file
		sys.stderr.write("\no Extracting features from training corpus " + corpus + "\n")
		train_feats = feats.extract_depfeatures(train)

		elapsed = time.time() - corpus_start_time
		sys.stderr.write("\nTime training on corpus:\n")
		sys.stderr.write(str(timedelta(seconds=elapsed)) + "\n\n")

	sys.stderr.write("\nTotal time:\n")
	elapsed = time.time() - start_time
	sys.stderr.write(str(timedelta(seconds=elapsed)) + "\n\n")

