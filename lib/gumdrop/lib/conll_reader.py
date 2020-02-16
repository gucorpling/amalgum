#!/usr/bin/python
# -*- coding: utf-8 -*-

import io, re, os,sys
from collections import defaultdict
from random import shuffle, seed
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Allow package level imports in module
script_dir = os.path.dirname(os.path.realpath(__file__))
lib = os.path.abspath(script_dir + os.sep + "..")
sys.path.append(lib)

from exec import exec_via_temp
from tt2conll import conllize
from depfeatures import DepFeatures


script_dir = os.path.dirname(os.path.realpath(__file__))
lib = script_dir

def get_case(word):
	if word.isdigit():
		return "d"
	elif word.isupper():
		return "u"
	elif word.islower():
		return "l"
	elif word.istitle():
		return "t"
	else:
		return "o"


def read_conll(infile,mode="seg",genre_pat=None,as_text=False,cap=None,char_bytes=False, usestanford=False):
	"""
	Read a DISRPT shared task format .conll file

	:param infile: file path for input
	:param mode: 'seg' to read discourse unit segmentation as labels or
	             'sent' to read sentence breaks as labels
	:param genre_pat: Regex pattern with capturing group to extract genre from document names
	:param as_text: Boolean, whether the input is a string, rather than a file name to read
	:param cap: Maximum tokes to read, after which reading stops at next newdoc boundary
	:param char_bytes: if True, first and last letters are replaced with first and last byte of string (for Chinese)
	:return: list of tokens, each a dictionary of features and values including a gold label, and
	         vocabulary frequency list
	"""

	if as_text:
		lines = infile.split("\n")
	else:
		lines = io.open(infile,encoding="utf8").readlines()

	# use Stanford parse
	if usestanford:
		from reparse import usestanfordparse
		lines = usestanfordparse(lines, language="zh", conversion=True)


	docname = infile if len(infile) < 100 else "doc1"
	output = []  # List to hold dicts of each observation's features
	cache = []  # List to hold current sentence tokens before adding complete sentence features for output
	toks = []  # Plain list of token forms
	firsts = set([])  # Attested first characters of words
	lasts = set([])  # Attested last characters of words
	vocab = defaultdict(int)  # Attested token vocabulary counts
	sent_start = True
	tok_id = 0  # Track token ID within document
	line_id = 0
	sent_id = 1
	genre = "_"
	open_quotes = set(['"','«','``','”'])
	close_quotes = set(['"','»','“',"''"])
	open_brackets = set(["(","[","{","<"])
	close_brackets = set([")","]","}",">"])
	used_feats = ["VerbForm","PronType","Person","Mood"]
	in_quotes = 0
	in_brackets = 0
	last_quote = 0
	last_bracket = 0
	total = 0
	total_sents = 0
	doc_sents = 1
	heading_first = "_"
	heading_last = "_"
	for r, line in enumerate(lines):
		if "\t" in line:
			fields = line.split("\t")
			if "-" in fields[0]:  # conllu super-token
				continue
			total +=1
			# Handle extremely long sentence, break up artificially
			# TODO: find a better solution for really long sentences
			if int(fields[0]) > 249:
				fields[0] = str(int(fields[0])-248)
				fields[6] = "1"
				if fields[7] == "root":
					fields[7] = "dep"
			if int(fields[6]) >= 249:
				fields[6] = "1"
			if int(fields[0]) == 249:
				fields[0] = "1"
				fields[6] = "0"
				fields[7] = "root"
				sent_start = True
				if len(cache)>0:
					if mode == "seg":  # Don't add s_len in sentencer learning mode
						sent = " ".join([t["word"] for t in cache])
						if sent[0] == sent[0].upper() and len(cache) < 6 and sent[-1] not in [".","?","!",";","！","？","。"]:
							# Uppercase short sentence not ending in punctuation - possible heading affecting subsequent data
							heading_first = str(sent.encode("utf8")[0]) if char_bytes else sent[0]
							heading_last = str(sent.encode("utf8")[-1]) if char_bytes else sent[-1]
						# Get s_type features
						s_type = get_stype(cache)
						for tok in cache:
							tok["s_len"] = len(cache)
							tok["s_id"] = sent_id
							tok["heading_first"] = heading_first
							tok["heading_last"] = heading_last
							tok["s_type"] = s_type
						sent_id +=1
						doc_sents += 1
						total_sents += 1
					output += cache
					if mode == "seg":
						if len(output) > 0:
							for t in output[-int(249):]:
								# Add sentence percentile of document length in sentences
								t["sent_doc_percentile"] = t["s_num"]/doc_sents
					cache = []

			word, lemma, pos, cpos, feats, head, deprel = fields[1:-2]
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
					label = "Sent"
				else:
					label = "_"
			else:
				raise ValueError("read_conll mode must be one of: seg|sent\n")
			# Compose a categorical feature from morphological features of interest
			feats = [f for f in feats.split("|") if "=" in f]
			feat_string = ""
			for feat in feats:
				name, val = feat.split("=")
				if name in used_feats:
					feat_string += val
			if feat_string == "":
				feat_string = "_"
			vocab[word] += 1
			case = get_case(word)
			head_dist = int(fields[0]) - int(head)
			if len(word.strip()) == 0:
				raise ValueError("! Zero length word at line " + str(r) + "\n")
			toks.append(word)
			first_char = word[0]
			last_char = word[-1]
			if char_bytes:
				try:
					first_char = str(first_char.encode("utf8")[0])
					last_char = str(last_char.encode("utf8")[-1])
				except:
					pass
			firsts.add(first_char)
			lasts.add(last_char)
			cache.append({"word":word, "lemma":lemma, "pos":pos, "cpos":cpos, "head":head, "head_dist":head_dist, "deprel":deprel,
						   "docname":docname, "case":case,"tok_len":len(word),"label":label,"first":first_char,"last":last_char,
						   "tok_id": tok_id, "genre":genre,"wid":int(fields[0]),"quote":in_quotes,"bracket":in_brackets,"morph":feat_string,
						   "heading_first": heading_first, "heading_last": heading_last,"depchunk":"_","conj":"_", "line_id":line_id})
			if mode == "seg":
				cache[-1]["s_num"] = doc_sents

			tok_id += 1
			line_id += 1
			sent_start = False
			if word in open_quotes:
				in_quotes = 1
				last_quote = tok_id
			elif word in close_quotes:
				in_quotes = 0
			if word in open_brackets:
				in_brackets = 1
				last_bracket = tok_id
			elif word in close_brackets:
				in_brackets = 0
			if tok_id - last_quote > 100:
				in_quotes = 0
			if tok_id - last_bracket > 100:
				in_brackets = 0

		elif "# newdoc id = " in line:
			if cap is not None:
				if total > cap:
					break
			docname = re.search(r"# newdoc id = (.+)",line).group(1)
			if genre_pat is not None:
				genre = re.search(genre_pat,docname).group(1)
			else:
				genre = "_"
			doc_sents =1
			tok_id = 1
		elif len(line.strip())==0:
			sent_start = True
			if len(cache)>0:
				if mode == "seg":  # Don't add s_len in sentencer learning mode
					sent = " ".join([t["word"] for t in cache])
					if sent[0] == sent[0].upper() and len(cache) < 6 and sent[-1] not in [".","?","!",";","！","？","。"]:
						# Uppercase short sentence not ending in punctuation - possible heading affecting subsequent data
						heading_first = str(sent.encode("utf8")[0]) if char_bytes else sent[0]
						heading_last = str(sent.encode("utf8")[-1]) if char_bytes else sent[-1]
					# Get s_type features
					s_type = get_stype(cache)
					for tok in cache:
						tok["s_len"] = len(cache)
						tok["s_id"] = sent_id
						tok["heading_first"] = heading_first
						tok["heading_last"] = heading_last
						tok["s_type"] = s_type
					sent_id +=1
					doc_sents += 1
					total_sents += 1
				output += cache
				if mode == "seg":
					if len(output) > 0:
						for t in output[-int(fields[0]):]:
							# Add sentence percentile of document length in sentences
							t["sent_doc_percentile"] = t["s_num"]/doc_sents
				cache = []

	# Flush last sentence if no final newline
	if len(cache)>0:
		if mode == "seg":  # Don't add s_len in sentencer learning mode
			sent = " ".join([t["word"] for t in cache])
			if sent[0] == sent[0].upper() and len(cache) < 6 and sent[-1] not in [".","?","!",";","！","？","。"]:
				# Uppercase short sentence not ending in punctuation - possible heading
				heading_first = str(sent.encode("utf8")[0]) if char_bytes else sent[0]
				heading_last = str(sent.encode("utf8")[-1]) if char_bytes else sent[-1]
			# Get s_type features
			s_type = get_stype(cache)
			for tok in cache:
				tok["s_len"] = len(cache)
				tok["s_id"] = sent_id
				tok["heading_first"] = heading_first
				tok["heading_last"] = heading_last
				tok["s_type"] = s_type

		output += cache
		if mode == "seg":
			for t in output[-int(fields[0]):]:
				# Add sentence percentile of document length in sentences
				t["sent_doc_percentile"] = 1.0

	if mode == "seg":
		df = DepFeatures()
		output = df.extract_depfeatures(output)

	return output, vocab, toks, firsts, lasts


def read_conll_conn(infile,mode="seg",genre_pat=None,as_text=False,cap=None,char_bytes=False):
	"""
	Read a DISRPT shared task format .conll file

	:param infile: file path for input
	:param mode: 'seg' to read discourse unit segmentation as labels or
	             'sent' to read sentence breaks as labels
	:param genre_pat: Regex pattern with capturing group to extract genre from document names
	:param as_text: Boolean, whether the input is a string, rather than a file name to read
	:param cap: Maximum tokes to read, after which reading stops at next newdoc boundary
	:param char_bytes: if True, first and last letters are replaced with first and last byte of string (for Chinese)
	:return: list of tokens, each a dictionary of features and values including a gold label, and
	         vocabulary frequency list
	"""

	if as_text:
		lines = infile.split("\n")
	else:
		lines = io.open(infile,encoding="utf8").readlines()
	docname = infile if len(infile) < 100 else "doc1"
	output = []  # List to hold dicts of each observation's features
	cache = []  # List to hold current sentence tokens before adding complete sentence features for output
	toks = []  # Plain list of token forms
	firsts = set([])  # Attested first characters of words
	lasts = set([])  # Attested last characters of words
	vocab = defaultdict(int)  # Attested token vocabulary counts
	sent_start = True
	tok_id = 0  # Track token ID within document
	line_id = 0
	sent_id = 1
	genre = "_"
	open_quotes = set(['"','«','``','”'])
	close_quotes = set(['"','»','“',"''"])
	open_brackets = set(["(","[","{","<"])
	close_brackets = set([")","]","}",">"])
	used_feats = ["VerbForm","PronType","Person","Mood"]
	in_quotes = 0
	in_brackets = 0
	last_quote = 0
	last_bracket = 0
	total = 0
	total_sents = 0
	doc_sents = 1
	heading_first = "_"
	heading_last = "_"
	for r, line in enumerate(lines):
		if "\t" in line:
			fields = line.split("\t")
			if "-" in fields[0]:  # conllu super-token
				continue
			total +=1
			word, lemma, pos, cpos, feats, head, deprel = fields[1:-2]
			if mode=="seg":
				# if "BeginSeg=Yes" in fields[-1]:
				# 	label = "_"
				if "Seg=B-Conn" in fields[-1]:
					label = "Seg=B-Conn"
				elif "Seg=I-Conn" in fields[-1]:
					label = "Seg=I-Conn"
				# elif "Seg=S-Conn" in fields[-1]:
				# 	label = "Seg=S-Conn"
				else:
					label = "_"
			elif mode == "sent":
				if sent_start:
					label = "Sent"
				else:
					label = "_"
			else:
				raise ValueError("read_conll_conn mode must be one of: seg|sent\n")
			# Compose a categorical feature from morphological features of interest
			feats = [f for f in feats.split("|") if "=" in f]
			feat_string = ""
			for feat in feats:
				name, val = feat.split("=")
				if name in used_feats:
					feat_string += val
			if feat_string == "":
				feat_string = "_"
			vocab[word] += 1
			case = get_case(word)
			head_dist = int(fields[0]) - int(head)
			if len(word.strip()) == 0:
				raise ValueError("! Zero length word at line " + str(r) + "\n")
			toks.append(word)
			first_char = word[0]
			last_char = word[-1]
			if char_bytes:
				try:
					first_char = str(first_char.encode("utf8")[0])
					last_char = str(last_char.encode("utf8")[-1])
				except:
					pass
			firsts.add(first_char)
			lasts.add(last_char)
			cache.append({"word":word, "lemma":lemma, "pos":pos, "cpos":cpos, "head":head, "head_dist":head_dist, "deprel":deprel,
						   "docname":docname, "case":case,"tok_len":len(word),"label":label,"first":first_char,"last":last_char,
						   "tok_id": tok_id, "genre":genre,"wid":int(fields[0]),"quote":in_quotes,"bracket":in_brackets,"morph":feat_string,
						   "heading_first": heading_first, "heading_last": heading_last,"depchunk":"_","conj":"_", "line_id":line_id})
			if mode == "seg":
				cache[-1]["s_num"] = doc_sents

			tok_id += 1
			line_id += 1
			sent_start = False
			if word in open_quotes:
				in_quotes = 1
				last_quote = tok_id
			elif word in close_quotes:
				in_quotes = 0
			if word in open_brackets:
				in_brackets = 1
				last_bracket = tok_id
			elif word in close_brackets:
				in_brackets = 0
			if tok_id - last_quote > 100:
				in_quotes = 0
			if tok_id - last_bracket > 100:
				in_brackets = 0

		elif "# newdoc id = " in line:
			if cap is not None:
				if total > cap:
					break
			docname = re.search(r"# newdoc id = (.+)",line).group(1)
			if genre_pat is not None:
				genre = re.search(genre_pat,docname).group(1)
			else:
				genre = "_"
			doc_sents =1
			tok_id = 1
		elif len(line.strip())==0:
			sent_start = True
			if len(cache)>0:
				if mode == "seg":  # Don't add s_len in sentencer learning mode
					sent = " ".join([t["word"] for t in cache])
					if sent[0] == sent[0].upper() and len(cache) < 6 and sent[-1] not in [".","?","!",";","！","？","。"]:
						# Uppercase short sentence not ending in punctuation - possible heading affecting subsequent data
						heading_first = str(sent.encode("utf8")[0]) if char_bytes else sent[0]
						heading_last = str(sent.encode("utf8")[-1]) if char_bytes else sent[-1]
					# Get s_type features
					s_type = get_stype(cache)
					for tok in cache:
						tok["s_len"] = len(cache)
						tok["s_id"] = sent_id
						tok["heading_first"] = heading_first
						tok["heading_last"] = heading_last
						tok["s_type"] = s_type
					sent_id +=1
					doc_sents += 1
					total_sents += 1
				output += cache
				if mode == "seg":
					if len(output) > 0:
						for t in output[-int(fields[0]):]:
							# Add sentence percentile of document length in sentences
							t["sent_doc_percentile"] = t["s_num"]/doc_sents
				cache = []

	# Flush last sentence if no final newline
	if len(cache)>0:
		if mode == "seg":  # Don't add s_len in sentencer learning mode
			sent = " ".join([t["word"] for t in cache])
			if sent[0] == sent[0].upper() and len(cache) < 6 and sent[-1] not in [".","?","!",";","！","？","。"]:
				# Uppercase short sentence not ending in punctuation - possible heading
				heading_first = str(sent.encode("utf8")[0]) if char_bytes else sent[0]
				heading_last = str(sent.encode("utf8")[-1]) if char_bytes else sent[-1]
			# Get s_type features
			s_type = get_stype(cache)
			for tok in cache:
				tok["s_len"] = len(cache)
				tok["s_id"] = sent_id
				tok["heading_first"] = heading_first
				tok["heading_last"] = heading_last
				tok["s_type"] = s_type

		output += cache
		if mode == "seg":
			for t in output[-int(fields[0]):]:
				# Add sentence percentile of document length in sentences
				t["sent_doc_percentile"] = 1.0

	if mode == "seg":
		df = DepFeatures()
		output = df.extract_depfeatures(output)

	return output, vocab, toks, firsts, lasts



def get_stype(tokens):
	q = "NoQ"
	root_child_funcs = []
	# Get root
	for t in tokens:
		if t["deprel"] == "root":
			root = t["wid"]
			root_pos = t["cpos"] if t["cpos"] != "_" else t["pos"]
		if t["word"] in ["?","？"]:
			q = "Q"
	for t in tokens:
		try:
			if t["head"] == "root":
				root_child_funcs.append(t["deprel"])
		except:
			raise IOError("! Found input sentence without a root label: " + " ".join([t["word"] for t in tokens]) + "\n")
	if any(["subj" in f for f in root_child_funcs]):
		subj = "Subj"
	else:
		subj = "NoSubj"
	if any(["acl" in f or "rcmod" in f for f in root_child_funcs]):
		acl = "Acl"
	else:
		acl = "NoAcl"
	if any(["advcl" in f for f in root_child_funcs]):
		advcl = "Advcl"
	else:
		advcl = "NoAdvcl"
	if "conj" in root_child_funcs:
		conj = "Conj"
	else:
		conj = "NoConj"
	if "cop" in root_child_funcs:
		cop = "Cop"
	else:
		cop = "NoCop"

	s_type = "_".join([q,subj,conj,cop,advcl,acl])

	return s_type


def space_join(conllu,sentence_wise=False):
	"""Takes conllu input and returns:

	All tokens separated by space (if sentence_wise is False), OR
	A list of sentences, each a space separated string of tokens

	"""
	lines = conllu.replace("\r","").strip().split("\n")
	lines.append("")  # Ensure last blank
	just_text = []
	sentences = []
	length = 0
	for line in lines:
		if "\t" in line:
			fields = line.split("\t")
			if "." in fields[0]:  # ellipsis tokens
				continue
			if "-" in fields[0]:  # need to get super token and ignore next n tokens
				just_text.append(fields[1])
				start, end = fields[0].split("-")
				start = int(start)
				end = int(end)
				length = end-start+1
			else:
				if length > 0:
					length -= 1
					continue
				just_text.append(fields[1])
		elif len(line.strip())==0 and sentence_wise:  # New sentence
			sent_text = " ".join(just_text)
			sentences.append(sent_text)
			just_text = []

	if sentence_wise:
		return sentences
	else:
		text = " ".join(just_text)
		return text


def udpipe_tag(conllu,udpipe_model):
	"""
	Tag a conllu file using a given UDPipe model. Existing tags/dependencies/sentence splits will be removed

	:param conllu: Input in conllu format (only the token column matters)
	:param udpipe_model: Path to a trained UDPipe model
	:return: conllu string with tagged data
	"""

	tok_conllu = []
	counter = 1

	# Remove gold syntax
	for line in conllu.split("\n"):
		if "\t" in line:
			fields = line.split("\t")
			if "-" in fields[0]:
				continue
			tok = fields[1]
			outline = "\t".join([str(counter),tok,"_","_","_","_","0","_","_","_"])
			tok_conllu.append(outline)
			counter +=1
		elif line.startswith("#"):
			if "newdoc" in line:  # Preserve doc splits
				counter = 1
				tok_conllu.append("")
			tok_conllu.append(line)
		elif len(line.strip())==0:
			continue
			tok_conllu.append("")
			counter=1
	tok_conllu = "\n".join(tok_conllu)

	udpipe_path = os.path.abspath(os.path.join(lib,"udpipe")) + os.sep
	cmd = [udpipe_path + "udpipe","--tag",udpipe_model,"tempfilename"]
	tagged = exec_via_temp(tok_conllu, cmd)

	return tagged


def text2conllu(text):
	tokens = text.split(" ")
	output = []
	counter =1
	for tok in tokens:
		outline = "\t".join([str(counter),tok,"_","_","_","_","0","_","_","_"])
		counter+=1
		output.append(outline)

	return "\n".join(output)


def tt_tag(conllu,lang="eng",preserve_sent=False):
	"""
	Tag a conllu file using a given TreeTagger model. Existing tags/dependencies/sentence splits will be removed

	:param conllu: Input in conllu format (only the token column matters)
	:param lang: 3 letter language code corresponding to the TreeTagger model name in lib/treetagger/
	:return: conllu string with tagged data
	"""

	toks = []
	docid = ""
	first_doc = True
	if preserve_sent:
		toks.append("<s>")  # First sentence SGML open tag

	# Remove gold syntax
	for line in conllu.split("\n"):
		if "\t" in line:
			fields = line.split("\t")
			if "-" in fields[0]:
				continue
			tok = fields[1]
			toks.append(tok)
		elif line.startswith("# newdoc id"):
			docid = re.search("= *([^\s]+)",line).group(1)
			if not first_doc:
				toks.append("</newdoc>")
			toks.append('<newdoc id="'+docid+'">')
			first_doc = False
		elif len(line.strip())==0 and preserve_sent:
			toks.append("</s>")
			toks.append("<s>")

	if not toks[-1] == "</s>" and preserve_sent:
		toks.append("</s>")
	if docid != "":
		toks.append("</newdoc>")

	toks = "\n".join(toks)

	tt_path = os.path.abspath(os.path.join(lib,"treetagger")) + os.sep
	cmd = [
		tt_path + "tree-tagger",
		"-token",
		"-lemma",
		"-no-unknown",
		# "-cap-heuristics", # not found in 3.2.2 for some reason?
		"-hyphen-heuristics",
		lib+os.sep+"treetagger"+os.sep+lang+".par",
		"tempfilename"
	]
	print(cmd)
	tagged = exec_via_temp(toks, cmd)

	conllized = conllize(tagged, element="s", newdoc="newdoc", ten_cols=True)

	return conllized + "\n"


def shuffle_cut_conllu(conllu_in,limit=50000):
	"""
	Randomly shuffles conllu documents by '# newdoc id' comments and returns limit lines

	:param conllu_in: string with conllu input
	:param limit: line number to return
	:return: conllu data cut to limit lines
	"""

	seed(42)
	line_count = conllu_in.count("\n")
	if line_count > limit:
		sys.stderr.write("o Dataset with "+str(line_count)+" lines too large, selecting "+str(limit)+" lines from randomized documents\n")

		# Shuffle input by documents
		docs = conllu_in.split("# newdoc")
		if "" in docs:
			docs.remove("")
		shuffle(docs)
		conllu_in = "# newdoc".join(docs).strip()
		if not conllu_in.startswith("# newdoc") and conllu_in.startswith("id"):
			conllu_in = "# newdoc " + conllu_in

		# Cut shuffled training to limit
		lines = conllu_in.split("\n")
		lines = lines[:limit]
		conllu_in = "\n".join(lines)

	return conllu_in


def get_multitrain_preds(clf,X,y,multifolds):

	all_preds = []
	all_probas = []
	X_folds = np.array_split(X, multifolds)
	y_folds = np.array_split(y, multifolds)
	for i in range(multifolds):
		X_train = np.vstack(tuple([X_folds[j] for j in range(multifolds) if j!=i]))
		y_train = np.concatenate(tuple([np.array(y_folds[j]) for j in range(multifolds) if j!=i]))
		X_heldout = X_folds[i]
		sys.stderr.write("o Training on fold " + str(i+1) +" of " + str(multifolds) + "\n")
		clf.fit(X_train,y_train)
		probas = clf.predict_proba(X_heldout)
		preds = [str(int(p[1] > 0.5)) for p in probas]
		probas = [str(p[1]) for p in probas]
		all_preds += preds
		all_probas += probas

	pairs = list(zip(all_preds,all_probas))
	pairs = ["\t".join(pair) for pair in pairs]

	return "\n".join(pairs)


def feats2rnn(feat_dicts, scaler_dict=None, sep=" ", bracket_names=True, cat_labels=None, num_labels=None):
	"""

	:param feat_dicts: List with one dictionary of features per token, created by read_conll()
	:param scaler_dict: Optional dictionary of features names to existing sklearn StandardScaler() objects
	:param sep: Separator to use to separate features in output
	:param bracket_names: Whether to prepend feature name in brackets to each line (NCRFpp format)
	:param cat_labels: List of categorical features to include in the output; 'word' must be the first category
	:param num_labels: List of numerical features to scale and include in the output
	:return: String of line separated sentences, each one token per line with features delimited by sep, and dictionary of scalers
	"""

	# Chosen features here
	# Note 'word' must be first!
	if cat_labels is None:
		cat_labels = ["word","pos","genre","deprel","case","morph"]  # "cpos"
		cat_labels = ["word","pos","genre","deprel","case","morph","head_dist","wid"]  # "cpos"
	if num_labels is None:
		num_labels = ["tok_len","s_len","wid","head_dist","sent_doc_percentile"]
		num_labels = []

	all_feats = cat_labels + num_labels + ["label"]

	facts = pd.DataFrame(feat_dicts, columns=all_feats)

	if "Seg=B-Conn" in facts["label"].unique():  # Connective detection, BIO
		facts["label"] = np.where(facts["label"]=="Seg=B-Conn","B-CON",facts["label"])
		facts["label"] = np.where(facts["label"]=="Seg=I-Conn","I-CON",facts["label"])
		facts["label"] = np.where(facts["label"]=="_","O",facts["label"])
	else:  # Discourse unit segmentation, S/O
		facts["label"] = np.where(facts["label"]=="_","O","S-SEG")

	# Bin head_dist

	bins = [-1000, -4, -1.1, -0.01, 0.1, 1, 4, 1000]
	labels = ["RFAR","RCLOSE","RNEXT","ROOT","LNEXT","LCLOSE","LFAR"]
	facts['head_dist'] = pd.cut(facts['head_dist'], bins=bins, labels=labels)

	facts["genre"] = np.where(facts["wid"]==1,facts["genre"],"_")

	if scaler_dict is None:
		scaler_dict = {}
		for feat in num_labels:
			s = StandardScaler()
			facts[feat] = s.fit_transform(facts[feat].astype(float).values.reshape(-1, 1))
			scaler_dict[feat] = s
	else:
		for feat in num_labels:
			s = scaler_dict[feat]
			facts[feat] = s.transform(facts[feat].astype(float).values.reshape(-1, 1))

	col_vals = [facts[col].values for col in all_feats]
	output = []

	for i in range(facts.shape[0]):
		row = []
		if col_vals[all_feats.index("wid")][i] == 1 and i > 0:
			output.append("")  # Line space between sentences
		for j, col in enumerate(all_feats):
			if col == "word" or col == "label":
				row.append(col_vals[j][i])
			else:
				if bracket_names:
					row.append("["+col+"]" + str(col_vals[j][i]))
				else:
					row.append(str(col_vals[j][i]))
		output.append(sep.join(row))

	return "\n".join(output) + "\n", scaler_dict


def get_seg_labs(infile,as_text=False):
	if as_text:
		lines = infile.split("\n")
	else:
		lines = io.open(infile,encoding="utf8").readlines()
	labs = []

	for line in lines:
		if "\t" in line:
			fields = line.split("\t")
			if "-" in fields[0]:
				continue
			lab = fields[-1].strip()
			lab = 1 if "Seg=" in lab else 0
			labs.append(lab)
	return labs


