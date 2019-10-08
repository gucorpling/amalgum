#!/usr/bin/python
# -*- coding: utf-8 -*-
from random import choice

import re, io, os, sys
from argparse import ArgumentParser
from glob import glob

# Allow package level imports in module
script_dir = os.path.dirname(os.path.realpath(__file__))
lib = os.path.abspath(script_dir + os.sep + "..")
sys.path.append(lib)

from exec import exec_via_temp
from conll_reader import space_join

class UDPipeSentencer:

	def __init__(self,lang="eng"):
		lang_map = {"deu":"german","eng":"english","spa":"spanish","fra":"french","nld":"dutch","rus":"russian","eus":"basque","por":"portuguese","zho":"chinese", "tur":"turkish"}
		self.lang = lang
		self.long_lang = lang_map[lang] if lang in lang_map else lang
		self.name = "UDPipeSentencer"
		try:
			self.udpipe_model = glob(os.path.abspath(os.path.join(lib,"udpipe",self.long_lang+"*.udpipe")))[0]
		except:
			sys.stderr.write("! Model not found for language " + self.long_lang + "*.udpipe in " + os.path.abspath(os.path.join([lib,"udpipe",self.long_lang+"*.udpipe"]))+"\n")
			sys.exit(0)
		self.udpipe_path = os.path.abspath(os.path.join(lib,"udpipe")) + os.sep

	def run_udpipe(self,text):

		cmd = [self.udpipe_path + "udpipe","--tokenize",self.udpipe_model,"tempfilename"]
		tokenized = exec_via_temp(text, cmd, workdir=self.udpipe_path)
		return tokenized

	def predict(self,conllu):

		# Reconstruct text with heuristics
		text = space_join(conllu)
		tokens = text.split()
		text = re.sub(r" ([.,，!?;；:：！？。)\]}%])",r'\1',text)  # Heuristic detokenization
		text = re.sub(r"([$([{]) ",r'\1',text)

		# Run UDPipe sentencer
		conllu = self.run_udpipe(text)
		sents = space_join(conllu,sentence_wise=True)

		# Realign to input tokens
		tabbed = "\t".join(sents)
		tabbed = "\t" + tabbed.replace(" ","")

		output = []
		for tok in tokens:
			if tabbed.startswith("\t"):  # This is a split point
				output.append((1,1.0))  # Prediction is 1 (='segment') probability is 1.0
				tabbed = tabbed[1:]
			else:
				output.append((0,0.0)) # Prediction is 0 (='non segment') probability is 0.0
			if tabbed.startswith(tok):
				tabbed = tabbed[len(tok):]

		# Verify we are returning as many predictions as we received input tokens
		assert len(tokens) == len(output)

		return output


if __name__ == "__main__":

	p = ArgumentParser()
	p.add_argument("-f","--file",default=None,help="file to tokenize")
	p.add_argument("-l","--lang",default="eng",help="language 3 letter code",choices=["eng","spa","fra","deu","eus","nld","rus","por","zho", "tur"])

	opts = p.parse_args()

	infile = opts.file
	lang = opts.lang

	# Run test
	sentencer = UDPipeSentencer(lang=lang)

	if infile is None:
		# Some default test tokens if no file provided
		conllu = """1	In	_	_	_	_	_	_	_	SpacesBefore=\s
2	Sorge	_	_	_	_	_	_	_	_
3	Schon	_	_	_	_	_	_	_	_
4	machen	_	_	_	_	_	_	_	_
5	die	_	_	_	_	_	_	_	_
6	ersten	_	_	_	_	_	_	_	_
7	Witze	_	_	_	_	_	_	_	_
8	die	_	_	_	_	_	_	_	_
9	Runde	_	_	_	_	_	_	_	SpaceAfter=No
10	.	_	_	_	_	_	_	_	_
11	Darin	_	_	_	_	_	_	_	_
12	kommt	_	_	_	_	_	_	_	_
13	weißes	_	_	_	_	_	_	_	_
14	Pulver	_	_	_	_	_	_	_	_
15	vor	_	_	_	_	_	_	_	SpaceAfter=No
16	.	_	_	_	_	_	_	_	_
"""
	else:
		conllu = io.open(infile,encoding="utf8").read()

	sent_starts = sentencer.predict(conllu)
	tokens = space_join(conllu).split()
	print([(tok, boundary) for tok, boundary in (zip(tokens,sent_starts))])


