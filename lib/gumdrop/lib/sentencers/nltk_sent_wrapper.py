#!/usr/bin/python
# -*- coding: utf-8 -*-

from nltk import sent_tokenize, word_tokenize
import re, io, os, sys
from argparse import ArgumentParser
# Allow package level imports in module
script_dir = os.path.dirname(os.path.realpath(__file__))
lib = os.path.abspath(script_dir + os.sep + "..")
sys.path.append(lib)
from conll_reader import space_join

class NLTKSentencer:

	def __init__(self,lang="eng"):
		lang_map = {"deu":"german","eng":"english","spa":"spanish","fra":"french","nld":"dutch","rus":"english","eus":"spanish","por":"portuguese", "tur":"turkish"}
		self.lang = lang
		self.name = "NLTKSentencer"
		self.long_lang = lang_map[lang] if lang in lang_map else lang

	def predict(self,conllu):

		if isinstance(conllu, list):
			tokens = conllu
		else:
			tokens = space_join(conllu)
			tokens = tokens.split()

		# Reconstruct text with heuristics
		text = " ".join(tokens)
		text = re.sub(r" ([.,，!?;；:：！？。)\]}%])",r'\1',text)
		text = re.sub(r"([$([{]) ",r'\1',text)

		# Run NLTK sentence tokenize
		if self.lang == "zho":  # Just split on sentence final punctuation
			endpunct = "[！？。.!?]"
			text = re.sub("("+endpunct+")",r'\1//<->//',text)
			sents = re.split('//<->//',text)
			sents = [s for s in sents if len(s.strip())>0]
		else:
			sents = sent_tokenize(text,language=self.long_lang)

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
	sentencer = NLTKSentencer(lang=lang)

	if infile is None:
		# Some default test tokens if no file provided
		if lang == "zho":
			tokens = ['闽','台','经贸','合作','的','深入','发展','为','福建','汽车','工业','注入','了','生机','。','去年','初','以来','，','台湾','最','具','实力','的','汽车','公司','——','裕隆','集团','中华','汽车','公司','多','次','组','团','访','闽','，','就','合作','发展','汽车','工业','进行','了','积极','的','蹉商','；','福建','方面','则','成立','了','由','省委','书记','贾庆林','、','省长','陈明义','任','正','、','副','组长','的','省','汽车','工业','领导','小组','，','将','发展','本','省','汽车','工业','摆上','重要','议事','日程','。']
		else:
			tokens = ['Introduction', 'Research', 'has', 'shown', 'examples', '.', 'But', 'we', 'need', 'more', '.']
	else:
		text = io.open(infile,encoding="utf8").read()
		tokens = word_tokenize(text)

	sent_starts = sentencer.predict(tokens)
	print([(tok, boundary) for tok, boundary in (zip(tokens,sent_starts))])


