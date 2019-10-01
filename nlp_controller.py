#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
nlp_controller.py
A general purpose interface to add annotations to data coming from all genres
"""

import os, io, sys, re, platform
from argparse import ArgumentParser
from lib.whitespace_tokenize import tokenize as tt_tokenize
from lib.utils import exec_via_temp, get_col

PY3 = sys.version_info[0] == 3
script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
lib_dir = script_dir + "lib" + os.sep
bin_dir = script_dir + "bin" + os.sep
xml_out = script_dir + "nlped" + os.sep + "xml" + os.sep

class AutoGumNLP:

	def __init__(self):
		pass

	def pos_tag(self, tokenized_document):
		data = tokenized_document.strip().split("\n")
		orig_data_split = data
		data = [word.strip().replace("’","'") for word in data]  # Use plain quotes for easier tagging
		data = "\n".join(data)

		# Choose tagger here
		tagger = "TT"
		outfile = False

		if tagger == "TT":
			tt_path = bin_dir + "treetagger" + os.sep + "bin" + os.sep
			tag = [tt_path + 'tree-tagger', tt_path + 'english.par', '-lemma', '-no-unknown', '-sgml', 'tempfilename']
		elif tagger == "marmot":
			outfile = True
			if platform.system() == "Windows":
				tag = ["java","-Dfile.encoding=UTF-8","-Xmx2g","-cp","marmot.jar;trove.jar","marmot.morph.cmd.Annotator","-model-file","eng.marmot","-lemmatizer-file","eng.lemming","-test-file","form-index=0,tempfilename","-pred-file","tempfilename2"]
			else:
				tag = ["java","-Dfile.encoding=UTF-8","-Xmx2g","-cp","marmot.jar:trove.jar","marmot.morph.cmd.Annotator","-model-file","eng.marmot","-lemmatizer-file","eng.lemming","-test-file","form-index=0,tempfilename","-pred-file","tempfilename2"]

		tagged = exec_via_temp(data, tag, outfile=outfile)
		tagged = tagged.strip().replace("\r","").split("\n")
		if tagger == "marmot":
			tags = get_col(tagged,5)
			lemmas = get_col(tagged,3)
		else:
			tags = get_col(tagged,0)
			lemmas = get_col(tagged,1)

		data = orig_data_split

		outlines = []
		counter = 0
		for i, line in enumerate(data):
			if not line.startswith("<") and not line.endswith(">"):
				pos, lemma = tags[counter], lemmas[counter]
				lemma = lemma.replace('"',"''")
				if data[i].strip() == "“":
					pos = "``"
				elif data[i].strip() == "”":
					pos = "''"
				elif data[i].strip() == "[":
					pos = "("
				elif data[i].strip() == "]":
					pos = ")"
				outline = "\t".join([line,pos,lemma])
				outlines.append(outline)
				counter += 1
			else:
				outlines.append(line)
		tagged = "\n".join(outlines)

		return tagged

	def tokenize(self, xml_data):
		"""Tokenize input XML or plain text into TT SGML format.

		:param xml_data: input string of a single document
		:return: TTSGML with exactly one token or opening tag or closing tag per line

		example input:

			<text id="autogum_voyage_doc3" title="Aakirkeby">
			<head>Aakirkeby</head>

			<p><hi rend="bold">Aakirkeby</hi> is a city on <ref target="Bornholm">Bornholm</ref>,

		example output:

			<text id="autogum_voyage_doc3" title="Aakirkeby">
			<head>
			Aakirkeby
			</head>
			<p>
			<hi rend="bold">
			Aakirkeby
			</hi>
			is
			a
			city
			...
		"""

		def postprocess_tok(TTSGML):
			# Phone numbers
			if "☏" in TTSGML:
				phone_exp = re.findall(r'((?:☏|(?:fax|phone)\n:)\n(?:\+?[0-9]+\n|-\n)+)',TTSGML,flags=re.UNICODE)
				for phone in phone_exp:
					fused = phone.replace("\n","").replace("☏","☏\n").replace("fax:","fax\n:\n").replace("phone:","phone\n:\n") + "\n"
					TTSGML = TTSGML.replace(phone,fused)

			# Currency
			TTSGML = re.sub(r'([¥€\$])([0-9,.]+)\n',r'\1\n\2\n',TTSGML)

			# Ranges
			TTSGML = re.sub(r'(¥|\$|€)\n?([0-9.,]+)-([0-9.,]+\n)',r'\1\n\2\n-\n\3',TTSGML)  # Currency
			TTSGML = re.sub(r'([12]?[0-9]:[0-5][0-9])(-)([12]?[0-9]:[0-5][0-9])\n',r'\1\n\2\n\3\n',TTSGML)  # Time
			TTSGML = re.sub(r'((?:sun|mon|tues?|wed|thu(?:rs)|fri|sat(?:ur)?)(?:day)?)-((?:sun|mon|tues?|wed|thu(?:rs)|fri|sat(?:ur)?)(?:day)?)\n',
							r'\1\n-\n\2\n',TTSGML,flags=re.IGNORECASE)  # Days




			fixed = TTSGML
			return fixed

		abbreviations = lib_dir + "english-abbreviations"

		# Separate n/m dashes
		xml_data = xml_data.replace("–"," – ").replace("—"," — ")

		tokenized = tt_tokenize(xml_data,abbr=abbreviations)
		tokenized = postprocess_tok(tokenized)

		return tokenized


if __name__ == "__main__":

	nlp = AutoGumNLP()

	test = script_dir + "out" + os.sep + "voyage" + os.sep + "autogum_voyage_doc4.xml"

	files = [test]

	for file_ in files:
		raw_xml = io.open(file_,encoding="utf8").read()
		tokenized = nlp.tokenize(raw_xml)

		tok_count = len([t for t in tokenized.strip().split("\n") if not t.startswith("<")])

		# Skip documents that are too big or small
		if tok_count < 300 or tok_count > 5000:
			continue

		# POS tag
		tagged = nlp.pos_tag(tokenized)

		doc = os.path.basename(file_)

		with io.open(xml_out + doc, 'w', encoding="utf8", newline="\n") as f:
			f.write(tagged)
