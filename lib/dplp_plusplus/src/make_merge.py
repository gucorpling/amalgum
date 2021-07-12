import io, os, sys, re
from collections import defaultdict
from glob import glob
from textblob import TextBlob
from flair.data import Sentence
from flair.models import TextClassifier
import torch, flair

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep

# Sentence classifier model to predict discourse functions
class Sequencer:
	def __init__(self, model_path=None):
		if model_path is None:
			model_path = script_dir + ".." + os.sep + "models" + os.sep + "rstdt_collapsed.pt"
		self.tagger = TextClassifier.load(model_path,)  # .load_from_file

	def clear_embeddings(self, sentences, also_clear_word_embeddings=False):
		"""
		Clears the embeddings from all given sentences.
		:param sentences: list of sentences
		"""
		for sentence in sentences:
			sentence.clear_embeddings(also_clear_word_embeddings=also_clear_word_embeddings)

	def predict_proba(self, sentences):
		from flair import __version__
		from flair.data import Sentence

		# Sort sentences and keep order
		sents = [(len(s.split()),i,s) for i, s in enumerate(sentences)]
		sents.sort(key=lambda x:x[0], reverse=True)

		sentences = [s[2] for s in sents]

		major, minor = str(__version__).split(".")[0:2]
		if int(major) > 0 or int(minor) > 4:
			sentences = [Sentence(s, use_tokenizer=lambda q: q.split()) for s in sentences]

		preds = self.tagger.predict(sentences)

		if preds is None:  # Newer versions of flair have void predict method, use modified Sentence list
			preds = sentences

		# sort back
		sents = [tuple(list(sents[i]) + [s]) for i, s in enumerate(preds)]
		sents.sort(key=lambda x:x[1])
		sents = [s[3] for s in sents]

		output = []
		for s in sents:
			output.append((s.labels[0].value,s.labels[0].score))

		return output


def merge(rst, xml, dep, filename, seq=None, as_text=True, outdir=""):
	#flair.device = torch.device('cpu')

	if as_text:
		rst_lines = rst.split("\n")
		xml_lines = xml.split("\n")
		dep_lines = dep.split("\n")
		format = "rs3"
		if filename.count("_") == 2:
			genre = filename.split("_")[1]
		else:
			genre = "None"
	else:
		rst_lines = io.open(rst, encoding="utf8").read().strip().split("\n")
		xml_lines = io.open(xml, encoding="utf8").read().strip().split("\n")
		dep_lines = io.open(dep, encoding="utf8").read().strip().split("\n")
		if ".dis" in rst:
			format = "dis"
		elif ".rs3" in rst:
			format = "rs3"
		genre = os.path.basename(rst).split("_")[1]

	# files = glob(paths["dis"] + "*.dis")
	# if len(files) == 0:
	# 	files = glob(paths["dis"] + "*.rs3")
	# if len(files) ==0:
	# 	sys.stderr.write("No RST files found in " + paths["dis"] + "\nQuitting...")
	# 	quit()

	# for file_ in files:
	# 	docname = re.sub(r'\.[^.]*','',os.path.basename(file_))
	# 	if docname.count("_") == 2:
	# 		# Expecting document names like GUM_interview_xyz -> genre=interview
	# 		genre = docname.split("_")[1]  # TODO: make better genre featureconfiguration
	# 	else:
	# 		genre = "NONE"

		# Look for .rs3 or .dis file to get EDUs

		# if os.path.exists(paths["dis"] + docname + ".dis"):
		# 	format = "dis"
		# elif os.path.exists(paths["dis"] + docname + ".rs3"):
		# 	format = "rs3"
		# else:
		# 	sys.stderr.write("RST file " + docname + " missing... skipping\n")
		# 	continue

	# rst_lines = io.open(paths["dis"] + docname + "." + format,encoding="utf8").read().strip().split("\n")

	tok2edu = {}  # Which EDU each token in a document is in
	edu2tok = {}  # Start and end tokens for each edu
	edu2sentiment = {}
	edu2seqpred = {}
	edus = []

	counter = 0
	for line in rst_lines:
		if "<segment" in line and format == "rs3":
			edu_num = re.search('id="([^"]+)"',line).group(1)
			text = re.search(r'>(.*)<',line).group(1)
		elif "text _!" in line and format == "dis":
				m = re.search(r'leaf ([0-9]+).*?text _!(.*)_!', line)
				if m is not None:
					edu_num = m.group(1)
					text = m.group(2)
		else:
			continue

		toks = text.split()
		start = counter
		for tok in toks:
			tok2edu[counter] = edu_num
			counter += 1
		end = counter - 1
		edu2tok[edu_num] = (start,end)
		edu2sentiment[edu_num] = TextBlob(text).sentiment
		edus.append(text.strip())

	seq_preds = seq.predict_proba(edus)

	# xml_lines = io.open(paths["xml"] + docname + ".xml",encoding="utf8").read().strip().split("\n")

	tok2feat = defaultdict(lambda: "_")  # XML features for tokens

	counter = 0
	elems = ["head","caption","sp","p","item"]
	xml_tag = "_"
	for line in xml_lines:
		if line.startswith("<"):
			if xml_tag == "_" or "/" in xml_tag:
				for elem in elems:
					if "<" + elem + ">" in line or "<" + elem + " " in line:
						xml_tag  = "<" + elem + ">"
						break
			if xml_tag == "_":
				for elem in elems:
					if "</" + elem + ">" in line:
						xml_tag  = "</" + elem + ">"
						break
		if "\t" in line:
			if "/" in xml_tag and counter != 0:
				tok2feat[counter-1] = xml_tag
			else:
				tok2feat[counter] = xml_tag
			xml_tag = "_"
			counter += 1
	if xml_tag != "_":
		tok2feat[counter-1] = xml_tag

	# dep_lines = io.open(paths['dep'] + docname + ".conllu", encoding="utf8").read().strip().split("\n")

	sent_num = 0
	output = []
	counter = 0
	s_type = "other"
	output_lines = ""
	for line in dep_lines:
		if "s_type" in line:
			s_type = re.search(r'=\s*([^\n]+)',line).group(1)
		elif "\t" in line:
			fields = line.split("\t")
			if "." in fields[0] or "-" in fields[0]:
				continue
			tokid, word, lemma, upos, xpos, morph, head, deprel, _, _ = fields
			# Check if head is outside EDU to left or right
			parent_dir = "NONE"
			edu_num = tok2edu[counter]
			edu_head = "0"
			edu_func = "NONE"
			if deprel != "punct" and head != "0":
				dist2head = int(head) - int(tokid)
				try:
					head_edu = tok2edu[counter+dist2head]
				except:
					raise IOError("Corrupt input: Missing EDU or relation in discourse file ")
				if head_edu > edu_num:
					parent_dir = "RIGHT"
					edu_head = tokid
					edu_func = deprel
				elif head_edu < edu_num:
					parent_dir = "LEFT"
					edu_head = tokid
					edu_func = deprel
			elif head=="0":
				edu_head = tokid
				edu_func = "root"

			subj, senti = edu2sentiment[edu_num]
			subj = str(subj)
			senti = str(senti)
			if int(edu_num) <= len(seq_preds):
				seq_lab, seq_conf = seq_preds[int(edu_num)-1]
			else:
				raise IOError("Corrupt input: Missing EDU or relation in discourse file ")
			seq_conf = str(seq_conf)

			output.append("\t".join([str(sent_num),tokid,word,lemma,xpos,deprel,head,tok2feat[counter] + "|" + genre + "|"+edu_func + "|" + parent_dir,s_type + "|"+subj+"|"+senti+"|"+seq_lab+"|"+seq_conf,tok2edu[counter]]))
			counter += 1
		elif len(line.strip())==0:  # New sentence
			sent_num += 1
			output.append("")

	with io.open(outdir + filename + ".merge", 'w', encoding="utf8", newline="\n") as f:
		f.write("\n".join(output)+"\n")

	output_lines = "\n".join(output)+"\n"

	return output_lines


#seq = Sequencer()
