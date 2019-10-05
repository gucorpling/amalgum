import io, re, sys, os
from glob import glob
from random import shuffle, seed, choice
from lib.utils import Document


def get_paragraphs(text):
	"""Divides a string into a list of paragraphs based on multiple white space lines"""
	text = re.sub('(^ +|\t)','',text.strip())  # Trim all leading space and tabs in lines
	text = re.sub(' +','♡❤♡',text)  # Protect spaces
	paragraphs = re.split(r'\n\n+',text.strip())
	output = []
	for para in paragraphs:
		para = para.replace("♡❤♡"," ")
		#para = re.sub(r'\n',' ',para)  # All lines end in space
		para = re.sub(' +',' ',para)
		output.append(para.strip())

	return output


def detect_headings(paragraphs):
	"""Classifies paragraphs into headings or not based on line count and other heuristics"""

	output = []
	for i, para in enumerate(paragraphs):
		if i > 48:
			a=3
		if para.count("\n") > 0:  # Multiline can't be heading
			output.append(0)
			continue
		else:
			if re.match(r'[a-z\[]',para[0]) is not None:  # Lower case or figure, footnote can't be heading
				output.append(0)
				continue
			elif "copyright" in para.lower():
				output.append(0)
				continue
			elif re.match(r'".*"$',para):  # Direct speech line paragraph
				output.append(0)
				continue
			elif re.search(r'[A-Za-z]',para) is None:  # No text in paragraph
				output.append(0)
				continue
			else:
				output.append(1)
	return output




seed(42)

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
data_dir = script_dir + "data" + os.sep

files = glob(data_dir + "fiction" + os.sep + "*.txt")

docnum = 0
for file_ in files:
	text = io.open(file_,encoding="utf8").read()

	# Collapse multiline brackets (e.g. multiline figure captions)
	collapsed = ""
	lines = text.split("\n")
	open_bracket = False
	for line in lines:
		if "[" in line and not "]" in line:
			open_bracket = True
		if open_bracket:
			collapsed += line
		else:
			collapsed += line + "\n"
		if "]" in line:
			open_bracket = False
	text = collapsed

	paragraphs = get_paragraphs(text)
	heading_idx = detect_headings(paragraphs)
	hds = [i for i, h in enumerate(heading_idx) if h == 1]
	hd_before_text = []
	last = 1000000
	for i in hds[::-1]:
		if i != last - 1:
			if i+1 < len(paragraphs)-1:
				if len(paragraphs[i+1])>200:  # Next paragraph must be somewhat long
					hd_before_text.append(i)
	if len(hd_before_text) == 0:
		continue
	start = choice(hd_before_text)

	spaces = 0
	par_num = start
	output = []
	for para in paragraphs[start:]:
		spaces += para.count(" ")
		if par_num in hds:
			para = "<head>" + para + "</head>"
		else:
			para = "<p>" + para + "</p>"
		para = para.replace("\n"," ")
		para = re.sub(' +',' ', para)
		para = re.sub(r'_([^_]+)_',r'<hi rend="italic">\1</hi>',para)  # italic
		# Footnotes
		fn = re.search(r'\[footnote( [0-9]+)?:?([^\]]+)\]',para,flags=re.IGNORECASE)
		if fn is not None:
			if fn.group(1) is not None:
				note_num = ' n="'+fn.group(1)+'"'
			else:
				note_num = ""
			para = para.replace(fn.group(0),'<note'+note_num+'>'+fn.group(2)+'</note>')

		# Figures
		para = re.sub(r'\[illustration:?([^\]]*)\]',r'<figure><caption>\1</caption></figure>',para,flags=re.IGNORECASE)
		para = para.replace("<figure><caption></caption></figure>","<figure/>")

		par_num +=1
		output.append(para)

		if spaces > 1000:
			break
	if spaces < 300:
		continue

	# Remove alpha-less paragraphs in head/tail, ignoring XML:
	cleaned = []
	start = False
	for para in output:
		noxml = re.sub(r'<[^<>]+>','',para)
		if re.search('[A-Za-z]',noxml) is None:
			if not start:
				continue
		else:
			start = True
		cleaned.append(para)
	output = cleaned
	for i in list(range(len(cleaned)))[::-1]:
		noxml = re.sub(r'<[^<>]+>','',output[i])
		if re.search(r'[A-Za-z]',noxml) is None:
			output.pop(i)
		else:
			break

	doc = Document()
	doc.url = "https://www.gutenberg.org/"
	doc.genre = "fiction"
	doc.author, doc.title = os.path.basename(file_).replace(".txt","").split("___")
	doc.text = "\n\n".join(output)
	doc.docnum = docnum

	if re.search(r'\bpoe',doc.title) is not None:  # Avoid poetry
		continue

	doc.serialize()

	docnum+=1

