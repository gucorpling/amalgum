import io, re, sys, os
from glob import glob
from random import shuffle, seed, choice
from lib.utils import Document
from gutenberg.acquire import load_etext
from gutenberg.acquire.text import _format_download_uri
from gutenberg.cleanup import strip_headers


# Constants for limiting text sizes based on space count
MAX_SPACES = 1000
MIN_SPACES = 300
TARGET_SPACES = 800
TOTAL_GENRE_SIZE = 430000  # approx. ratio of 0.85 spaces to tokens, so just over 0.5 M tokens

def detect_hyphenation(text):
	"""Heuristically detects texts with unrestored hyphenation"""
	bad = ["sug- g","dis- t","ig- n","un- "]
	if any([b in text for b in bad]):
		return True
	return False


def detect_non_fiction(text):
	if "endnote" in text.lower():
		return True
	if "<head>contents" in text.lower():
		return True
	if "<head>table of contents" in text.lower():
		return True
	if "<head>list of chapters" in text.lower():
		return True
	if "<p>contents.</p>" in text.lower():
		return True
	if "<head>illustrations</head>" in text.lower():
		return True
	if "<head>acknowledgments" in text.lower():
		return True
	if " trans. by" in text.lower():
		return True

	# Archaic language:
	if " doth " in text or " hath " in text or " thou art " in text:
		return True

	return False

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
			elif para.startswith('"') or para.startswith("'") or para.startswith("‘") or para.startswith("“"):
				# Possible direct speech line paragraph
				output.append(0)
				continue
			elif re.search(r'[A-Za-z]',para) is None:  # No text in paragraph
				output.append(0)
				continue
			elif para[-1] in [")","(","]","[",".","?","!", ":", "-", "”", '"', "'","’"]:  # Heading should not end in punctuation
				output.append(0)
				continue
			else:
				output.append(1)
	return output


seed(42)

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
data_dir = script_dir + "data" + os.sep

meta = io.open("gutenberg_meta_filtered.tab",encoding="utf8").read().strip().split("\n")

docnum = 0

total_spaces = 0
accepted = 0

for i, book in enumerate(meta):

	if i+1 % 10 == 0:
		sys.stderr.write("\rSeen " + str(i+1) + " books, accepted " + str(accepted) +
						 " with " + str(total_spaces)+ " spaces              ")

	book_id, title, author = book.split("\t")
	try:
		e_text = load_etext(int(book_id),mirror="http://gutenberg.readingroo.ms")
		url = _format_download_uri(int(book_id),mirror="http://gutenberg.readingroo.ms")
	except:  # can't find URI, e.g. UnknownDownloadUriException
		sys.stderr.write("WARN: could not download text ID:" + str(book_id) + "\n")
		continue

	text = strip_headers(e_text).strip().replace("\r","")

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
		para = re.sub(r'\[(?:[Ii]llustration|[Pp]icture):?([^\]]*)\]',r'<figure><caption>\1</caption></figure>',para,flags=re.IGNORECASE)
		para = para.replace("<figure><caption></caption></figure>","<figure/>")

		par_num +=1
		output.append(para)

		if spaces > TARGET_SPACES:
			break
	if spaces < MIN_SPACES:
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
	doc.url = url
	doc.genre = "fiction"
	doc.author, doc.title = author, title  # os.path.basename(file_).replace(".txt","").split("___")
	doc.text = "\n\n".join(output)
	doc.docnum = docnum

	if doc.text.count(" ") > MAX_SPACES:
		continue
	if detect_hyphenation(doc.text):
		continue
	if detect_non_fiction(doc.text):
		continue

	doc.serialize()
	accepted += 1

	total_spaces += doc.text.count(" ")

	if total_spaces > TOTAL_GENRE_SIZE:
		break

	docnum+=1

