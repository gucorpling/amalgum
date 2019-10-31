import urllib.request, bz2, io, os, sys, re, shutil
#from dateparser.search import search_dates  # This one is worse than datefinder
from datefinder import find_dates  # Maybe try sutime instead? (but needs Java)
from collections import OrderedDict
from multiprocessing import cpu_count
from glob import glob
from random import shuffle, seed
from lib.WikiExtractor import process_dump
from lib.utils import Document

seed(42)

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep


def get_dump(dump_file="enwikivoyage-latest-pages-articles.xml"):
	"""Downloads dump file"""

	dump_url = "https://dumps.wikimedia.org/enwikivoyage/latest/enwikivoyage-latest-pages-articles.xml.bz2"

	req = urllib.request.urlopen(dump_url)
	CHUNK = 16 * 1024

	decompressor = bz2.BZ2Decompressor()
	with io.open(dump_file, 'wb') as fp:
		while True:
			chunk = req.read(CHUNK)
			if not chunk:
				break
			fp.write(decompressor.decompress(chunk))
	req.close()


def reformat_text(text):

	def leap(year):
		year = int(year)
		if year % 4 > 0:
			return False
		elif year % 100 > 0:
			return True
		elif year % 400  >0:
			return False
		else:
			return True

	# Remove XML except hyperlinks and headings
	text = text.replace("&gt;",">").replace("&lt;","<").replace("&amp;","&").replace("&nbsp;"," ")
	text = re.sub(r"<(/a|a [^<>]*)>",r"#@Q\1Q@#",text)
	text = re.sub(r"<(/?h[0-9])>",r"#@Q\1Q@#",text)
	text = re.sub('<[^<>]+>','',text)
	text = text.replace("#@Q","<").replace("Q@#",">")

	# Mark up
	text = re.sub(r'<h[0-9]>([^\n<>]+)</h[0-9]>', r"<head>\1</head>", text)
	text = re.sub(r'(<a href)', r"<ref target", text)
	text = re.sub(r'(target="[^"]+)%3A', r"\1:", text)
	text = re.sub(r'(</?)a([ >])', r"\1ref\2", text)
	text = re.sub(r'BULLET::::-?([^\n]+)', r"<item>\1</item>", text)
	text = re.sub(r'^([^<\s][^\n]*)$',r'<p>\1</p>',text,flags=re.MULTILINE)
	text = re.sub(r'♡❤♡([^❤♡]+)♡❤♡', r'<hi rend="bold">\1</hi>', text)
	text = re.sub(r'♡❤([^❤♡]+)♡❤', r'<hi rend="italic">\1</hi>', text)
	text = re.sub(r' +', r" ", text)


	pats = [(r'((<item>.*?</item>\n?)+)',r'<list type="unordered">\n\1</list>\n')]

	text = text.replace("&quot;",'"')
	for pat in pats:
		text = re.sub(pat[0], pat[1], text,flags=re.MULTILINE|re.DOTALL)

	dates = find_dates(text,source=True,strict=True)  # Non-strict finds garbage
	unique_dates = set([])
	if dates is not None:
		for d in dates:
			date_spec, date_text = d
			if len(date_text) < 4:
				continue
			year, month, day = date_spec.year, date_spec.month, date_spec.day
			if year == 0:
				year = "xxxx"
			else:
				year = str(year)
			if month == 0 and day == 0:
				when = '<date when="'+year+'">'
				unique_dates.add((date_text,when))
				continue
			elif month == 0:
				continue
			month = str(month)
			if len(month) ==1:
				month = "0" + month
			if day == 0:
				start = year + "-" + month + "-01"
				end = "31" if month in ["01","03","05","07","08","10","12"] else "30"
				if month == "02":
					if leap(year):
						end = "29"
					else:
						end = "28"
				end = year + "-" + month + "-" + end
				when = '<date notBefore="'+start+'" notAfter="' + end + '">'
			else:  # Full date
				day = str(day)
				if len(day) == 1:
					day = "0"+day
				when = '<date when="' + year + "-" + month + "-" + day + '">'
			unique_dates.add((date_text,when))
	for t, d in unique_dates:
		text = text.replace(t,d+t+"</date>")
	return text


if __name__ == "__main__":

	MAX_PER_FILE = 10

	dump_file = "enwikivoyage-latest-pages-articles.xml"
	if not os.path.exists(script_dir + dump_file):
		sys.stderr.write("o Downloading data\n")
		get_dump(dump_file)
	else:
		sys.stderr.write("o Found cached download data\n")

	if not os.path.exists(script_dir + "data"):
		os.mkdir(script_dir + "data")
	if not os.path.exists(script_dir + "data" + os.sep + "voyage"):
		os.mkdir(script_dir + "data" + os.sep + "voyage")

	# Set file size for WikiExtractor output
	file_size = "1M"
	power = 'kmg'.find(file_size[-1].lower()) + 1
	file_size = int(file_size[:-1]) * 1024 ** power

	# Get Wiki content
	if not os.path.exists(script_dir + "data" + os.sep + "voyage" + os.sep + "AA"):
		sys.stderr.write("o Extracting data with WikiExtractor\n")
		template_file = script_dir + "lib" + os.sep + "voyage_templ.txt"
		default_process_count = max(1, cpu_count() - 1)
		default_process_count =1
		# If templates file does not exist yet, this will start a LONG pass of all data to harvest templates, then run
		# the extraction
		from types import SimpleNamespace
		options = SimpleNamespace(toHTML=True, keepLists=True)

		process_dump(script_dir+dump_file, template_file, script_dir + "data" + os.sep + "voyage" + os.sep, file_size,
				 False, default_process_count)
	else:
		sys.stderr.write("o Found processed files in data" + os.sep + "voyage" + os.sep + "AA\n")

	# Sample documents from 50 randomly selected files
	all_docs = []

	files = glob(script_dir + "data" + os.sep + "voyage" + os.sep + "**" + os.sep + "wiki*",recursive=True)
	shuffle(files)
	#files = files[:1]

	for file_ in files:
		current_file_docs = 0
		docs = io.open(file_,encoding="utf8").read().split("</doc>")
		if len(all_docs) > 600:
			break

		for doc in docs:
			if current_file_docs > MAX_PER_FILE:
				break
			if len(all_docs) > 600:
				break
			if "<doc" not in doc:
				continue
			current_doc = Document(genre="voyage")
			m = re.search('<doc id="([^"]*)" url="([^"]*)" title="([^"]*)"', doc)
			current_doc.url = m.group(2)
			current_doc.title = m.group(3)
			text = re.sub('<doc[^<>]*?>', '', doc)
			text = reformat_text(text)
			if text.count(" ") < 400:
				continue
			if text.count(" ") > 1000:
				# Truncate text at first section exceeding 1000 spaces
				sections = text.split("<head>")
				current_text = []
				for section in sections:
					prev_text = current_text
					current_text.append(section)
					if "".join(current_text).count(" ") > 1000 and "".join(prev_text).count(" ") > 400:
						break
				text = "<head>".join(current_text)


			# Remove empty section
			text = re.sub(r'<head>[^<>]+</head>\s*(?=(<head>|</text>|$))',r'',text)


			current_doc.text = text
			all_docs.append(current_doc)
			current_file_docs += 1


	if not os.path.exists(script_dir + "out"):
		os.mkdir(script_dir + "out")
	if not os.path.exists(script_dir + "out" + os.sep + "voyage"):
		os.mkdir(script_dir + "out" + os.sep + "voyage")
	else:
		shutil.rmtree(script_dir + "out" + os.sep + "voyage")
		os.mkdir(script_dir + "out" + os.sep + "voyage")

	out_dir = script_dir + "out" + os.sep + "voyage" + os.sep
	shuffle(all_docs)
	for i, doc in enumerate(all_docs):
		doc.docnum = i
		doc.author= "Wikivoyage community (see URL)"
		#doc.date_created = ?
		#doc.date_modified = ?
		doc.serialize(out_dir)

	sys.stdout.write("\no Wrote " + str(len(all_docs)) + " documents\n")

