#!/usr/bin/python
# -*- coding: utf-8 -*-

import json, sys, re, os, io
import praw
import csv, tempfile, random
from glob import glob
from langdetect import detect, DetectorFactory
from collections import defaultdict
from lib.utils import Document, utc_to_date
DetectorFactory.seed = 0

PY3 = sys.version_info[0] == 3

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
out_dir = script_dir + "out" + os.sep + "reddit" + os.sep


if sys.platform == "win32" and not PY3:
	import os, msvcrt
	msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)


def bad_reddit(xml):
	if "> > >" in xml:  # plain text quotation structures:
		return True
	if xml.count("target=") > 20:  # Could be just a bunch of links
		return True
	if xml.count("*") > 10:  # Unprocessed bold/bullets, who knows what
		return True
	if xml.count("@") > 20:  # List of e-mail addresses?
		return True
	if re.search(r'\bbot\b',xml.lower()):
		return True

	return False

def unescape_reddit(xml):
	xml = xml.replace(r'\(','(').replace(r'\)',')').replace(r'\-','-').replace("\_","_")
	xml = xml.replace(r'\+','+').replace(r'\>','&gt;').replace(r'\<','&lt;')
	return xml

def get_via_praw(post_id, post_type, praw_cred):

	if praw_cred is None:
		raise IOError("Missing praw credentials")

	from praw import Reddit

	reddit = Reddit(client_id=praw_cred["client_id"], client_secret=praw_cred["client_secret"],
						 password=praw_cred["password"], user_agent=praw_cred["user_agent"],username=praw_cred["username"])

	if post_type == "post":
		submission = reddit.submission(post_id)
		try:
			created_utc = submission.mod.thing.created_utc
		except:
			return ""
		selftext = submission.mod.thing.selftext
		selftext = re.sub(r'\s+',' ',selftext)
		#selftext = selftext.replace("\\","\\\\").replace('"', '\\"').replace("\t"," ").replace("\t","\\t").replace("\r","\\r").replace("\n","\\n")
		selftext = json.dumps(selftext)
		title = submission.mod.thing.title
		title = title.replace('"',"&quot;")
		title = json.dumps(title)
		#title = title.replace('"', '\\"').replace("\t","\\t").replace("\r","\\r").replace("\n","\\n")#replace("'","\\'")
		author = submission.mod.thing.author
		if author is not None:
			author = author.name
		else:
			author = "unknown"
		out_json = '{"id":"'+post_id+'","selftext":'+selftext+',"created_utc":'+str(int(created_utc))+\
				   ',"title":'+title+',"author":"'+author+'"}'
	else:
		submission = reddit.comment(post_id)
		created_utc = submission.mod.thing.created_utc
		selftext = submission.mod.thing.body
		selftext = re.sub(r'\s+',' ',selftext)
		selftext = selftext.replace('"', '\\"').replace("'","\\'")
		out_json = "[{'id':'"+post_id+"','body':'"+selftext+"','created_utc':"+str(int(created_utc))+"}]"
	try:
		out_json = json.loads(out_json)
	except:
		print("Invalid json: " + out_json)
		quit()

	return out_json


def escape_reddit(lines):
	lines = re.sub(r'(?<!\\)(\\[nr])+', r' ___NEWLINE__', lines)  # Replace new lines with real newline
	lines = re.sub(r'(?<!\\)\\(["\\/bfnrt])', r' ', lines)  # Remove literal escapes in data
	lines = lines.replace("<","&lt;")
	lines = lines.replace(">","&gt;")
	lines = re.sub(" +"," ",lines)
	return lines

def fix_bullets(text):
	if "</list>" in text:
		lists = text.split("</list>")
		processed = ""
		for i, l in enumerate(lists[:-1]):
			parts = l.split("</item>")
			prolog, part1 = parts[0].split("<item")
			part1 = "<item" + part1
			epilog = parts[-1]
			parts = [part1] + parts[1:-1]
			output = []
			for part in parts:
				part = re.sub('\*+','</item>\n<item>',part)
				output.append(part)
			output = "</item>".join(output) + "</item>"
			processed += prolog + output + epilog + "</list>"
		text = processed + lists[-1]
	return text

def make_para(text):
	text = text.split("___NEWLINE__")
	text = "</p>\n<p>".join(text)
	text = "<p>" + text + "</p>"
	text = text.replace("<p></p>","").strip()
	# Handle bullets
	text = re.sub(r'<p>(\s*)\*([^\n]*?[^*\n]\s*)</p>',r'\1<item>\2</item>',text)
	text = re.sub(r'((\s*<item>.*?</item>\s*\n?)+)',r'\n<list type="unordered">\1\n</list>',text)
	# Hyperlinks
	text = re.sub(r'\[(.*?)\]\((.*?)\)',r'<ref target="\2">\1</ref>',text)
	# Bold/italic
	#text = re.sub(r'(?<=\W)\*+([^\n<>]+?)\*+(?=[^\w\*])',r'<hi rend="bold">\1</hi>',text)
	#text = re.sub(r'(?<=\W)_+([^\n<>]+?)\*+(?=[^\w_])',r'<hi rend="italic">\1</hi>',text)
	text = re.sub(r'(?<=[ >])\*+([^\n<>\*]+?)\*+(?=[ !?.,:<])',r' <hi rend="bold">\1</hi> ',text)
	text = re.sub(r'(?<=[ >])_+([^\n<>_]+?)_+(?= [ !?.,:<])',r' <hi rend="italic">\1</hi> ',text)
	return text

def flattenjson(b, delim):
	val = {}
	for i in b.keys():
		if isinstance( b[i], dict ):
			get = flattenjson( b[i], delim )
			for j in get.keys():
				val[ i + delim + j ] = get[j]
		else:
			val[i] = b[i]

	return val


# AutoGum settings
min_length = 25  # Min length for some comment in the thread
max_length = 500  # Max length for some comment in the thread
max_tokens_per_slice = 9000  # Max amount of tokens out of any single month dump of reddit
min_doc_length = 400  # Min length for total thread comprising one document
max_doc_length = 1000  # Max length for total thread comprising one document

max_spaces_per_slice = 0.85*max_tokens_per_slice  # Approximate spaces to tokens ratio = 0.85

# Set up praw credentials to get reddit data
praw_cred = io.open(script_dir + "praw.txt",encoding="utf8")
praw_dict = {}

for line in praw_cred.read().split("\n"):
	if "\t" in line and not line.startswith("#"):
		key, val = line.split("\t")
		praw_dict[key] = val

total_docs = 0
used = set([])  # prevent hitting the same OP from two posts

# Cache to avoid asking reddit for same ID twice
submission_cache = {}
if not os.path.exists("submission_cache.tab"):
	io.open("submission_cache.tab", 'w').close()
with io.open("submission_cache.tab",encoding="utf8") as f:
	lines = f.readlines()
	for line in lines:
		if "\t" in line:
			subid, subjson = line.split("\t")
			submission_cache[subid] = subjson

threads = defaultdict(list)
thread_modified = defaultdict(int)
thread_created = defaultdict(int)
thread_utcs = defaultdict(list)


files = glob("data" + os.sep + "reddit" + os.sep + "*.json")[::-1]
docs = []


## OLD FILES
old_files = glob("out" + os.sep + "reddit" + os.sep + ".xml")
old_files += glob("out" + os.sep + "reddit" + os.sep + "done" + os.sep + ".xml")
for file_ in old_files:
	text = io.open(file_,encoding="utf8").read()
	t = re.search(r'sourceURL="http://redd.it/([^"]+)"',text).group(1)
	used.add(t)

for file_idx, file_ in enumerate(files):
	sys.stderr.write("o reading file " + file_ + "\n")

	spaces_so_far = 0

	with io.open(file_,encoding="utf8") as f:
		lines = f.read().strip()
		lines = escape_reddit(lines)
		lines = lines.split("\n")
		# Randomize posts
		random.seed(42)
		random.shuffle(lines)
		joined = "[" + ",".join(lines) + "]"
		data = json.loads(joined, encoding="utf8")

	rows = list(map(lambda x: flattenjson(x, "__"), data))

	if file_idx == 0:  # determine column order once
		columns = [x for row in rows for x in row.keys()]
		columns = list(set(columns))
		columns.remove('body')
		columns = ['body'] + columns
	parent_id_col = columns.index("parent_id")

	for row in rows:
		fields = []
		length = "0"
		for col in columns:
			if col in row:
				if PY3:
					fields.append(str(row[col]).replace("\r","").replace("\n","").replace("\t"," "))
				else:
					fields.append(unicode(row[col]).replace("\r","").replace("\n","").replace("\t"," "))
			else:
				fields.append("")
			if col == "body":
				length = str(row["body"].count(" ") + 1)

		thread_id = fields[parent_id_col].split("_")[-1]
		if thread_id in used:
			continue
		if PY3:
			line = "\t".join(fields + [length])
		else:
			line = "\t".join(fields + [length]).encode("utf8")
		lang = ""
		if int(length) >= min_length and int(length) <= max_length:
			try:
				lang = detect(fields[0])
			except:
				lang = "unknown"
				continue
		else:
			pass
		if not line.startswith("[deleted]"):
			if "â" not in line:  # Drop mixed Latin-1 data
				if int(length) >= min_length and int(length) <= max_length:
					if lang == 'en':  # Check that this is probably in English
						meta = []
						this_post = ""
						for idx, col in enumerate(columns):
							if idx == 0:
								this_post += '<sp'
							else:
								if not col == "body":
									if col == "author":
										this_post += ' who="#' + fields[idx] + '"'
									elif col == "id":
										pass
									elif col == "created_utc":
										this_utc = fields[idx]
										if thread_created[thread_id] == 0 or int(thread_created[thread_id]) > int(this_utc):
											thread_created[thread_id] = this_utc
										if thread_modified[thread_id] == 0 or int(thread_modified[thread_id]) < int(this_utc):
											thread_modified[thread_id] = this_utc

						this_post += ">\n"
						text = make_para(fields[0])
						this_post += text + "\n</sp>\n"
						threads[thread_id].append(this_post)
						thread_utcs[thread_id].append(this_utc)

	# Get OP submissions for comments
	for thread_id in threads:
		if thread_id in used:
			continue
		else:
			used.add(thread_id)
		if thread_id in submission_cache:
			json_result = submission_cache[thread_id]
			if len(json_result.strip())>0:
				json_result = json.loads(json_result)
			else:
				json_result = ""
		else:
			json_result = get_via_praw(thread_id,"post",praw_dict)
			with io.open("submission_cache.tab",'a',encoding="utf8",newline="\n") as f:
				if json_result != "":
					f.write(thread_id + "\t" + json.dumps(json_result)+"\n")
				else:
					f.write(thread_id + "\t\n")
		if json_result == "":
			sys.stderr.write("o Unable to retrieve post: " + thread_id + "\n")
			continue

		heading = "<head>" + json_result["title"] + "</head>\n"
		text = make_para(json_result["selftext"])
		thread_created[thread_id] = json_result["created_utc"]
		out_xml = heading
		if text.strip() != "":
			out_xml += '<sp who="#' + json_result["author"] + '">\n' + text + "\n</sp>\n"
		for comment in sorted(threads[thread_id],key=lambda x: int(thread_utcs[thread_id][threads[thread_id].index(x)])):
			if out_xml.count(" ") > max_doc_length:
				break
			out_xml += comment
		if out_xml.count(" ") > min_doc_length:
			if bad_reddit(out_xml):
				continue
			out_xml = unescape_reddit(out_xml)
			out_xml = fix_bullets(out_xml)
			doc = Document(genre="reddit")
			doc.docnum = total_docs + 598
			doc.title = json_result["title"]
			doc.author= "Reddit community (see URL)"
			doc.url = "http://redd.it/" + json_result["id"]
			doc.date_created = utc_to_date(thread_created[thread_id])
			doc.date_modified = utc_to_date(thread_modified[thread_id])
			doc.text = out_xml

			doc.serialize()

			total_docs += 1
			spaces_so_far += doc.text.count(" ")
			if spaces_so_far > max_spaces_per_slice:
				# Jump to next month
				spaces_so_far = 0
				break
		if total_docs > 700:
			quit()

