import io, os, sys, re
from glob import glob
from collections import defaultdict
from argparse import ArgumentParser
from random import shuffle, seed

seed(42)

p = ArgumentParser()
p.add_argument("-g","--genre",default="fiction")
opts = p.parse_args()

xml_dir = "nlped" + os.sep + "xml" + os.sep

genre = opts.genre
files = []
for g in ["academic","bio","fiction","interview","news","reddit","whow","voyage"]:
	files += glob(xml_dir + "autogum_" + genre + "*.xml")


nouns = defaultdict(int)
propns = defaultdict(int)

known_nouns = set([])
with io.open('C:\\uni\\cl\\xrenner\\xrenner\\models\\eng\\entity_heads.tab',encoding="utf8") as f:
	for line in f.readlines():
		if "\t" in line:
			known_nouns.add(line.split("\t")[0])

file_stats = []
lemmas = defaultdict(str)
ended = False
shuffle(files)
for file_ in files:
	file_tokens = 0
	for line in io.open(file_,encoding="utf8").readlines():
		if "\t" in line:
			file_tokens += 1
			fields = line.split("\t")
			if fields[1] in ["NN","NNS"]:
				nouns[fields[0]] += 1
				lemmas[fields[0]] = fields[2].strip()
			elif fields[1] in ["NNP","NNPS"]:
				propns[fields[0]] += 1
	file_stats.append(file_tokens)
	if sum(file_stats)>500000 and not ended:
		ended = True
		print("Reached " + str(sum(file_stats)) + " after " + str(len(file_stats)) + " at " + os.path.basename(file_))

print("Total files in "+opts.genre+": " + str(len(files)))
print("Mean file size: " + str(sum(file_stats)/len(files)))
print("Total tokens: " + str(sum(file_stats)))

print("Seen " + str(len(known_nouns)) + " nouns with " + str(sum(nouns.values())))
known_count = 0
with io.open("nouns.tab",'w',encoding='utf8',newline="\n") as f:
	for i, tup in enumerate(sorted(list(nouns.items()),key=lambda x:x[1],reverse=True)):
		known = "known" if (str(tup[0]) in known_nouns or lemmas[tup[0]] in known_nouns) else "unknown"
		if known == "known":
			known_count += 1
		if known != "known":
			if tup[0].endswith("s"):
				if tup[0][:-1] in known_nouns:
					continue
			if tup[0].lower() in known_nouns:
				continue
			f.write(tup[0] + "\t" + str(tup[1]) + "\t" + known + "\n")
		#if i > 1000:
		#	break
		#if i - known_count > 200:
			#break

print("Known nouns in top "+str(i)+": " + str(known_count/i))

with io.open("propns.tab",'w',encoding='utf8',newline="\n") as f:
	for i, tup in enumerate(sorted(list(propns.items()),key=lambda x:x[1],reverse=True)):
		f.write(tup[0] + "\t" + str(tup[1]) + "\n")
		if i > 1000:
			break
