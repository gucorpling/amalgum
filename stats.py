import io, os, sys, re
from glob import glob
from collections import defaultdict
from argparse import ArgumentParser

p = ArgumentParser()
p.add_argument("-g","--genre",default="fiction")
opts = p.parse_args()

xml_dir = "nlped" + os.sep + "xml" + os.sep

files = glob(xml_dir + "autogum_" + opts.genre + "*.xml")

file_stats = []
lemmas = defaultdict(str)
ended = False
for file_ in files:
	file_tokens = 0
	for line in io.open(file_,encoding="utf8").readlines():
		if "\t" in line:
			file_tokens += 1
	file_stats.append(file_tokens)
	if sum(file_stats)>500000 and not ended:
		ended = True
		print("Reached " + str(sum(file_stats)) + " after " + str(len(file_stats)) + " at " + os.path.basename(file_))

print("Total files in "+opts.genre+": " + str(len(files)))
print("Mean file size: " + str(sum(file_stats)/len(files)))
print("Total tokens: " + str(sum(file_stats)))

