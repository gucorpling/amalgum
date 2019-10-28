import io, json, sys, os, re

def bad_subj(subj):
	forbidden = ["poem","poet","verse","address","constitution","bible"]
	subj = subj.lower()
	if any([f in subj for f in forbidden]):
		return True
	return False


meta_string = io.open("gutenberg-metadata.json",encoding="utf8").read()
meta_dict = json.loads(meta_string)

output = []

total = 0
accepted = 0

for book_id in meta_dict:
	total += 1
	this_book = []
	meta = meta_dict[book_id]
	if "language" in meta:
		if len(meta["language"]) > 0:
			if meta["language"][0] != "en":
				continue
		else:
			continue
	else:
		continue
	if "title" in meta:
		this_book.append(str(book_id))
		if len(meta["title"])>0:
			this_book.append(meta["title"][0])
		else:
			continue
	if "author" in meta:
		if len(meta["author"])>0:
			this_book.append(meta["author"][0])
		else:
			this_book.append("unknown")
	else:
		this_book.append("unknown")
	if "subject" in meta:
		# May not contain forbidden subjects
		if any([bad_subj(subj) for subj in meta["subject"]]):
			continue
		# Must contain fiction in subject
		if all(["fiction" not in subj.lower() for subj in meta["subject"]]):
			continue
	accepted += 1
	row = "\t".join(this_book).replace("\n"," ").replace("\r"," ").replace('"',"''")
	row = re.sub(' +',' ',row)
	output.append(row)

print("Total book IDs: " + str(total))
print("Accepted IDs: " + str(accepted))

with io.open("gutenberg_meta_filtered.tab",'w', encoding="utf8",newline="\n") as f:
	f.write("\n".join(output))
