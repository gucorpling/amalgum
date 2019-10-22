import os, sys, tempfile, subprocess, io, re
import datetime
import unidecode

now = datetime.datetime.now()
today = now.strftime("%Y-%m-%d")

PY3 = sys.version_info[0] == 3

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
root_dir = script_dir + ".." + os.sep


def get_col(data, colnum):
	if not isinstance(data,list):
		data = data.split("\n")

	splits = [row.split("\t") for row in data if "\t" in row]
	return [r[colnum] for r in splits]


def exec_via_temp(input_text, command_params, workdir="", outfile=False):
	temp = tempfile.NamedTemporaryFile(delete=False)
	if outfile:
		temp2 = tempfile.NamedTemporaryFile(delete=False)
	output = ""
	try:
		temp.write(input_text.encode("utf8"))
		temp.close()

		if outfile:
			command_params = [x if 'tempfilename2' not in x else x.replace("tempfilename2",temp2.name) for x in command_params]
		command_params = [x if 'tempfilename' not in x else x.replace("tempfilename",temp.name) for x in command_params]
		if workdir == "":
			proc = subprocess.Popen(command_params, stdout=subprocess.PIPE,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
			(stdout, stderr) = proc.communicate()
		else:
			proc = subprocess.Popen(command_params, stdout=subprocess.PIPE,stdin=subprocess.PIPE,stderr=subprocess.PIPE,cwd=workdir)
			(stdout, stderr) = proc.communicate()
		if outfile:
			if PY3:
				output = io.open(temp2.name,encoding="utf8").read()
			else:
				output = open(temp2.name).read()
			temp2.close()
			os.remove(temp2.name)
		else:
			if PY3:
				output = stdout.decode("utf8")
			else:
				output = stdout
		#print(stderr)
		#proc.terminate()
	except Exception as e:
		print(e)
	finally:
		os.remove(temp.name)
		return output


class Document:

	def __init__(self,genre="voyage"):
		self.title = ""
		self.short_title = ""
		self.text = ""
		self.author = ""
		self.url = ""
		self.lines = []
		self.genre = genre
		self.docnum = 0

	def make_short_title(self):
		letters = set(list("abcdefghijklmnopqrstuvwxyz"))
		stoplist = ["a","able","about","across","after","all","almost","also","am","among","an","and","any","are","as","at","be","because","been","but","by","can","cannot","could","dear","did","do","does","either","else","ever","every","for","from","get","got","had","has","have","he","her","hers","him","his","how","however","i","if","in","into","is","it","its","just","least","let","like","likely","may","me","might","most","must","my","neither","no","nor","not","of","off","often","on","only","or","other","our","own","rather","said","say","says","she","should","since","so","some","than","that","the","their","them","then","there","these","they","this","tis","to","too","twas","us","wants","was","we","were","what","when","where","which","while","who","whom","why","will","with","would","yet","you","your"]
		stop_re = r"\b("+"|".join(stoplist)+r")\b"
		title = unidecode.unidecode(self.title.lower())
		title = re.sub(stop_re,'',title)
		title = re.sub(' +'," ",title)
		words = title.split()
		title = ""
		for word in words:
			word = word
			if len(title)>20:
				break
			word = "".join([c for c in word if c in letters])
			title+=word
		return title

	def serialize(self,out_dir=None):

		self.short_title = self.make_short_title()

		if out_dir is None:
			out_dir = root_dir + "out" + os.sep + self.genre + os.sep

		if not os.path.exists(out_dir):
			os.mkdir(out_dir)

		docname = 'autogum_'+ self.genre +'_doc' + str(self.docnum)

		# TODO: more metadata
		header = '<text id="' + docname + '" title="' + self.title + '"'
		header += ' shortTile="' + self.short_title + '"'
		if self.author!= "":
			header += ' author="'+self.author+'"'

		header += ' type="'+self.genre+'"'
		header += ' dateCollected="'+today+'"'
		if self.url != "":
			header+= ' sourceURL="'+ self.url +'"'
		header += '>\n'
		output = header + self.text.strip() + "\n</text>\n"

		with io.open(out_dir + docname + ".xml",'w',encoding="utf8",newline="\n") as f:
			f.write(output)
