import os, sys, tempfile, subprocess, io

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
		self.text = ""
		self.author = ""
		self.url = ""
		self.lines = []
		self.genre = genre
		self.docnum = 0

	def serialize(self,out_dir=None):
		if out_dir is None:
			out_dir = root_dir + "out" + os.sep + self.genre + os.sep

		if not os.path.exists(out_dir):
			os.mkdir(out_dir)

		docname = 'autogum_'+ self.genre +'_doc' + str(self.docnum)

		# TODO: more metadata
		header = '<text id="' + docname + '" title="' + self.title + '"'
		if self.author!= "":
			header += ' author="'+self.author+'"'
		header += '>\n'
		output = header + self.text.strip() + "\n</text>\n"
		with io.open(out_dir + docname + ".xml",'w',encoding="utf8",newline="\n") as f:
			f.write(output)
