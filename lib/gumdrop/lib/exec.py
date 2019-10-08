import sys, tempfile, subprocess, os

PY3 = sys.version_info[0] == 3

def exec_via_temp(input_text, command_params, workdir=""):
	temp = tempfile.NamedTemporaryFile(delete=False)
	exec_out = ""
	try:
		if PY3:
			# try:
			temp.write(input_text.encode("utf8"))
			# except:
			# 	temp.write(input_text)
		else:
			temp.write(input_text)
		temp.close()

		command_params = [x if x != 'tempfilename' else temp.name for x in command_params]
		if workdir == "":
			proc = subprocess.Popen(command_params, stdout=subprocess.PIPE,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
			(stdout, stderr) = proc.communicate()
		else:
			proc = subprocess.Popen(command_params, stdout=subprocess.PIPE,stdin=subprocess.PIPE,stderr=subprocess.PIPE,cwd=workdir)
			(stdout, stderr) = proc.communicate()

		exec_out = stdout
	except Exception as e:
		print(e)
	finally:
		os.remove(temp.name)
		if PY3:
			try:
				exec_out = exec_out.decode("utf8").replace("\r","")
			except:
				pass
		return exec_out
