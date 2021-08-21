# This is a script (the first of two) to add the SpaceAfter=No annotations to existing conllu files, along
# with multiword token (mwt) annotations, and update the text comments to reflect the new spacing annotations.
# This scripts reads in conllu files and their corresponding xml files from the "input_amalgum_conllu" directory and
# the "input_amalgum_xml" directory respectively. The script outputs updated version of the conllu files into 
# the "intermediate_output" directory that have SpaceAfter=No annotations added to them and the text comments 
# for the sentences removed. This script should be followed by the bash script 'mwt_annotation.bash' to finish 
# the updates to the annotations.

import os
import re
from lxml import etree
from lxml import html
from conllu import parse

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)) + os.sep
XML_DIR = SCRIPT_DIR + "input_amalgum_xml" + os.sep
CONLLU_DIR = SCRIPT_DIR + "input_extra_amalgum_conllu" + os.sep
OUT_DIR = SCRIPT_DIR + "intermediate_output_extra" + os.sep

# Takes in the filepath of an xml file and returns a string of the file contents without 
# xml tags and without extra space formatting
def strip_xml(xml_filepath):
	# remove xml tags
	try: 
		# parse the xml file
		tree = etree.parse(xml_filepath)
		plain_text = etree.tostring(tree, encoding='utf8', method='text').decode('utf-8')
	except:
		# removing/replacing non etree parseable characters/expressions and re-parsing
		xml_string = open(xml_filepath, 'r').read()
		xml_string = re.sub(r'\&nbsp;', r' ', xml_string)
		xml_string = re.sub(r'<3', r'&lt;3', xml_string)
		xml_string = re.sub(r'&mdash;', r'—', xml_string)
		xml_string = re.sub(r'&ndash;', r'–', xml_string)
		xml_string = re.sub(r'<ref target="#datatag{.*}"></ref>', r'', xml_string)
		tree = etree.XML(xml_string)
		plain_text = etree.tostring(tree, encoding='utf8', method='text').decode('utf-8')
	# replace newlines with spaces
	plain_text = re.sub(r'\n', r' ', plain_text)
	plain_text = re.sub(r'□', r' ', plain_text)
	# replace various spaces with simple space
	plain_text = re.sub(r'\s', r' ', plain_text)
	# replace multiple spaces with single space
	plain_text = re.sub(r' +', r' ', plain_text)
	# remove leading and trailing space on ends of the file
	plain_text = plain_text[1:-1]
	return plain_text

def realign_interally_spaced_token(form, text_string):
	# if the text of the conllu token form matches up with the beginning of xml string text_string when extra spaces and dashes are ignored, 
	# returns the index in the text_string after the content of the conllu token ends,
	# otherwise returns -1
	text_string_index = 0
	form_index = 0
	while (form_index < len(form)): # while we are still looking to match the characters within form
		if (form[form_index] == text_string[text_string_index]):
			# increment the form_index and the text_string index when they match
			form_index += 1
			text_string_index += 1
		elif (text_string[text_string_index] == ' ' or text_string[text_string_index] == '-'): 
			# increment just the text_string index when we encounter a space or dash we want to skip over for matching
			text_string_index += 1
		else:
			# if a characrer that is not a space or a dash is encountered in text_string,
			# we do not have a match to the conllu token text, and we return -1 to indicate a non-match
			text_string_index = -1
			break

	return text_string_index

def main():
	conllu_files = os.listdir(CONLLU_DIR)
	for conllu_file in conllu_files:
		
		# get filepath for current conllu file and corresponding xml file
		conllu_filepath = CONLLU_DIR + conllu_file
		xml_file = re.sub(r'.conllu', '.xml', conllu_file)
		xml_filepath = XML_DIR + xml_file

		# get data for current conllu file and corresponding xml file
		conllu_data = parse(open(conllu_filepath, 'r').read())
		plain_text = strip_xml(xml_filepath)
		
		misalign = False

		# Adding SpaceAfter=No annotations
		for sentence in conllu_data:
			if misalign == True:
				# if the content of the conllu and xml files are already misaligned, don't go through the rest of the file  
				break
			# removing text comment for each sentence
			sentence.metadata.pop('text') 
			for word in sentence:

				if (type(word['id']) is not int):
					# skip multiword token annotations
					continue

				if ((word['form'] + ' ') == plain_text[:len(word['form'])+1]):
					# Space after token, chop start of xml string and move on
					plain_text = plain_text[len(word['form'])+1:]
				elif (word['form'] == plain_text[:len(word['form'])]):
					# No space after token, add no space label, chop start of xml string, and move on
					if(type(word['misc']) is dict):
						word['misc']['SpaceAfter'] = 'No'
					else:
						word['misc'] = {'SpaceAfter': 'No'}
					plain_text = plain_text[len(word['form']):]
				else:
					# re-align if the xml file has extra spaces or dashes within the current single conllu token itself
					cut_at = realign_interally_spaced_token(word['form'], plain_text)
					if (cut_at != -1):
						plain_text = plain_text[cut_at:]
						if (plain_text[0] == ' '):
							# Space after token, chop start of xml string and move on
							plain_text = plain_text[1:]
						else:
							# No space after token, add no space label; start of xml string is already next token, okay to move on
							if(type(word['misc']) is dict):
								word['misc']['SpaceAfter'] = 'No'
							else:
								word['misc'] = {'SpaceAfter': 'No'}
					else:
						print('Could not fix the misalignment for the following file: ')
						print(conllu_filepath)
						misalign = True
						break

		# remove SpaceAfter=No annotation from last word of last sentence - is this wrong?
		misc_of_last_word_in_last_sentence = conllu_data[len(conllu_data)-1][len(conllu_data[len(conllu_data)-1])-1]['misc']
		if (type(misc_of_last_word_in_last_sentence) is dict and 'SpaceAfter' in misc_of_last_word_in_last_sentence): 
			misc_of_last_word_in_last_sentence.pop('SpaceAfter')
			if (not bool(misc_of_last_word_in_last_sentence)):
				# misc_of_last_word_in_last_sentence is empty
				new_misc = '_'
			else:
				new_misc = misc_of_last_word_in_last_sentence
			conllu_data[len(conllu_data)-1][len(conllu_data[len(conllu_data)-1])-1]['misc'] = new_misc

		with open(OUT_DIR + conllu_file, 'w') as f:
			# outputting updated conllu data
			f.writelines([sentence.serialize() for sentence in conllu_data])

if __name__ == "__main__":
	main()
	