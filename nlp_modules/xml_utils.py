import re
import os
import shlex
from conllu import parse

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)) + os.sep

# for a given conllu file with xml annotations, use the annotations to reconstruct the xml and write it to a file
def reconstruct_xml(conllu_filepath, updated_xml_filepath=None):
	conllu_data = parse(open(conllu_filepath, 'r').read())
	xml_string = ''
	trailing_xml = ''
	#xml_string = '<?xml version=\'1.0\' encoding=\'utf8\'?>\n'
	
	# add metadata from the top of the conllu file to the 'text' attribute
	metadata = {}
	conllu_lines = open(conllu_filepath, 'r').readlines()
	for line in conllu_lines:
		if (line[:8] == '# meta::'):
			(key, value) = line[8:].split(' = ', 1)
			metadata[key] = value[:-1]
	metadata_str = '<text '
	for attr in metadata:
		metadata_str += attr + '=' + '\"' + metadata[attr] + '\"' + ' '
	metadata_str = metadata_str[:-1] + '>'
	xml_string += metadata_str + '\n'

	# initialize span tracker for open tags
	span_tracker = []

	for sentence in conllu_data:

		# close tags that have finished their span

		# close last sentence tag
		if(sentence.metadata['sent_id'][-2:] != '-1'):
			xml_string += '</s>\n'

		# add trailing xml from previous sentence
		if (trailing_xml != ''):
			trailing_xml_lines = re.findall('<.*?>', trailing_xml) # split things contained in angle brackets
			for line in trailing_xml_lines:
				xml_string += re.sub(':::', '=', line) + '\n'
		
		# empty trailing xml string
		trailing_xml = ''

		# add close for tags that have finished their span and remove them from the tracker, decrement all spans by 1
		span_index = 0
		for span in span_tracker:
			if(span[1] - 1 == 0):
				xml_string += '</' + span[0] + '>\n'
			span_tracker[span_index][1] -= 1
			span_index += 1
		span_tracker = [span for span in span_tracker if span[1] > 0]

		# parse sentence level comment tags
		sentence_tag = ''
		for entry in sentence.metadata:

			if (entry == 's_type'):
				sentence_tag = '<s type=\"' + sentence.metadata[entry] + '\">' 

			# construct open tag line from the comment
			if ('newpar' in entry):
				for element_info in sentence.metadata[entry].split(" | "):
					entry_info = shlex.split(element_info)
					tag_name = entry_info[0]
					tag_span = entry_info[-2][1:]
					entry_tag = '<' + tag_name
					tag_attribs = entry_info[1:-2]
					for attribute in tag_attribs:
						key, value = attribute.split(":::", 1)
						entry_tag += ' ' + key + '=\"' + value + '\"'
					entry_tag += '>'

					# store the number of sentences that the tag spans
					if(int(tag_span) != 0):
						span_tracker = [[tag_name, int(tag_span)]] + span_tracker

					# add the line to the xml string
					xml_string += re.sub(':::', '=', entry_tag) + '\n'

					# close the tag if it doesn't span any sentences
					if(int(tag_span) == 0):
						xml_string += '</' + tag_name + '>\n'

			if(entry == 'trailing_xml'):
				trailing_xml = sentence.metadata[entry]

		# close any tags whose span has reached 0 and remove them from the tracker
		for span in span_tracker:
			if(span[1] == 0):
				xml_string += '</' + span[0] + '>\n'
		span_tracker = [span for span in span_tracker if span[1] > 0]

		# add the sentence tag
		if (sentence_tag == ''):
			sentence_tag = '<s>'
		xml_string += sentence_tag + '\n'

		for token in sentence:
			if (type(token['id']) is not int):
				# skip multiword token entries
				continue
			# if token has xml annotation
			if (type(token['misc']) is dict and 'XML' in token['misc']):
				# if there are open tags
				if (token['misc']['XML'][:2] != '</'):
					# grab them and add them to the xml string
					split_index = token['misc']['XML'].find('</')
					if (split_index != -1):
						open_tags = token['misc']['XML'][:split_index]
						open_tags = re.sub(':::', '=', open_tags)
					else:
						open_tags = token['misc']['XML']
					open_tags = re.sub(':::', '=', open_tags)
					open_tags_list = re.findall(r'<.*?>', open_tags)
					for tag in open_tags_list:
						xml_string += tag + '\n'
				# then add the token to the xml string
				xml_string += token['form'] + '\t' + token['xpos'] + '\t' + token['lemma'] + '\n'
				# if there are close tags
				if ('</' in token['misc']['XML']):
					# grab them and add them
					if (token['misc']['XML'][:2] == '</'):
						close_tags = token['misc']['XML']
					else:
						close_tags = token['misc']['XML'][split_index:]
					close_tags_list = re.findall(r'<.*?>', close_tags)
					for tag in close_tags_list:
						xml_string += tag + '\n'
			else:
				# if there is no xml annotation, just add the token line
				xml_string += token['form'] + '\t' + token['xpos'] + '\t' + token['lemma'] + '\n'
	
	xml_string += '</s>\n'

	# add xml that directly trails the sentence close
	if (trailing_xml != ''):
		trailing_xml_lines = re.findall('<.*?>', trailing_xml) # split things contained in angle brackets
		for line in trailing_xml_lines:
			xml_string += re.sub(':::', '=', line) + '\n'

	# close any remaining tags
	for span in span_tracker:
		xml_string += '</' + span[0] + '>\n'

	xml_string += '</text>'

	if (updated_xml_filepath):
		with open(updated_xml_filepath, 'w') as f:
			# write updated conllu data to new file
			f.write(xml_string)			

	return xml_string

if __name__ == "__main__":
	# test
	conllu_filepath = SCRIPT_DIR + "../amalgum/academic/dep/AMALGUM_academic_acrylamide.conllu"
	#updated_xml_filepath = SCRIPT_DIR + "test.xml"
	xml_string = reconstruct_xml(conllu_filepath)
	print(xml_string)