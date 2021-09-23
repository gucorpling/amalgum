# This file has several functions for simplifiying/cleaning up the amalgum xml files

import os
from lxml import etree

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)) + os.sep
XML_DIRS = [SCRIPT_DIR + "../amalgum/amalgum/academic/xml" + os.sep,
SCRIPT_DIR + "../amalgum/amalgum/bio/xml" + os.sep,
SCRIPT_DIR + "../amalgum/amalgum/fiction/xml" + os.sep,
SCRIPT_DIR + "../amalgum/amalgum/interview/xml" + os.sep,
SCRIPT_DIR + "../amalgum/amalgum/news/xml" + os.sep,
SCRIPT_DIR + "../amalgum/amalgum/voyage/xml" + os.sep,
SCRIPT_DIR + "../amalgum/amalgum/whow/xml" + os.sep,
SCRIPT_DIR + "../amalgum/amalgum_extra/academic/xml" + os.sep,
SCRIPT_DIR + "../amalgum/amalgum_extra/fiction/xml" + os.sep,
SCRIPT_DIR + "../amalgum/amalgum_extra/interview/xml" + os.sep,
SCRIPT_DIR + "../amalgum/amalgum_extra/news/xml" + os.sep,
SCRIPT_DIR + "../amalgum/amalgum_extra/voyage/xml" + os.sep,
SCRIPT_DIR + "../amalgum/amalgum_extra/whow/xml" + os.sep,
SCRIPT_DIR + "../xml_into_conllu/reddit_xml" + os.sep]
REDUCED_DIR = SCRIPT_DIR + "../xml_into_conllu/reduced_amalgum_xml" + os.sep

# writes new xml file with slightly simplified tags
def reducing_xml_files():

	for XML_DIR in XML_DIRS:
		xml_files = os.listdir(XML_DIR)
		for xml_file in xml_files:
			xml_filepath = XML_DIR + xml_file
			new_filepath = SCRIPT_DIR + "/reduced_amalgum_xml/" + xml_file
			tree = etree.parse(xml_filepath)
			etree.strip_tags(tree,'dd', 'dl', 'dt', 'tbody')
			root = tree.getroot()
			reduce_xml_tags(root)
			remove_redundant_attributes(root, xml_filepath)
			# remove leaf nodes that cover no text
			nodes_removed = remove_empty_nodes(root)
			while (nodes_removed > 0):
				nodes_removed = remove_empty_nodes(root)
			# write to new file
			with open(new_filepath, 'wb') as f:
				f.write(etree.tostring(root))	

# removes leaf nodes that cover no text
def remove_empty_nodes(node):
	
	if(len(list(node)) == 0):
		if (node.text == '\n' and node.tag not in ['figure', 'ref', 'sp']):
			preserve_tail_before_delete(node)
			node.getparent().remove(node)
			return 1
		else:
			return 0
	else:
		nodes_removed = 0
		for child in list(node):
			nodes_removed += remove_empty_nodes(child)

		return nodes_removed

def preserve_tail_before_delete(node):
    if node.tail: # preserve the tail
        previous = node.getprevious()
        if previous is not None: # if there is a previous sibling it will get the tail
            if previous.tail is None:
                previous.tail = node.tail
            else:
                previous.tail = previous.tail + node.tail
        else: # The parent get the tail as text
            parent = node.getparent()
            if parent.text is None:
                parent.text = node.tail
            else:
                parent.text = parent.text + node.tail

def remove_redundant_attributes(node, xml_filepath):
	if(len(list(node)) == 0):
		return
	else:
		for child in list(node):
			for attrib in child.attrib:
				if(attrib in node.attrib and child.attrib[attrib] == node.attrib[attrib]):
					child.attrib.pop(attrib)
			remove_redundant_attributes(child, xml_filepath)

def reduce_xml_tags(node):
	if(node.tag == 'b'):
		node.tag = 'hi'
		node.set('rend', 'bold')
	elif(node.tag == 'blockquote'):
		node.tag = 'quote'
	elif(node.tag == 'h3'):
		node.tag = 'head'
		node.set('n', '3')
	elif(node.tag == 'h4'):
		node.tag = 'head'
		node.set('n', 'h4')
	elif(node.tag == 'head' and 'id' in node.attrib):
		node.attrib.pop('id')
	elif(node.tag == 'hi' and 'lang' in node.attrib):
		# rend should maybe also be split out if present?
		node.tag = 'lang'
		node.set('id', node.attrib['lang'])
		node.attrib.pop('lang')
	elif(node.tag == 'item'):
		if('style' in node.attrib):
			node.attrib.pop('style')
		if('id' in node.attrib):
			node.attrib.pop('id')
	elif(node.tag == 'list'):
		if(len(node.attrib) == 0):
			node.set('type', 'unordered')
	elif(node.tag == 'p'):
		if('style' in node.attrib):
			node.attrib.pop('style')
	elif(node.tag == 'ref'):
		if('resource' in node.attrib):
			node.attrib.pop('resource')
		if('style' in node.attrib):
			node.attrib.pop('style')
	elif(node.tag == 'table'):
		if('class' in node.attrib):
			node.attrib.pop('class')
	elif(node.tag == 'td'):
		node.tag = 'cell'
		if('align' in node.attrib):
			node.attrib.pop('align')
		if('class' in node.attrib):
			node.attrib.pop('class')
		if('valign' in node.attrib):
			node.attrib.pop('valign')
	elif(node.tag == 'tr'):
		node.tag = 'row'
	if (len(list(node)) == 0):
		return
	else:
		for child in list(node):
			reduce_xml_tags(child)
	return

# moves sentence internal ref tags that span no tokens to have them cover the previous token
def edit_files():
	xml_files = os.listdir(REDUCED_DIR)
	for xml_file in xml_files:

		xml_filepath = REDUCED_DIR + xml_file
		
		xml_data = open(xml_filepath, 'r').readlines()
		in_sentence = False
		last_line = ''
		line_index = 0

		for line in xml_data:

			if (line[:3] == '<s ' or line[:3] == '<s>'):
				in_sentence = True

			if (line[:4] == '</s>' or line[:4] == '</s '):
				in_sentence = False

			if(in_sentence and last_line[:5] == '<ref ' and line == '</ref>\n'):
				# we have a ref tag around no token, move the opening tag back one line
				xml_data[line_index-1] = xml_data[line_index-2]
				xml_data[line_index-2] = last_line

			last_line = line
			line_index += 1

		xml_string = ''
		for line in xml_data:
			if (line != '\n'):
				xml_string += line

		with open(xml_filepath, 'w') as f:
			# write updated xml to file
			f.write(xml_string)
	return

if __name__ == "__main__":
	reducing_xml_files()
	edit_files()