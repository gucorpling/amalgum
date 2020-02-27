from pip._vendor.distlib.util import get_package_data
from pyparsing import nestedExpr, QuotedString
import io, re, sys
from collections import defaultdict

class Node:

	def __init__(self,nid,relname,children,text):
		self.nid = nid
		self.relname = relname
		self.children = []
		if children is not None:
			for child in children:
				if isinstance(child,list):
					self.children.append(child[0].replace(":",""))
				else:
					self.children.append(re.sub(':.*','',child))
		self.text = text
		self.type = None
		self.satellite = False

def debinarize(node,parent_dict,all_nodes):

	if node.nid in parent_dict:
		grand_parent = parent_dict[node.nid]
	else:
		grand_parent = None
	children = node.children
	to_remove = []
	child_ids = children[:]
	for child_id in child_ids:
		if child_id not in all_nodes:
			continue
		if "1023" in child_ids or node.nid == "1023":
			a=3
		child = all_nodes[child_id]
		try:
			if child.child_rel in ["joint","sequence","contrast","restatement"] and node.type=="multinuc":
				if node.child_rel == child.child_rel:
					parent_dict[child.nid] = grand_parent
					to_remove.append(node.nid)
		except:
			a=3
		parent_dict, all_nodes = debinarize(child,parent_dict,all_nodes)
		for nid in to_remove:
			if nid in parent_dict:
				del parent_dict[nid]
			if nid in all_nodes:
				del all_nodes[nid]
		to_remove = []

	return parent_dict, all_nodes


def to_rs3(text):
	def get_parents(node,parent_dict,all_nodes):
		nid, relname = node[0].replace(":",""), node[1]
		if nid == '1011':
			a=3
		children = node[2:]
		all_nodes[nid] = Node(nid,relname,children,None)
		for child in children:
			if isinstance(child,list):  # group
				child_id = child[0].replace(":","")
				parent_dict[child_id] = Node(nid,relname,children,None)
				parent_dict, all_nodes = get_parents(child,parent_dict,all_nodes)
			else:  # edu
				child_id = re.sub(':.*','',child)
				parent_dict[child_id] = Node(nid,relname,children,None)
				child_text = re.sub(r'^[0-9]+: *','',child).strip()
				all_nodes[child_id] = Node(child_id,None,None,child_text)
		return parent_dict, all_nodes


	edu_text = QuotedString(quoteChar="[", endQuoteChar="]")
	stack = nestedExpr('(',')',content=None,ignoreExpr=edu_text).parseString(text).asList()
	parent_dict = {}
	all_nodes = {}
	parent_dict, all_nodes = get_parents(stack[0],parent_dict,all_nodes)

	for nid in all_nodes:
		node = all_nodes[nid]
		if node.relname is None:
			pass  # EDU, its relname is on its parent
		elif node.relname.startswith("NN"):
			pass # Multinuc, parentage is correct
			node.type = "multinuc"
			for child in node.children:
				all_nodes[child].child_rel = node.relname[3:]
		else:  # Has a sat child
			if node.relname.startswith("SN"):  # Satellite first
				parent_dict[node.children[0]] = all_nodes[node.children[1]]
				all_nodes[node.children[0]].child_rel = node.relname[3:]
				all_nodes[node.children[0]].satellite = True
				all_nodes[node.children[1]].child_rel = "span"
				node.children.pop(0)
			elif node.relname.startswith("NS"):
				parent_dict[node.children[1]] = all_nodes[node.children[0]]
				all_nodes[node.children[1]].child_rel = node.relname[3:]
				all_nodes[node.children[1]].satellite = True
				all_nodes[node.children[0]].child_rel = "span"
				node.children.pop(1)

	# De-binarize multinucs
	new_parents = {}

	root = all_nodes['1000']
	parent_dict, all_nodes = debinarize(root,parent_dict,all_nodes)



	rst_rels = set([])
	multinuc_rels = set([])
	edus = []
	groups = ['<group id="1000" type="span" />']

	for nid in parent_dict:
		node = all_nodes[nid]
		parent = parent_dict[nid]
		try:
			par = parent.nid
		except:
			s=3
		if node.text is not None:
			text = node.text.replace("-RRB-",")").replace("-LRB-","(").replace("-RSB-","]").replace("-LSB-","[")
			edus.append('<segment id="'+nid+'" parent="'+par+'" relname="'+node.child_rel+'">'+text+'</segment>')
		else:
			ntype = "span"
			if node.type == "multinuc":
				ntype = "multinuc"
			groups.append('<group id="'+nid+'" type="'+ntype+'" parent="'+par+'" relname="'+node.child_rel+'"/>')
		if node.child_rel != "span":
			if not node.satellite:
				multinuc_rels.add(node.child_rel)
			else:
				rst_rels.add(node.child_rel)

	output = ""
	output += """<rst>
<header>
	<relations>\n"""
	for rel in rst_rels:
		output += '\t\t\t<rel name="'+rel+'" type="rst"/>\n'
	for rel in multinuc_rels:
		output += '\t\t\t<rel name="'+rel+'" type="multinuc"/>\n'
	output += """		</relations>
</header>
<body>
"""
	for edu in edus:
		output += edu + "\n"

	for group in groups:
		output += group +"\n"

	output += """\t</body>
</rst>
"""
	return output



if __name__ == "__main__":
	file_ = sys.argv[1]
	text = io.open(file_,encoding="utf8").read()
	rs3 = to_rs3(text)
	with io.open("output.rs3",'w',encoding="utf8",newline="\n") as f:
		f.write(rs3)
