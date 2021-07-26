import logging
import re, os, io
from lib.whitespace_tokenize import tokenize as tt_tokenize
from .base import NLPModule, PipelineDep
from xml.dom import minidom

bad_period_followers = {"The", "You", "If", "This", "It", "In", "For", "Be", "When", "But", "As", "Here", "Even", "Just", "To", "Whatever",
     "See", "Once", "Never", "Read", "Years", "We", "Man", "Seattle", "They", "Thats", "So", "Gonna", "Marked", "She",
     "Given", "Prices", "Fort", "Shoes", "Bus", "Rakvere", "Santa", "Note", "Based", "Within", "Regional", "Intercity",
     "Soon", "Then", "Avoid", "Enter", "Less", "Keep", "Put", "Speak", "Or", "Many", "Please", "And", "Embrace", "Have",
     "Love", "What", "Touch", "Try", "After", "Use", "Choose", "Rather", "Make", "Mix", "Search", "Also", "Thus",
     "Knowing", "Currency", "Write", "Although", "Glassware", "Heat", "Glass", "Cast", "Silicon", "Stainless",
     "Getting", "Take", "Warli", "Always", "Ask", "Books", "Visiting", "Image"}
bad_period_followers = "|".join(list(bad_period_followers))
bad_period_followers = re.compile(r"([a-z]\.)(" + bad_period_followers + ")")


def postprocess_text(TTSGML):
    # Likely verbal VVN, probably not an amod
    VVN = "been|called|made|found|seen|done|based|taken|born|considered|got|located|said|told|started|shown|become|put|gone|created|had|asked"
    ART = "the|this|that|those|a|an"

    # Phone numbers
    phone_exp = re.findall(
        r"((?:☏|(?:fax|phone)\n:)\n(?:\+?[0-9]+\n|-\n)+)", TTSGML, flags=re.UNICODE
    )
    for phone in phone_exp:
        fused = (
            phone.replace("\n", "")
            .replace("☏", "☏\n")
            .replace("fax:", "fax\n:\n")
            .replace("phone:", "phone\n:\n")
            + "\n"
        )
        TTSGML = TTSGML.replace(phone, fused)

    # Currency
    TTSGML = re.sub(r"([¥€\$])([0-9,.]+)\n", r"\1\n\2\n", TTSGML)

    # Common image credit error:
    TTSGML = TTSGML.replace(".Image\n",".\nImage\n")

    # Ranges
    TTSGML = re.sub(r"(¥|\$|€)\n?([0-9.,]+)-([0-9.,]+\n)", r"\1\n\2\n-\n\3", TTSGML)
    TTSGML = re.sub(r"(¥|\$|€)\n?([0-9.,]+)-(¥|\$|€)([0-9.,]+\n)", r"\1\n\2\n-\n\3\n\4", TTSGML)

    # Chemistry
    TTSGML = re.sub(r"(≡[A-Za-z]+)\n([-–])\n([A-Za-z]+)\n([-–])\n([A-Za-z]+≡)",r'\1\2\3\4\5',TTSGML)

    # Time
    TTSGML = re.sub(
        r"((?:sun|mon|tues?|wed|thu(?:rs)|fri|sat(?:ur)?)(?:day)?)-((?:sun|mon|tues?|wed|thu(?:rs)|fri|sat(?:ur)?)(?:day)?)\n",
        r"\1\n-\n\2\n",
        TTSGML,
        flags=re.IGNORECASE,
    )
    TTSGML = re.sub(r"([12]?[0-9]:[0-5][0-9])(-)([12]?[0-9]:[0-5][0-9])\n", r"\1\n\2\n\3\n", TTSGML)

    # Days
    TTSGML = re.sub(r"(Su|M|Tu|W|Th|Fr?|Sa)-(Su|M|Tu|W|Th|Fr?|Sa)\n", r"\1\n-\n\2\n", TTSGML)  # Short days

    # Measurement symbols
    TTSGML = re.sub(r"\n(k?m)²\n", r"\n\1\n²\n", TTSGML)  # Squared
    TTSGML = re.sub(r"([0-9])°\n", r"\1\n°", TTSGML)  # Degree symbol

    # Latin abbreviations
    TTSGML = TTSGML.replace("\ni.\ne.\n", "\ni.e.\n").replace("\ne.\ng.\n", "\ne.g.\n")

    # Trailing periods in section headings like "1. Introduction", usually following an XML tag
    TTSGML = re.sub(r"^(\n[0-9]+)\n(\.\n)", r"\1\2", TTSGML)

    # en dash spelled --
    TTSGML = re.sub(r"([^\n])--", r"\1\n--", TTSGML)
    TTSGML = re.sub(r"--([^\n]+)\n", r"--\n\1\n", TTSGML)

    # Find missing contraction spellings
    TTSGML = re.sub(r"\n([Ii]t)(s\nnot\n)", r"\n\1\n\2", TTSGML)
    TTSGML = TTSGML.replace("\nIve\n", "\nI\nve\n")
    TTSGML = re.sub(r"\n(did|do|was|were|would|should|had|must)nt\n", r"\n\1\nnt\n", TTSGML)

    # Fix grammar-dependant tokenizations
    TTSGML = re.sub(r"(\n[Ii]t)(s\n(?:" + VVN + ART + r")\n)", r"\1\n\2", TTSGML)

    # Fix apostrophes
    TTSGML = re.sub(r"^\n'\ns\n", r"\n's\n", TTSGML)

    # Fix parentheses in tokens like parent(s)
    TTSGML = re.sub(r"\n(\w+)\(s\n\)\n", r"\n\1(s)\n", TTSGML)

    # LS-like number broken from period right after tag
    TTSGML = re.sub(r'^(\n?([0-9]+\.)+[0-9]+?)\n\.\n',r'\1.\n', TTSGML)

    fixed = TTSGML
    return fixed


def escape_treetagger(tt_sgml):
    def replace_xml_chars(text):
        text = (
            text.replace("&", "&amp;")
            .replace("&amp;amp;", "&amp;")
            .replace(">", "&gt;")
            .replace("&amp;gt;", "&gt;")
            .replace("<", "&lt;")
            .replace("&amp;lt;", "&lt;")
            .replace('"', "&quot;")
            .replace("&amp;quot;", "&quot;")
            .replace("'", "&apos;")
            .replace("&amp;apos;", "&apos;")
        )
        return text

    regexxmltag = r"<.*>"
    regextitlestr = (
        r"(?<=title=).*(?=shortTile)"
    )  # if shortTile doesnt follow title, the cleaning process will break
    regexreftarget = r'(?<=target=).*(?=">)'
    new_lines = []
    for line in tt_sgml.split("\n"):
        if not re.match(regexxmltag, line):  # not an element
            new_lines.append(replace_xml_chars(str(line)))
        else:
            if line.startswith("<ref"):  # refs have links with ampersands
                match = re.search(regexreftarget, line)
                if match is not None:
                    matchtext = match.group(0).strip()[1:]
                    matchtext = replace_xml_chars(matchtext)
                    line = line.replace(match.group(0).strip(), '"' + matchtext)

            elif line.startswith(
                "<text"
            ):  # the first tag. The problem here is usually with " present in the title text
                match = re.search(regextitlestr, line)
                if match is not None:
                    matchtext = match.group(0).strip()[1:-1]
                    matchtext = replace_xml_chars(matchtext)
                    line = line.replace(match.group(0).strip(), '"' + matchtext + '"')

            new_lines.append(line)

        """
        if line.startswith("<") and line.endswith(">"):
            new_lines.append(line)
        else:
            new_lines.append(
                line.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&apos;")
            )
        """
    return "\n".join(new_lines)


class TreeTaggerTokenizer(NLPModule):
    requires = ()
    provides = (PipelineDep.TOKENIZE,)

    def __init__(self, config):
        self.LIB_DIR = config["LIB_DIR"]
        self.regexxmltag = r"<.*>"
        self.regexxmlunfriendly = r'[&"\'<>]'

    def test_dependencies(self):
        pass

    def tokenize(self, xml_data):
        """Tokenize input XML or plain text into TT SGML format.

        :param xml_data: input string of a single document
        :return: TTSGML with exactly one token or opening tag or closing tag per line

        example input:

            <text id="autogum_voyage_doc3" title="Aakirkeby">
            <head>Aakirkeby</head>

            <p><hi rend="bold">Aakirkeby</hi> is a city on <ref target="Bornholm">Bornholm</ref>,

        example output:

            <text id="autogum_voyage_doc3" title="Aakirkeby">
            <head>
            Aakirkeby
            </head>
            <p>
            <hi rend="bold">
            Aakirkeby
            </hi>
            is
            a
            city
            ...
        """
        # Separate en/em dashes
        xml_data = xml_data.replace("–", " – ").replace("—", " — ")

        # Separate common period fusions
        xml_data = bad_period_followers.sub(r'\1 \2',xml_data)

        abbreviations = self.LIB_DIR + "english-abbreviations"
        tokenized = tt_tokenize(xml_data, abbr=abbreviations)

        # TreeTagger doesn't escape XML chars. We need to fix it before we parse.
        tokenized = escape_treetagger(tokenized)

        xml = minidom.parseString(tokenized)

        def postprocess_text_nodes(node):
            if hasattr(node, "childNodes") and list(node.childNodes):
                for child in node.childNodes:
                    postprocess_text_nodes(child)
            elif node.nodeType == minidom.Node.TEXT_NODE:
                node.data = postprocess_text(node.data)

        postprocess_text_nodes(xml)
        tokenized = xml.toxml()

        tok_count = len(
            [t for t in tokenized.strip().split("\n") if not t.startswith("<")]
        )
        if not (300 <= tok_count <= 2000):
            top_elt = tokenized.strip().split("\n")[0]
            id = re.findall(r'id="(.*?)"', tokenized)
            logging.warning(
                f"Document '{top_elt if len(id) == 0 else id[0]}' has {tok_count} tokens."
            )

        return tokenized

    def run(self, input_dir, output_dir):
        # Identify a function that takes data and returns output at the document level
        processing_function = self.tokenize

        # use process_files, inherited from NLPModule, to apply this function to all docs
        self.process_files(
            input_dir, output_dir, processing_function, multithreaded=True
        )

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
    lib = script_dir + ".." + os.sep + "lib" + os.sep
    tok = TreeTaggerTokenizer({"LIB_DIR":lib})
    data = io.open(script_dir + ".." + os.sep + "out_tiny_24" + os.sep + "amalgum_academic_leucine.xml").read()
    tokenized = tok.tokenize(data)
    print(tokenized)