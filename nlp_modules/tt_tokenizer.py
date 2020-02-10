import re
from lib.whitespace_tokenize import tokenize as tt_tokenize
from .base import NLPModule, PipelineDep
from xml.dom import minidom


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

    # Ranges
    TTSGML = re.sub(
        r"(¥|\$|€)\n?([0-9.,]+)-([0-9.,]+\n)", r"\1\n\2\n-\n\3", TTSGML
    )  # Currency
    TTSGML = re.sub(
        r"([12]?[0-9]:[0-5][0-9])(-)([12]?[0-9]:[0-5][0-9])\n", r"\1\n\2\n\3\n", TTSGML
    )  # Time
    TTSGML = re.sub(
        r"((?:sun|mon|tues?|wed|thu(?:rs)|fri|sat(?:ur)?)(?:day)?)-((?:sun|mon|tues?|wed|thu(?:rs)|fri|sat(?:ur)?)(?:day)?)\n",
        r"\1\n-\n\2\n",
        TTSGML,
        flags=re.IGNORECASE,
    )  # Days
    TTSGML = re.sub(
        r"(Su|M|Tu|W|Th|Fr?|Sa)-(Su|M|Tu|W|Th|Fr?|Sa)\n", r"\1\n-\n\2\n", TTSGML
    )  # Short days

    # Measurement symbols
    TTSGML = re.sub(r"\n(k?m)²\n", r"\n\1\n²\n", TTSGML)  # Squared
    TTSGML = re.sub(r"([0-9])°\n", r"\1\n°", TTSGML)  # Degree symbol

    # Latin abbreviations
    TTSGML = TTSGML.replace(" i. e. ", " i.e. ").replace(" e. g. ", " e.g. ")

    # Trailing periods in section headings like "1. Introduction", usually following an XML tag
    TTSGML = re.sub(r"(>\n[0-9]+)\n(\.\n)", r"\1\2", TTSGML)

    # en dash spelled --
    TTSGML = re.sub(r"([^\n])--", r"\1\n--", TTSGML)
    TTSGML = re.sub(r"--([^\n]+)\n", r"--\n\1\n", TTSGML)

    # Find missing contraction spellings
    TTSGML = re.sub(r"\n([Ii]t)(s\nnot\n)", r"\n\1\n\2", TTSGML)
    TTSGML = TTSGML.replace("\nIve\n", "\nI\nve\n")
    TTSGML = re.sub(
        r"\n(did|do|was|were|would|should|had|must)nt\n", r"\n\1\nnt\n", TTSGML
    )

    # Fix grammar-dependant tokenizations
    TTSGML = re.sub(r"(\n[Ii]t)(s\n(?:" + VVN + ART + r")\n)", r"\1\n\2", TTSGML)

    # Fix apostrophes
    TTSGML = re.sub(r">\n'\ns\n", r">\n's\n", TTSGML)

    # Fix parentheses in tokens like parent(s)
    TTSGML = re.sub(r"\n(\w+)\(s\n\)\n", r"\n\1(s)\n", TTSGML)

    fixed = TTSGML
    return fixed


def escape_treetagger(tt_sgml):
    new_lines = []
    for line in tt_sgml.split("\n"):
        # probably an element
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
    return "\n".join(new_lines)


class TreeTaggerTokenizer(NLPModule):
    requires = ()
    provides = (PipelineDep.TOKENIZE,)

    def __init__(self, config):
        self.LIB_DIR = config["LIB_DIR"]

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

        return tokenized

    def run(self, input_dir, output_dir):
        # Identify a function that takes data and returns output at the document level
        processing_function = self.tokenize

        # use process_files, inherited from NLPModule, to apply this function to all docs
        self.process_files(input_dir, output_dir, processing_function)
