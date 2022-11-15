import logging
import re, os, io
try:
    from .base import NLPModule, PipelineDep
except:
    from base import NLPModule, PipelineDep
from xml.dom import minidom
from trankit import Pipeline

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep

bad_period_followers = {"The", "You", "If", "This", "It", "In", "For", "Be", "When", "But", "As", "Here", "Even", "Just", "To", "Whatever",
     "See", "Once", "Never", "Read", "Years", "We", "Man", "Seattle", "They", "Thats", "So", "Gonna", "Marked", "She",
     "Given", "Prices", "Fort", "Shoes", "Bus", "Rakvere", "Santa", "Note", "Based", "Within", "Regional", "Intercity",
     "Soon", "Then", "Avoid", "Enter", "Less", "Keep", "Put", "Speak", "Or", "Many", "Please", "And", "Embrace", "Have",
     "Love", "What", "Touch", "Try", "After", "Use", "Choose", "Rather", "Make", "Mix", "Search", "Also", "Thus",
     "Knowing", "Currency", "Write", "Although", "Glassware", "Heat", "Glass", "Cast", "Silicon", "Stainless",
     "Getting", "Take", "Warli", "Always", "Ask", "Books", "Visiting", "Image"}
bad_period_followers = "|".join(list(bad_period_followers))
bad_period_followers = re.compile(r"([a-z]\.)(" + bad_period_followers + ")")


def remove_xml(xml_data):
    stack = []
    stack_tag = ''
    xml_tags = []
    idx = 0
    plain_text = ''
    for char in xml_data:
        if char == '<':
            stack.append(idx)
            stack_tag += char
        elif char == '>':
            insert_idx = stack.pop()
            stack_tag += char
            xml_tags.append((insert_idx, stack_tag))
            stack_tag = ''
        else:
            if stack:
                stack_tag += char
                continue
            else:
                if char not in '\n ':
                    idx += 1
                plain_text += char
    plain_text = [sent.strip() for sent in plain_text.split('\n') if sent]
    return xml_tags, plain_text


class TrankitTokenizer(NLPModule):
    requires = ()
    provides = (PipelineDep.TOKENIZE,)

    def __init__(self, config):
        self.LIB_DIR = config["LIB_DIR"]
        self.regexxmltag = r"<.*>"
        self.regexxmlunfriendly = r'[&"\'<>]'
        self.p = Pipeline(lang='customized', cache_dir=script_dir+'tokenize-dependencies'+os.sep+'trankit')

    def test_dependencies(self):
        import trankit
        trankit.verify_customized_pipeline(
            category='customized',  # pipeline category
            save_dir=script_dir + 'tokenize-dependencies',  # directory used for saving models in previous steps
            embedding_name='xlm-roberta-base'
            # embedding version that we use for training our customized pipeline, by default, it is `xlm-roberta-base`
        )
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
        # Separate common period fusions
        xml_data = bad_period_followers.sub(r'\1 \2', xml_data)

        # Store xml tags and get plain text
        xml_tags, plain_text = remove_xml(xml_data)

        tokenized_doc = [self.p.tokenize(sent, is_sent=True) for sent in plain_text if len(sent)>0]
        tokenized = []
        for s in tokenized_doc:
            sent = []
            for i in s['tokens']:
                token = i['text']
                if ' ' in token:
                    token = token.split()
                    sent.extend(token)
                else:
                    sent.append(token)
            tokenized.append(sent)

        # tokenized = [[i['text'] for i in s['tokens']] for s in tokenized_doc]

        char_index = 0
        xml_index = 0
        out = []
        flat_tokenized = [tok for sents in tokenized for tok in sents]

        # Restore XML tags
        for token in flat_tokenized:
            while xml_tags[xml_index][0] <= char_index:
                out.append(xml_tags[xml_index][1])
                xml_index += 1
            char_index += len(token)
            out.append(token)

        while xml_index < len(xml_tags):
            out.append(xml_tags[xml_index][1])
            xml_index += 1

        tok_count = len(
            [t for t in out if not t.startswith("<")]
        )
        if not (300 <= tok_count <= 2000):
            top_elt = out[0]
            id = re.findall(r'id="(.*?)"', top_elt)
            logging.warning(
                f"Document '{top_elt if len(id) == 0 else id[0]}' has {tok_count} tokens."
            )

        return '\n'.join(out)

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
    tok = TrankitTokenizer({"LIB_DIR": lib})
    data = io.open(script_dir + ".." + os.sep + "out_tiny" + os.sep + "autogum_academic_doc000.xml").read()
    tokenized = tok.tokenize(data)
    print(tokenized)
