from nlp_modules.s_typer import test_main as test_stype
from nlp_modules.xrenner_coreferencer import test_main as test_coref
from nlp_modules.ace_entities import test_main as test_ace
from nlp_modules.rst_parser import test_main as test_rst

from argparse import ArgumentParser

p = ArgumentParser()
p.add_argument("-m", "--module", choices=["stype", "coref", "ace", "rst"], default="ace", help="module to test")

opts = p.parse_args()
if opts.module == "coref":
	test_coref()
elif opts.module == "stype":
	test_stype()
elif opts.module == "ace":
	test_ace()
elif opts.module == "rst":
	test_rst()
