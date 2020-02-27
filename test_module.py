from nlp_modules.s_typer import test_main as test_stype
from nlp_modules.xrenner_coreferencer import test_main as test_coref
from argparse import ArgumentParser

p = ArgumentParser()
p.add_argument("-m","--module",choices=["stype","coref"],default="stype",help="module to test")

opts = p.parse_args()
if opts.module == "coref":
	test_coref()
else:
	test_stype()