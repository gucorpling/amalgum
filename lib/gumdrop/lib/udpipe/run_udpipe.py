# This file is part of UDPipe <http://github.com/ufal/udpipe/>.
#
# Copyright 2016 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Adapted for module import by Amir Zeldes

import sys

from ufal.udpipe import Model, Pipeline, ProcessingError # pylint: disable=no-name-in-module

def udpipe(conllu_in, model_path):

	model = Model.load(model_path)
	if not model:
		sys.stderr.write("Cannot load model from file '%s'\n" % sys.argv[3])
		sys.exit(1)

	pipeline = Pipeline(model, "conllu", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")
	error = ProcessingError()

	# Process data
	processed = pipeline.process(conllu_in, error)
	if error.occurred():
		sys.stderr.write("An error occurred when running run_udpipe: ")
		sys.stderr.write(error.message)
		sys.stderr.write("\n")
		sys.exit(1)

	return processed
