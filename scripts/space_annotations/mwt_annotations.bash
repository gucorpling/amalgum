#!/bin/bash
# Runs DepEdit on all conllu files in the 'intermediate_output' directory and saves the output in the 'output_updated_amalgum_conllu' directory
# The file 'config.ini' includes rules for adding multiword tokens, and the -t flag adds in text comments for each sentence

cd ./intermediate_output/
for FILE in *.conllu; 
	do 
		python -m depedit -c ../config.ini -t $FILE > ../output_updated_amalgum_conllu/$FILE
		echo $FILE; 
	done