#!/usr/bin/python
# -*- coding: utf-8 -*-

import io, os, sys, re, copy, pickle
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from glob import glob
from argparse import ArgumentParser
from datetime import datetime
from random import shuffle
from torch.autograd import Variable
import torch
import torch.cuda
import torch.nn
import copy
import time


script_dir = os.path.dirname(os.path.realpath(__file__))
lib = os.path.abspath(script_dir + os.sep + "lib")

sys.path.append(script_dir + os.sep)


from typing import Tuple, List, Dict
from collections import defaultdict
from reader.reader import Reader
from config import config
from util.utils import save_dynamic_config
from model.sequence_labeling import BiRecurrentConvCRF4NestedNER
from training.utils import adjust_learning_rate, clip_model_grad, create_opt
from training.utils import pack_target, unpack_prediction
from util.evaluate import evaluate
from util.utils import Alphabet, load_dynamic_config



class ShibuyaEntities:

	def __init__(self):
		self.name = "ShibuyaEntities"
		
	def conllu2acegold(self, conllustr):
		lines = [x.strip().split('\n') for x in conllustr.strip().split('\n\n')]
		acegoldstr = ""
		
		for line in lines:
			acegoldstr += " ".join([x.split('\t')[1] for x in line if '\t' in x]) + "\n"
			# Add fake-gold entities
			acegoldstr += '0,1 person\n\n'

		return acegoldstr
	
	def txt2acegold(self, txtstr):
		acegoldstr = txtstr.replace('\n', '\n0,1 person\n\n') + '\n1,2 person\n\n\n'
		return acegoldstr
		
		
	def gen_data(self, dataset="amalgum"):
		reader = Reader(config.bert_model)
		reader.read_all_data("./data/" + dataset + "/", dataset + ".train", dataset + ".dev",
		                     dataset + ".test")
		
		train_batches, dev_batches, test_batches = reader.to_batch(config.batch_size)
		f = open(os.path.normpath('./data/' + dataset + '_train.pkl'), 'wb')
		pickle.dump(train_batches, f)
		f.close()
		
		f = open(os.path.normpath('./data/' + dataset + '_dev.pkl'), 'wb')
		pickle.dump(dev_batches, f)
		f.close()
		
		f = open(os.path.normpath('./data/' + dataset + '_test.pkl'), 'wb')
		pickle.dump(test_batches, f)
		f.close()
		
		# misc config
		misc_dict = save_dynamic_config(reader)
		f = open(os.path.normpath('./data/' + dataset + '_config.pkl'), 'wb')
		
		pickle.dump(misc_dict, f)
		f.close()
		
		print("o Remember to scp word vectors as well")
		
		
	
	def get_f1(self, model: BiRecurrentConvCRF4NestedNER, mode: str, file_path: str = None,
	           f=None, voc_dict=None, label_dict=None) -> float:
		with torch.no_grad():
			model.eval()
			
			test_input_ids_batches, \
			test_input_mask_batches, \
			test_first_sub_tokens_batches, \
			test_label_batches, \
			test_mask_batches \
				= pickle.load(f)
			f.close()
			
			pred_all, pred, recall_all, recall = 0, 0, 0, 0
			gold_cross_num = 0
			pred_cross_num = 0
			if mode == 'dev':
				batch_zip = zip(dev_input_ids_batches,
				                dev_input_mask_batches,
				                dev_first_sub_tokens_batches,
				                dev_label_batches,
				                dev_mask_batches)
				totaliters = max(len(dev_input_ids_batches),
				                 len(dev_input_mask_batches),
				                 len(dev_first_sub_tokens_batches),
				                 len(dev_label_batches),
				                 len(dev_mask_batches))
			elif mode == 'test':
				batch_zip = zip(test_input_ids_batches,
				                test_input_mask_batches,
				                test_first_sub_tokens_batches,
				                test_label_batches,
				                test_mask_batches)
				totaliters = max(len(test_input_ids_batches),
				                 len(test_input_mask_batches),
				                 len(test_first_sub_tokens_batches),
				                 len(test_label_batches),
				                 len(test_mask_batches))
			
			else:
				raise ValueError
			
			
			outputstr = ""
			
			iter = 0
			
			for input_ids_batch, input_mask_batch, first_sub_tokens_batch, label_batch, mask_batch in batch_zip:
				
				input_ids_batch_var = Variable(torch.LongTensor(np.array(input_ids_batch)))
				input_mask_batch_var = Variable(torch.LongTensor(np.array(input_mask_batch)))
				mask_batch_var = Variable(torch.ByteTensor(np.array(mask_batch, dtype=np.uint8)))
				if config.if_gpu:
					input_ids_batch_var = input_ids_batch_var \
						# .cuda()
					input_mask_batch_var = input_mask_batch_var \
						# .cuda()
					mask_batch_var = mask_batch_var \
						# .cuda()
				
				pred_sequence_entities = model.predict(input_ids_batch_var,
				                                       input_mask_batch_var,
				                                       first_sub_tokens_batch,
				                                       mask_batch_var)
				pred_entities = unpack_prediction(model, pred_sequence_entities)
				p_a, p, r_a, r = evaluate(label_batch, pred_entities)
				
				gold_cross_num += 0
				pred_cross_num += 0
				
				pred_all += p_a
				pred += p
				recall_all += r_a
				recall += r
				
				if file_path is not None:
					for input_ids, input_mask, first_sub_tokens, mask, label, preds \
							in zip(input_ids_batch, input_mask_batch, first_sub_tokens_batch,
							       mask_batch, label_batch, pred_entities):
						words = []
						for t, m in zip(input_ids, input_mask):
							if m == 0:
								break
							words.append(voc_dict.get_instance(t))
						outputstr += ' '.join(words) + '\n'
						
						labels = []
						for l in sorted(label, key=lambda x: (x[0], x[1], x[2])):
							s = first_sub_tokens[l[0]]
							e = first_sub_tokens[l[1]] if l[1] < len(first_sub_tokens) else len(words) - 1
							labels.append("{},{} {}".format(s, e, label_dict.get_instance(l[2])))
						outputstr += '|'.join(labels) + '\n'
						
						labels = []
						for p in sorted(preds, key=lambda x: (x[0], x[1], x[2])):
							s = first_sub_tokens[p[0]]
							e = first_sub_tokens[p[1]] if p[1] < len(first_sub_tokens) else len(words) - 1
							labels.append("{},{} {}".format(s, e, label_dict.get_instance(p[2])))
						outputstr += '|'.join(labels) + '\n'
						
						outputstr += '\n'
				
				print('o Done predicting %d/%d stacks' % (iter, totaliters), end='\r')
				iter += 1
			
			
			pred = pred / pred_all if pred_all > 0 else 1.
			recall = recall / recall_all if recall_all > 0 else 1.
			f1 = 2 / ((1. / pred) + (1. / recall)) if pred > 0. and recall > 0. else 0.
			
			return outputstr, f1
	
	
	
	def predict(self, dataset="amalgum", serialnumber="200226_153935"):
		"""
		Predict sentence NNER
		"""
		
		f = open(os.path.normpath('./data/' + dataset + '_test.pkl'), 'rb')
		
		
		this_model_path = config.model_path + "_" + serialnumber
		
		# misc info
		misc_config: Dict[str, Alphabet] = pickle.load(
			open(os.path.normpath('./data/' + dataset + '_config.pkl'), 'rb'))
		
		voc_dict, label_dict = load_dynamic_config(misc_config)
		config.voc_size = voc_dict.size()
		config.label_size = label_dict.size()
		
		device = torch.device('cpu')
		
		model = BiRecurrentConvCRF4NestedNER(config.bert_model, config.label_size, hidden_size=config.hidden_size,
		                                     layers=config.layers, lstm_dropout=config.lstm_dropout)
		
		model.load_state_dict(torch.load(this_model_path + '.pt', map_location=device))
		
		print('o before f1 score')
		
		cur_time = time.time()
		outputstr, f1 = self.get_f1(model, 'test', file_path=this_model_path + '_' + dataset + '_pred.result.txt',
		                            f=f, voc_dict=voc_dict, label_dict=label_dict)
		print("test step took {:.4f} seconds".format(time.time() - cur_time))
		
		return outputstr, f1
		
	
	
	def subtok2tok(self, outputstr, tokenizedstr):
		
		in_lines = outputstr.strip().split('\n')
		
		goldtokenized_lines = [x for idx, x in enumerate(tokenizedstr.strip().split('\n')) if idx % 3 == 0]
		
		out_lines = []
		
		# always four lines in sequence: subtoks, golds, preds, empty
		for multiplier in range(len(in_lines) // 4 + 1):
			if len(in_lines) <= 4 * multiplier + 2:
				continue
			
			subtoks_string = re.sub(r'(\s+)|(##)', '',
			                        re.sub(r'^\[CLS\](.*)\[SEP\]$', r'\1', in_lines[4 * multiplier + 0])).replace('##',
			                                                                                                      '')
			
			# find matching gold string
			goldtoks = [x for x in goldtokenized_lines if re.sub(r'(\s+)|(##)', '', x) == subtoks_string]
			
			if len(goldtoks) > 1:
				for otherid in range(1, len(goldtoks)):
					assert goldtoks[0] == goldtoks[otherid]
			
			goldtokenized_lines.remove(goldtoks[0])
			goldtoks = goldtoks[0].split()
			
			# subtoks = re.sub(r'^\[CLS\](.*)\[SEP\]$', r'\1', in_lines[4 * multiplier + 0]).strip().split()
			subtoks = re.sub(r'^\[CLS\](.*)\[SEP\]$', r'\1', in_lines[4 * multiplier + 0]).strip().replace('##',
			                                                                                               '').split()
			
			assert len(subtoks) >= len(goldtoks)
			
			emptygold = False
			emptypred = False
			
			# read start, end,
			if in_lines[4 * multiplier + 1].strip() != '':
				golds = [[int(y) if re.match(r'\d+', y) else y for y in re.split(r'[, ]', x.strip())] for x in
				         in_lines[4 * multiplier + 1].strip().split('|')]
			else:
				golds = in_lines[4 * multiplier + 1].strip()
				emptygold = True
			
			if in_lines[4 * multiplier + 2].strip() != '':
				preds = [[int(y) if re.match(r'\d+', y) else y for y in re.split(r'[, ]', x.strip())] for x in
				         in_lines[4 * multiplier + 2].strip().split('|')]
			else:
				preds = in_lines[4 * multiplier + 2].strip()
				emptypred = True
			
			# looping in reverse order
			idgoldtok = len(goldtoks) - 1
			
			for idsub in range(len(subtoks) - 1, -1, -1):
				# if subtoks[idsub].startswith('##'):
				# 	subtoks[idsub-1] += subtoks[idsub][2:]
				# 	subtoks[idsub] = ''
				
				if subtoks[idsub] != goldtoks[idgoldtok]:
					subtoks[idsub - 1] += subtoks[idsub]
					subtoks[idsub] = ''
					
					if not emptygold:
						for idgold in range(len(golds)):
							
							if golds[idgold][0] >= idsub + 1:
								golds[idgold][0] = golds[idgold][0] - 1
							if golds[idgold][1] >= idsub + 2:
								golds[idgold][1] = golds[idgold][1] - 1
					
					if not emptypred:
						for idpred in range(len(preds)):
							if preds[idpred][0] >= idsub + 1:
								preds[idpred][0] = preds[idpred][0] - 1
							if preds[idpred][1] >= idsub + 2:
								preds[idpred][1] = preds[idpred][1] - 1
				
				else:
					# assert subtoks[idsub] == goldtoks[idgoldtok]
					idgoldtok = idgoldtok - 1
			
			subtoks = [x for x in subtoks if x != '']
			subtoks = ' '.join(subtoks)
			
			if not emptygold:
				golds = '|'.join(['%d,%d %s' % (x[0], x[1], x[2]) for x in golds])
			if not emptypred:
				preds = '|'.join(['%d,%d %s' % (x[0], x[1], x[2]) for x in preds])
			
			out_lines += [subtoks, golds, preds, '']
	
		return '\n'.join(out_lines)
	


if __name__ == "__main__":

	p = ArgumentParser()
	p.add_argument('--dataset', '-d', default='amalgum', help='Input dataset')
	p.add_argument('--serialnumber', '-s', default='200226_153935', help='Best model pt serial number')
	p.add_argument('--inputfile', '-i', default="shortsample.conllu", help='Input file')
	opts = p.parse_args()

	print('Step 0: Processing dataset ' + opts.dataset)
	
	# Predict sentence splits
	e = ShibuyaEntities()
	
	inputstr = io.open(opts.inputfile, 'r', encoding="utf8").read()
	
	assert opts.inputfile.endswith(".txt") or opts.inputfile.endswith(".conllu")
	if opts.inputfile.endswith(".txt"):
		assert '\n\n' not in inputstr.strip()
		acegoldstr = e.txt2acegold(inputstr)
		outputfilename = opts.inputfile.replace('.txt', '.ace')
	elif opts.inputfile.endswith(".conllu"):
		assert '\n\n' in inputstr and '\t' in inputstr
		acegoldstr = e.conllu2acegold(inputstr)
		outputfilename = opts.inputfile.replace('.conllu', '.ace')
	
	
	with io.open(os.path.join('.', 'data', opts.dataset, opts.dataset+'.test'), 'w', encoding='utf8') as ftest:
		ftest.write(acegoldstr)
	print("Step 1: File written to ACE format")
	
	# Pickle test data
	e.gen_data(dataset="amalgum")
	print("Step 2: File converted to pickle")

	# predicts and outputs subtoks
	outputstr, f1 = e.predict(dataset="amalgum", serialnumber=opts.serialnumber)
	print("Step 3: File predicted into BERT subtokens")

	
	# convert subtoks to toks
	outputstr = e.subtok2tok(outputstr, acegoldstr)
	print("Step 4: File converted into tokens")
	
	
	with io.open(outputfilename, 'w', encoding='utf8') as face:
		face.write(outputstr)

	if int(f1)!=0 and int(f1)!=100:
		print("o F1 score is", f1)
	
	print("o Done!")
	
	
	
	
	
	






