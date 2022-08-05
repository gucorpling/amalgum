import copy
import os
import random
import json


def process_configs(arg_parser):
    args, _ = arg_parser.parse_known_args()
    run_args = _yield_configs(arg_parser, args)

    if run_args.seed == -1:
        run_args.seed = random.randint(0, 1000)
    return run_args


def _read_config(path):
    lines = open(path).readlines()

    runs = []
    run = [1, dict()]
    for line in lines:
        stripped_line = line.strip()

        # continue in case of comment
        if stripped_line.startswith('#'):
            continue

        if not stripped_line:
            if run[1]:
                runs.append(run)

            run = [1, dict()]
            continue

        if stripped_line.startswith('[') and stripped_line.endswith(']'):
            repeat = int(stripped_line[1:-1])
            run[0] = repeat
        else:
            key, value = stripped_line.split('=')
            key, value = (key.strip(), value.strip())
            run[1][key] = value

    if run[1]:
        runs.append(run)

    return runs


def _convert_config(config):
    config_list = []
    for k, v in config.items():
        if v.lower() == 'true':
            config_list.append('--' + k)
        elif v.lower() != 'false':
            config_list.extend(['--' + k] + v.split(' '))
    return config_list


def _yield_configs(arg_parser, args, verbose=True):
    args.config = './ssn/eval.conf'
    config = _read_config(args.config)

    for run_repeat, run_config in config:
        args_copy = copy.deepcopy(args)
        run_config = copy.deepcopy(run_config)
        config_list = _convert_config(run_config)
        run_args = arg_parser.parse_args(config_list, namespace=args_copy)

        run_args_list = []
        # batch eval
        save_path = run_args.model_path
        for dirpath,dirnames,filenames in sorted(os.walk(save_path),key=lambda x:x[0]):
            args_path = "/".join(dirpath.split("/")[:-1])+"/args.json"
            args_dict = json.load(open(args_path))

            run_args.model_path = dirpath
            run_args.tokenizer_path = dirpath
            run_args.types_path = args_dict["types_path"]
            run_args.log_path = args_dict["log_path"]
            run_args.weight_decay = args_dict["weight_decay"]
            run_args.eval_batch_size = args_dict["eval_batch_size"]
            run_args.prop_drop = args_dict["prop_drop"]
            run_args.seed = args_dict["seed"]
            run_args.lstm_layers = args_dict["lstm_layers"]
            run_args.decoder_layers = args_dict["decoder_layers"]
            run_args.lstm_drop = args_dict["lstm_drop"]
            run_args.pos_size = args_dict["pos_size"]
            run_args.char_lstm_layers = args_dict["char_lstm_layers"]
            run_args.char_lstm_drop = args_dict["char_lstm_drop"]
            run_args.char_size = args_dict["char_size"]
            run_args.use_pos = args_dict["use_pos"]
            run_args.use_glove = args_dict["use_glove"]
            run_args.use_char_lstm = args_dict["use_char_lstm"]
            run_args.wordvec_path = args_dict["wordvec_path"]
            run_args.reduce_dim = args_dict["reduce_dim"]
            run_args.bert_before_lstm = args_dict["bert_before_lstm"]
            run_args.confidence = args_dict["confidence"]
            run_args.num_query = args_dict["num_query"]

            run_args_list.append(copy.deepcopy(run_args))

    return run_args
