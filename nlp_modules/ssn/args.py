import argparse


def _add_common_args(arg_parser):
    # Input
    arg_parser.add_argument('--types_path', type=str, default='./nner-dependencies/gum_types.json', help="Path to type specifications")

    # Preprocessing
    arg_parser.add_argument('--tokenizer_path', type=str, help="Path to tokenizer")
    arg_parser.add_argument('--lowercase', action='store_true', default=False,
                            help="If true, input is lowercased during preprocessing")
    arg_parser.add_argument('--sampling_processes', type=int, default=4,
                            help="Number of sampling processes. 0 = no multiprocessing for sampling")

    # Logging
    arg_parser.add_argument('--label', type=str, help="Label of run. Used as the directory name of logs/models")
    arg_parser.add_argument('--log_path', type=str, help="Path do directory where training/evaluation logs are stored")
    arg_parser.add_argument('--store_predictions', action='store_true', default=False,
                            help="If true, store predictions on disc (in log directory)")
    arg_parser.add_argument('--store_examples', action='store_true', default=False,
                            help="If true, store evaluation examples on disc (in log directory)")
    arg_parser.add_argument('--example_count', type=int, default=None,
                            help="Count of evaluation example to store (if store_examples == True)")
    arg_parser.add_argument('--debug', action='store_true', default=False, help="Debugging mode on/off")

    # Model / Training / Evaluation
    arg_parser.add_argument('--model_path', type=str, default='./nner-dependencies/model',
                            help="Path to directory that contains model checkpoints")
    arg_parser.add_argument('--dependency_path', type=str, default='./nner-dependencies')
    arg_parser.add_argument('--model_type', type=str, default="ssn", help="Type of model")
    arg_parser.add_argument('--cpu', action='store_true', default=True,
                            help="If true, train/evaluate on CPU even if a CUDA device is available")
    arg_parser.add_argument('--device_id', type=str, default="0", help="gpu device id")
    arg_parser.add_argument('--eval_batch_size', type=int, default=1, help="Evaluation batch size")
    arg_parser.add_argument('--prop_drop', type=float, default=0.1, help="Probability of dropout used in SSN")
    arg_parser.add_argument('--freeze_transformer', action='store_true', default=False, help="Freeze BERT weights")
    arg_parser.add_argument('--no_overlapping', action='store_true', default=False,
                            help="If true, do not evaluate on overlapping entities\
                                 and relations with overlapping entities")
    arg_parser.add_argument('--confidence', type=float, default=0.6, help="prediction confidence received")
    arg_parser.add_argument('--num_query', type=int, default=50, help="number of entity queries")
    arg_parser.add_argument('--decoder_layers', type=int, default=3)
    arg_parser.add_argument('--lstm_drop', type=float, default=0.1)
    arg_parser.add_argument('--lstm_layers', type=int, default=1)
    arg_parser.add_argument('--pos_size', type=int, default=25)
    arg_parser.add_argument('--char_lstm_layers', type=int, default=1)
    arg_parser.add_argument('--char_size', type=int, default=25)
    arg_parser.add_argument('--char_lstm_drop', type=float, default=0.1)
    arg_parser.add_argument('--use_glove', action='store_true', default=False)
    arg_parser.add_argument('--use_pos', action='store_true', default=False)
    arg_parser.add_argument('--use_char_lstm', action='store_true', default=False)
    arg_parser.add_argument('--pool_type', type=str, default = "max")
    arg_parser.add_argument('--wordvec_path', type=str, default = "../glove/glove.6B.300d.txt")
    arg_parser.add_argument('--reduce_dim', action='store_true', default=False)
    arg_parser.add_argument('--bert_before_lstm', action='store_true', default=False)
    arg_parser.add_argument('--pretrained_model', type=str, default='bert-base-cased')
    arg_parser.add_argument('--vec_size', type=str, default="100")

    # Misc
    arg_parser.add_argument('--seed', type=int, default=-1, help="Seed")
    arg_parser.add_argument('--cache_path', type=str, default=None,
                            help="Path to cache transformer models (for HuggingFace transformers library)")


def eval_argparser():
    arg_parser = argparse.ArgumentParser()

    # Input
    arg_parser.add_argument('--dataset_path', type=str, help="Path to dataset")
    arg_parser.add_argument('--corpus', type=str)

    _add_common_args(arg_parser)

    return arg_parser
