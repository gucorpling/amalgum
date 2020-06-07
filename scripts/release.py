import argparse
import os
import re
from uuid import uuid1 as uuid
from shutil import copytree, rmtree
from glob import glob
import xml.etree.ElementTree as ET

import conllu
from tqdm import tqdm
from pmap import pmap
ID_PATTERN = re.compile(r' id="([^"]*)"')


# general --------------------------------------------------------------------------------
def progress(msg):
    print('⌛ ' + msg)
    
    
def warn(msg):
    print('⚠️  ' + msg)


def ok(msg):
    print('✔️  ' + msg)


# helpers --------------------------------------------------------------------------------
def dirtype2ext(dir_type):
    if dir_type == "dep":
        return "conllu"
    elif dir_type == "rst":
        return "rs3"
    else:
        return dir_type


def conllu_tokens(conllu_filepath):
    with open(conllu_filepath, 'r') as f:
        return [t['form'] for tl in conllu.parse(f.read()) for t in tl]


def xml_tokens(xml_filepath):
    unescape = (
        lambda s: s.replace('&amp;', '&')
            .replace('&lt;', '<')
            .replace('&gt;', '>')
            .replace('&quot;', '"')
            .replace('&apos;', "'")
    )
    with open(xml_filepath, 'r') as f:
        xml_lines = f.read().split("\n")
    return [
        unescape(line.split('\t')[0])
        for line in xml_lines if '\t' in line
    ]


def tsv_tokens(tsv_filepath):
    with open(tsv_filepath, 'r') as f:
        tsv_lines = f.read().split("\n")
    return [
        line.split('\t')[2]
        for line in tsv_lines
        if '\t' in line and not line.lstrip().startswith('#')
    ]


def rs3_tokens(rs3_filepath):
    tree = ET.parse(rs3_filepath)
    root = tree.getroot()
    return [tok for seg in root.iter('segment') for tok in seg.text.split(' ')]


DIR_TYPE_TOKENS_FUNCTION = {
    'xml': xml_tokens,
    'dep': conllu_tokens,
    'tsv': tsv_tokens,
    'rst': rs3_tokens
}
def read_tokens(filepath, dir_type):
    if dir_type not in DIR_TYPE_TOKENS_FUNCTION:
        print(f'Unknown dir type {dir_type}, returning empty list of tokens')
        return []
    else:
        return DIR_TYPE_TOKENS_FUNCTION[dir_type](filepath)


def count_tokens(dir, genre=None):
    filenames = glob(f'{dir}/dep/*{genre if genre is not None else ""}*.conllu')
    count = 0
    for filename in filenames:
        with open(filename, 'r') as f:
            for line in f:
                if '\t' in line:
                    count += 1
    return count


# tests ----------------------------------------------------------------------------------
def check_filename_parity(args, working_dir):
    progress('Checking whether all dir types have the same filenames...')
    # glob over '.../dep/*.conllu', etc.
    filenames_by_dir_type = {
        dir_type: [
            dpath.split('/')[-1][:-(len(dirtype2ext(dir_type))+1)]
            for dpath in sorted(glob(f'{working_dir}/{dir_type}/*.{dirtype2ext(dir_type)}'))
        ]
        for dir_type in args.dir_types
    }
    filenames = list(filenames_by_dir_type.values())
    assert all(filenames[0] == filenames[i] for i in range(1, len(filenames))), "Not all filenames were equal across all directories!"
    ok(f"Filenames OK, {len(filenames[0])} documents found")


def check_token_parity(args, working_dir):
    progress('Checking whether all documents across all dir types have the same tokens...')
    bare_filenames = [
        filepath.split('/')[-1][:-(len(dirtype2ext(args.dir_types[0]))+1)]
        for filepath in glob(f'{working_dir}/{args.dir_types[0]}/*.{dirtype2ext(args.dir_types[0])}')
    ]
    for filename in sorted(bare_filenames):
        token_lists = []
        for dir_type in args.dir_types:
            tokens = read_tokens(f'{working_dir}/{dir_type}/{filename}.{dirtype2ext(dir_type)}', dir_type)
            if len(tokens) > 0:
                token_lists.append(tokens)

        # check equal length
        assert len(set(len(tl) for tl in token_lists)) == 1

        for i in range(max(len(tl) for tl in token_lists)):
            if not all(token_lists[0][i] == tl[i] for tl in token_lists[1:]):
                print(i, "\t".join(f'"{tl[i]}"' if i < len(tl) else '_' for tl in token_lists))
        assert all(token_lists[0] == tl for tl in token_lists[1:]), f'Not all tokens matched for dir types {args.dir_types}'

    ok(f'Tokens equal across all dir types {args.dir_types}. Token count: f{count_tokens(working_dir)}')


def filter_docs_by_length(args, working_dir):
    progress('Checking whether all documents across all dir types have the same tokens...')
    print("Token counts by genre:")
    for genre in args.genres:
        print(genre, count_tokens(working_dir, genre=genre))

    bare_filenames = [
        filepath.split('/')[-1][:-(len(dirtype2ext(args.dir_types[0]))+1)]
        for filepath in glob(f'{working_dir}/{args.dir_types[0]}/*.{dirtype2ext(args.dir_types[0])}')
    ]
    for filename in sorted(bare_filenames):
        filepaths = [f'{working_dir}/{dir_type}/{filename}.{dirtype2ext(dir_type)}' for dir_type in args.dir_types]

        tokens = read_tokens(filepaths[0], args.dir_types[0])
        rm = False
        if len(tokens) < args.min_length:
            rm = True
            warn(f"{filename} has {len(tokens)} < {args.min_length} tokens, removing")
        elif len(tokens) > args.max_length:
            rm = True
            warn(f"{filename} has {len(tokens)} > {args.max_length} tokens, removing")

        if rm:
            for filepath in filepaths:
                os.remove(filepath)

    print("Token counts by genre after removing documenths with bad length:")
    for genre in args.genres:
        print(genre, count_tokens(working_dir, genre=genre))
    ok(f'Removed documents with length not between {args.min_length} and {args.max_length}')


def fix_xml_id(filepath):
    with open(filepath, "r") as f:
        s = f.read()
    filename = filepath.split(os.sep)[-1][:-4]
    filename = filename.replace('autogum', 'amalgum')
    match = re.search(ID_PATTERN, s)
    s = s.replace(f' id="{match.group(1)}"', f' id="{filename}"')
    with open(filepath, "w") as f:
        f.write(s)


def compact(args, working_dir):
    """Make sure genre numbers are contiguous. Also, s/autogum/amalgum """
    progress("Compacting document IDs")
    bare_filenames = [
        filepath.split('/')[-1][:-(len(dirtype2ext(args.dir_types[0]))+1)]
        for filepath in glob(f'{working_dir}/{args.dir_types[0]}/*.{dirtype2ext(args.dir_types[0])}')
    ]

    for genre in args.genres:
        bare_genre_filenames = sorted([x for x in bare_filenames if genre in x])
        #print(f"{len(bare_genre_filenames)} in {genre}")

        for i, bare_filename in enumerate(bare_genre_filenames):
            new_bare_filename = bare_filename[:-3] + str(i).zfill(3)
            new_bare_filename = new_bare_filename.replace('autogum', 'amalgum')
            #print('Renaming', bare_filename, '->', new_bare_filename)
            for dir_type in args.dir_types:
                old_filepath = f'{working_dir}/{dir_type}/{bare_filename}.{dirtype2ext(dir_type)}'
                new_filepath = f'{working_dir}/{dir_type}/{new_bare_filename}.{dirtype2ext(dir_type)}'
                os.rename(old_filepath, new_filepath)
                # xml has an id attribute that needs to be updated
                if dir_type == 'xml':
                    fix_xml_id(new_filepath)
    ok('Document IDs compacted.')


def balance(args, working_dir):
    progress('Balancing genres by token count...')
    print("Token counts by genre:")
    for genre in args.genres:
        print(genre, count_tokens(working_dir, genre=genre))

    bare_filenames = [
        filepath.split('/')[-1][:-(len(dirtype2ext(args.dir_types[0]))+1)]
        for filepath in glob(f'{working_dir}/{args.dir_types[0]}/*.{dirtype2ext(args.dir_types[0])}')
    ]

    for genre in args.genres:
        token_count = count_tokens(working_dir, genre=genre)
        genre_bare_filenames = sorted([x for x in bare_filenames if genre in x])

        while token_count > args.target_token_count:
            filename = genre_bare_filenames.pop()
            filepaths = [f'{working_dir}/{dir_type}/{filename}.{dirtype2ext(dir_type)}' for dir_type in args.dir_types]
            token_count -= len(read_tokens(filepaths[0], args.dir_types[0]))
            # stop one document early to get a bigger-looking number
            if token_count < args.target_token_count:
                break
            #print(f"Removing {filename} current {genre} token count: {token_count}")
            for dir_type in args.dir_types:
                os.remove(f'{working_dir}/{dir_type}/{filename}.{dirtype2ext(dir_type)}')

    print("Token counts by genre after balancing:")
    for genre in args.genres:
        print(genre, count_tokens(working_dir, genre=genre))
    ok(f'Balanced genres')


def main(args, working_dir):
    # check for equality of all filenames (minus exts)
    check_filename_parity(args, working_dir)
    print()

    # check for equality of all tokens across formats
    check_token_parity(args, working_dir)
    print()

    filter_docs_by_length(args, working_dir)
    print()

    compact(args, working_dir)
    print()

    copytree(working_dir, args.output_dir)
    ok(f'Copied unbalanced documents to {args.output_dir}')
    print()

    balance(args, working_dir)
    print()

    copytree(working_dir, args.output_dir + args.balanced_suffix)
    ok(f'Copied balanced documents to {args.output_dir + args.balanced_suffix}')
    print()


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Use this script to package the output of the AMALGUM pipeline for release. "
                    "Documents from source-dir are processed and put in two directories: one contains "
                    "all documents, and the other contains a subset of those documents to make "
                    "token counts approximately balanced across genres."
    )
    p.add_argument(
        "source_dir",
        help="The directory holding the output from the LAST module of the NLP pipeline"
    )
    p.add_argument(
        "--dir-types",
        default=['xml', 'dep', 'tsv', 'rst'],
        nargs='+',
        help="The subdirectories expected under the source dir, each holding a version of every document."
    )
    p.add_argument(
        "--output-dir",
        default="amalgum",
        help="Directory to write documents to"
    )
    p.add_argument(
        "--balanced-suffix",
        default="_balanced",
        help="suffix for genre-balanced directories"
    )
    p.add_argument(
        "--genres",
        default=['academic', 'bio', 'fiction', 'interview', 'news', 'voyage', 'whow'],
        nargs='+',
        help='expected genres'
    )
    p.add_argument(
        "--min-length",
        default=450,
        help="Minimum document length allowed, in tokens."
    )
    p.add_argument(
        "--max-length",
        default=1600,
        help="Maximum document length allowed, in tokens."
    )
    p.add_argument(
        "--target-token-count",
        default=500000,
        help="The maximum number of tokens a genre should have. "
    )

    args = p.parse_args()

    # copy data to a working dir so we can modify it
    working_dir = f'/tmp/{uuid()}'
    copytree(args.source_dir, working_dir)
    ok(f"Made working dir {working_dir}")
    print()
    try:
        main(args, working_dir)
    finally:
        print("Exited main, removing working dir...", end=" ")
        #rmtree(working_dir)
        print("OK.")