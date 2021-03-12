from numpy import dot
from numpy.linalg import norm
from gensim.models import KeyedVectors
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--embedding', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()

    return args

def eval_sembias(emb, output):
    sembias = [l.strip() for l in open('data/sembias.txt')]

    gender_vector = emb['he'] - emb['she']

    definition_num = 0
    none_num = 0
    stereotype_num = 0
    total_num = 0

    sub_definition_num = 0
    sub_none_num = 0
    sub_stereotype_num = 0
    sub_sembias_size = 40
    sub_sembias_start_idx = len(sembias) - sub_sembias_size

    for sembias_idx, l in enumerate(sembias):
        word_pairs = l.split()
        best_score = -float('inf')
        for word_pair_idx, word_pair in enumerate(word_pairs):
            word_pair = word_pair.split(':')
            diff_vector = emb[word_pair[0]] - emb[word_pair[1]]
            score = dot(gender_vector, diff_vector) \
                  / (norm(gender_vector) * norm(diff_vector))
            if score > best_score:
                best_idx = word_pair_idx
                best_score  = score

        if best_idx == 0:
            definition_num += 1
            if sembias_idx >= sub_sembias_start_idx:
                sub_definition_num += 1
        elif best_idx == 1 or best_idx == 2:
            none_num += 1
            if sembias_idx >= sub_sembias_start_idx:
                sub_none_num += 1
        elif best_idx == 3:
            stereotype_num += 1
            if sembias_idx >= sub_sembias_start_idx:
                sub_stereotype_num += 1

        total_num += 1

    with open(args.output, 'w') as fw:
        definition_score = definition_num / total_num * 100
        stereotype_score = stereotype_num / total_num * 100
        none_score = none_num / total_num * 100
        fw.write(f'definition: {definition_score}\n')
        fw.write(f'stereotype: {stereotype_score}\n')
        fw.write(f'none: {none_score}\n')

        sub_definition_score = sub_definition_num / sub_sembias_size * 100
        sub_stereotype_score = sub_stereotype_num / sub_sembias_size * 100
        sub_none_score = sub_none_num / sub_sembias_size * 100
        fw.write(f'sub definition: {sub_definition_score}\n')
        fw.write(f'sub stereotype: {sub_stereotype_score}\n')
        fw.write(f'sub none: {sub_none_score}\n')

def main(args):
    if args.embedding.endswith('bin'):
        binary = True
    else:
        binary = False
    emb = KeyedVectors.load_word2vec_format(args.embedding, binary=binary)
    eval_sembias(emb, args.output)

if __name__ == '__main__':
    args= parse_args()
    main(args)
