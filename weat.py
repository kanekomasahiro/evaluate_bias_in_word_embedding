import numpy as np
from gensim.models import KeyedVectors
import sys
import argparse
import json
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--embedding', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1111)

    args = parser.parse_args()

    return args


def word_assoc(w, A, B, emb):
    return emb.n_similarity([w],A) - emb.n_similarity([w],B)


def diff_assoc(X, Y, A, B, emb):
    word_assoc_X = np.array(list(map(lambda x : word_assoc(x, A, B, emb), X)))
    word_assoc_Y = np.array(list(map(lambda y : word_assoc(y, A, B, emb), Y)))
    mean_diff = np.mean(word_assoc_X) - np.mean(word_assoc_Y)
    std = np.std(np.concatenate((word_assoc_X, word_assoc_Y), axis=0))
    return mean_diff / std


def random_choice(word_pairs, subset_size):
    return np.random.choice(word_pairs,
                            subset_size,
                            replace=False)


def get_bias_scores_mean_err(word_pairs, emb):
    subset_size_target = min(len(word_pairs['X']), len(word_pairs['Y'])) // 2
    subset_size_attr = min(len(word_pairs['A']), len(word_pairs['B'])) // 2
    bias_scores = [diff_assoc(
        random_choice(word_pairs['X'], subset_size_target),
        random_choice(word_pairs['Y'], subset_size_target),
        random_choice(word_pairs['A'], subset_size_attr),
        random_choice(word_pairs['B'], subset_size_attr),
        emb) for _ in range(5000)]
    return np.mean(bias_scores), stats.sem(bias_scores)


def run_test(config, emb):
    word_pairs = {}
    min_len = sys.maxsize
    for word_list_name, word_list in config.items():
        if word_list_name in ['X', 'Y', 'A', 'B']:
            word_list_filtered = list(filter(lambda x: x in emb and np.count_nonzero(emb[x]) > 0, word_list))
            word_pairs[word_list_name] = word_list_filtered
            if len(word_list_filtered) < 2:
                print('ERROR: Words from list {} not found in embedding\n {}'.\
                format(word_list_name, word_list))
                print('All word groups must contain at least two words')
                return None, None
    return get_bias_scores_mean_err(word_pairs, emb)


def eval_weat(emb, output):
    config = json.load(open('data/weat.json'))
    with open(output, 'w') as fw:
        for name_of_test, test_config in config['tests'].items():
            mean, err = run_test(test_config, emb)
            if mean is not None:
                mean = str(round(mean, 4))
                err = str(round(err, 4))
                fw.write(f'{name_of_test}\n')
                fw.write(f'Score: {mean}\n')
                fw.write(f'P-value: {err}\n')


def main(args):
    if args.embedding.endswith('bin'):
        binary = True
    else:
        binary = False
    emb = KeyedVectors.load_word2vec_format(args.embedding, binary=binary)
    eval_weat(emb, args.output)


if __name__ == '__main__':
    args= parse_args()
    np.random.seed(args.seed)
    main(args)
