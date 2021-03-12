import numpy as np
import argparse
from gensim.models import KeyedVectors
from scipy.stats import pearsonr


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--embedding', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()

    return args

def cos_sim(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def eval_wat(emb, output, path, word1_list, word2_list):
    gold_d = {}
    for l in open(path):
        word, score = l.strip().split('\t')
        if word in emb:
            gold_d[word] = float(score)
    emb_d = dict()
    for key in gold_d.keys():
        if key in word1_list or key in word2_list:
            continue
        e = emb[key]
        score = [cos_sim(e, emb[w1]) - cos_sim(e, emb[w2])
                 for w1, w2 in zip(word1_list, word2_list)]
        score = sum(score) / len(score)
        emb_d[key] = score

    g_l = []
    e_l = []
    for key in emb_d.keys():
        g_l += [gold_d[key]]
        e_l += [emb_d[key]]
    r, p = pearsonr(np.array(g_l), np.array(e_l))

    with open(output, 'w') as fw:
        fw.write(f'Pearson’s correlation coefficient: {r}\n')
        fw.write(f'P-value: {p}\n')

def main(args):
    embedding = args.embedding
    if embedding.endswith('bin'):
        binary = True
    else:
        binary = False
    emb = KeyedVectors.load_word2vec_format(embedding, binary=binary)
    emb = {word: emb[word] for word in emb.vocab.keys()}

    word1_list = ['he', 'father', 'son', 'husband', 'grandfather',
                  'brother', 'man', 'boy', 'uncle', 'gentleman']
    word2_list = ['she', 'mother', 'daughter', 'wife', 'grandmother',
                  'sister', 'woman', 'girl', 'aunt', 'lady']

    eval_wat(emb, args.output, 'data/wat_bi.txt', word1_list, word2_list)

if __name__ == "__main__":
    args = parse_args()
    main(args)
