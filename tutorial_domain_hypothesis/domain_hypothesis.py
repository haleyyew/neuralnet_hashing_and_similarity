# Set Unionability
def compute_sets(A, B):
    t = []

    n_a = len(A)
    n_b = len(B)

    D = []
    for i in range(n_a):
        a_i = A[i]
        try:
            j = B.index(a_i)
            del B[j]
            D.append(a_i)
            t.append(a_i)
        except ValueError:
            D.append(a_i)
    D.extend(B)

    n_D = len(D)

    # print(D)
    # print(n_D)
    return n_a, n_b, D, n_D, t

import operator as op
from functools import reduce
def nCr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom


def hypergeometric(s, n_a, n_b, n_D):
    return nCr(n_a, s) * nCr(n_D - n_a, n_b - s) / nCr(n_D, n_b)

def cdf(t, n_a, n_b, n_D):
    n_t = len(t)

    cdf_t = 0
    for s in range(n_t):
        p_s_na_nb_nD = hypergeometric(s, n_a, n_b, n_D)
        # print(s, p_s_na_nb_nD)
        cdf_t += p_s_na_nb_nD

    return cdf_t


def example_set():
    A = ['a', 'a', 'b', 'b', 'b']
    B = ['b', 'b', 'c', 'c', 'd']
    B_copy = B.copy()
    n_a, n_b, D, n_D, t = compute_sets(A, B)
    print(t)
    B = B_copy

    U_set = cdf(t, n_a, n_b, n_D)
    print(U_set)

    alpha = 0.95
    if U_set < alpha:
        same_domain = False # reject same domain hypothesis
    else:
        same_domain = True

# NLP unionability
from gensim.models import FastText
from gensim.test.utils import common_texts
from gensim.test.utils import get_tmpfile
import numpy as np

def add_new_sentences(new_sentences, model):
    model.build_vocab(new_sentences, update=True)
    model.train(sentences=new_sentences, total_examples=len(new_sentences), epochs=model.epochs)

    return model

def create_new_model(model_name):
    word_vec_num_dim = 4
    model = FastText(size=word_vec_num_dim, window=3, min_count=1)
    model.build_vocab(sentences=common_texts)
    model.train(sentences=common_texts, total_examples=len(common_texts), epochs=10)
    return model

def create_set_of_vectors(sentences, model):
    return [model.wv[word] for context in sentences for word in context]

def compute_mean(attr_vectors):
    return sum([word_vec for word_vec in attr_vectors]) / len(attr_vectors)

def compute_covariance(mean, attr_vectors):
    return 1/(len(attr_vectors)) * sum([(word_vec - mean).dot(np.transpose(word_vec - mean)) for word_vec in attr_vectors])

def conmpute_pooled_covar(n_a, n_b, covar_a, covar_b):
    return ((n_a - 1) * covar_a + (n_b - 1) * covar_b) / (n_a + n_b - 2)

def compute_hotelling(n_attr1, n_attr2, attr1_mean, attr2_mean, pooled_covar):
    return n_attr1 * n_attr2 / (n_attr1 + n_attr2) * (attr1_mean - attr2_mean).dot(
        (1 / pooled_covar) * np.transpose((attr1_mean - attr2_mean)))

import scipy
def compute_f_distr_cdf(hotelling, word_vec_num_dim, n_a, n_b):
    return scipy.stats.f.cdf(hotelling, word_vec_num_dim, n_a + n_b - word_vec_num_dim - 1)

def example_nlp():
    model_dir = "/Users/haoran/Documents/neuralnet_hashing_and_similarity/model/"

    # attr1
    model_name = get_tmpfile(model_dir + "fasttext_new.model")
    # attr2
    model_name2 = get_tmpfile(model_dir + "fasttext_new2.model")

    # use facebook word vectors instead for all attrs
    new_sentences = [
        ['computer', 'aided', 'design'],
        ['computer', 'science'],
        ['computational', 'complexity'],
        ['military', 'supercomputer'],
        ['central', 'processing', 'unit'],
        ['onboard', 'car', 'computer'],
        ['a', 'b', 'c'],
    ]

    new_sentences2 = [
        ['computer', 'aided', 'design'],
        ['computer', 'science'],
        ['computational', 'complexity'],
        ['b', 'c', 'd'],
    ]

    word_vec_num_dim = 4
    model = create_new_model(model_name)
    model2 = create_new_model(model_name2)

    model = add_new_sentences(new_sentences, model)
    model2 = add_new_sentences(new_sentences2, model2)

    model.save(model_name)
    model = FastText.load(model_name)
    model2.save(model_name2)
    model2 = FastText.load(model_name2)

    attr1_vectors = create_set_of_vectors(new_sentences, model)
    attr2_vectors = create_set_of_vectors(new_sentences2, model2)
    attr1_mean = compute_mean(attr1_vectors)
    attr1_covar = compute_covariance(attr1_mean, attr1_vectors)
    attr2_mean = compute_mean(attr2_vectors)
    attr2_covar = compute_covariance(attr2_mean, attr2_vectors)

    print(attr1_mean, attr1_covar)
    print(attr2_mean, attr2_covar)

    n_attr1 = len(attr1_vectors)
    n_attr2 = len(attr2_vectors)

    pooled_covar = conmpute_pooled_covar(n_attr1, n_attr2, attr1_covar, attr2_covar)
    # Hotelling’s two sample statistics
    hotelling = compute_hotelling(n_attr1, n_attr2, attr1_mean, attr2_mean, pooled_covar)
    print(pooled_covar, hotelling)

    alpha = 0.95
    U_nlp = compute_f_distr_cdf(hotelling, word_vec_num_dim, n_attr1, n_attr2)

    print(U_nlp)
    if U_nlp < alpha:
        same_domain = False # reject same domain hypothesis
    else:
        same_domain = True

example_set()
# example_nlp()