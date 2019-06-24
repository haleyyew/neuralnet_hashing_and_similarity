from gensim.models import FastText
from gensim.test.utils import common_texts

print(common_texts[0])
print(len(common_texts))

# model = FastText(size=4, window=3, min_count=1)  # instantiate
# model.build_vocab(sentences=common_texts)
# model.train(sentences=common_texts, total_examples=len(common_texts), epochs=10)  # train
#
import numpy as np
# bool = np.allclose(model.wv['computer'], model.wv['computer'])
# print(bool)

# TODO use the streaming version instead

model_dir = "/Users/haoran/Documents/neuralnet_hashing_and_similarity/model/"

from gensim.test.utils import get_tmpfile
fname = get_tmpfile(model_dir + "fasttext.model")
# model.save(fname)
model = FastText.load(fname)

print(1, 'computation' in model.wv.vocab)
old_vector = np.copy(model.wv['computation'])
print(2, old_vector)

new_sentences = [
     ['computer', 'aided', 'design'],
     ['computer', 'science'],
     ['computational', 'complexity'],
     ['military', 'supercomputer'],
     ['central', 'processing', 'unit'],
     ['onboard', 'car', 'computer'],
 ]

model.build_vocab(new_sentences, update=True)
model.train(new_sentences, total_examples=len(new_sentences), epochs=model.epochs)

new_vector = model.wv['computation']

print(3, np.allclose(old_vector, new_vector, atol=1e-4))
print(4, 'computation' in model.wv.vocab)

existent_word = "computer"
print(5, existent_word in model.wv.vocab)
computer_vec = model.wv[existent_word]

oov_word = "graph-out-of-vocab"
print(6, oov_word in model.wv.vocab)
oov_vec = model.wv[oov_word]

similarities = model.wv.most_similar(positive=['computer', 'human'], negative=['interface'])
most_similar = similarities[0]
print(7, most_similar)

similarities = model.wv.most_similar_cosmul(positive=['computer', 'human'], negative=['interface'])
most_similar = similarities[0]
print(8, most_similar)

not_matching = model.wv.doesnt_match("human computer interface tree".split())
print(9, not_matching)

sim_score = model.wv.similarity('reading book', 'library store')
print(10, sim_score)

try:
    sim_score = model.wv.similarity('reading book', 'library store')
except Exception:
    # words do not appear together for the 1st or 2nd sequence
    pass

# # Correlation with human opinion on word similarity:
# from gensim.test.utils import datapath
# similarities = model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))
#
# # And on word analogies:
# analogies_result = model.wv.evaluate_word_analogies(datapath('questions-words.txt'))

# python 3.7