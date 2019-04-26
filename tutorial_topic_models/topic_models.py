from nltk.corpus import brown
# nltk.download('brown')

import re
from gensim import models, corpora
from nltk import word_tokenize
from nltk.corpus import stopwords
# nltk.download('stopwords')

from gensim.test.utils import common_texts, get_tmpfile
import pickle

from gensim import similarities

def get_brown_data():
    data = []

    for fileid in brown.fileids():
        document = ' '.join(brown.words(fileid))
        data.append(document)

    NO_DOCUMENTS = len(data)
    print(NO_DOCUMENTS)
    print(data[:5])
    return data

def clean_text(text):
    tokenized_text = word_tokenize(text.lower())
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    return cleaned_text


NUM_TOPICS = 10
STOPWORDS = stopwords.words('english')

def build_lda_model(data):

    # For gensim we need to tokenize the data and filter out stopwords
    tokenized_data = []
    for text in data:
        tokenized_data.append(clean_text(text))

    # Build a Dictionary - association word to numeric id
    dictionary = corpora.Dictionary(tokenized_data)

    # Transform the collection of texts to a numerical form
    corpus = [dictionary.doc2bow(text) for text in tokenized_data]

    # Have a look at how the 20th document looks like: [(word_id, count), ...]
    print(corpus[20])

    # Build the LDA model
    lda_model = models.LdaModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)

    # print("LDA Model:")
    #
    # for idx in range(NUM_TOPICS):
    #     # Print the first 10 most representative topics
    #     print("Topic #%s:" % idx, lda_model.print_topic(idx, 10))

    return lda_model, dictionary, corpus


from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


def svd_dimension_reduction(data):
    vectorizer = CountVectorizer(min_df=5, max_df=0.9,
                                 stop_words='english', lowercase=True,
                                 token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
    data_vectorized = vectorizer.fit_transform(data)

    svd = TruncatedSVD(n_components=2)
    documents_2d = svd.fit_transform(data_vectorized)

    df1 = pd.DataFrame(columns=['x', 'y', 'document'])
    df1['x'], df1['y'], df1['document'] = documents_2d[:, 0], documents_2d[:, 1], range(len(data))


    svd = TruncatedSVD(n_components=2)
    words_2d = svd.fit_transform(data_vectorized.T)

    df2 = pd.DataFrame(columns=['x', 'y', 'word'])
    df2['x'], df2['y'], df2['word'] = words_2d[:, 0], words_2d[:, 1], vectorizer.get_feature_names()

    return df1, df2


def example():

    # data = get_brown_data()
    # lda_model, dictionary, corpus = build_lda_model(data)

    model_dir = "/Users/haoran/Documents/neuralnet_hashing_and_similarity/model/"
    path = get_tmpfile(model_dir+"lda_brown.model")
    dict_path = get_tmpfile(model_dir+"dict_brown.model")
    corpus_path = get_tmpfile(model_dir+"corpus_brown.pkl")
    data_path = get_tmpfile(model_dir+"data_brown.pkl")
    # lda_model.save(path)
    # dictionary.save(dict_path)
    # pickle.dump(corpus, open(corpus_path, "wb"))
    # pickle.dump(data, open(data_path, "wb"))

    dictionary = corpora.Dictionary.load(dict_path)
    lda_model = models.LdaModel.load(path)
    corpus = pickle.load(open(corpus_path, "rb"))
    data = pickle.load(open(data_path, "rb"))

    text = "The economy is working better than ever"
    bow = dictionary.doc2bow(clean_text(text))  # the query

    print(lda_model[bow])


    lda_index = similarities.MatrixSimilarity(lda_model[corpus])
    # Let's perform some queries
    similarities_to_query = lda_index[lda_model[bow]]
    # Sort the similarities
    similarities_to_query = sorted(enumerate(similarities_to_query), key=lambda item: -item[1])

    # Top most similar documents:
    print(similarities_to_query[:10])

    # Let's see what's the most similar document
    document_id, similarity = similarities_to_query[0]
    print(document_id, similarity )
    print(data[document_id][:1000])


    # SVD
    # df1, df2 = svd_dimension_reduction(data)
    svd_doc_path = get_tmpfile(model_dir+"svd_doc.csv")
    svd_word_path = get_tmpfile(model_dir+"svd_word.csv")
    # df1.to_csv(svd_doc_path, encoding='utf-8')
    # df2.to_csv(svd_word_path, encoding='utf-8')

    svd_doc = pd.read_csv(svd_doc_path)
    svd_word = pd.read_csv(svd_word_path)

    print(svd_doc.head())
    print(svd_word.head())

# example()