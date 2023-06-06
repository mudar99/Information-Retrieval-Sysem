import math

import numpy as np
from nltk import word_tokenize
from nltk.corpus import wordnet

from OffileOperations import load_corpus, load_trained_model
from TextPreprocessing import preprocess


def calculate_query_tfidf(query, inverted_index, name):
    tfidf_vector = {}
    terms = word_tokenize(query)
    corpus = load_corpus(name)
    for term in set(terms):
        tf = terms.count(term) / len(terms)
        idf = 0  # Default IDF value for terms not in the inverted index
        if term in inverted_index:
            idf = math.log(len(corpus) / len(inverted_index[term]))
        tfidf_vector[term] = tf * idf

    return tfidf_vector


def query_word_embedding(word_embedding_model, query):
    query_embedding = np.zeros(word_embedding_model.vector_size)
    query_tokens = query.lower().split()
    num_tokens = 0
    for word in query_tokens:
        # word_embedding_model.wv retrieves the word vector for the given word.
        if word in word_embedding_model.wv:
            query_embedding += word_embedding_model.wv[word]
            num_tokens += 1
    if num_tokens > 0:
        query_embedding /= num_tokens
    # average of all the word vectors
    return query_embedding


# def expand_query_wordnet(query, num_expansions):
#     expanded_terms = set()
#
#     for term in query.split():
#         # Get synonyms for each term using WordNet
#         synonyms = wordnet.synsets(term)
#         for syn in synonyms:
#             for lemma in syn.lemmas():
#                 expanded_terms.add(lemma.name())
#
#     # Select a subset of expanded terms
#     expanded_terms = list(expanded_terms)[:num_expansions]
#
#     # Combine original query and expanded terms
#     expanded_query = query + " " + " ".join(expanded_terms)
#
#     return expanded_query


def retrieve_documents(query, name, k):
    from Indexing import vector_space_model_we
    from Indexing import vector_space_model_tfidf
    results = vector_space_model_we(query, name, k)
    return results


def expand_query_word2vec(query, name):
    # similarity score between words
    similarity_threshold = 0.9
    word_embedding_model = load_trained_model(name)
    expanded_terms = []
    terms = word_tokenize(query)

    for term in terms:
        #  checks if the current term exists in the vocabulary of the word embedding model
        if term in word_embedding_model.wv.key_to_index:
            similar_terms = word_embedding_model.wv.most_similar(term, topn=3)
            for t in similar_terms:
                if t[1] >= similarity_threshold:
                    expanded_terms.append(t[0])

    expanded_query = query + " " + " ".join(expanded_terms)
    return expanded_query


# Main function
def query_expansion(query, name, k):
    ppq = preprocess(query)
    print("ppq: ", ppq)
    expanded_query = expand_query_word2vec(ppq, name)
    print("Expanded Query:", expanded_query)
    ppq = preprocess(expanded_query)
    final_results = retrieve_documents(ppq, name, k)
    return final_results
