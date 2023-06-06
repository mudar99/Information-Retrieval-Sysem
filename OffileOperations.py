import json
import math
from collections import defaultdict

import ir_datasets
import numpy as np
from gensim.models import Word2Vec
from nltk import word_tokenize

from TextPreprocessing import preprocess


def create_corpus(docs_iter, name):
    corpus = {}
    i = 0
    for doc in docs_iter:
        corpus[doc.doc_id] = preprocess(doc.text)
        print(i)
        i += 1
    if name == "A":
        output_file = 'Offline Documents/Antique/corpus.json'
    if name == "Q":
        output_file = 'Offline Documents/Quora/corpus.json'
    with open(output_file, 'w') as f:
        json.dump(corpus, f)


def load_corpus(name):
    if name == "A":
        output_file = 'Offline Documents/Antique/corpus.json'
    if name == "Q":
        output_file = 'Offline Documents/Quora/corpus.json'
    with open(output_file, "r") as f:
        corpus_data = json.load(f)
    return corpus_data


def create_training_corpus(docs_iter, name):
    corpus = {}
    for doc in docs_iter:
        corpus[doc.doc_id] = preprocess(doc.text)
    if name == "A":
        output_file = 'Offline Documents/Antique/training_corpus.json'
    if name == "Q":
        output_file = 'Offline Documents/Quora/training_corpus.json'
    with open(output_file, 'w') as f:
        json.dump(corpus, f)


def load_training_corpus(name):
    if name == "A":
        output_file = 'Offline Documents/Antique/training_corpus.json'
    if name == "Q":
        output_file = 'Offline Documents/Quora/training_corpus.json'
    with open(output_file, "r") as f:
        corpus_data = json.load(f)
    return corpus_data


def create_inverted_index(name):
    corpus = load_corpus(name)
    inverted_index = defaultdict(list)
    for doc_id, doc_content in corpus.items():
        print(doc_content)
        terms = word_tokenize(doc_content)
        for term in terms:
            inverted_index[term].append(doc_id)
    if name == "A":
        output_file = 'Offline Documents/Antique/inverted_index.json'
    if name == "Q":
        output_file = 'Offline Documents/Quora/inverted_index.json'
    with open(output_file, 'w') as file:
        json.dump(dict(inverted_index), file)


def load_inverted_index(name):
    if name == "A":
        output_file = 'Offline Documents/Antique/inverted_index.json'
    if name == "Q":
        output_file = 'Offline Documents/Quora/inverted_index.json'
    with open(output_file, "r") as f:
        inverted_index = json.load(f)
    # print(len(inverted_index))
    return inverted_index


def calculate_doc_tfidf_vectors(inverted_index, name):
    if name == "A":
        output_file = 'Offline Documents/Antique/tfidf_doc_vectors.json'
    if name == "Q":
        output_file = 'Offline Documents/Quora/tfidf_doc_vectors.json'

    corpus = load_corpus(name)
    # print(corpus)
    tfidf_vectors = {}
    for doc_id, doc_text in corpus.items():
        tfidf_vector = {}
        # print(doc_text)
        terms = word_tokenize(doc_text)

        for term in terms:
            tf = terms.count(term) / len(terms)
            # idf = 0
            if term in inverted_index:
                idf = math.log(len(corpus) / len(inverted_index[term]))
            tfidf_vector[term] = tf * idf

        tfidf_vectors[doc_id] = tfidf_vector

    with open(output_file, 'w') as f:
        json.dump(tfidf_vectors, f)


def load_tfidf_doc_vectors(name):
    if name == "A":
        output_file = 'Offline Documents/Antique/tfidf_doc_vectors.json'
    if name == "Q":
        output_file = 'Offline Documents/Quora/tfidf_doc_vectors.json'
    with open(output_file, "r") as f:
        inverted_index = json.load(f)
    return inverted_index


def train_model(name,window):
    if name == "A":
        model_path = 'Offline Documents/Antique/word_embedding_model.bin'
        output_file = 'Offline Documents/Antique/document_embedding.json'
    if name == "Q":
        model_path = 'Offline Documents/Quora/word_embedding_model.bin'
        output_file = 'Offline Documents/Quora/document_embedding.json'
    loaded_corpus = load_training_corpus(name)
    tokenized_corpus = [text.split() for text in loaded_corpus.values()]
    # Load or train the Word2Vec model
    try:
        word_embedding_model = Word2Vec.load(model_path)
    except FileNotFoundError:
        word_embedding_model = Word2Vec(tokenized_corpus,
                                        vector_size=100,  # Adjust the vector size
                                        window=window,  # Adjust the window size
                                        min_count=2,  # Adjust the min_count threshold
                                        sg=1)  # Use skip-gram instead of CBOW
        word_embedding_model.save(model_path)
    # Convert documents to word embeddings
    document_embeddings = {}
    for doc_id, doc in loaded_corpus.items():
        doc_tokens = doc.split()
        # this line of code creates an array of zeros to serve as a placeholder
        doc_embedding = np.zeros(word_embedding_model.vector_size)
        num_tokens = 0
        for token in doc_tokens:
            if token in word_embedding_model.wv:
                doc_embedding += word_embedding_model.wv[token]
                num_tokens += 1
        if num_tokens > 0:
            doc_embedding /= num_tokens
            document_embeddings[doc_id] = doc_embedding.tolist()

    # Save the document embeddings to a file
    with open(output_file, 'w') as f:
        json.dump(document_embeddings, f)

    print("Document embeddings saved successfully.")


def load_trained_model(name):
    if name == "A":
        model_path = 'Offline Documents/Antique/word_embedding_model.bin'
    if name == "Q":
        model_path = 'Offline Documents/Quora/word_embedding_model.bin'
    word_embedding_model = Word2Vec.load(model_path)
    return word_embedding_model


def load_document_embedding(name):
    if name == "A":
        model_path = 'Offline Documents/Antique/document_embedding.json'
    if name == "Q":
        model_path = 'Offline Documents/Quora/document_embedding.json'
    with open(model_path, "r") as f:
        document_embedding = json.load(f)

    return document_embedding


def qrels_parsing(dataset, name):
    if name == "A":
        output_file = 'Offline Documents/Antique/antique_qrels.json'
    if name == "Q":
        output_file = 'Offline Documents/Quora/Quora_qrels.json'
    parsed_data = {}
    for query in dataset.queries_iter():
        query_id = query.query_id
        query_text = query.text
        parsed_data[query_text] = []
        for qrel in dataset.qrels_iter():
            if qrel.query_id == query_id:
                doc_id = qrel.doc_id
                relevance = qrel.relevance
                parsed_data[query_text].append((doc_id, relevance))

    with open(output_file, 'w') as f:
        json.dump(parsed_data, f)


def load_qrels(name):
    if name == "A":
        output_file = 'Offline Documents/Antique/antique_qrels.json'
    if name == "Q":
        output_file = 'Offline Documents/Quora/Quora_qrels.json'

    with open(output_file, "r") as f:
        qrels_file = json.load(f)
    return qrels_file


def get_dataset_queries(dataset, name):
    if name == "A":
        output_file = 'Offline Documents/Antique/queries.txt'
    if name == "Q":
        output_file = 'Offline Documents/Quora/queries.txt'
    queries_list = []

    for query in dataset.queries_iter():
        queries_list.append(query.text)

    with open(output_file, 'w') as f:
        json.dump(queries_list, f)


def load_dataset_queries(name):
    if name == "A":
        output_file = 'Offline Documents/Antique/queries.txt'
    if name == "Q":
        output_file = 'Offline Documents/Quora/queries.txt'
    with open(output_file, 'r') as f:
        queries_list = json.load(f)
    return queries_list


def update_antique():
    name = "A"
    antique_dataset = ir_datasets.load('antique')
    antique_test_dataset = ir_datasets.load('antique/test')
    antique_train_dataset = ir_datasets.load('antique/train')
    dataset_docs = antique_dataset.docs_iter()
    dataset_train = antique_train_dataset.docs_iter()
    create_corpus(dataset_docs, name)
    print('create_corpus Done')
    create_training_corpus(dataset_train, name)
    print('create_training_corpus Done')
    create_inverted_index(name)
    print('create_inverted_index Done')
    inverted_index = load_inverted_index(name)
    qrels_parsing(antique_test_dataset, name)
    print('qrels_parsing Done')
    calculate_doc_tfidf_vectors(inverted_index, name)
    print('calculate_doc_tfidf_vectors Done')
    get_dataset_queries(antique_test_dataset, name)
    print('get_dataset_queries Done')
    train_model(name)


def update_quora():
    name = "Q"
    quora_dataset = ir_datasets.load("beir/quora")
    quora_test_dataset = ir_datasets.load("beir/quora/test")
    dataset_docs = quora_dataset.docs_iter()
    create_corpus(dataset_docs, name)
    print('create_corpus Done')
    create_training_corpus(dataset_docs, name)
    print('create_training_corpus Done')
    create_inverted_index(name)
    print('create_inverted_index Done')
    inverted_index = load_inverted_index(name)
    qrels_parsing(quora_test_dataset, name)
    calculate_doc_tfidf_vectors(inverted_index, name)
    print('calculate_doc_tfidf_vectors Done')
    get_dataset_queries(quora_test_dataset, name)
    print('get_dataset_queries Done')
    train_model(name)

# Offline Operations
# update_antique()
# update_quora()
#
# train_model("A",5)
# train_model("Q",2)


