from OffileOperations import load_trained_model, load_document_embedding, load_tfidf_doc_vectors, load_inverted_index

# Using Word_Embedding
word_embedding_quora = None
document_embeddings_quora = None
word_embedding_antique = None
document_embeddings_antique = None

# Using TF_IDF
inverted_index_antique = None
tfidf_vectors_antique = None
inverted_index_quora = None
tfidf_vectors_quora = None


def load_antique_utils_tfidf():
    global inverted_index_antique, tfidf_vectors_antique
    inverted_index_antique = load_inverted_index("A")
    print("A load_inverted_index")
    tfidf_vectors_antique = load_tfidf_doc_vectors("A")
    print("A load_tfidf_doc_vectors")


def load_quora_utils_tfidf():
    global inverted_index_quora, tfidf_vectors_quora
    inverted_index_quora = load_inverted_index("Q")
    print("Q load_inverted_index")
    tfidf_vectors_quora = load_tfidf_doc_vectors("Q")
    print("Q load_tfidf_doc_vectors")


def load_antique_utils_we():
    global word_embedding_antique, document_embeddings_antique, inverted_index_antique
    word_embedding_antique = load_trained_model("A")
    print("A load_trained_model")
    document_embeddings_antique = load_document_embedding("A")
    print("A load_document_embedding")
    inverted_index_antique = load_inverted_index("A")
    print("A load_inverted_index")


def load_quora_utils_we():
    global word_embedding_quora, document_embeddings_quora, tfidf_vectors_quora
    word_embedding_quora = load_trained_model("Q")
    print("Q load_trained_model")
    document_embeddings_quora = load_document_embedding("Q")
    print("Q load_document_embedding")
    tfidf_vectors_quora = load_tfidf_doc_vectors("Q")
    print("Q load_tfidf_doc_vectors")
