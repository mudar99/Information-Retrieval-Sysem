import math

import numpy as np
from nltk import word_tokenize

from dataset_utils import word_embedding_antique, document_embeddings_antique, word_embedding_quora, \
    document_embeddings_quora, inverted_index_quora, tfidf_vectors_quora, inverted_index_antique, tfidf_vectors_antique


def manual_cosine_similarity(vec1, vec2):
    #  multiplying their corresponding elements and summing up the results
    dot_product = np.dot(vec1, vec2)
    #  Euclidean norms
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0  # Return default similarity value when division by zero occurs

    similarity = dot_product / (norm1 * norm2)

    if np.isnan(similarity):
        return 0  # Return default similarity value when encountering NaN

    return similarity


def vector_space_model_tfidf(query, name, k):
    from QueryProcessing import calculate_query_tfidf

    if name == "A":
        query_terms = query.lower().split()
        query_vector = calculate_query_tfidf(query, inverted_index_antique, name)

        similarity_scores = {}
        for term in query_terms:
            if term in inverted_index_antique:
                relevant_docs = inverted_index_antique[term]
                for doc_id in relevant_docs:
                    if doc_id not in similarity_scores:
                        similarity_scores[doc_id] = 0
                    similarity_scores[doc_id] += query_vector[term] * tfidf_vectors_antique[doc_id][term]

        for doc_id, score in similarity_scores.items():
            query_magnitude = math.sqrt(sum(query_vector[term] ** 2 for term in query_terms))
            doc_magnitude = math.sqrt(sum(tfidf_vectors_antique[doc_id][term] ** 2 for term in tfidf_vectors_antique[doc_id]))
            if int(query_magnitude) != 0 and int(doc_magnitude) != 0:
                similarity_scores[doc_id] = score / (query_magnitude * doc_magnitude)

        docs = sorted(similarity_scores, key=similarity_scores.get, reverse=True)[:k]
        return docs

    if name == "Q":
        query_terms = query.lower().split()
        query_vector = calculate_query_tfidf(query, inverted_index_quora, name)

        similarity_scores = {}
        for term in query_terms:
            if term in inverted_index_quora:
                relevant_docs = inverted_index_quora[term]
                for doc_id in relevant_docs:
                    if doc_id not in similarity_scores:
                        similarity_scores[doc_id] = 0
                    similarity_scores[doc_id] += query_vector[term] * tfidf_vectors_quora[doc_id][term]

        for doc_id, score in similarity_scores.items():
            query_magnitude = math.sqrt(sum(query_vector[term] ** 2 for term in query_terms))
            doc_magnitude = math.sqrt(
                sum(tfidf_vectors_quora[doc_id][term] ** 2 for term in tfidf_vectors_quora[doc_id]))
            if int(query_magnitude) != 0 and int(doc_magnitude) != 0:
                similarity_scores[doc_id] = score / (query_magnitude * doc_magnitude)

        docs = sorted(similarity_scores, key=similarity_scores.get, reverse=True)[:k]
        return docs
    return None

def vector_space_model_we(query, name, k):
    from QueryProcessing import query_word_embedding

    if name == "A":
        query_embedding = query_word_embedding(word_embedding_antique, query)
        similarity_scores = {}
        query_terms = word_tokenize(query)

        for term in query_terms:
            if term in inverted_index_antique:
                relevant_doc_ids = inverted_index_antique[term]  # Retrieve relevant document IDs from the inverted index
                for doc_id in relevant_doc_ids:
                    try:
                        doc_embedding = document_embeddings_antique[doc_id]  # Retrieve the document embedding
                        similarity = manual_cosine_similarity(query_embedding, doc_embedding)
                        if not np.isnan(similarity):  # Check for division by zero or NaN values
                            similarity_scores[doc_id] = similarity
                    except KeyError:
                        continue

        sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        sorted_ids = [doc_id for doc_id, _ in sorted_scores]  # Extract only the document IDs
        return sorted_ids

    if name == "Q":
        query_embedding = query_word_embedding(word_embedding_quora, query)
        similarity_scores = {}
        query_terms = word_tokenize(query)

        for term in query_terms:
            if term in inverted_index_quora:
                relevant_doc_ids = inverted_index_quora[term]  # Retrieve relevant document IDs from the inverted index
                for doc_id in relevant_doc_ids:
                    try:
                        doc_embedding = document_embeddings_quora[doc_id]  # Retrieve the document embedding
                        similarity = manual_cosine_similarity(query_embedding, doc_embedding)
                        if not np.isnan(similarity):  # Check for division by zero or NaN values
                            similarity_scores[doc_id] = similarity
                    except KeyError:
                        continue

        sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        sorted_ids = [doc_id for doc_id, _ in sorted_scores]  # Extract only the document IDs
        return sorted_ids
    return None


# def vector_space_model_tfidf(query, name, k):
#     from QueryProcessing import calculate_query_tfidf
#
#     if name == "A":
#         query_terms = query.lower().split()
#         query_vector = calculate_query_tfidf(query, inverted_index_antique, name)
#
#         similarity_scores = {}
#         for doc_id, term_vectors in tfidf_vectors_antique.items():
#             # dot_product measures the similarity.
#             dot_product = 0
#             for term in query_terms:
#                 if term in term_vectors:
#                     dot_product += query_vector[term] * term_vectors[term]
#             query_magnitude = math.sqrt(sum(query_vector[term] ** 2 for term in query_terms))
#             # calculates the magnitude of the term vectors for a specific document.
#             doc_magnitude = math.sqrt(sum(term_vectors[term] ** 2 for term in term_vectors))
#             # Handle division by zero
#             if int(query_magnitude) == 0 or int(doc_magnitude) == 0:
#                 similarity_scores[doc_id] = 0
#             else:
#                 similarity_scores[doc_id] = dot_product / (query_magnitude * doc_magnitude)
#         docs = sorted(similarity_scores, key=similarity_scores.get, reverse=True)[:k]
#         return docs
#
#     if name == "Q":
#         query_terms = query.lower().split()
#         query_vector = calculate_query_tfidf(query, inverted_index_quora, name)
#
#         similarity_scores = {}
#         for doc_id, term_vectors in tfidf_vectors_quora.items():
#             # dot_product measures the similarity.
#             dot_product = 0
#             for term in query_terms:
#                 if term in term_vectors:
#                     dot_product += query_vector[term] * term_vectors[term]
#             query_magnitude = math.sqrt(sum(query_vector[term] ** 2 for term in query_terms))
#             # calculates the magnitude of the term vectors for a specific document.
#             doc_magnitude = math.sqrt(sum(term_vectors[term] ** 2 for term in term_vectors))
#             # Handle division by zero
#             if int(query_magnitude) == 0 or int(doc_magnitude) == 0:
#                 similarity_scores[doc_id] = 0
#             else:
#                 similarity_scores[doc_id] = dot_product / (query_magnitude * doc_magnitude)
#         docs = sorted(similarity_scores, key=similarity_scores.get, reverse=True)[:k]
#
#         return docs


# def vector_space_model_we(query, name, k):
#     from QueryProcessing import query_word_embedding
#
#     if name == "A":
#         query_embedding = query_word_embedding(word_embedding_antique, query)
#         similarity_scores = {}
#         for doc_id, doc_embedding in document_embeddings_antique.items():
#             similarity = manual_cosine_similarity(query_embedding, doc_embedding)
#             if not np.isnan(similarity):  # Check for division by zero or NaN values
#                 similarity_scores[doc_id] = similarity
#
#         sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:k]
#         sorted_ids = [doc_id for doc_id, _ in sorted_scores]  # Extract only the document IDs
#         return sorted_ids
#
#     if name == "Q":
#         query_embedding = query_word_embedding(word_embedding_quora, query)
#         similarity_scores = {}
#         for doc_id, doc_embedding in document_embeddings_quora.items():
#             similarity = manual_cosine_similarity(query_embedding, doc_embedding)
#             if not np.isnan(similarity):  # Check for division by zero or NaN values
#                 similarity_scores[doc_id] = similarity
#
#         sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:k]
#         sorted_ids = [doc_id for doc_id, _ in sorted_scores]  # Extract only the document IDs
#         return sorted_ids
#
#     return None
