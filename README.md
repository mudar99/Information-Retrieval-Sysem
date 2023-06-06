# Information-Retrieval-System
A information retrieval system has been built using two datasets:

1) Antique:

* Contains 404K documents
* Includes 200 queries
* Consists of 6.6K qrels (relevance judgments)

2) Beir/quora:

* Contains 523K documents
* Includes 10K queries
* Consists of 16K qrels (relevance judgments)
* Both datasets include both testing and training data.

# Text Preprocessing
The Preprocess function performs various text preprocessing steps.

* Replace possessive forms using Contractions.
* Correct spelling mistakes using the autocorrect library.
* Tokenize the text into individual words or special characters using the word_tokenize function.
* Convert to lowercase.
* Remove punctuation marks.
* Remove stop words: Remove predefined set of stop words (e.g., a, an, the, over, etc.).
* Stemming: Reduce words to their word stem or root form.
* POS: Part Of Speech to identify the type of each word. Perform lemmatization using WordNetLemmatizer to obtain the base or dictionary form based on POS.
* Remove auxiliary verbs and delete single-character words.


# Offline Operations
We have represented files in several formats for the purpose of their use

functions:
* create_corpus(docs_iter, name): Creates a corpus by preprocessing and storing the documents.
* load_corpus(name): Loads the corpus data from a JSON file.
* create_training_corpus(docs_iter, name): Creates a training corpus by preprocessing and storing the training documents.
* load_training_corpus(name): Loads the training corpus data from a JSON file.
* create_inverted_index(name): Creates an inverted index from the corpus.
* load_inverted_index(name): Loads the inverted index from a JSON file.
* calculate_doc_tfidf_vectors(inverted_index, name): Calculates the TF-IDF vectors for each document in the corpus.
* load_tfidf_doc_vectors(name): Loads the TF-IDF document vectors from a JSON file.
* train_model(name): Trains a Word2Vec model on the training corpus and saves the document embeddings.
* load_trained_model(name): Loads the trained Word2Vec model.
* load_document_embedding(name): Loads the document embeddings from a JSON file.
* qrels_parsing(dataset, name): Parses the relevance judgments (qrels) from the dataset.
* load_qrels(name): Loads the relevance judgments from a JSON file.
* get_dataset_queries(dataset, name): Retrieves the queries from the dataset and saves them to a text file.
* load_dataset_queries(name): Loads the queries from a text file.
* update_antique(): Updates the Antique dataset by performing various preprocessing and training steps.
* update_quora(): Updates the Quora dataset by performing various preprocessing and training steps.

# Query Processing

functions:
* calculate_query_tfidf(query, inverted_index, name): Calculates the TF-IDF vector for a given query based on the inverted index and corpus.
* query_word_embedding(word_embedding_model, query): Generates the word embedding representation of a query using a trained word embedding model.
* expand_query_wordnet(query, num_expansions): Expands the query by adding synonyms for each term using WordNet.
* retrieve_documents(query, name, k): Retrieves the top-k documents relevant to the given query using the vector space model.
* expand_query_word2vec(query, name): Expands the query by adding similar terms based on word embeddings.
* query_expansion(query, name, k): Performs query expansion by combining multiple techniques and retrieves relevant documents based on the expanded query.

# Vector Space Model

functions:
* manual_cosine_similarity(vec1, vec2): Calculates the cosine similarity between two vectors using manual computation. It computes the dot product of the vectors and normalizes them based on their magnitudes. Returns the similarity score.
* vector_space_model_tfidf(query, name, k): Performs vector space model retrieval using TF-IDF weighting. Takes a query, dataset name, and the number of documents to retrieve (k) as input. Calculates the TF-IDF vector for the query, computes the similarity between the query and each document using TF-IDF vectors, and returns the top-k most relevant document IDs.
* vector_space_model(query, name, k): Performs vector space model retrieval using word embeddings. Takes a query, dataset name, and the number of documents to retrieve (k) as input. Converts the query to a word embedding vector, calculates the similarity between the query and each document using document embeddings, and returns the top-k most relevant document IDs.

# Dataset Server
The code runs a server and waits for a request from the client to input the desired dataset along with the query. After that, it calls the query_expansion function, which performs the aforementioned operations, including matching the results and returning the closest documents.

# Evaluation

functions:
* calculate_average_precision(relevant_docs, retrieved_docs): Calculates the average precision for a given set of relevant documents and retrieved documents. It measures the average relevance of the retrieved documents.
* calculate_reciprocal_rank(relevant_docs, retrieved_docs): Calculates the reciprocal rank for a given set of relevant documents and retrieved documents. It determines the rank of the first relevant document in the list of retrieved documents.
* calculate_precision(relevant_docs, retrieved_docs): Calculates the precision for a given set of relevant documents and retrieved documents. It measures the proportion of retrieved documents that are relevant.
* calculate_recall(relevant_docs, retrieved_docs): Calculates the recall for a given set of relevant documents and retrieved documents. It measures the proportion of relevant documents that are retrieved.

# How to communicate between services
1. Run the server and wait for user input.
2. Perform preprocessing on the query.
3. Perform query expansion using Word2Vec or WordNet.
4. Perform preprocessing on the expanded query.
5. Call the Vector Space Model (VSM) to calculate TF-IDF for the query or calculate an embedding vector, depending on the chosen feature.
6. The VSM calculates the cosine similarity between the document vectors and the query vector, where higher values indicate higher similarity.
7. Reorder the results based on the highest cosine similarity.
8. Return the results to the user.
