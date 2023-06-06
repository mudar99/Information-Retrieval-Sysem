from Evaluation.Docs import get_relevant_docs, get_retrieved_docs
from dataset_utils import load_antique_utils_we, load_quora_utils_we, load_antique_utils_tfidf, load_quora_utils_tfidf


def calculate_precision(relevant_docs, retrieved_docs):
    scores = []
    print(relevant_docs)
    print(retrieved_docs)
    # [[742812, 3], [132153, 4], [321421, 1]] ---> [(742812, 3), (132153, 4),(321421, 1)]
    relevant_docs = [tuple(doc) for doc in relevant_docs]  # Convert elements to tuples
    relevant_docs = set(relevant_docs)

    for doc in relevant_docs:
        doc_id = doc[0]
        if doc_id in retrieved_docs:
            scores.append(doc)
    if len(retrieved_docs) != 0:
        precision = len(scores) / len(retrieved_docs)
    else:
        precision = 0.0
    return precision


def evaluate_precision(queries, name):
    if name == "A":
        output_file = "C:/Users/mudar/PycharmProjects/pythonProject/Evaluation/Results/Antique/TFIDF_precision@10.txt"
    if name == "Q":
        output_file = "C:/Users/mudar/PycharmProjects/pythonProject/Evaluation/Results/Quora/WE_precision@10.txt"
    total_queries = len(queries)
    total_precision = 0.0

    with open(output_file, 'w') as f:
        for query in queries:
            relevant_docs = get_relevant_docs(query, name)
            retrieved_docs = get_retrieved_docs(query, name, 10)
            query_precision = calculate_precision(relevant_docs, retrieved_docs)
            print(f"Query: {query}\tPrecision: {query_precision}\n")
            f.write(f"Query: {query}\tPrecision: {query_precision}\n")
            total_precision += query_precision
        vsm_precision = total_precision / total_queries
        f.write(f"VSM Precision: {vsm_precision}\n")

    return vsm_precision


def get_precision(query, name):
    if name == "A":
        load_antique_utils_we()
        # load_antique_utils_tfidf()
    if name == "Q":
        load_quora_utils_we()
        # load_quora_utils_tfidf()

    relevant_docs = get_relevant_docs(query, name)
    retrieved_docs = get_retrieved_docs(query, name, len(relevant_docs))

    precision = calculate_precision(relevant_docs, retrieved_docs)
    print(f"Query: {query}\tPrecision: {precision}\n")


def get_whole_precision(name):
    from OffileOperations import load_dataset_queries
    # Example usage
    queries = load_dataset_queries(name)
    if name == "A":
        load_antique_utils_we()
        load_antique_utils_tfidf()
    if name == "Q":
        load_quora_utils_we()
        load_quora_utils_tfidf()
    vsm_precision = evaluate_precision(queries, name)
    print("VSM Precision : ", vsm_precision)
