from Evaluation.Precision import get_relevant_docs, get_retrieved_docs
from dataset_utils import load_antique_utils_we, load_quora_utils_we, load_antique_utils_tfidf, load_quora_utils_tfidf


def calculate_average_precision(relevant_docs, retrieved_docs):
    precision_sum = 0.0
    num_relevant_retrieved = 0
    # assigns an index value to each element using enumerate
    for i, doc in enumerate(retrieved_docs, start=1):
        if doc in [doc[0] for doc in relevant_docs]:
            num_relevant_retrieved += 1
            precision_sum += num_relevant_retrieved / i

    if num_relevant_retrieved == 0:
        return 0.0

    if len(relevant_docs) != 0:
        average_precision_for_query = precision_sum / len(relevant_docs)
    else:
        average_precision_for_query = 0.0
    return average_precision_for_query


# Get Average Precision For The Whole Corpus Or For My VSM Model
def evaluate_mean_average_precision(queries, name):
    average_precision = 0.0
    total_queries = len(queries)
    if name == "A":
        output_file = "C:/Users/mudar/PycharmProjects/pythonProject/Evaluation/Results/Antique/TFIDF_MAP.txt"
    if name == "Q":
        output_file = "C:/Users/mudar/PycharmProjects/pythonProject/Evaluation/Results/Quora/WE_MAP.txt"

    with open(output_file, 'w') as f:
        for query in queries:
            relevant_docs = get_relevant_docs(query, name)
            retrieved_docs = get_retrieved_docs(query, name, len(relevant_docs))
            # Get Average Precision For Each Query
            query_ap = calculate_average_precision(relevant_docs, retrieved_docs)
            print(f"Query: {query}\tAP: {query_ap}\n")
            f.write(f"Query: {query}\tAP: {query_ap}\n")
            average_precision += query_ap
        mean_average_precision = average_precision / total_queries
        f.write(f"MAP: {mean_average_precision}\n")

    return mean_average_precision


def get_mean_average_precision(name):
    from OffileOperations import load_dataset_queries
    # Example usage
    queries = load_dataset_queries(name)
    if name == "A":
        load_antique_utils_we()
        load_antique_utils_tfidf()
    if name == "Q":
        load_quora_utils_we()
        load_quora_utils_tfidf()
    avg_precision = evaluate_mean_average_precision(queries, name)
    print('MAP : ', avg_precision)
