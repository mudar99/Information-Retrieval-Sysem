from past.translation import hooks

from Evaluation.Docs import get_relevant_docs, get_retrieved_docs
from dataset_utils import load_antique_utils_we, load_quora_utils_we, load_antique_utils_tfidf, load_quora_utils_tfidf


def calculate_reciprocal_rank(relevant_docs, retrieved_docs):
    print(relevant_docs)
    print(retrieved_docs)
    # get highest relevance
    highest_relevance = max([relevance for _, relevance in relevant_docs])
    # get ids for highest relevance relevant docs
    highest_relevant_docs = [doc_id for doc_id, relevance in relevant_docs if relevance == highest_relevance]
    print("-------------highest_relevant_docs----------")
    print(highest_relevant_docs)
    for i, doc in enumerate(retrieved_docs, start=1):
        if doc in highest_relevant_docs:
            return 1 / i
    return 0.0

def evaluate_mean_reciprocal_rank(queries, name):
    if name == "A":
        output_file = "C:/Users/mudar/PycharmProjects/pythonProject/Evaluation/Results/Antique/TFIDF_MRR.txt"
    if name == "Q":
        output_file = "C:/Users/mudar/PycharmProjects/pythonProject/Evaluation/Results/Quora/WE_MRR.txt"

    total_queries = len(queries)
    reciprocal_rank_sum = 0.0

    with open(output_file, 'w') as f:
        for query in queries:
            relevant_docs = get_relevant_docs(query, name)
            retrieved_docs = get_retrieved_docs(query, name, len(relevant_docs))
            reciprocal_rank = calculate_reciprocal_rank(relevant_docs, retrieved_docs)
            print(f"Query: {query}\tRR: {reciprocal_rank}\n")
            f.write(f"Query: {query}\tRR: {reciprocal_rank}\n")
            reciprocal_rank_sum += reciprocal_rank
        mean_reciprocal_rank = reciprocal_rank_sum / total_queries
        f.write(f"MRR: {mean_reciprocal_rank}\n")

    return mean_reciprocal_rank


def get_mean_reciprocal_rank(name):
    from OffileOperations import load_dataset_queries
    # Example usage
    queries = load_dataset_queries(name)
    if name == "A":
        load_antique_utils_we()
        load_antique_utils_tfidf()
    if name == "Q":
        load_quora_utils_we()
        load_quora_utils_tfidf()
    mrr = evaluate_mean_reciprocal_rank(queries, name)
    print("MRR : ", mrr)
