from Evaluation.Docs import get_relevant_docs, get_retrieved_docs
from dataset_utils import load_antique_utils_we, load_quora_utils_we, load_antique_utils_tfidf, load_quora_utils_tfidf


def calculate_recall(relevant_docs, retrieved_docs):
    num_relevant_docs = len(relevant_docs)
    # Extract document IDs from relevant_docs
    relevant_ids = [doc[0] for doc in relevant_docs]
    TP = sum(1 for doc in retrieved_docs if doc in relevant_ids)
    FN = num_relevant_docs - TP
    # Calculate the recall
    if TP + FN > 0:
        recall = TP / (TP + FN)
    else:
        recall = 0.0
    return recall


def evaluate_recall(queries, name):
    if name == "A":
        output_file = "C:/Users/mudar/PycharmProjects/pythonProject/Evaluation/Results/Antique/TFIDF_Recall.txt"
    if name == "Q":
        output_file = "C:/Users/mudar/PycharmProjects/pythonProject/Evaluation/Results/Quora/WE_Recall.txt"
    total_queries = len(queries)
    total_recall = 0.0

    with open(output_file, 'w') as f:
        for query in queries:
            relevant_docs = get_relevant_docs(query, name)
            retrieved_docs = get_retrieved_docs(query, name,len(relevant_docs))
            query_recall = calculate_recall(relevant_docs, retrieved_docs)
            print(f"Query: {query}\tRecall: {query_recall}\n")
            f.write(f"Query: {query}\tRecall: {query_recall}\n")
            total_recall += query_recall
        average_recall = total_recall / total_queries
        f.write(f"Average Recall: {average_recall}\n")

    return average_recall


def get_recall(name):
    from OffileOperations import load_dataset_queries
    # Example usage
    queries = load_dataset_queries(name)
    if name == "A":
        load_antique_utils_we()
        load_antique_utils_tfidf()
    if name == "Q":
        load_quora_utils_we()
        load_quora_utils_tfidf()
    recall = evaluate_recall(queries, name)
    print("Average Recall : ", recall)
