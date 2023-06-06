def get_retrieved_docs(query, name, k):
    from QueryProcessing import query_expansion

    retrieved_docs = query_expansion(query, name, k)
    print("---------------- retrieved_docs ----------------")
    print(len(retrieved_docs))
    return retrieved_docs


def get_relevant_docs(query, name):
    from OffileOperations import load_qrels
    qrels = load_qrels(name)
    relevant_docs_sorted = sorted(qrels[query], key=lambda x: x[1], reverse=True)
    relevant_docs = relevant_docs_sorted
    print("---------------- relevant_docs ----------------")
    print(len(relevant_docs))
    return relevant_docs
