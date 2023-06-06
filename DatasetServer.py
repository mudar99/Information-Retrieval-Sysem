import ir_datasets
from flask import Flask, request, jsonify
from flask_cors import CORS
from QueryProcessing import query_expansion
from dataset_utils import load_antique_utils_we, load_quora_utils_we,load_quora_utils_tfidf,load_antique_utils_tfidf

app = Flask(__name__)
CORS(app, origins='http://127.0.0.1:5500')
CORS(app, origins='*')


def get_docs(relevant_docs, docstore):
    docs = docstore.get_many(relevant_docs)
    res = {}
    for doc_id, doc in docs.items():
        res[doc_id] = doc.text
    return res


load_antique_utils_we()
# load_antique_utils_tfidf()
load_quora_utils_we()
# load_quora_utils_tfidf()


@app.route('/query', methods=['POST'])
def post_query():
    query_data = request.get_json()
    query_text = query_data['query']
    dataset_name = query_data['dataset_name']  # Extract the additional parameter
    print(query_text, dataset_name)

    if dataset_name == "A":
        antique_dataset = ir_datasets.load("antique")
        antique_docstore = antique_dataset.docs_store()
        expanded_results = query_expansion(query_text, dataset_name, 10)
        res = get_docs(expanded_results, antique_docstore)
        print("Antique", res)
    if dataset_name == "Q":
        quora_dataset = ir_datasets.load("beir/quora")
        quora_docstore = quora_dataset.docs_store()
        expanded_results = query_expansion(query_text, dataset_name, 10)
        res = get_docs(expanded_results, quora_docstore)
        print("Quora", res)
    response = {'relevant_docs': res}
    return jsonify(response)


if __name__ == '__main__':
    app.run()
