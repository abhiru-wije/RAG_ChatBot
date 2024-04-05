from flask import Flask, request, jsonify
from queryretriever import QueryRetriever
from sentence_transformers import SentenceTransformer
import chromadb
import os

app = Flask(__name__)

CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = 8000
chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
bot_emb = chroma_client.get_collection("whatsapp")
model = SentenceTransformer('all-MiniLM-L6-v2')
query_retriever = QueryRetriever(model, bot_emb)

@app.route('/queryretriever', methods=['POST'])
def handle_query():
    # Extract query from request
    data = request.json
    query_text = data.get('query', '')

    # Use the QueryRetriever to process the query
    if query_text:
        results = query_retriever.query(query_text, n_results=1)
        return jsonify(results), 200
    else:
        return jsonify({"error": "Empty query provided"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
