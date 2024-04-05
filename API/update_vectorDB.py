from flask import Flask, request, jsonify
import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Initialize the CHROMA client
CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = 8000
chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# Load your sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

FOLDER_PATH = "../documents"

# Assume this function reads PDF files and returns their content along with metadata
def read_pdf_files_from_folder(folder_path):
    file_data = []

    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return file_data

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in {folder_path}.")
        return file_data

    for file_name in pdf_files:
        try:
            doc = fitz.open(os.path.join(folder_path, file_name))
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            file_data.append({"file_name": file_name, "content": text})
        except Exception as e:
            print(f"Failed to read {file_name}: {e}")

    if not file_data:
        print("PDF files found, but none could be read.")
    return file_data

# folder_path = "../documents"
# file_data = read_pdf_files_from_folder(folder_path)
    
# Function to process files and create embeddings
def process_files_and_create_embeddings(folder_path):
    file_data = read_pdf_files_from_folder(folder_path)
    if not file_data:
        return []

    documents, embeddings, metadatas, ids = [], [], [], []
    for index, data in enumerate(file_data):
        documents.append(data['content'])
        embedding = model.encode(data['content']).tolist()
        embeddings.append(embedding)
        metadatas.append({'source': data['file_name']})
        ids.append(str(index + 1))
    
    # Create or get collection in chromadb and add documents
    collection_name = "bot"
    bot_emb = chroma_client.get_or_create_collection(name=collection_name)
    bot_emb.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
    return bot_emb

bot_emb = process_files_and_create_embeddings(FOLDER_PATH)

# Flask route to process PDFs and query
@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({"error": "Query is required"}), 400

    if not bot_emb:
        return jsonify({"error": "Embeddings not created. Ensure documents are processed."}), 500

    # Query the collection
    input_em = model.encode(query).tolist()
    results = bot_emb.query(query_embeddings=[input_em], n_results=1)
    
    documents = results.get("documents", [])
    simplified_results = [doc[0] for doc in documents if doc]
    
    if simplified_results:
        return jsonify({"answer": simplified_results[0]})
    else:
        return jsonify({"answer": "No results found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)