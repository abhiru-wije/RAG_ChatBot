from flask import Flask, request, jsonify
import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
# Placeholder imports - replace with your actual modules
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from decouple import config

load_dotenv()

app = Flask(__name__)

# Initialize the CHROMA client
CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = 8000
chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# Load the sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

FOLDER_PATH = "../documents"

# Function to read PDF files from a folder and return their content
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
    return file_data

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
    collection_name = "bot"
    bot_emb = chroma_client.get_or_create_collection(name=collection_name)
    bot_emb.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
    return bot_emb

bot_emb = process_files_and_create_embeddings(FOLDER_PATH)

class ChromaRetriever:
    def __init__(self, chroma_client):
        self.chroma_client = chroma_client

    def retrieve(self, query_embeddings, n_results):
        results = self.chroma_client.search(query_embeddings, n_results)
        return results
    
    def to_dict(self):
        return {"type": "chroma", "client": self.chroma_client}

chroma_retriever = ChromaRetriever(chroma_client)
chroma_retriever_dict = chroma_retriever.to_dict()

# Initialize the ChatOpenAI instance and other components for conversational retrieval
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.5)
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=memory,
    retriever=chroma_retriever_dict,
    chain_type="refine"
)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    query = data.get('query')
    history = data.get('history')
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    # Query the vector database
    input_em = model.encode(query).tolist()
    vector_db_results = bot_emb.query(query_embeddings=[input_em], n_results=1)
    
    # Now, process the query and history with your RAG function
    answer = rag_function(query, history, vector_db_results)
    return jsonify({"answer": answer})

def rag_function(query: str, history: list, vector_db_results) -> str:
    # Incorporate vector_db_results with history and query to generate an answer
    # Placeholder implementation - adapt based on your actual logic
    conversation_context = {"query": query, "history": history, "vector_db_results": vector_db_results}
    response = qa_chain(conversation_context)  # Assuming this is how you've set up your chain to accept context
    return response.get("answer")

if __name__ == '__main__':
    app.run(host='0.0.0.0', Port=5000, debug=True)
