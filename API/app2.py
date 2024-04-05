from flask import Flask, request, jsonify
from decouple import config
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import chromadb
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()

app = Flask(__name__)

# Initialize the CHROMA client
CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = 8000
chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_db = Chroma(
    client=chroma_client,
    collection_name="bot",
    embedding_function=embedding_function
)

llm = ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"), temperature=0.5)


def rag_function(question: str, history: list = None) -> str:
    """
    This function takes a question as input and returns the answer to the question.
    
    :param question: String value of the question or the prompt from the user.
    :return: String value of the answer to the user question.
    """
    context = ""
    if history:
        for msg in history:
            # memory.add_message(body=msg["body"], role=msg["role"], timestamp=msg["timestamp"])
            context += f"{msg['role']}: {msg['body']}\n"
    
    full_question = context + question
    
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history", initial_messages=[{"body": full_question, "role": "user"}])
    
    qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=memory,
    retriever= vector_db.as_retriever(search_kwargs={"fetch": 4, "k": 3}, search_type="mmr"),
    chain_type="refine"
    )
    
    response = qa_chain({"question": full_question})
    return response.get("answer")

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    history = data.get('history', [])
    
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        answer = rag_function(question, history)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
