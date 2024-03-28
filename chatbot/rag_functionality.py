from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from decouple import config

# embedding function
embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory= "../vector_db",
    collection_name="megareward_collection",
    embedding_function=embedding_function
)

llm = ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"), temperature=0.5)

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=memory,
    retriever=vector_db.as_retriever(
        search_kwargs={"fetch": 4, "k": 3}, search_type="mmr"),
    chain_type="refine"
)

def rag_function(question: str) -> str:
    """
    This function takes a question as input and returns the answer to the question.
    
    :param: question: String value of the question or the prompt from the user.
    :return: String value of the answer to the user question.
    """
    response = qa_chain({"question": question})
    
    return response.get("answer")
    


