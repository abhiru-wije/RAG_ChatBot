from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

FILE_PATH = "../documents/Zappy-AI.pdf"

# Create loader
loader = PyPDFLoader(FILE_PATH)

# Split document into sentences
pages = loader.load_and_split()

print(len(pages))

# Emebedding function
embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Create a vector store
vectordb = Chroma.from_documents(
    documents=pages,
    embedding= embedding_function,
    persist_directory= "../vector_db",
    collection_name = "megareward_collection"
)

# Make persistent
vectordb.persist()