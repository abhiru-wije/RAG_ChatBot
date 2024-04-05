import chromadb
from chromadb.config import Settings
import os

client = chromadb.Client(Settings(chroma_api_impl="rest",
                                  chroma_server_host=os.environ['CHROMADB_HOST'],
                                  chroma_server_http_port="8000"
                                  ))
