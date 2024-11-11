import pyodbc
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import MarkdownTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from dotenv import dotenv_values
from bs4 import BeautifulSoup
from langchain_pinecone import PineconeVectorStore
import os

config = dotenv_values(".env")


# MS SQL Database configuration
connection_string = config["DB_URI"]

# pc = Pinecone(api_key=config["PINECONE_API_KEY"])

# Pinecone index configuration
index_name = "finda"


embedder = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=config["OPENAI_KEY"])
os.environ['PINECONE_API_KEY'] = config["PINECONE_API_KEY"]
vector_store = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedder, )


results = vector_store.similarity_search(
    "integration",
    k=2,
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")