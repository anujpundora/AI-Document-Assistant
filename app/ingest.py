import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load PDF
loader = PyPDFLoader("data/sample.pdf")
documents = loader.load()

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")

# ✅ LOCAL EMBEDDINGS (FREE)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# Store vectors
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

vectorstore.save_local("vectorstore")

print("✅ Vector DB created using local embeddings!")