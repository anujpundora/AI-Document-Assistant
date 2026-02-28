import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

# Load embedding model (same as ingestion)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load stored vector DB
vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.3
)

while True:
    query = input("\nAsk question (type exit): ")

    if query.lower() == "exit":
        break

    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer the question using ONLY the context below.

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    print("\n✅ Answer:\n")
    print(response.content)