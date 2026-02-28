
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted

load_dotenv()

#If query is too long, its automatically truncated by HuggingFaceEmbeddings

# -------------------------------
# SAFE GEMINI CALL WRAPPER
# -------------------------------
def safe_invoke(llm, prompt):
    try:
        return llm.invoke(prompt)

    except ResourceExhausted:
        print("\n Gemini API limit crossed.")
        print("Please try again later.\n")
        return None

    except Exception as e:
        print(f"\nUnexpected Error: {e}")
        return None


# EMBEDDINGS (LOCAL)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# LOAD VECTOR DATABASE

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# GEMINI MODEL

llm = ChatGoogleGenerativeAI(
    model="models/gemini-flash-lite-latest",
    temperature=0.3
)

# DOCUMENT SUMMARIZATION

def summarize_document(llm, vectorstore):

    print("\nUsing 5 representative chunks for summary...\n")

    docs = vectorstore.similarity_search(
        "main ideas and overall summary of the document",
        k=10
    )

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
    You are an AI document assistant.

    Provide a concise summary using ONLY
    the context below.

    Context:
    {context}
    """

    response = safe_invoke(llm, prompt)

    if response:
        return response.content
    else:
        return "API limit crossed. Try again later."

# MAIN LOOP

while True:

    query = input(
        "\nAsk question "
        "(exit to quit | summarize to summarize document): "
    )

    if query.lower() == "exit":
        break

    # -------- SUMMARY MODE --------
    if query.lower() == "summarize":
        summary = summarize_document(llm, vectorstore)
        print("\n✅ Document Summary:\n")
        print(summary)
        continue

    # -------- QUESTION ANSWERING --------
    clean_query = compress_query(llm, query)
    docs = retriever.invoke(clean_query)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    #If question is unrelated to document
    prompt = f"""
    You are a document assistant.

    Answer ONLY using the provided context.
    If answer is not present, say:
    "I could not find this information in the document."

    Context:
    {context}

    Question:
    {query}
    """

    response = safe_invoke(llm, prompt)

    if response:
        print("\n✅ Answer:\n")
        print(response.content)