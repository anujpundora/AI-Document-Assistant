# 📄 AI Document Assistant (RAG-based)

An AI-powered Document Assistant that allows users to upload PDFs and interact with them using natural language queries.

The system uses **Retrieval-Augmented Generation (RAG)** to enable semantic search, contextual question answering, and document summarization using modern AI workflows.

---

## 🚀 Project Overview

Traditional document search relies on keyword matching.

This project enables **semantic understanding** of documents by converting text into vector embeddings and retrieving relevant information before generating answers using an LLM.

Users can:

✅ Ask questions about a PDF  
✅ Generate document summaries  
✅ Perform semantic search  
✅ Get context-aware AI responses  

---

## 🧠 What is RAG?

**Retrieval-Augmented Generation (RAG)** combines:

1. **Retrieval System**
   - Finds relevant information from documents.

2. **Generation Model (LLM)**
   - Uses retrieved context to generate accurate responses.

### RAG Flow

![RAG flow structure](image.png)

Unlike normal LLM responses, answers are grounded in uploaded documents.

---

## 🧩 System Architecture

![SYSTEM ARCHITECTURE PNG ](structure.png)

---

## ⚙️ Technologies Used

| Component | Technology |
|------------|------------|
| Language | Python |
| Framework | LangChain |
| LLM | Google Gemini API |
| Embeddings | HuggingFace Sentence Transformers |
| Vector Database | FAISS |
| PDF Parsing | PyPDFLoader |
| Environment Management | dotenv |

---

## 🔎 Core Concepts Explained

---

### ✅ Embeddings

Embeddings convert text into numerical vectors representing **semantic meaning**.

---

## ⚙️ Technologies Used

| Component | Technology |
|------------|------------|
| Language | Python |
| Framework | LangChain |
| LLM | Google Gemini API |
| Embeddings | HuggingFace Sentence Transformers |
| Vector Database | FAISS |
| PDF Parsing | PyPDFLoader |
| Environment Management | dotenv |

---

## 🔎 Core Concepts Explained

---

### ✅ Embeddings

Embeddings convert text into numerical vectors representing **semantic meaning**.

Similar meanings produce nearby vectors.

---

### ✅ Why Chunking?

LLMs and embedding models have token limits.

Large documents are split into smaller parts:
Overlap preserves contextual continuity.

---

### ✅ Vector Database (FAISS)

FAISS stores embeddings and enables fast similarity search.

Instead of keyword search:
"What is backend language?"

System retrieves semantically similar content like:

Python is used for backend development.


---

### ✅ Semantic Retrieval

User query → embedding → nearest vectors selected.

Only **top-k relevant chunks** are sent to Gemini.

This reduces:
- hallucinations
- token usage
- API cost

---

### ✅ Why Not Send Full PDF?

Sending entire documents:

❌ exceeds token limits  
❌ increases latency  
❌ increases cost  

RAG retrieves only relevant knowledge.

---

## 📄 Features

- ✅ PDF-based Question Answering
- ✅ Semantic Document Search
- ✅ Intelligent Document Summarization
- ✅ Query Compression for long inputs
- ✅ API Rate Limit Handling
- ✅ Context-Grounded Responses
- ✅ Efficient Token Usage

---

## 🏗️ Project Structure

![PROJECT STRUCTURE PNG](image.png)

---

## ▶️ How to Run

### 1️⃣ Clone Repository
python3 -m venv venv
source venv/bin/activate

---

### 3️⃣ Install Dependencies

pip install -r requirements.txt

---

### 4️⃣ Add Gemini API Key

Create `.env`

GOOGLE_API_KEY=your_api_key

---

### 5️⃣ Create Vector Database

---

### 6️⃣ Run Assistant

---

## 🧠 Interview Discussion Points

This project demonstrates:

- Retrieval-Augmented Generation (RAG)
- Semantic Search Systems
- Vector Databases
- Embedding Models
- LLM Integration
- Token Optimization
- API Rate Limit Handling
- AI System Design

---

## 👨‍💻 Author

Built as part of hands-on learning in **AI Engineering & Agentic Systems**.