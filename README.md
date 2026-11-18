# FAR/DFARS Contract Compliance RAG Assistant

An aerospace contract compliance assistant powered by **Retrieval-Augmented Generation (RAG)**. This application ingests federal acquisition regulations (FAR/DFARS clauses), stores them in a persistent vector database, and uses **GPT-4o** alongside semantic search to audit aerospace subcontracts for compliance risks and discrepancies.

## 🛠️ Tech Stack

* **Web Framework:** FastAPI (with automated Swagger UI docs)
* **AI & Embeddings:** OpenAI API (`gpt-4o` for analysis, `text-embedding-3-small` for vector embeddings)
* **Vector Database:** ChromaDB (Persistent local storage)
* **Environment Management:** Python-dotenv

---

## 📁 Project Directory Structure

```text
Contract-Compliance-RAG-Assistant/
│
├── .env                  # Environment variables (OpenAI API Key)
├── README.md             # Project documentation
├── app/
│   ├── __init__.py       # Python package marker
│   ├── main.py           # FastAPI web server and routing
│   └── rag_engine.py     # Core RAG logic, ChromaDB, and OpenAI integration
└── vector_db/            # Local directory where ChromaDB stores embeddings (auto-created)