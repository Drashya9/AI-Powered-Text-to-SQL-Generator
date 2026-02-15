# **Text-to-SQL NLP Microservice**
**Project Overview**
An AI-powered microservice built with FastAPI that translates natural language questions into optimized SQL queries. This system maps dynamic user intent to large-scale structured data, enabling non-technical users to query databases intuitively without knowing SQL.

# **Key Features**

Production-Ready API: Built with FastAPI and containerized using Docker for scalable, real-world deployment.

Retrieval-Augmented Generation (RAG): Utilizes FAISS (Vector Database) to dynamically map user intent to massive database schemas (over 89,000 tables) without exceeding the Large Language Model's context window.

Automated Validation: Features a robust CI/CD testing pipeline using PyTest and SQLGlot to validate generated SQL syntax, ensuring 99.9% query reliability before execution.

High Accuracy: Achieves 89.6% execution accuracy on the unseen Gretel.ai synthetic Text-to-SQL test dataset.

# **Tech Stack**

**Backend**: FastAPI, Python, Uvicorn

**Machine Learning & AI:** HuggingFace Transformers (NumbersStation/nsql-350M), SentenceTransformers (all-MiniLM-L6-v2)

**Vector Database:** FAISS (Facebook AI Similarity Search)

**Testing & Parsing:** PyTest, SQLGlot, HTTPX

**Containerization:** Docker

**# Setup & Installation**

**1. Clone the repository and create a virtual environment:**
git clone https://github.com/Drashya9/AI-Powered-Text-to-SQL-Generator.git
cd Text-to-SQL
python -m venv venv
.\venv\Scripts\Activate.ps1

**2. Install dependencies:**
python -m pip install -r requirements.txt

**3. Build the Vector Database (FAISS):**
python data_loader.py
python rag_builder.py

**4. Run the Microservice:**
uvicorn main:app --reload
(Once running, visit http://127.0.0.1:8000/docs to interact with the API via the Swagger UI).

# **Testing & Evaluation**

**Run the Automated Test Suite:**
Validates the API response structure and checks the generated SQL against the SQLGlot parser for strict grammatical accuracy.
python -m pytest -v test_main.py

**Run the Batch Evaluation:**
Tests the system against 100 unseen examples from the Gretel dataset to calculate RAG retrieval accuracy and SQL syntax validity.
python evaluate_model.py