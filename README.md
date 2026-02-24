Annex 1 for UNICAF Dissertation module

This project is a research prototype of a curriculum-aware ICT tutor. It runs locally using Ollama for inference and a FAISS cosine index for retrieval-augmented generation (RAG). A controller orchestrates lesson flow, chapter progression, and an assessor-style evaluation call.

What’s included

Inside the project folder:

scripts/tutor_controller_chat.py — main controller + tutor session logic

scripts/run_tutor_controller_chat.py — terminal runner (loads FAISS + embeddings, starts chat)



Requirements
Programs

Python 3.10+

Ollama installed and running locally

Ollama model

The runner uses a local model string (as configured in code). Make sure the model exists in Ollama (example below):

ollama pull llama3.2:3b-instruct-q4_K_M

Start Ollama if needed:

ollama serve
Python dependencies

Create and activate a virtual environment, then install:

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install faiss-cpu sentence-transformers ollama numpy

Building the index

Only do this once.

scripts/extract_text.py, scripts/prepare_text.py — text extraction/cleaning helpers

scripts/chunk_text_heading_metadata_idea.py — chunking script

scripts/build_index_cosine_idea.py — index builder

scripts/test_retrieval_idea.py — retrieval sanity test

Run Tutor (terminal)

run run_tutor_controller_chat.py

This will:

load the FAISS index from ../data/index_metadata_cosine_idea/

use sentence-transformers for query embeddings

start a terminal chat loop (you type learner messages; the tutor responds)
