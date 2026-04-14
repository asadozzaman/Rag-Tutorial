"""
Day 1: Code Example — Your First RAG Pipeline
================================================
This script demonstrates the complete RAG loop:
  Load → Chunk → Embed → Store → Retrieve → Generate

Prerequisites:
  pip install langchain langchain-openai faiss-cpu python-dotenv

Setup:
  Create a .env file in the same directory with:
  OPENAI_API_KEY=sk-your-key-here
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

load_dotenv()

# ──────────────────────────────────────────────────────────────
# STEP 1: Prepare your "knowledge base"
# ──────────────────────────────────────────────────────────────
# In production, you'd load from files, databases, or APIs.
# Here we use hardcoded text so you can focus on the pipeline.

documents = [
    "RAG stands for Retrieval-Augmented Generation. It combines information "
    "retrieval with text generation to produce grounded, factual answers. "
    "The technique was introduced by Facebook AI Research in 2020.",

    "Vector databases like FAISS and Chroma store document embeddings as "
    "high-dimensional vectors, enabling fast similarity search. FAISS was "
    "developed by Facebook AI Research and is widely used for its speed.",

    "Chunking is the process of splitting large documents into smaller pieces. "
    "Good chunking preserves semantic meaning and improves retrieval accuracy. "
    "Common strategies include fixed-size, recursive, and semantic chunking.",

    "LLMs can hallucinate — generating plausible but incorrect information — "
    "especially when they lack domain-specific knowledge. RAG mitigates this "
    "by providing relevant source documents as context at generation time.",

    "The embedding model converts text into numerical vectors. Similar texts "
    "produce vectors that are close together in vector space, enabling "
    "semantic search rather than simple keyword matching.",

    "In a RAG pipeline, the retriever fetches the top-K most relevant chunks "
    "for a given query. K=2 to K=5 is typical for most use cases. Too many "
    "chunks can overwhelm the LLM context; too few may miss relevant info.",
]

print("=" * 60)
print("  DAY 1: YOUR FIRST RAG PIPELINE")
print("=" * 60)
print(f"\nKnowledge base: {len(documents)} documents\n")

# ──────────────────────────────────────────────────────────────
# STEP 2: Chunk the documents
# ──────────────────────────────────────────────────────────────
# RecursiveCharacterTextSplitter tries to split on natural
# boundaries (paragraphs → sentences → words) to keep chunks
# semantically coherent.

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,       # Max characters per chunk
    chunk_overlap=30,     # Overlap between chunks to preserve context
)
splits = splitter.create_documents(documents)
print(f"[CHUNK]  Created {len(splits)} chunks from {len(documents)} documents")

# Let's inspect what chunking actually does:
print(f"         Average chunk size: {sum(len(s.page_content) for s in splits) // len(splits)} chars")
print(f"         Smallest chunk: {min(len(s.page_content) for s in splits)} chars")
print(f"         Largest chunk: {max(len(s.page_content) for s in splits)} chars\n")

# ──────────────────────────────────────────────────────────────
# STEP 3: Embed and store in FAISS
# ──────────────────────────────────────────────────────────────
# The embedding model converts each chunk into a 1536-dimensional
# vector. FAISS indexes these for fast similarity search.

print("[EMBED]  Creating embeddings with text-embedding-3-small...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(splits, embeddings)
print("[STORE]  Vector store created with FAISS\n")

# ──────────────────────────────────────────────────────────────
# STEP 4: Build the retrieval chain
# ──────────────────────────────────────────────────────────────
# RetrievalQA ties retriever + LLM into one callable chain.
# When you invoke it, it:
#   1. Embeds your query
#   2. Retrieves top-K similar chunks
#   3. Builds a prompt with those chunks as context
#   4. Sends it to the LLM for generation

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,  # So we can inspect what was retrieved
)
print("[CHAIN]  RetrievalQA chain ready (k=2)\n")

# ──────────────────────────────────────────────────────────────
# STEP 5: Ask questions and see the full RAG loop in action
# ──────────────────────────────────────────────────────────────
queries = [
    "Why does RAG help with hallucination?",
    "What is a vector database and how does it work?",
    "How does chunking affect retrieval quality?",
]

print("=" * 60)
print("  QUERYING THE RAG PIPELINE")
print("=" * 60)

for i, query in enumerate(queries, 1):
    result = qa_chain.invoke({"query": query})

    print(f"\n{'─' * 60}")
    print(f"  Question {i}: {query}")
    print(f"{'─' * 60}")
    print(f"\n  Answer: {result['result']}\n")

    print(f"  Retrieved {len(result['source_documents'])} source chunks:")
    for j, doc in enumerate(result["source_documents"]):
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"    [{j+1}] \"{preview}...\"")

print(f"\n{'=' * 60}")
print("  DONE! You've built your first RAG pipeline.")
print("=" * 60)


# ──────────────────────────────────────────────────────────────
# BONUS: Try manual retrieval to see what the retriever returns
# ──────────────────────────────────────────────────────────────
print("\n\n--- BONUS: Manual Retrieval (no LLM) ---\n")

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
manual_query = "What is an embedding?"
retrieved_docs = retriever.invoke(manual_query)

print(f"Query: \"{manual_query}\"")
print(f"Retrieved {len(retrieved_docs)} chunks:\n")
for i, doc in enumerate(retrieved_docs):
    print(f"  Chunk {i+1}:")
    print(f"  {doc.page_content}")
    print()
