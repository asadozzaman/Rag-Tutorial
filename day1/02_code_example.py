"""
Day 1: Code Example - Your First RAG Pipeline
=============================================
This script demonstrates the complete RAG loop:
  Load -> Chunk -> Embed -> Store -> Retrieve -> Generate

Prerequisites:
  pip install langchain langchain-google-genai faiss-cpu python-dotenv

Setup:
  Create a .env file in the same directory with:
  GEMINI_API_KEY=your-gemini-api-key
"""

import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

load_dotenv()


def configure_gemini_api() -> str:
    """Read the Gemini API key from the environment."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Missing API key. Set GEMINI_API_KEY in your environment or .env file."
        )

    return api_key


def chunk_texts(texts, chunk_size=300, chunk_overlap=30):
    """Split strings into overlapping chunks without extra dependencies."""
    chunks = []
    step = max(1, chunk_size - chunk_overlap)

    for text in texts:
        start = 0
        while start < len(text):
            chunk = text[start : start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            if start + chunk_size >= len(text):
                break
            start += step

    return chunks


def answer_with_context(llm, retrieved_docs, query):
    """Generate an answer grounded only in the retrieved chunks."""
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = f"""You are a helpful RAG assistant.

Use only the context below to answer the question. If the answer is not in the
context, say so clearly.

Context:
{context}

Question: {query}

Answer:"""
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


API_KEY = configure_gemini_api()

# STEP 1: Prepare your knowledge base.
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
    "LLMs can hallucinate, generating plausible but incorrect information, "
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

# STEP 2: Chunk the documents.
splits = chunk_texts(
    documents,
    chunk_size=300,
    chunk_overlap=30,
)
print(f"[CHUNK]  Created {len(splits)} chunks from {len(documents)} documents")
print(f"         Average chunk size: {sum(len(s) for s in splits) // len(splits)} chars")
print(f"         Smallest chunk: {min(len(s) for s in splits)} chars")
print(f"         Largest chunk: {max(len(s) for s in splits)} chars\n")

# STEP 3: Embed and store in FAISS.
print("[EMBED]  Creating embeddings with models/gemini-embedding-001...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    api_key=API_KEY,
)
vectorstore = FAISS.from_texts(splits, embeddings)
print("[STORE]  Vector store created with FAISS\n")

# STEP 4: Build the retriever + LLM.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=API_KEY,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
print("[CHAIN]  Retriever and Gemini model ready (k=2)\n")

# STEP 5: Ask questions and see the full RAG loop in action.
queries = [
    "Why does RAG help with hallucination?",
    "What is a vector database and how does it work?",
    "How does chunking affect retrieval quality?",
]

print("=" * 60)
print("  QUERYING THE RAG PIPELINE")
print("=" * 60)

for i, query in enumerate(queries, 1):
    retrieved_docs = retriever.invoke(query)
    answer = answer_with_context(llm, retrieved_docs, query)

    print(f"\n{'-' * 60}")
    print(f"  Question {i}: {query}")
    print(f"{'-' * 60}")
    print(f"\n  Answer: {answer}\n")

    print(f"  Retrieved {len(retrieved_docs)} source chunks:")
    for j, doc in enumerate(retrieved_docs, 1):
        preview = doc.page_content[:100].replace("\n", " ")
        print(f'    [{j}] "{preview}..."')

print(f"\n{'=' * 60}")
print("  DONE! You've built your first RAG pipeline.")
print("=" * 60)

# BONUS: Try manual retrieval to see what the retriever returns.
print("\n\n--- BONUS: Manual Retrieval (no LLM) ---\n")

manual_query = "What is an embedding?"
retrieved_docs = retriever.invoke(manual_query)

print(f'Query: "{manual_query}"')
print(f"Retrieved {len(retrieved_docs)} chunks:\n")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"  Chunk {i}:")
    print(f"  {doc.page_content}")
    print()
