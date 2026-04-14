"""
Day 1 Mini Project: "Explain Like I'm 5" RAG Bot
==================================================
Loads a Wikipedia article on Quantum Computing, chunks it,
embeds it into FAISS, and answers questions in simple language
a 5-year-old would understand.

Prerequisites:
  pip install langchain langchain-openai langchain-community faiss-cpu
  pip install python-dotenv beautifulsoup4 lxml

Setup:
  Create a .env file with:
  OPENAI_API_KEY=sk-your-key-here
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()


def load_wikipedia_article(url: str):
    """Load and return documents from a Wikipedia URL."""
    print(f"Loading article from: {url}")
    loader = WebBaseLoader(url)
    docs = loader.load()
    char_count = len(docs[0].page_content)
    print(f"Loaded {len(docs)} page(s), {char_count:,} characters")
    return docs


def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def create_vectorstore(chunks, save_path="faiss_eli5_index"):
    """Embed chunks and store in FAISS. Saves index to disk."""
    print("Creating embeddings (this may take a moment)...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save so we don't re-embed every run
    vectorstore.save_local(save_path)
    print(f"Index saved to ./{save_path}/")
    return vectorstore


def load_vectorstore(save_path="faiss_eli5_index"):
    """Load a previously saved FAISS index."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        save_path, embeddings, allow_dangerous_deserialization=True
    )
    print(f"Loaded existing index from ./{save_path}/")
    return vectorstore


def build_eli5_chain(vectorstore):
    """Build a RetrievalQA chain with an ELI5 system prompt."""
    eli5_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a friendly, patient teacher explaining things to a curious 5-year-old.

Rules:
- Use simple words and short sentences
- Use fun analogies and comparisons to everyday things
- Avoid jargon — if you must use a big word, explain it immediately
- Keep answers to 3-5 sentences max
- If the context doesn't contain the answer, say "Hmm, I don't see that in my notes!"

Context from the article:
{context}

Question: {question}

Simple explanation:""",
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": eli5_prompt},
    )
    return chain


def run_demo_questions(chain):
    """Run a set of demo questions to show the bot working."""
    demo_questions = [
        "What is quantum computing?",
        "What is a qubit?",
        "Why is quantum computing faster than regular computers?",
        "Who invented quantum computing?",
        "What problems can quantum computers solve?",
    ]

    print("\n" + "=" * 55)
    print("  DEMO: Watch the ELI5 bot answer questions")
    print("=" * 55)

    for q in demo_questions:
        result = chain.invoke({"query": q})
        print(f"\n  Q: {q}")
        print(f"  A: {result['result']}")
        print(f"     (Used {len(result['source_documents'])} source chunks)")
        print("-" * 55)


def run_interactive_mode(chain):
    """Let the user ask their own questions."""
    print("\n" + "=" * 55)
    print("  INTERACTIVE MODE — Ask anything!")
    print("  Type 'quit' to exit")
    print("=" * 55 + "\n")

    while True:
        try:
            user_q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_q.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if not user_q:
            continue

        result = chain.invoke({"query": user_q})
        print(f"Bot: {result['result']}\n")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  ELI5 RAG Bot — Quantum Computing Edition")
    print("=" * 55 + "\n")

    INDEX_PATH = "faiss_eli5_index"

    # Check if we already have a saved index
    if os.path.exists(INDEX_PATH):
        vectorstore = load_vectorstore(INDEX_PATH)
    else:
        # Step 1: Load
        raw_docs = load_wikipedia_article(
            "https://en.wikipedia.org/wiki/Quantum_computing"
        )

        # Step 2: Chunk
        chunks = chunk_documents(raw_docs, chunk_size=500, chunk_overlap=50)

        # Step 3: Embed & Store
        vectorstore = create_vectorstore(chunks, save_path=INDEX_PATH)

    # Step 4: Build chain
    chain = build_eli5_chain(vectorstore)
    print("\nELI5 chain ready!\n")

    # Step 5: Run
    run_demo_questions(chain)
    run_interactive_mode(chain)
