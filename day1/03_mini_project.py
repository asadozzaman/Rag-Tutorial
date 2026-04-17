"""
Day 1 Mini Project: "Explain Like I'm 5" RAG Bot
================================================
Loads a Wikipedia article on Quantum Computing, chunks it,
embeds it into FAISS, and answers questions in simple language
a 5-year-old would understand.

Prerequisites:
  pip install langchain langchain-google-genai langchain-community faiss-cpu
  pip install python-dotenv beautifulsoup4 lxml

Setup:
  Create a .env file with:
  GEMINI_API_KEY=your-gemini-api-key
"""

import os

import requests
from bs4 import BeautifulSoup
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


API_KEY = configure_gemini_api()


def load_wikipedia_article(url: str):
    """Load and return documents from a Wikipedia URL."""
    print(f"Loading article from: {url}")
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": os.environ.get("USER_AGENT", "day1-rag-learning/1.0")},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    content_root = soup.select_one("#mw-content-text") or soup
    paragraphs = [
        paragraph.get_text(" ", strip=True)
        for paragraph in content_root.select("p")
        if paragraph.get_text(" ", strip=True)
    ]
    article_text = "\n\n".join(paragraphs)

    print(f"Loaded 1 page(s), {len(article_text):,} characters")
    return [article_text]


def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    """Split documents into overlapping text chunks."""
    chunks = []
    step = max(1, chunk_size - chunk_overlap)

    for doc in docs:
        text = doc
        start = 0
        while start < len(text):
            chunk = text[start : start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            if start + chunk_size >= len(text):
                break
            start += step

    print(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def create_vectorstore(chunks, save_path="faiss_eli5_index_gemini"):
    """Embed chunks and store in FAISS. Saves index to disk."""
    print("Creating embeddings (this may take a moment)...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        api_key=API_KEY,
    )
    vectorstore = FAISS.from_texts(chunks, embeddings)
    vectorstore.save_local(save_path)
    print(f"Index saved to ./{save_path}/")
    return vectorstore


def load_vectorstore(save_path="faiss_eli5_index_gemini"):
    """Load a previously saved FAISS index."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        api_key=API_KEY,
    )
    vectorstore = FAISS.load_local(
        save_path, embeddings, allow_dangerous_deserialization=True
    )
    print(f"Loaded existing index from ./{save_path}/")
    return vectorstore


def build_eli5_chain(vectorstore):
    """Build the retriever + Gemini model pair for ELI5 answers."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        api_key=API_KEY,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return {"llm": llm, "retriever": retriever}


def answer_like_eli5(llm, retrieved_docs, question):
    """Generate a simple answer grounded in the retrieved context."""
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = f"""You are a friendly, patient teacher explaining things to a curious 5-year-old.

Rules:
- Use simple words and short sentences
- Use fun analogies and comparisons to everyday things
- Avoid jargon - if you must use a big word, explain it right away
- Keep answers to 3-5 sentences max
- If the context doesn't contain the answer, say "Hmm, I don't see that in my notes!"

Context from the article:
{context}

Question: {question}

Simple explanation:"""
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


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
        retrieved_docs = chain["retriever"].invoke(q)
        answer = answer_like_eli5(chain["llm"], retrieved_docs, q)
        print(f"\n  Q: {q}")
        print(f"  A: {answer}")
        print(f"     (Used {len(retrieved_docs)} source chunks)")
        print("-" * 55)


def run_interactive_mode(chain):
    """Let the user ask their own questions."""
    print("\n" + "=" * 55)
    print("  INTERACTIVE MODE - Ask anything!")
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

        retrieved_docs = chain["retriever"].invoke(user_q)
        answer = answer_like_eli5(chain["llm"], retrieved_docs, user_q)
        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    os.environ.setdefault("USER_AGENT", "day1-rag-learning/1.0")

    print("=" * 55)
    print('  ELI5 RAG Bot - Quantum Computing Edition')
    print("=" * 55 + "\n")

    index_path = "faiss_eli5_index_gemini"

    if os.path.exists(index_path):
        vectorstore = load_vectorstore(index_path)
    else:
        raw_docs = load_wikipedia_article("https://en.wikipedia.org/wiki/Quantum_computing")
        chunks = chunk_documents(raw_docs, chunk_size=500, chunk_overlap=50)
        vectorstore = create_vectorstore(chunks, save_path=index_path)

    chain = build_eli5_chain(vectorstore)
    print("\nELI5 chain ready!\n")

    run_demo_questions(chain)
    run_interactive_mode(chain)
