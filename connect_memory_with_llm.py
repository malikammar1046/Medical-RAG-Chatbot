# connect_memory_with_llm.py
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from citation_utils import format_document_references

load_dotenv()

# -----------------------------
# Small-talk / intent detection
# -----------------------------
SMALL_TALK = [
    "hi", "hello", "hey",
    "how are you", "who are you",
    "what can you do", "thanks"
]

def is_small_talk(query: str) -> bool:
    q = query.lower().strip()
    return any(phrase in q for phrase in SMALL_TALK)

# -----------------------------
# GROQ LLM setup
# -----------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5,
    max_tokens=512,
    api_key=GROQ_API_KEY,
)

# -----------------------------
# Load FAISS vector store
# -----------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

# -----------------------------
# RAG Prompt (clean markdown output)
# -----------------------------
PROMPT_TEMPLATE = """You are MedBot, a professional medical information assistant developed by Team Amigos.
Your role is to provide accurate, well-structured answers based strictly on the provided context.

If the answer is not present in the context, respond with:
"I do not have enough information in the provided documents to answer this."

FORMATTING RULES:
- Use "## Section Title" for each major section heading (Overview, Key Points, Additional Details)
- Use "- " for bullet points under each section
- Keep bullet points concise — one idea per bullet
- Do NOT write long paragraphs
- Do NOT copy sentences verbatim from the source
- Do NOT use asterisks for bold or italics
- Do NOT add extra blank lines between bullet points
- Always start with the Overview section

OUTPUT FORMAT (follow this exactly):

## Overview
- [2-3 sentence summary as bullet points]

## Key Points
- [Point 1]
- [Point 2]
- [Point 3]

## Additional Details
- [Detail 1, if applicable]
- [Detail 2, if applicable]

Context:
{context}

Question:
{input}

Answer:
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "input"]
)

# -----------------------------
# Build RAG chain
# -----------------------------
combine_docs_chain = create_stuff_documents_chain(
    llm,
    prompt
)

rag_chain = create_retrieval_chain(
    db.as_retriever(search_kwargs={"k": 3}),
    combine_docs_chain
)

# -----------------------------
# Interactive loop
# -----------------------------
if __name__ == "__main__":
    while True:
        user_query = input("\nWrite Your Question Here (or 'exit'): ").strip()
        if user_query.lower() == "exit":
            break

        # Small talk → direct LLM
        if is_small_talk(user_query):
            reply = llm.invoke(user_query)
            print("\nRESULT:")
            print(reply.content)
            continue

        # Knowledge → RAG
        response = rag_chain.invoke({"input": user_query})

        print("\nRESULT:")
        print(response["answer"])

        print("\nSOURCES:")
        references = format_document_references(response.get("context", []))
        if not references:
            print("No relevant documents found.")
        else:
            for ref in references:
                print(f"- {ref['file']}, Page {ref['page']}")
