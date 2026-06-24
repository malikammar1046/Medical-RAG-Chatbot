# app.py
import os
import json
import tempfile

from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from citation_utils import format_document_references

load_dotenv()

app = Flask(__name__, static_folder="templates", static_url_path="")

# --------------------------------------------------
# Config
# --------------------------------------------------
DB_FAISS_PATH   = "vectorstore/db_faiss"
DOCS_META_PATH  = "vectorstore/documents.json"   # persists doc names across restarts
UPLOAD_FOLDER   = "uploaded_docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("vectorstore", exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf"}

# --------------------------------------------------
# Shared resources (loaded once)
# --------------------------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5,
    max_tokens=512,
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Load or initialise vector store
if os.path.exists(DB_FAISS_PATH):
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
else:
    db = None   # no docs yet — upload required before querying

# Load or initialise doc metadata list
if os.path.exists(DOCS_META_PATH):
    with open(DOCS_META_PATH, "r") as f:
        loaded_docs: list[dict] = json.load(f)
else:
    # Seed from any files already present in UPLOAD_FOLDER
    loaded_docs = []
    for fname in os.listdir(UPLOAD_FOLDER):
        if fname.lower().endswith(".pdf"):
            loaded_docs.append({"name": fname, "pages": "?", "source": "preloaded"})

def save_docs_meta():
    with open(DOCS_META_PATH, "w") as f:
        json.dump(loaded_docs, f, indent=2)

# --------------------------------------------------
# RAG prompt
# --------------------------------------------------
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

# --------------------------------------------------
# Small-talk detection
# --------------------------------------------------
SMALL_TALK_PHRASES = ["hi", "hello", "hey", "how are you", "who are you",
                      "what can you do", "thanks", "thank you"]

def is_small_talk(query: str) -> bool:
    q = query.lower().strip()
    return any(phrase in q for phrase in SMALL_TALK_PHRASES)

# --------------------------------------------------
# Helper: build RAG chain from current db
# --------------------------------------------------
def build_rag_chain():
    combine_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(
        db.as_retriever(search_kwargs={"k": 3}),
        combine_chain
    )

# --------------------------------------------------
# Helper: process an uploaded PDF into FAISS
# --------------------------------------------------
def process_pdf(filepath: str, filename: str) -> int:
    """Loads, chunks, embeds a PDF and merges it into the global db.
    Returns the number of pages processed."""
    global db

    loader = PyPDFLoader(filepath)
    pages  = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)

    # Tag every chunk with the original filename so sources are accurate
    for chunk in chunks:
        chunk.metadata["source"] = filename

    new_store = FAISS.from_documents(chunks, embedding_model)

    if db is None:
        db = new_store
    else:
        db.merge_from(new_store)

    # Persist the updated index
    db.save_local(DB_FAISS_PATH)

    return len(pages)

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/documents", methods=["GET"])
def get_documents():
    """Return the list of all loaded documents."""
    return jsonify({"documents": loaded_docs})


@app.route("/upload", methods=["POST"])
def upload_document():
    """Accept a PDF, embed it, merge into FAISS, return updated doc list."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Only PDF files are supported"}), 400

    filename = file.filename

    # Check for duplicates
    existing_names = [d["name"] for d in loaded_docs]
    if filename in existing_names:
        return jsonify({"error": f'"{filename}" is already loaded.'}), 409

    # Save to disk
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    try:
        page_count = process_pdf(save_path, filename)
    except Exception as e:
        os.remove(save_path)
        return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500

    # Record metadata
    doc_entry = {"name": filename, "pages": page_count, "source": "uploaded"}
    loaded_docs.append(doc_entry)
    save_docs_meta()

    return jsonify({
        "message": f'"{filename}" uploaded and indexed successfully.',
        "document": doc_entry,
        "documents": loaded_docs
    })


@app.route("/ask", methods=["POST"])
def ask():
    """Answer a question using RAG over all loaded documents."""
    data     = request.get_json(force=True)
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Small talk — bypass RAG
    if is_small_talk(question):
        reply = llm.invoke(question)
        return jsonify({"answer": reply.content, "sources": []})

    if db is None:
        return jsonify({
            "answer": "No documents have been loaded yet. Please upload a PDF first.",
            "sources": []
        })

    try:
        rag_chain = build_rag_chain()
        response  = rag_chain.invoke({"input": question})
        answer    = response["answer"]

        sources = format_document_references(response.get("context", []))

        return jsonify({"answer": answer, "sources": sources})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
