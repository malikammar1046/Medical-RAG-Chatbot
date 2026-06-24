import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from citation_utils import format_document_references, format_references_markdown

# ------------------------
# Setup
# ------------------------
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"

DB_FAISS_PATH = "vectorstore/db_faiss"

st.set_page_config(page_title="Medical Chatbot")

# ------------------------
# Cached resources
# ------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=512,
        api_key=os.environ.get("GROQ_API_KEY"),
    )

# ------------------------
# Prompt
# ------------------------
PROMPT_TEMPLATE = """
You are a professional assistant named "MedBot", specialized in providing accurate and concise information based on the provided context from various documents.
You are developed by Team Amigos.

Answer the question using ONLY the provided context.
If the answer is not present, say:
"I do not know based on the provided documents."

Structure your response naturally with appropriate headings when helpful, but feel free to write in complete sentences and paragraphs rather than strict bullet points.

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

# ------------------------
# UI
# ------------------------
st.title("🩺 Medical Chatbot")

query = st.chat_input("Ask a medical question")

if query:
    st.chat_message("user").markdown(query)

    try:
        vectorstore = load_vectorstore()
        llm = load_llm()

        combine_docs_chain = create_stuff_documents_chain(
            llm,
            prompt
        )

        rag_chain = create_retrieval_chain(
            vectorstore.as_retriever(search_kwargs={"k": 3}),
            combine_docs_chain
        )

        response = rag_chain.invoke({"input": query})
        answer = response["answer"]
        references = format_document_references(response.get("context", []))
        references_markdown = format_references_markdown(references)

        with st.chat_message("assistant"):
            st.markdown(answer)
            if references_markdown:
                st.markdown(references_markdown)

    except Exception as e:
        st.error(str(e))
