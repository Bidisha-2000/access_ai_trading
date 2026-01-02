# rag/index.py

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .docs import RAG_DOCUMENTS
from .loader import fetch_latest_news
from .chunking import chunk_text
from dotenv import load_dotenv
import os
load_dotenv()
_vectorstore = None

api_key=os.getenv('MY_TOKEN')

def build_vectorstore():
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )

    documents = []

    # 1️⃣ Static RAG docs
    for d in RAG_DOCUMENTS:
        documents.append(
            Document(
                page_content=d["text"],
                metadata=d.get("metadata", {})
            )
        )

    # 2️⃣ Latest news (chunked)
    for news in fetch_latest_news():
        chunks = chunk_text(news["content"])

        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"type": "news", "title": news["title"]},
                )
            )

    _vectorstore = FAISS.from_documents(documents, embeddings)

    return _vectorstore


