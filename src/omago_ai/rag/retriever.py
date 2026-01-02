# rag/retriever.py

from .index import build_vectorstore

def retrieve_news_docs(query: str, k: int = 4):
    store = build_vectorstore()
    results = store.similarity_search(query, k=k)
    return results   # 👈 return full Document objects
