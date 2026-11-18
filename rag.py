"""
RAG Vector Retriever & Indexing Engine for the AI Personal Trainer.
Manages ChromaDB persistence, local HuggingFace embeddings, and semantic retrieval
of sports science, biomechanics, hypertrophy, and nutrition guidelines.
"""

import os
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


class RAGEngine:
    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "sports_science_kb",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initializes the ChromaDB vector store and local embedding model.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Local, lightweight embedding model (no API key required for vectorization)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # Initialize or load existing ChromaDB collection
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def add_documents(self, documents: List[Document]) -> None:
        """
        Indexes a list of LangChain Document chunks into ChromaDB.
        """
        if not documents:
            print("[RAGEngine] Warning: No documents provided for indexing.")
            return

        self.vector_store.add_documents(documents)
        print(f"[RAGEngine] Successfully indexed {len(documents)} chunks into '{self.collection_name}'.")

    def retrieve(self, query: str, top_k: int = 4, filter_metadata: Optional[dict] = None) -> List[str]:
        """
        Performs semantic similarity search and returns the top-k relevant text chunks.
        Called directly by PersonalTrainerAgent.chat().
        """
        try:
            results = self.vector_store.similarity_search(
                query=query,
                k=top_k,
                filter=filter_metadata
            )
            return [doc.page_content for doc in results]
        except Exception as e:
            print(f"[RAGEngine] Retrieval error: {e}")
            return []

    def get_collection_count(self) -> int:
        """Returns the total number of indexed chunks currently stored in ChromaDB."""
        return self.vector_store._collection.count()


if __name__ == "__main__":
    # Quick diagnostic test
    rag = RAGEngine()
    total_chunks = rag.get_collection_count()
    print(f"Total indexed chunks in ChromaDB: {total_chunks}")

    if total_chunks > 0:
        sample_query = "hypertrophy training volume and RIR autoregulation"
        docs = rag.retrieve(sample_query, top_k=2)
        for i, doc in enumerate(docs, 1):
            print(f"\n--- Result {i} ---\n{doc[:300]}...")
    else:
        print("Database is empty. Run your ingestion pipeline (processors.py) to populate ChromaDB.")