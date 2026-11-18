import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader

# Internal RAG engine
from src.rag.rag_engine import RAGEngine


class DocumentProcessor:
    def __init__(
        self,
        raw_data_dir: str = "data/raw",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initializes the document processor with chunking parameters.
        """
        self.raw_data_dir = raw_data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Configure splitter optimized for sports science texts and manuals
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_raw_documents(self) -> List[Document]:
        """
        Loads all text (.txt) and PDF (.pdf) files from the data/raw directory safely.
        """
        if not os.path.exists(self.raw_data_dir):
            os.makedirs(self.raw_data_dir, exist_ok=True)
            print(f"[DocumentProcessor] Created missing directory: '{self.raw_data_dir}'. Place your raw files here.")
            return []

        documents = []

        # Load Text files (.txt) with explicit error handling for empty directories
        if any(f.endswith(".txt") for f in os.listdir(self.raw_data_dir)):
            try:
                txt_loader = DirectoryLoader(
                    self.raw_data_dir,
                    glob="**/*.txt",
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"}
                )
                txt_docs = txt_loader.load()
                documents.extend(txt_docs)
            except Exception as e:
                print(f"[DocumentProcessor] Note on text loading: {e}")

        # Load PDF files (.pdf - ACSM/NSCA manuals, textbooks)
        if any(f.lower().endswith(".pdf") for f in os.listdir(self.raw_data_dir)):
            try:
                pdf_loader = DirectoryLoader(
                    self.raw_data_dir,
                    glob="**/*.pdf",
                    loader_cls=PyPDFLoader
                )
                pdf_docs = pdf_loader.load()
                documents.extend(pdf_docs)
            except Exception as e:
                print(f"[DocumentProcessor] Note on PDF loading: {e}")

        print(f"[DocumentProcessor] Successfully loaded {len(documents)} raw document(s) from '{self.raw_data_dir}'.")
        return documents

    def process_and_index(self, rag_engine: RAGEngine) -> int:
        """
        Loads raw documents, splits them into semantic chunks, and indexes them into ChromaDB.
        """
        raw_docs = self.load_raw_documents()
        if not raw_docs:
            print("[DocumentProcessor] No raw documents found to process. Add .txt or .pdf files to data/raw/")
            return 0

        # Split documents into manageable chunks
        chunked_docs = self.text_splitter.split_documents(raw_docs)
        print(f"[DocumentProcessor] Split raw documents into {len(chunked_docs)} semantic chunks.")

        # Index chunks into ChromaDB
        rag_engine.add_documents(chunked_docs)
        return len(chunked_docs)


if __name__ == "__main__":
    # Execution entry point for ETL pipeline
    print("Starting Document Ingestion & Vector Indexing Pipeline...")
    
    rag = RAGEngine()
    processor = DocumentProcessor(raw_data_dir="data/raw")
    
    total_indexed = processor.process_and_index(rag)
    print(f"Pipeline finished! Total chunks indexed in ChromaDB: {total_indexed}")
    print(f"Total collection count: {rag.get_collection_count()}")
