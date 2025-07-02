from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import logging

class VectorStoreManager:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", logger: logging=None):
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        self.vectorstore = None
        self.logger = logger

    def create_store(self, documents):
        self.vectorstore = FAISS.from_documents(documents, self.embedding_model)

    def get_retriever(self, k=5):
        if not self.vectorstore:
            self.logger.error("Vector store not initialized. Call create_store first.")
            raise ValueError("Vector store not initialized. Call create_store first.")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
