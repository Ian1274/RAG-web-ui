from typing import List, Any
from pymilvus import Collection, Connections, utility
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_milvus.vectorstores.milvus import Milvus
from app.core.config import settings

from .base import BaseVectorStore


class MilvusVectorStore(BaseVectorStore):
    """Milvus vector store implementation"""
    
    def __init__(self, collection_name: str, embedding_function: Embeddings, **kwargs):
        """Initialize Milvus vector store"""
        self._client = Connections()
        self._client.connect(
                alias="default",
                uri=settings.MILVUS_URL,
            )
        self._store = Milvus(
            collection_name=collection_name,
            embedding_function=embedding_function,
            connection_args={"uri": settings.MILVUS_URL},
            auto_id=True,
            enable_dynamic_field=True
        )

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to Milvus"""
        self._store.add_documents(documents)
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents from Milvus"""
        self._store.delete(ids)
    
    def as_retriever(self, **kwargs: Any):
        """Return a retriever interface"""
        return self._store.as_retriever(**kwargs)
    
    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """Search for similar documents in Milvus"""
        return self._store.similarity_search(query, k=k, **kwargs)
    
    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """Search for similar documents in Milvus with score"""
        return self._store.similarity_search_with_score(query, k=k, **kwargs)

    def delete_collection(self, kb_name: str) -> None:
        """Delete the entire collection"""
        if utility.has_collection(kb_name):
                utility.drop_collection(kb_name)