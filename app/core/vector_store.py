"""Vector store module for Qdrant operations."""

from functools import lru_cache
from typing import Any 
from uuid import uuid4

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import VectorParams, Distance

from app.utils.logger import get_logger
from app.core.embeddings import get_embeddings
from app.config import get_settings

logger=get_logger(__name__)
settings=get_settings()

EMBEDDING_DIMENSION=1536

@lru_cache

def get_qdrant_client() -> QdrantClient:
    """Get cached Qdrant client instance.

    Returns:
        Configured QdrantClient instance
    """
    logger.info(f"Connecting to qdrant at : {settings.QDRANT_URL}")
    client=QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )
    
    logger.info("Qdrant client connected successfully")
    return client

class VectoreStoreService:
    """Initialize vector store service.

        Args:
            collection_name: Name of the Qdrant collection (default from settings)
    """
    def __init__(self, collection_name : str | None = None):
        
        self.collection_name=collection_name or settings.COLLECTION_NAME
        self.client=get_qdrant_client()
        self.embeddings=get_embeddings()
        
        self._ensure_collection()
        
        self.vector_store=QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        
        logger.info(f"Vectore store initialized for collection : {self.collection_name}")
        
    def _ensure_collection(self) -> None:
        """Ensure the collection exists, create if not."""
        try:
            collection_info=self.client.get_collection(self.collection_name)
            logger.info(
                f"collection '{self.collection_name} exists with"
                f"{collection_info.points_count} points"
            )
        except UnexpectedResponse:
            logger.info(f"Creating collection : {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Collection '{self.collection_name}' Created Successfully")
        
    def add_documents(self, documents: list[Document]) -> list[str]:
        
        """Add documents to the vector store.

        Args:
            documents: List of Document objects to add

        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents to add")
            return []
        logger.info(f"Adding {len(documents)} documents to collection")
            
        # Generating unique ids for each documents
        ids=[str(uuid4()) for _ in documents]
            
        # Add to vector store
        self.vector_store.add_documents(documents, ids)
        logger.info(f"{len(documents)} documents added successfully")
        return ids
        
    def search(self, query : str, k : int | None = None) -> list[Document]:
        
        """Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return (default from settings)

        Returns:
            List of similar Document objects
        """
        k=k or settings.RETRIEVAL_K
        logger.debug(f"searching for : {query[:50]}... (k={k})")
        results=self.vector_store.similarity_search(query, k)
        logger.debug(f"found results {len(results)}")
        return results
        
    def search_with_scores(self, query : str, k : int | None = None) -> list[tuple[Document, float]]:
        """Search for similar documents with relevance scores.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of (Document, score) tuples
        """
        k=k or settings.RETRIEVAL_K
        logger.debug(f"Searching with scores : {query[:50]}... (k={k})")
        results=self.vector_store.similarity_search_with_score(query, k=k)
            
        logger.debug(f"found {len(results)} results with scores")
        return results
        
    def get_retriever(self, k : int | None=None) -> Any:
        """Get a retriever for the vector store.

        Args:
            k: Number of documents to retrieve

        Returns:
            LangChain retriever object
        """    
        k=k or settings.RETRIEVAL_K
        return self.vector_store.as_retriever(
            search_type='similarity',
            search_kwargs={'k':k},
        )
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        logger.warning(f"Deleting collection: {self.collection_name}")
        self.client.delete_collection(self.collection_name)
        logger.info(f"Collection '{self.collection_name}' deleted")
        
    def get_collection_info(self) -> dict:
        """Get information about the collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            info=self.client.get_collection(self.collection_name)
            return {
                'name': self.collection_name,
                'points_count': info.points_count,
                'indexed_vectors_count': info.indexed_vectors_count,
                'status': info.status.value,
            }
        except UnexpectedResponse:
            return {
                'name': self.collection_name,
                'points_count': 0,
                'indexed_vectors_count': 0,
                'status': 'Not found',
            }
            
    def health_check(self) -> bool:
        """Check if vector store is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return False
            
                
            
        
        