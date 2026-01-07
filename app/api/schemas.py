"""Pydantic schemas for API request/response models."""

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field

# ============== Health Schemas ==============

class HealthResponse(BaseModel):
    """Health check response."""
    status:str=Field(..., description="Service Status")
    timestamp: datetime=Field(
        default_factory=datetime.utcnow,
        description='Response Timestamp',
    )
    version: str=Field(..., description="Application Version")
    
class ReadinessResponse(BaseModel):
    """Readiness check response."""
    status: str=Field(..., description="Service Status")
    qdrant_connected: bool=Field(..., description="Qdrant Connection Status")
    collection_info: dict=Field(..., description="Collection Information")
    
# ============== Document Schemas ==============

class DocumentUploadResponse(BaseModel):
    """Response after document upload."""
    message: str=Field(..., description="Status Message")
    filename: str=Field(..., description="Uploaded Filename")
    chunks_created: int=Field(..., description="Number of Chunks Created")
    document_ids: list[str]=Field(..., description="List of Document Ids")
    
class DocumentInformation(BaseModel):
    """Document information."""
    source: str=Field(..., description="Document Source/Filename")
    metadata: dict[str, Any]=Field(
        default_factory=dict,
        description="Document Metadata",
    )
    
class DocumentListResponse(BaseModel):
    """Response for listing documents."""
    collection_name: str=Field(..., description="Collection Name")
    total_documents: int=Field(..., description="Total Document Count")
    status: str=Field(..., description="Collection Status")
    
# ============== Query Schemas ==============