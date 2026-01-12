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
    
class DocumentInfo(BaseModel):
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

class QueryRequest(BaseModel):
    """Request for RAG query."""
    question: str = Field(
        ...,
        description="Question to Ask",
        min_length=1,
        max_length=1000,
    )
    
    include_sources: bool=Field(
        default=True,
        description="Include Source documents in response",
    )
    
    enable_evaluation: bool=Field(
        default=False,
        description="Enable RAGAS Evaluation (faithfulness, answer relevancy)",
    )
    
    model_config={
        "json_schema_extra":{
            "examples": [
                {
                    question: "what is RAG",
                    include_sources: True,
                    enable_evaluation: False,
                }
            ]
        }
    }
    
class SourceDocument(BaseModel):
    """Source document information."""
    content: str=Field(..., description="Document Content Information")
    metadata: dict[str, Any]=Field(...,description="Document Metadata")
    
    
class EvaluationScores(BaseModel):
    """RAGAS evaluation scores."""
    faithfulness: float | None =Field(
        None,
        description="Faithfulness Score (0-1): measures factual consistency with sources",
        ge=0.0,
        le=1.0,
    )
    
    answer_relevancy: float | None=Field(
        None, 
        description="Answer Relevancy Score (0-1): measures relevance to question",
        ge=0.0,
        le=1.0,
    )
    
    evaluation_time_ms: float | None =Field(
        None,
        description="Time taken for evaluation in milliseconds",
    )
    
    error: str | None = Field(
        None,
        description="Error message if evaluation Failed",
    )
    
    
class QueryResponse(BaseModel):
    
    question: str =Field(..., description="Original question")
    answer: str=Field(..., description="Generated answer")
    sources: list[SourceDocument] | None=Field(
        None,
        description="Source document used",
    )
    
    processing_time_ms: float =Field(
        ...,
        description="Query processing time in milliseconds",
    )
    
    evaluation: EvaluationScores | None =Field(
        None, 
        description="RAGAS Evaluation scores (if requested)",
    )



# ============== Error Schemas ==============


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Detailed error information")


class ValidationErrorResponse(BaseModel):
    """Validation error response."""

    error: str = Field(default="Validation Error", description="Error type")
    message: str = Field(..., description="Error message")
    errors: list[dict] = Field(..., description="Validation errors")