"""Document management endpoints."""

from fastapi import APIRouter, File, UploadFile, HTTPException
from app.core.document_processor import DocumentProcessor
from app.core.vector_store import VectoreStoreService
from app.utils.logger import get_logger

from app.api.schemas import DocumentUploadResponse, DocumentListResponse, ErrorResponse

logger=get_logger(__name__)
router=APIRouter(prefix='/documents', tags=['Documents'])

@router.post(
    '/upload',
    response_model=DocumentUploadResponse,
    responses={
        400 : {"model": ErrorResponse, "description": "Invalid file type"},
        500 : {"model": ErrorResponse, "description": "processing error"},
    },
    summary="Upload and Ingest a document",
    description="Upload a document (PDF, Text, or CSV) to be processed and added to vector store.",
)

async def upload_document(
    file: UploadFile=File(..., description="document file to upload")
) -> DocumentUploadResponse:
    """upload and process a document"""
    logger.info(f"Received document upload : {file.filename}")
    
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is Required!!"
        )
    try:
        # Process a document
        processor=DocumentProcessor()
        chunks=processor.process_upload(file.file, file.filename)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No content could be extracted from the document"
            )
        # Add to vector store
        vector_store=VectoreStoreService()
        document_ids=vector_store.add_document(chunks)
        
        logger.info(
            f"Successfully Processed : {file.filename}"
            f"({len(chunks)} chunks, {len(document_ids)} documents)"
        )
        
        return  DocumentUploadResponse(
            message="Document uploaded and Processed Successfully",
            filename=file.filename,
            chunks_created=len(chunks),
            document_ids=document_ids,
        )
    except ValueError as e:
        logger.warning(f"Invalid file type : {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing document : {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}",
        )
    
@router.get(
    '/info',
    response_model=DocumentListResponse,
    summary="Get Document Collection",
    description="Get information about the document collection.",
)


async def get_collection_info() -> DocumentListResponse:
    """Get information about the document collection."""
    logger.debug("collection info requested")
    try: 
        
        vector_store=VectoreStoreService()
        info=vector_store.get_collection_info()
        
        return DocumentListResponse(
            collection_name=info['name'],
            total_documents=info['points_count'],
            status=info['status'],
            
        )
    except Exception as e:
        logger.error(f"Error getting collection info : {e}")
        raise HTTPException(
            status_code=400, 
            detail=f"Error getting collection info : {str(e)}"
        )
        
@router.delete(
    '/collection',
    responses={
        200: {'description': "collection deleted successfully"},
        500: {'model': ErrorResponse, 'description': "Deletion Error"},
    },
    summary="Delete entire collection",
    description="Delete all documents from the vector store. use with caution",
)
    
async def delete_collection()->dict:
    """Delete the entire document collection."""
    logger.warning("collection deletion requested")
    try:
        
        vector_store=VectoreStoreService()
        vector_store.delete_collection()
        
        return {'message': "collection deleted successfully"}
    
    except Exception as e:
        logger.error(f"Error deleting documents : {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting collection : {e}"
        )
