"""FastAPI application entry point."""

# IMPORTANT: Load .env file FIRST, before any LangChain imports
# This ensures LangSmith environment variables are available for tracing
# ruff: noqa: E402, I001

from dotenv import load_dotenv


load_dotenv()

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.utils.logger import get_logger, setup_logger
from app.config import get_settings
from app.api.routes import documents, query, health

settings=get_settings()
import os

# Get the path of the current file (app/main.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Move up one level to the root where 'static' lives
static_dir = os.path.join(os.path.dirname(BASE_DIR), "static")

# Safety: Create the directory if it's missing so the app doesn't crash
os.makedirs(static_dir, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    #Startup
    setup_logger(settings.LOG_LEVEL)
    logger=get_logger(__name__)
    logger.info(f"Starting {settings.APP_NAME} v{__version__}")
    logger.info(f"Log Level : {settings.LOG_LEVEL}")
    
    yield
    
    # Shutdown
    logger.info(f"Shutting down application")

# Create FastAPI application
app=FastAPI(
    title=settings.APP_NAME,
    description="""
    ## RAG Q&A System API

    A Retrieval-Augmented Generation (RAG) question-answering system built with:
    - **FastAPI** for the API layer
    - **LangChain** for RAG orchestration
    - **Qdrant Cloud** for vector storage
    - **OpenAI** for embeddings and LLM

    ### Features
    - Upload PDF, TXT, and CSV documents
    - Ask questions and get AI-powered answers
    - View source documents for transparency
    - Streaming responses for real-time feedback
    """,
    version=__version__,
    docs_url='/docs',
    redoc_url='/redoc',
    openapi_url='/openapi.json',
    
    lifespan=lifespan,
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)
app.mount("/static", StaticFiles(directory=static_dir), name='static')

# Include routers
app.include_router(query.router)
app.include_router(documents.router)
app.include_router(health.router)


@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def root():
    """Serve the main UI."""
    index_path = os.path.join(static_dir, "index.html")
    
    if not os.path.exists(index_path):
        return HTMLResponse(f"index.html not found at {index_path}", status_code=404)
        
    with open(index_path, "r") as f:
        return f.read()
    
    
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger = get_logger(__name__)
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
        },
    )
    
if __name__=="__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )