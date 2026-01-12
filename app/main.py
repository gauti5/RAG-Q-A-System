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
app.mount("/static", StaticFiles(directory='static'), name='static')

# Include routers
app.include_router(query.router)
app.include_router(documents.router)
app.include_router(health.router)


@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def root():
    """Serve the main UI."""
    with open("static/index.html", "r") as f:
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