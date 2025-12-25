from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class settings(BaseSettings):
    model_config=SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore',
        
    )
    # Google API Key Configuration
    GOOGLE_API_KEY : str

    # Qdrant Configuration
    QDRANT_URL:str
    QDRANT_API_KEY:str

    # Collection Settings
    COLLECTION_NAME:str='rag_documents'

    # Document Processing Settings
    CHUNK_SIZE:int=1000
    CHUNK_OVERLAP:int=200

    # Model Configuration
    EMBEDDING_MODEL:str='models/text-embedding-004'
    LLM_MODEL:str='gemini-2.5-flash'
    LLM_TEMPERATURE:float=0.0

    # Retrieval Settings
    RETRIEVAL_K:int=4

    # Logging 
    LOG_LEVEL:str='INFO'

    # API Settings
    API_HOST:str='0.0.0.0'
    API_PORT:int=8000
    
    # RAGAS Evaluation Settings
    enable_ragas_evaluation: bool = True
    ragas_timeout_seconds: float = 30.0
    ragas_log_results: bool = True
    ragas_llm_model: str | None = None  # Defaults to llm_model if not set
    ragas_llm_temperature: float | None = None  # Defaults to llm_temperature if not set
    ragas_embedding_model: str | None = None  # Defaults to embedding_model if not set
    
    # Application Info
    app_name: str = "RAG Q&A System"
    app_version: str = "0.1.0"
    
@lru_cache
def get_settings()->settings:
    return settings