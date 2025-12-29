"""RAG chain module using LangChain LCEL."""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from app.utils.logger import get_logger 
from app.core.vector_store import VectoreStoreService
from app.config import get_settings

logger=get_logger()
settings=get_settings()


# RAG Prompt Template
RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question based on the provided context.

If you cannot answer the question based on the context, say "I don't have enough information to answer that question."

Do not make up information. Only use the context provided.

Context:
{context}

Question: {question}

Answer:"""

def format_docs(docs : list[Document]) -> str:
    """Format documents into a single context string.

    Args:
        docs: List of Document objects

    Returns:
        Formatted context string
    """
    return "\n\n--\n\n".join(doc.page_content for doc in docs)