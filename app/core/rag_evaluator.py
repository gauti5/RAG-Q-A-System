"""RAGAS evaluation module for RAG quality assessment."""

import asyncio
import time
from typing import Any

from datasets import Dataset
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

from app.config import get_settings
from app.utils.logger import get_logger

logger=get_logger(__name__)
settings=get_settings()


class RAGASEvaluator:
    def __init__(self):
        """Initialize RAGAS evaluator with metrics and models."""
        logger.info("Initalizing RAGAS Evaluator")
        # Get settings inside __init__ to allow mocking in tests
        self.settings=settings
        # Use RAGAS-specific LLM settings if provided, otherwise fall back to default
        eval_llm_model=self.settings.ragas_llm_model or self.settings.LLM_MODEL
        eval_llm_temperature=(
            self.settings.ragas_llm_temperature 
            if self.settings.ragas_llm_temperature is not None
            else self.settings.LLM_TEMPERATURE
        )
        eval_llm_embedding=self.settings.ragas_embedding_model or self.settings.EMBEDDING_MODEL
        
        # Initialize LLM for evaluation
        self.llm=ChatGoogleGenerativeAI(
            model=eval_llm_model,
            temperature=eval_llm_temperature,
            google_api_key=self.settings.GOOGLE_API_KEY,
        )
        # Initialize embeddings for evaluation
        self.embeddings=GoogleGenerativeAIEmbeddings(
            model=eval_llm_embedding,
            google_api_key=self.settings.GOOGLE_API_KEY,
        )
        # Initialize metrics (reference-free only)
        self.metrics=[
            faithfulness,
            answer_relevancy,
            
        ]
        
        logger.info(
            f"RAGAS Evaluator Intialized - "
            f"LLM : {eval_llm_model} (temp={eval_llm_temperature})"
            f"Embeddings : {eval_llm_embedding}"
            f"Metrics : {[metrics.name for metrics in self.metrics]}"
        )
    
    async def aevaluator(self, question: str, answer: str, contexts: list[str]) -> dict[str, Any]:
        """Execute async RAGAS evaluation.

        Args:
            question: The user's question
            answer: The generated answer
            contexts: List of retrieved context documents

        Returns:
            Dictionary with evaluation scores and metadata
        """
        logger.debug(f"Starting evaluation for question : {question[:100]}")
        start_time=time.time()
        
        try:
            # Prepare dataset for RAGAS
            dataset=self._prepare_dataset(question, answer, contexts)
            # Run evaluation in thread pool to avoid blocking event loop
            result=await asyncio.to_thread(
                self._evaluator_with_timeout,
                dataset,
            )
            
            evaluation_time_ms=(time.time()-start_time)*1000
            
            scores={
                'faithfulness': float(result['faithfulness']) if 'faithfulness' in result else None,
                'answer_relevancy': float(result['answer_relevancy']) if 'answer_relevancy' in result else None,
                'evaluation_time_ms': round(evaluation_time_ms, 2),
                'error': None,
            }
            
            if self.settings.ragas_log_results:
                logger.info(
                    f"Evaluation Completed - "
                    f"faithfulness = {scores['faithfulness']}, "
                    f"answer_relevancy = {scores['answer_relevancy']}, "
                    f"time = {scores['evaluation_time_ms']} ms"
                    
                    
                )
            return scores
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}", exc_info=True)
            return self._handle_evaluation_error(e)
        
    def _prepare_dataset(self, question: str, answer: str, contexts: list[str]) -> Dataset:
        """Convert RAG output to RAGAS Dataset format.

        Args:
            question: The user's question
            answer: The generated answer
            contexts: List of retrieved context documents

        Returns:
            Dataset object for RAGAS evaluation
        """
        Data={
            'question': question,
            'answer': answer,
            'contexts': contexts,
        }
        logger.debug(f"prepared the dataset with {len(contexts)} contexts" f"for question : {question[:100]}...")
        
        return Dataset.from_dict(Data)
    
    def _evaluator_with_timeout(self, dataset: Dataset) -> dict[str, Any]:
        """Execute RAGAS evaluation with timeout.

        Args:
            dataset: Prepared RAGAS dataset

        Returns:
            Evaluation results dictionary

        Raises:
            TimeoutError: If evaluation exceeds timeout
        """
        # Note: asyncio.timeout would be ideal, but RAGAS evaluate() is sync
        # For now, we rely on the async wrapper and trust RAGAS to complete
        # In production, consider using signal.alarm or threading.Timer
        result=evaluate(
            dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings,
        )
        # Convert to dictionary and extract scores
        return result.to_pandas().to_dict("records")[0]
    
    
    def _handle_evaluation_error(self, error: Exception) -> dict[str, Any]:
        """Return safe fallback scores on error.

        Args:
            error: The exception that occurred

        Returns:
            Dictionary with null scores and error message
        """
        logger.error(f"Returning fallback scores due to error: {error}")

        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "evaluation_time_ms": None,
            "error": str(error),
        }
            
        