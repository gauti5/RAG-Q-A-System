"""Document processing module for loading and chunking documents."""

import tempfile
from pathlib import Path
from typing import BinaryIO

from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.logger import get_logger
from app.config import get_setings

logger=get_logger(__name__)

class DocumentProcessor:
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv"}
    def __init__(
        self,
        chunk_size: int | None=None,
        chunk_overlap: int | None=None,
    ):
        """Initialize document processor.

        Args:
            chunk_size: Size of text chunks (default from settings)
            chunk_overlap: Overlap between chunks (default from settings)
        """
        settings=get_setings()
        self.chunk_size=chunk_size or settings.CHUNK_SIZE,
        self.chunk_overalap=chunk_overlap or settings.CHUNK_OVERLAP,
        
        self.text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overalap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            
        )
        
        logger.info(
            f"Document Processor initialized with chunk size = {self.chunk_size}, "
            f"chunk overlap = {self.chunk_overalap}"
        )
    
    # Loading PDF file
    def load_pdf(self, file_path : str | Path) -> list[Document]:
        
        """Load a PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            List of Document objects
        """
        
        file_path=Path(file_path)
        logger.info(f"Loading Pdf file : {file_path.name}")
        loader=PyPDFLoader(str(file_path))
        documents=loader.load()
        
        logger.info(f"loaded {len(documents)} pages from {file_path.name}")
        return documents
    
    # Loading TEXT file
    def load_text(self, file_path : str | Path) -> list[Document]:
        """Load a Text file.

        Args:
            file_path: Path to PDF file

        Returns:
            List of Document objects
        """
        
        file_path=Path(file_path)
        logger.info(f"Loaded text file : {file_path.name}")
        
        loader=TextLoader(str(file_path), encoding="utf-8")
        documents=loader.load()
        
        logger.info(f"Loaded {len(documents)} pages from {file_path.name}")
        
        return documents
    
    # Loading CSV file
    def load_csv(self, file_path : str | Path) -> list[Document]:
        
        """Load a Text file.

        Args:
            file_path: Path to PDF file

        Returns:
            List of Document objects
        """
        
        file_path=Path(file_path)
        logger.info(f"Loaded csv file : {file_path.name}")
        loader=CSVLoader(str(file_path), encoding="utf-8")
        documents=loader.load()
        
        logger.info(f"Loaded {len(documents)} pages from {file_path.name}")
        
        return documents
    
    def load_file(self, file_path : str | Path) -> list[Document]:
        
        """Load a file based on its extension.

        Args:
            file_path: Path to file

        Returns:
            List of Document objects

        Raises:
            ValueError: If file extension is not supported
        """
        file_path=Path(file_path)
        extension=file_path.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupoorted file extension : {extension}",
                f"Supported file extension : {self.SUPPORTED_EXTENSIONS}"
            )
        loaders={
            ".pdf": self.load_pdf,
            ".txt": self.load_text,
            ".csv": self.load_csv,
        }
        
        return loaders[extension](file_path)
    
    def load_from_upload(self,file : BinaryIO, filename : str) -> list[Document]:
        
        """Load document from uploaded file.

        Args:
            file: File-like object
            filename: Original filename

        Returns:
            List of Document objects
        """
        
        extension=Path(filename).suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension : {extension}"
                f"Supported file extension : {self.SUPPORTED_EXTENSIONS}"
                
            )
            
        # save to temp file for processing..
        
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temp_file:
            temp_file.write(file.read())
            temp_path=temp_file.name
        
        try:
            documents=self.load_file(temp_path)
            
            for doc in documents:
                # Update metadata with original filename
                doc.metadata["source"]=filename
            return documents
        finally:
            Path(temp_path).unlink(missing_ok=True)
            
    # Splitting the documents into chunks
    
    def split_documents(self, documents : list[Document]) -> list[Document]:
        
        """Split documents into chunks.

        Args:
            documents: List of Document objects

        Returns:
            List of chunked Document objects
        """
        logger.info(f"splitting {len(documents)} documents into chunks")
        
        chunks=self.text_splitter.split_documents(documents=documents)
        
        logger.info(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def process_file(self, file_path : str | Path) -> list[Document]:
        """Load and split a file in one step.

        Args:
            file_path: Path to file

        Returns:
            List of chunked Document objects
        """
        documents=self.load_file(file_path=file_path)
        return self.split_documents(documents)
    
    
    def process_upload(self, file : BinaryIO, filename : str) -> list[Document]:
        """Load and split an uploaded file.

        Args:
            file: File-like object
            filename: Original filename

        Returns:
            List of chunked Document objects
        """
        documents=self.load_from_upload(file, filename)
        return self.split_documents(documents=documents)

    
    
        