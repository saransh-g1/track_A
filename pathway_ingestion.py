"""
Pathway Ingestion Layer
Handles ingestion of full novels, backstory text, and metadata from various sources.
"""
import pathway as pw
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DocumentMetadata:
    """Metadata for ingested documents"""
    document_id: str
    source_type: str  # "novel", "backstory", "metadata"
    file_path: Optional[str] = None
    chapter_id: Optional[int] = None
    timestamp: Optional[datetime] = None
    additional_metadata: Dict = None


class PathwayIngestionLayer:
    """
    Pathway-based ingestion layer for novels and backstory.
    Supports local files, Google Drive, and cloud storage.
    """
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def ingest_from_local(self, file_path: str, source_type: str = "novel") -> pw.Table:
        """
        Ingest document from local file system using Pathway.
        
        Args:
            file_path: Path to the text file
            source_type: Type of document ("novel", "backstory", "metadata")
            
        Returns:
            Pathway Table with document data
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Use Pathway's file system connector
        # This creates a reactive table that updates when files change
        table = pw.io.fs.read(
            str(file_path),
            format="text",
            mode="static"  # Use "streaming" for real-time updates
        )
        
        # Add metadata columns
        table = table.select(
            text=pw.this.data,
            document_id=pw.apply(lambda x: f"{source_type}_{file_path.stem}", pw.this.data),
            source_type=pw.apply(lambda x: source_type, pw.this.data),
            file_path=pw.apply(lambda x: str(file_path), pw.this.data),
            timestamp=pw.apply(lambda x: datetime.now().isoformat(), pw.this.data)
        )
        
        return table
    
    def ingest_novel(self, novel_path: str) -> pw.Table:
        """Ingest full novel text file"""
        return self.ingest_from_local(novel_path, source_type="novel")
    
    def ingest_backstory(self, backstory_path: str) -> pw.Table:
        """Ingest backstory text file"""
        return self.ingest_from_local(backstory_path, source_type="backstory")
    
    def ingest_directory(self, directory: str, pattern: str = "*.txt") -> pw.Table:
        """
        Ingest all matching files from a directory.
        
        Args:
            directory: Directory path
            pattern: File pattern (e.g., "*.txt")
            
        Returns:
            Combined Pathway Table
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Use Pathway's glob pattern matching
        table = pw.io.fs.read(
            str(directory / pattern),
            format="text",
            mode="static"
        )
        
        # Extract source type from filename
        def extract_source_type(file_path: str) -> str:
            path = Path(file_path)
            if "backstory" in path.name.lower():
                return "backstory"
            elif "novel" in path.name.lower() or "current" in path.name.lower():
                return "novel"
            else:
                return "unknown"
        
        table = table.select(
            text=pw.this.data,
            document_id=pw.apply(
                lambda x: f"doc_{Path(x).stem}", 
                pw.this.path
            ),
            source_type=pw.apply(extract_source_type, pw.this.path),
            file_path=pw.this.path,
            timestamp=pw.apply(lambda x: datetime.now().isoformat(), pw.this.data)
        )
        
        return table
    
    def ingest_from_google_drive(self, file_id: str, credentials_path: str) -> pw.Table:
        """
        Ingest document from Google Drive.
        Requires Google Drive API credentials.
        
        Note: This is a placeholder for Google Drive integration.
        In production, use Pathway's Google Drive connector or custom implementation.
        """
        # Placeholder - would use pw.io.google_drive or custom connector
        raise NotImplementedError(
            "Google Drive ingestion requires Pathway Google Drive connector. "
            "For now, download files locally and use ingest_from_local."
        )
    
    def combine_documents(self, *tables: pw.Table) -> pw.Table:
        """
        Combine multiple Pathway tables into one.
        
        Args:
            *tables: Variable number of Pathway tables
            
        Returns:
            Combined table
        """
        if not tables:
            raise ValueError("At least one table required")
        
        # Pathway's concat operation
        combined = tables[0]
        for table in tables[1:]:
            combined = combined.concat(table)
        
        return combined
    
    def get_document_text(self, table: pw.Table, document_id: str) -> Optional[str]:
        """
        Extract text from a specific document in the table.
        
        Args:
            table: Pathway table
            document_id: ID of the document to retrieve
            
        Returns:
            Document text or None
        """
        # Filter table by document_id
        filtered = table.filter(pw.this.document_id == document_id)
        
        # In Pathway, we need to materialize to get actual values
        # This is a simplified version - in practice, use pw.run() or similar
        # For now, return the filtered table structure
        return filtered


def create_ingestion_pipeline(novel_path: str, backstory_path: str) -> Tuple[pw.Table, pw.Table]:
    """
    Create complete ingestion pipeline for Track A.
    
    Args:
        novel_path: Path to full novel text file
        backstory_path: Path to backstory text file
        
    Returns:
        Tuple of (novel_table, backstory_table)
    """
    ingestion = PathwayIngestionLayer()
    
    novel_table = ingestion.ingest_novel(novel_path)
    backstory_table = ingestion.ingest_backstory(backstory_path)
    
    return novel_table, backstory_table

