"""
Pathway Document Store + Vector Index
Handles temporal chunking, embeddings, and metadata-aware retrieval.
"""
import pathway as pw
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
import json


@dataclass
class Chunk:
    """Represents a chunk of text with metadata"""
    text: str
    chunk_id: str
    document_id: str
    chapter_id: Optional[int]
    position_ratio: float  # 0.0 = start, 1.0 = end
    start_char: int
    end_char: int
    metadata: Dict = None


class TemporalChunker:
    """
    Chunks documents sequentially with temporal awareness.
    Preserves chapter boundaries and position information.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(
        self, 
        text: str, 
        document_id: str,
        chapter_id: Optional[int] = None,
        total_length: Optional[int] = None
    ) -> List[Chunk]:
        """
        Chunk text sequentially with overlap.
        
        Args:
            text: Full text to chunk
            document_id: ID of source document
            chapter_id: Optional chapter identifier
            total_length: Total length of document (for position_ratio calculation)
            
        Returns:
            List of Chunk objects
        """
        if total_length is None:
            total_length = len(text)
        
        chunks = []
        start = 0
        chunk_idx = 0
        
        # Simple word-based chunking (can be improved with sentence boundaries)
        words = text.split()
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            
            # Calculate position ratio
            char_start = text.find(chunk_text, start * 5)  # Approximate
            char_end = char_start + len(chunk_text)
            position_ratio = char_start / total_length if total_length > 0 else 0.0
            
            chunk = Chunk(
                text=chunk_text,
                chunk_id=f"{document_id}_chunk_{chunk_idx}",
                document_id=document_id,
                chapter_id=chapter_id,
                position_ratio=position_ratio,
                start_char=char_start,
                end_char=char_end,
                metadata={"chunk_index": chunk_idx}
            )
            chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            chunk_idx += 1
        
        return chunks
    
    def chunk_with_sentences(
        self,
        text: str,
        document_id: str,
        chapter_id: Optional[int] = None
    ) -> List[Chunk]:
        """
        Chunk text respecting sentence boundaries.
        Better for narrative coherence.
        """
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        from nltk.tokenize import sent_tokenize
        
        sentences = sent_tokenize(text)
        total_length = len(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_idx = 0
        char_position = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = " ".join(current_chunk)
                position_ratio = char_position / total_length if total_length > 0 else 0.0
                
                chunk = Chunk(
                    text=chunk_text,
                    chunk_id=f"{document_id}_chunk_{chunk_idx}",
                    document_id=document_id,
                    chapter_id=chapter_id,
                    position_ratio=position_ratio,
                    start_char=char_position,
                    end_char=char_position + len(chunk_text),
                    metadata={"chunk_index": chunk_idx, "num_sentences": len(current_chunk)}
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-self.chunk_overlap//10:] if len(current_chunk) > self.chunk_overlap//10 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
                char_position += len(" ".join(overlap_sentences))
                chunk_idx += 1
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            position_ratio = char_position / total_length if total_length > 0 else 0.0
            
            chunk = Chunk(
                text=chunk_text,
                chunk_id=f"{document_id}_chunk_{chunk_idx}",
                document_id=document_id,
                chapter_id=chapter_id,
                position_ratio=position_ratio,
                start_char=char_position,
                end_char=char_position + len(chunk_text),
                metadata={"chunk_index": chunk_idx, "num_sentences": len(current_chunk)}
            )
            chunks.append(chunk)
        
        return chunks


class PathwayVectorStore:
    """
    Pathway-based vector store with FAISS backend.
    Stores embeddings and enables metadata-aware retrieval.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        store_path: Optional[str] = None
    ):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.store_path = Path(store_path) if store_path else None
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def add_chunks(self, chunks: List[Chunk]):
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of Chunk objects to add
        """
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Add to FAISS index
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        # Rebuild index
        self.index.reset()
        self.index.add(self.embeddings.astype('float32'))
        
        # Store chunks
        self.chunks.extend(chunks)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
        position_weight: bool = True
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieve top-k chunks for a query.
        
        Args:
            query: Query text
            top_k: Number of chunks to retrieve
            filter_metadata: Optional metadata filters
            position_weight: Whether to weight later positions higher
            
        Returns:
            List of (Chunk, score) tuples, sorted by relevance
        """
        # Encode query
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )[0]
        
        # Search in FAISS
        k = min(top_k * 2, len(self.chunks))  # Retrieve more for filtering
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k
        )
        
        # Convert distances to similarities (lower distance = higher similarity)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= len(self.chunks):
                continue
            
            chunk = self.chunks[idx]
            
            # Apply metadata filters
            if filter_metadata:
                if chunk.document_id != filter_metadata.get("document_id", chunk.document_id):
                    continue
                if filter_metadata.get("chapter_id") is not None:
                    if chunk.chapter_id != filter_metadata["chapter_id"]:
                        continue
            
            # Calculate similarity score (inverse of distance)
            similarity = 1.0 / (1.0 + distance)
            
            # Apply position weighting if enabled
            if position_weight:
                # Later positions get higher weight
                position_weight_val = 1.0 + (chunk.position_ratio * 0.2)
                similarity *= position_weight_val
            
            results.append((chunk, similarity))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def retrieve_multi_hop(
        self,
        query: str,
        top_k: int = 5,
        num_hops: int = 2
    ) -> List[Tuple[Chunk, float]]:
        """
        Multi-hop retrieval: retrieve chunks, then retrieve chunks similar to those.
        
        Args:
            query: Initial query
            top_k: Chunks per hop
            num_hops: Number of retrieval hops
            
        Returns:
            Combined list of chunks from all hops
        """
        all_results = []
        seen_chunk_ids = set()
        
        # First hop
        first_hop = self.retrieve(query, top_k=top_k)
        all_results.extend(first_hop)
        seen_chunk_ids.update(chunk.chunk_id for chunk, _ in first_hop)
        
        # Subsequent hops
        for hop in range(1, num_hops):
            # Use retrieved chunks as queries
            hop_results = []
            for chunk, _ in first_hop:
                if chunk.chunk_id in seen_chunk_ids:
                    continue
                
                # Retrieve chunks similar to this chunk
                similar = self.retrieve(chunk.text, top_k=top_k//2)
                for sim_chunk, sim_score in similar:
                    if sim_chunk.chunk_id not in seen_chunk_ids:
                        hop_results.append((sim_chunk, sim_score))
                        seen_chunk_ids.add(sim_chunk.chunk_id)
            
            all_results.extend(hop_results)
        
        # Deduplicate and sort
        unique_results = {}
        for chunk, score in all_results:
            if chunk.chunk_id not in unique_results:
                unique_results[chunk.chunk_id] = (chunk, score)
            else:
                # Keep higher score
                _, existing_score = unique_results[chunk.chunk_id]
                if score > existing_score:
                    unique_results[chunk.chunk_id] = (chunk, score)
        
        results = list(unique_results.values())
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k * num_hops]
    
    def save(self, path: Optional[str] = None):
        """Save vector store to disk"""
        save_path = Path(path) if path else self.store_path
        if save_path is None:
            raise ValueError("No save path specified")
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_path / "index.faiss"))
        
        # Save embeddings
        np.save(save_path / "embeddings.npy", self.embeddings)
        
        # Save chunks metadata
        chunks_data = [
            {
                "text": chunk.text,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "chapter_id": chunk.chapter_id,
                "position_ratio": chunk.position_ratio,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "metadata": chunk.metadata or {}
            }
            for chunk in self.chunks
        ]
        
        with open(save_path / "chunks.json", "w") as f:
            json.dump(chunks_data, f, indent=2)
    
    def load(self, path: Optional[str] = None):
        """Load vector store from disk"""
        load_path = Path(path) if path else self.store_path
        if load_path is None or not load_path.exists():
            raise ValueError(f"Load path does not exist: {load_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(load_path / "index.faiss"))
        
        # Load embeddings
        self.embeddings = np.load(load_path / "embeddings.npy")
        
        # Load chunks
        with open(load_path / "chunks.json", "r") as f:
            chunks_data = json.load(f)
        
        self.chunks = [
            Chunk(
                text=chunk_data["text"],
                chunk_id=chunk_data["chunk_id"],
                document_id=chunk_data["document_id"],
                chapter_id=chunk_data.get("chapter_id"),
                position_ratio=chunk_data["position_ratio"],
                start_char=chunk_data["start_char"],
                end_char=chunk_data["end_char"],
                metadata=chunk_data.get("metadata", {})
            )
            for chunk_data in chunks_data
        ]


class PathwayDocumentStore:
    """
    Main document store interface combining Pathway tables and vector store.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        store_path: Optional[str] = None
    ):
        self.chunker = TemporalChunker(chunk_size, chunk_overlap)
        self.vector_store = PathwayVectorStore(embedding_model_name, store_path)
    
    def index_document(
        self,
        text: str,
        document_id: str,
        chapter_id: Optional[int] = None
    ):
        """
        Index a document: chunk it and add to vector store.
        
        Args:
            text: Full document text
            document_id: Unique document identifier
            chapter_id: Optional chapter identifier
        """
        # Chunk with sentence boundaries
        chunks = self.chunker.chunk_with_sentences(
            text,
            document_id,
            chapter_id
        )
        
        # Add to vector store
        self.vector_store.add_chunks(chunks)
        
        return len(chunks)
    
    def retrieve_evidence(
        self,
        query: str,
        top_k: int = 5,
        multi_hop: bool = True
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieve evidence chunks for a query.
        
        Args:
            query: Query text (typically a claim)
            top_k: Number of chunks to retrieve
            multi_hop: Whether to use multi-hop retrieval
            
        Returns:
            List of (Chunk, score) tuples
        """
        if multi_hop:
            return self.vector_store.retrieve_multi_hop(query, top_k=top_k)
        else:
            return self.vector_store.retrieve(query, top_k=top_k)

