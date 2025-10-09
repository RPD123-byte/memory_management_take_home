"""
Semantic memory module for embedding-based similarity search and recall.

This module provides vector embedding generation and similarity search capabilities
to find semantically similar past tool executions.
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings


@dataclass
class SemanticMemory:
    """Represents a semantic memory with embeddings"""
    tool_id: str
    tool_name: str
    description: str
    embedding: List[float]
    metadata: Dict[str, Any]
    

@dataclass
class SimilarToolResult:
    """Represents a similar tool result from semantic search"""
    tool_id: str
    tool_name: str
    similarity_score: float
    description: str
    metadata: Dict[str, Any]


class SemanticMemoryService:
    """Service for semantic memory management using embeddings"""
    
    def __init__(self, api_key: str = None, model: str = "text-embedding-3-small"):
        """
        Initialize semantic memory service.
        
        Args:
            api_key: OpenAI API key
            model: Embedding model to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required for embeddings")
        
        # Initialize embeddings
        self.embeddings: Embeddings = OpenAIEmbeddings(
            openai_api_key=self.api_key,
            model=model
        )
        
        # In-memory vector store (tool_id -> SemanticMemory)
        self.memory_store: Dict[str, SemanticMemory] = {}
        
        # Embedding dimension (depends on model)
        self.embedding_dim = 1536 if model == "text-embedding-3-small" else 3072
        
    def create_tool_description(self, tool_name: str, parameters: Dict[str, Any], 
                               result: Dict[str, Any], summary: Optional[str] = None) -> str:
        """
        Create a rich text description of a tool execution for embedding.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            result: Tool execution result
            summary: Optional summary from LLM
            
        Returns:
            Rich text description suitable for embedding
        """
        parts = [f"Tool: {tool_name}"]
        
        # Add key parameters
        if parameters:
            # Extract key info from parameters
            if "command" in parameters:
                parts.append(f"Command: {parameters['command']}")
            if "file_path" in parameters:
                parts.append(f"File: {parameters['file_path']}")
            if "query" in parameters:
                parts.append(f"Query: {parameters['query']}")
            if "files" in parameters:
                parts.append(f"Files: {', '.join(parameters['files'])}")
        
        # Add result status
        status = result.get("status", "unknown")
        parts.append(f"Status: {status}")
        
        # Add summary if available
        if summary:
            parts.append(f"Summary: {summary}")
        
        # Add output snippet if available
        output = result.get("output", "")
        if output and isinstance(output, str):
            output_snippet = output[:200].replace('\n', ' ')
            parts.append(f"Output: {output_snippet}")
        
        return " | ".join(parts)
    
    async def add_memory_async(self, tool_id: str, tool_name: str, 
                               parameters: Dict[str, Any], result: Dict[str, Any],
                               summary: Optional[str] = None) -> SemanticMemory:
        """
        Add a tool execution to semantic memory (async).
        
        Args:
            tool_id: Unique tool ID
            tool_name: Name of the tool
            parameters: Tool parameters
            result: Execution result
            summary: Optional LLM-generated summary
            
        Returns:
            SemanticMemory object
        """
        description = self.create_tool_description(tool_name, parameters, result, summary)
        
        # Generate embedding
        embedding = await self.embeddings.aembed_query(description)
        
        # Create semantic memory
        memory = SemanticMemory(
            tool_id=tool_id,
            tool_name=tool_name,
            description=description,
            embedding=embedding,
            metadata={
                "parameters": parameters,
                "result": result,
                "summary": summary
            }
        )
        
        # Store in memory
        self.memory_store[tool_id] = memory
        
        return memory
    
    def add_memory(self, tool_id: str, tool_name: str, 
                  parameters: Dict[str, Any], result: Dict[str, Any],
                  summary: Optional[str] = None) -> SemanticMemory:
        """
        Add a tool execution to semantic memory (sync).
        
        Args:
            tool_id: Unique tool ID
            tool_name: Name of the tool
            parameters: Tool parameters
            result: Execution result
            summary: Optional LLM-generated summary
            
        Returns:
            SemanticMemory object
        """
        description = self.create_tool_description(tool_name, parameters, result, summary)
        
        # Generate embedding (sync)
        embedding = self.embeddings.embed_query(description)
        
        # Create semantic memory
        memory = SemanticMemory(
            tool_id=tool_id,
            tool_name=tool_name,
            description=description,
            embedding=embedding,
            metadata={
                "parameters": parameters,
                "result": result,
                "summary": summary
            }
        )
        
        # Store in memory
        self.memory_store[tool_id] = memory
        
        return memory
    
    def find_similar_tools(self, query_tool_name: str, query_parameters: Dict[str, Any],
                          top_k: int = 3, similarity_threshold: float = 0.7) -> List[SimilarToolResult]:
        """
        Find similar tool executions based on semantic similarity.
        
        Args:
            query_tool_name: Name of the tool being queried
            query_parameters: Parameters of the query
            top_k: Number of similar results to return
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of similar tool results, sorted by similarity
        """
        if not self.memory_store:
            return []
        
        # Create query description
        query_desc = self.create_tool_description(
            query_tool_name, 
            query_parameters, 
            {"status": "pending"}
        )
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query_desc)
        
        # Calculate similarities
        similarities = []
        for tool_id, memory in self.memory_store.items():
            similarity = self._cosine_similarity(query_embedding, memory.embedding)
            
            if similarity >= similarity_threshold:
                similarities.append(SimilarToolResult(
                    tool_id=tool_id,
                    tool_name=memory.tool_name,
                    similarity_score=similarity,
                    description=memory.description,
                    metadata=memory.metadata
                ))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x.similarity_score, reverse=True)
        return similarities[:top_k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score (0-1)
        """
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def get_memory(self, tool_id: str) -> Optional[SemanticMemory]:
        """
        Retrieve a semantic memory by tool ID.
        
        Args:
            tool_id: Tool ID to retrieve
            
        Returns:
            SemanticMemory if found, None otherwise
        """
        return self.memory_store.get(tool_id)
    
    def clear_memories(self):
        """Clear all semantic memories"""
        self.memory_store.clear()
    
    def search_by_natural_language(self, query: str, top_k: int = 5, 
                                   similarity_threshold: float = 0.6) -> List[SimilarToolResult]:
        """
        Search memories using natural language query.
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar tool results
        """
        if not self.memory_store:
            return []
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Calculate similarities
        similarities = []
        for tool_id, memory in self.memory_store.items():
            similarity = self._cosine_similarity(query_embedding, memory.embedding)
            
            if similarity >= similarity_threshold:
                similarities.append(SimilarToolResult(
                    tool_id=tool_id,
                    tool_name=memory.tool_name,
                    similarity_score=similarity,
                    description=memory.description,
                    metadata=memory.metadata
                ))
        
        # Sort and return top_k
        similarities.sort(key=lambda x: x.similarity_score, reverse=True)
        return similarities[:top_k]
