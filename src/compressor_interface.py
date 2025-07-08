from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
import json

from neo4j_service import Neo4jService
from token_counter import TokenCounter


class ToolCompressor(ABC):
    """Abstract base class for tool compression strategies"""
    
    def __init__(self, neo4j_service: Neo4jService, token_counter: TokenCounter, workflow_id: str):
        self.neo4j_service = neo4j_service
        self.token_counter = token_counter
        self.workflow_id = workflow_id
    
    @abstractmethod
    def get_compression_type(self) -> str:
        pass
    
    @abstractmethod
    def compress_and_store(self, tool_id: str, tool_content: Dict[str, Any], 
                          pre_generated_summary: Optional[Dict[str, Any]] = None) -> bool:
        pass
    
    @abstractmethod
    def retrieve_compressed(self, tool_id: str) -> Optional[str]:
        pass
    
    def _get_compression_metadata(self, tool_id: str) -> str:
        return f"compressed_{self.get_compression_type()}_{tool_id}"
    
    def _store_compressed_content(self, tool_id: str, compressed_content: str) -> bool:
        try:
            metadata = self._get_compression_metadata(tool_id)
            
            compression_data = {
                "tool_id": tool_id,
                "compression_type": self.get_compression_type(),
                "compressed_content": compressed_content,
                "token_count": self.token_counter.count_tokens(compressed_content),
                "timestamp": datetime.now().isoformat()
            }
            
            self.neo4j_service.update_node(
                metadata=metadata,
                summary=f"{self.get_compression_type()} compression of {tool_id}",
                content=json.dumps(compression_data),
                workflow_id=self.workflow_id
            )
            
            return True
            
        except Exception as e:
            print(f"Error storing compressed content for {tool_id}: {str(e)}")
            return False
    
    def _retrieve_compressed_content(self, tool_id: str) -> Optional[str]:
        try:
            metadata = self._get_compression_metadata(tool_id)
            
            node = self.neo4j_service.get_node_by_metadata(
                self.workflow_id, 
                metadata
            )
            
            if not node:
                return None
                
            compression_data = json.loads(node["content"])
            return compression_data.get("compressed_content")
            
        except Exception as e:
            print(f"Error retrieving compressed content for {tool_id}: {str(e)}")
            return None 