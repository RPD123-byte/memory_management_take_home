import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
from termcolor import colored

from neo4j_service import Neo4jService
from models import ToolResult, ToolSummary
from token_counter import TokenCounter
from llm_service import LLMService


class BaseKnowledgeGraphService(ABC):
    """Abstract base class for knowledge graph services"""
    
    def __init__(self, workflow_id: str, llm_service: LLMService, max_tokens: int = 100000):
        """
        Initialize the base knowledge graph service
        
        Args:
            workflow_id: Unique identifier for the workflow
            api_key: API key for LLM service
            max_tokens: Maximum tokens available for context
        """
        self.workflow_id = workflow_id
        self.neo4j_service = Neo4jService()
        self.token_counter = TokenCounter()
        self.llm_service = llm_service
        self.max_tokens = max_tokens
        
        # Initialize tool counter based on existing tools in the graph
        self.tool_counter = self._get_next_tool_counter()
        
    def _get_next_tool_counter(self) -> int:
        """Get the next tool counter based on existing tools in the graph"""
        try:
            nodes = self.neo4j_service.get_all_nodes(self.workflow_id)
            max_counter = 0
            
            for node in nodes:
                if node["id"].startswith("tool_result_TR-"):
                    # Extract counter from TR-X format
                    tool_id = node["id"].replace("tool_result_", "")
                    counter = int(tool_id.split("-")[1])
                    max_counter = max(max_counter, counter)
                    
            return max_counter
        except Exception as e:
            print(colored(f"Warning: Could not determine next tool counter: {str(e)}", "yellow"))
            return 0
        
    def close(self):
        """Close Neo4j connection"""
        self.neo4j_service.close()
        
    def _store_tool_result(self, tool_result: ToolResult, content: str):
        """Store tool result in Neo4j"""
        # Create tool result node
        metadata = f"tool_result_{tool_result.tool_id}"
        summary = f"{tool_result.action_type}: {self._extract_brief_params(tool_result.action)} - {tool_result.status.upper()}"
        
        self.neo4j_service.update_node(
            metadata=metadata,
            summary=summary,
            content=content,
            workflow_id=self.workflow_id
        )
        
    def _extract_brief_params(self, action: Dict[str, Any]) -> str:
        """Extract brief parameter description from action"""
        if isinstance(action, dict):
            if "command" in action:
                return action["command"]
            elif "file_path" in action:
                return action["file_path"]
            elif "query" in action:
                return action["query"]
            elif "code" in action:
                return f"code modification ({len(str(action['code']))} chars)"
        return str(action)[:50] + "..." if len(str(action)) > 50 else str(action)
    
    def get_all_tool_results(self) -> List[ToolResult]:
        """Get all tool results for display"""
        tool_results = []
        
        # Get all tool result nodes
        nodes = self.neo4j_service.get_all_nodes(self.workflow_id)
        
        for node in nodes:
            if node["id"].startswith("tool_result_"):
                tool_id = node["id"].replace("tool_result_", "")
                content = json.loads(node["content"])
                
                tool_result = ToolResult(
                    tool_id=tool_id,
                    action_type=content.get("action_type", "unknown"),
                    action=content.get("action", {}),
                    result=content.get("result", {}),
                    timestamp=content.get("timestamp", ""),
                    token_count=self.token_counter.count_tokens(json.dumps(content)),
                    status=content.get("result", {}).get("status", "unknown")
                )
                
                tool_results.append(tool_result)
                
        # Sort by tool ID number
        tool_results.sort(key=lambda x: int(x.tool_id.split("-")[1]))
        return tool_results
    
    def retrieve_tool_result_with_salient_data(self, tool_id: str) -> Optional[str]:
        """
        Retrieve summary with salient data for a tool
        
        Args:
            tool_id: Tool ID to retrieve
            
        Returns:
            Formatted string with summary and salient data
        """
        try:
            # Get summary node
            summary_node = self.neo4j_service.get_node_by_metadata(
                self.workflow_id,
                f"summary_{tool_id}"
            )
            
            if not summary_node:
                return None
                
            summary_content = json.loads(summary_node["content"])
            summary_text = summary_content.get("summary", "")
            salient_data = summary_content.get("salient_data")
            
            # Format the result with salient data
            if salient_data:
                if isinstance(salient_data, dict) and salient_data:
                    salient_parts = []
                    for key, value in salient_data.items():
                        if isinstance(value, str) and len(value) > 50:
                            value = value[:50] + "..."
                        salient_parts.append(f"{key}: {value}")
                    return f"{summary_text} ({', '.join(salient_parts)})"
                    
                elif isinstance(salient_data, str) and salient_data.strip():
                    return f"{summary_text} ({salient_data})"
                        
                elif isinstance(salient_data, list) and salient_data:
                    return f"{summary_text} ({', '.join(str(item) for item in salient_data)})"
            
            return summary_text
                
        except Exception as e:
            print(colored(f"Error retrieving summary with salient data for {tool_id}: {str(e)}", "red"))
            return None
    
    def retrieve_tool_result(self, tool_id: str, summary: bool = False) -> Optional[str]:
        """
        Retrieve full tool result or summary
        
        Args:
            tool_id: Tool ID to retrieve
            summary: If True, return summary instead of full result
            
        Returns:
            Formatted tool result or None if not found
        """
        try:
            if summary:
                # Get summary
                summary_node = self.neo4j_service.get_node_by_metadata(
                    self.workflow_id,
                    f"summary_{tool_id}"
                )
                
                if summary_node:
                    summary_content = json.loads(summary_node["content"])
                    return summary_content["summary"]
                else:
                    return f"Summary not available for {tool_id}"
                    
            else:
                # Get full result
                tool_node = self.neo4j_service.get_node_by_metadata(
                    self.workflow_id,
                    f"tool_result_{tool_id}"
                )
                
                if tool_node:
                    tool_content = json.loads(tool_node["content"])
                    return self._format_full_tool_result(tool_id, tool_content)
                else:
                    return f"Tool result not found for {tool_id}"
                    
        except Exception as e:
            print(colored(f"Error retrieving tool result {tool_id}: {str(e)}", "red"))
            return f"Error retrieving {tool_id}: {str(e)}"
    
    def _format_full_tool_result(self, tool_id: str, content: Dict[str, Any]) -> str:
        """Format full tool result for display"""
        action = content.get("action", {})
        result = content.get("result", {})
        
        lines = [
            f"[{tool_id}] {content.get('action_type', 'unknown')}:",
            f"Input: {json.dumps(action, indent=2)}",
            f"Result: {result.get('status', 'unknown')}",
            f"Output: {result.get('output', 'None')}",
            f"Error: {result.get('error', 'None')}"
        ]
        
        return "\n".join(lines)
    
    def reset_workflow(self):
        """Reset all data for this workflow"""
        self.neo4j_service.reset_graph_by_workflow(self.workflow_id)
        self.tool_counter = 0
        self._reset_additional_state()
        print(colored(f"Reset workflow {self.workflow_id}", "yellow"))
    
    def _reset_additional_state(self):
        """Reset additional state in subclasses - override if needed"""
        pass
    
    def add_tool_result(self, knowledge_entry: Dict[str, Any]) -> str:
        """
        Add a new tool result to the knowledge graph
        
        Args:
            knowledge_entry: Tool execution entry from knowledge sequence
            
        Returns:
            str: Tool ID (e.g., "TR-1")
        """
        # Create new tool
        self.tool_counter += 1
        tool_id = f"TR-{self.tool_counter}"
        
        # Calculate token count for the result
        result_text = json.dumps(knowledge_entry, indent=2)
        token_count = self.token_counter.count_tokens(result_text)
        
        # Create a copy of knowledge_entry and add token_count to it
        content_with_token_count = knowledge_entry.copy()
        content_with_token_count['token_count'] = token_count
        
        # Create tool result
        tool_result = ToolResult(
            tool_id=tool_id,
            action_type=knowledge_entry.get("action_type", "unknown"),
            action=knowledge_entry.get("action", {}),
            result=knowledge_entry.get("result", {}),
            timestamp=knowledge_entry.get("timestamp", datetime.now().isoformat()),
            token_count=token_count,
            status=knowledge_entry.get("result", {}).get("status", "unknown")
        )
        
        # Store in Neo4j
        content_json = json.dumps(content_with_token_count, indent=2)
        self._store_tool_result(tool_result, content_json)
        
        print(colored(f"Added tool result {tool_id} with {token_count} tokens", "green"))
        return tool_id
    
    @abstractmethod
    def generate_summary(self, tool_id: str) -> Optional[ToolSummary]:
        """
        Generate summary for a tool result
        
        Args:
            tool_id: Tool ID to summarize
            
        Returns:
            ToolSummary or None if failed
        """
        pass
    
    @abstractmethod
    def compress_tool_results(self, tool_ids: List[str]) -> bool:
        """
        Compress multiple tool results
        
        Args:
            tool_ids: List of tool IDs to compress
            
        Returns:
            bool: Success status
        """
        pass
    
    @abstractmethod
    def generate_dashboard(self, compressed_tool_groups: Optional[Dict[str, Dict[str, Any]]] = None,
                          expanded_tools: Optional[Set[str]] = None) -> str:
        """
        Generate tool dashboard
        
        Args:
            compressed_tool_groups: Dict mapping group_id -> {tool_ids, summary, timestamp}
            expanded_tools: Set of tool IDs that should show expanded details
            
        Returns:
            str: Formatted tool dashboard
        """
        pass 

    @abstractmethod
    def generate_tool_strings_for_dashboard(self, compressed_tool_groups: Optional[Dict[str, Dict[str, Any]]] = None,
                          expanded_tools: Optional[Set[str]] = None) -> List[str]:
        """
        Generate tool strings for dashboard
        
        Args:
            compressed_tool_groups: Dict mapping group_id -> {tool_ids, summary, timestamp}
            expanded_tools: Set of tool IDs that should show expanded details
            
        Returns:
            List[str]: List of string representations of tool results for dashboard
        """
        pass