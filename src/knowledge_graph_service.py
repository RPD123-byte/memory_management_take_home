import json
import os
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
from termcolor import colored

from base_knowledge_graph_service import BaseKnowledgeGraphService
from models import RelationshipType, ToolResult, ToolSummary
from tool_summary_prompts import TOOL_SUMMARY_PROMPT
from llm_service import LLMService


class KnowledgeGraphService(BaseKnowledgeGraphService):
    """Service for managing tool results using a knowledge graph approach"""
    
    def __init__(self, workflow_id: str, llm_service: LLMService):
        """
        Initialize the knowledge graph service
        
        Args:
            workflow_id: Unique identifier for the workflow
            llm_service: LLM service to generate
        """
        super().__init__(workflow_id, llm_service)

        
    def generate_summary(self, tool_id: str) -> Optional[ToolSummary]:
        """
        Generate summary for a tool result
        
        Args:
            tool_id: Tool ID to summarize
            
        Returns:
            ToolSummary or None if failed
        """
        try:
            # Get tool result from Neo4j
            tool_node = self.neo4j_service.get_node_by_metadata(
                self.workflow_id, 
                f"tool_result_{tool_id}"
            )
            
            if not tool_node:
                print(colored(f"Tool result {tool_id} not found", "red"))
                return None
                
            # Parse tool content
            tool_content = json.loads(tool_node["content"])
            
            # Generate summary using LLM
            summary_content, salient_data = self._generate_tool_summary(tool_content)
            
            # Calculate token count
            token_count_str = summary_content
            if salient_data:
                if isinstance(salient_data, (dict, list)):
                    token_count_str += json.dumps(salient_data)
                else:
                    token_count_str += str(salient_data)
            
            # Create summary object
            summary = ToolSummary(
                tool_id=tool_id,
                summary_content=summary_content,
                salient_data=salient_data,
                token_count=self.token_counter.count_tokens(token_count_str),
                timestamp=datetime.now().isoformat()
            )
            
            # Store summary in Neo4j
            self._store_tool_summary(summary)
            
            print(colored(f"Generated summary for {tool_id}", "green"))
            return summary
            
        except Exception as e:
            print(colored(f"Error generating summary for {tool_id}: {str(e)}", "red"))
            return None
            
    def _generate_tool_summary(self, tool_content: Dict[str, Any]) -> Tuple[str, Optional[Any]]:
        """Generate summary and salient data for a tool result"""
        try:
            result = self.llm_service.generate_summary(tool_content, TOOL_SUMMARY_PROMPT)
            
            summary = result.get("summary", "")
            salient_data = result.get("salient_data")
            
            return summary, salient_data
            
        except Exception as e:
            print(colored(f"Error in LLM summary generation: {str(e)}", "red"))
            return f"Summary generation failed: {str(e)}", None
            
    def _store_tool_summary(self, summary: ToolSummary):
        """Store tool summary in Neo4j"""
        # Create summary node
        summary_metadata = f"summary_{summary.tool_id}"
        
        # Prepare content
        summary_content = {
            "summary": summary.summary_content,
            "salient_data": summary.salient_data,
            "token_count": summary.token_count,
            "timestamp": summary.timestamp
        }
        
        # Serialize the entire content to JSON for storage
        content_json = json.dumps(summary_content)
        
        self.neo4j_service.update_node(
            metadata=summary_metadata,
            summary=f"Summary of {summary.tool_id}",
            content=content_json,
            workflow_id=self.workflow_id
        )
        
        # Create relationship between tool and summary
        self.neo4j_service.update_edge(
            source_metadata=f"tool_result_{summary.tool_id}",
            target_metadata=summary_metadata,
            relation_type=RelationshipType.SUMMARIZES,
            description=f"Summary of tool result {summary.tool_id}",
            workflow_id=self.workflow_id
        )
        

        
    def compress_tool_results(self, tool_ids: List[str]) -> bool:
        """
        Compress multiple tool results
        
        Args:
            tool_ids: List of tool IDs to compress
            
        Returns:
            bool: Success status
        """
        try:
            # Get individual summaries for all tools
            summaries = []
            
            for tool_id in tool_ids:
                # Try to get existing summary
                summary_node = self.neo4j_service.get_node_by_metadata(
                    self.workflow_id,
                    f"summary_{tool_id}"
                )
                
                if summary_node:
                    summary_content = json.loads(summary_node["content"])
                    summaries.append(f"[{tool_id}] {summary_content['summary']}")
                else:
                    # Generate summary if it doesn't exist
                    self.generate_summary(tool_id)
                    summary_node = self.neo4j_service.get_node_by_metadata(
                        self.workflow_id,
                        f"summary_{tool_id}"
                    )
                    if summary_node:
                        summary_content = json.loads(summary_node["content"])
                        summaries.append(f"[{tool_id}] {summary_content['summary']}")
                    else:
                        summaries.append(f"[{tool_id}] Summary not available")
            
            # Create compression node with individual summaries
            compression_id = f"compression_{'-'.join(tool_ids)}"
            compression_content = {
                "compressed_tools": tool_ids,
                "summary": " | ".join(summaries),
                "timestamp": datetime.now().isoformat()
            }
            
            self.neo4j_service.update_node(
                metadata=compression_id,
                summary=f"Compression of tools {', '.join(tool_ids)}",
                content=json.dumps(compression_content),
                workflow_id=self.workflow_id
            )
            
            # Create relationships to compressed tools
            for tool_id in tool_ids:
                self.neo4j_service.update_edge(
                    source_metadata=compression_id,
                    target_metadata=f"tool_result_{tool_id}",
                    relation_type=RelationshipType.COMPRESSES,
                    description=f"Compresses tool {tool_id}",
                    workflow_id=self.workflow_id
                )
                
            print(colored(f"Compressed tools {', '.join(tool_ids)}", "green"))
            return True
            
        except Exception as e:
            print(colored(f"Error compressing tools: {str(e)}", "red"))
            return False
        

    def generate_tool_strings_for_dashboard(self, compressed_tool_groups: Optional[Dict[str, Dict[str, Any]]] = None,
                          expanded_tools: Optional[Set[str]] = None) -> List[str]:
        if compressed_tool_groups is None:
            compressed_tool_groups = {}
        if expanded_tools is None:
            expanded_tools = set()
        
        # Get all tool results
        tool_results = self.get_all_tool_results()
        
        if not tool_results:
            return [];
        
        # Build a set of all compressed tool IDs for quick lookup
        compressed_tool_ids = set()
        for group_info in compressed_tool_groups.values():
            compressed_tool_ids.update(group_info.get("tool_ids", []))
        
        # Generate dashboard
        tools_str = []
        
        for tool in tool_results:
            tool_id = tool.tool_id
            
            tool_str = ""
            # Check if this tool is compressed
            if tool_id in compressed_tool_ids and tool_id not in expanded_tools:
                # Show compressed version
                summary_with_data = self.retrieve_tool_result_with_salient_data(tool_id)
                if summary_with_data:
                    tool_str = f"[{tool_id}] {summary_with_data} [COMPRESSED]"
                else:
                    summary = self.retrieve_tool_result(tool_id, summary=True)
                    tool_str = f"[{tool_id}] {summary} [COMPRESSED]"
            else:
                # Show full expanded view
                status = tool.status.upper()
                warning = " ⚠️" if tool.status == "error" or tool.token_count > 5000 else ""
                
                tool_str = f"[{tool_id}] {tool.action_type} - {status} ({tool.token_count:,} tokens){warning}\n"
                tool_str += f"Input: {json.dumps(tool.action)}\n"
                tool_str += f"Result: {status.lower()}\n"
                
                output = tool.result.get("output", "")
                if output:
                    tool_str += f"Output: {output}\n"
                
                error = tool.result.get("error", "")
                if error:
                    tool_str += f"Error: {error}\n"
            
            tools_str.append(tool_str)

        return tools_str;
    
    def generate_dashboard(self, compressed_tool_groups: Optional[Dict[str, Dict[str, Any]]] = None,
                          expanded_tools: Optional[Set[str]] = None) -> str:
        """
        Generate tool dashboard with compression/expansion state
        
        Args:
            compressed_tool_groups: Dict mapping group_id -> {tool_ids, summary, timestamp}
            expanded_tools: Set of tool IDs that should show expanded details
            
        Returns:
            str: Formatted tool dashboard
        """
        
        # Get all tool results
        tool_results = self.get_all_tool_results()
        
        if not tool_results:
            return "=== ACTIVE TOOL RESULTS ===\nNo tool results yet."
        
        # Generate dashboard
        lines = ["=== ACTIVE TOOL RESULTS ==="]
        tools_str = self.generate_tool_strings_for_dashboard(compressed_tool_groups, expanded_tools)
        lines.extend(tools_str)
        total_tokens = sum([tool.token_count for tool in tool_results])
        
        # Calculate dashboard token count
        dashboard_content = "\n".join(lines)
        dashboard_tokens = self.token_counter.count_tokens(dashboard_content)

        # Print comparison info
        print(f"\n ORIGINAL SERVICE STATS:")
        print(f"   Original: {total_tokens:,} tokens")
        print(f"   Dashboard: {dashboard_tokens:,} tokens")
        print(f"   Compression: {dashboard_tokens/total_tokens*100:.1f}%")
        print(f"   Saved: {total_tokens - dashboard_tokens:,} tokens")
        
        return dashboard_content 