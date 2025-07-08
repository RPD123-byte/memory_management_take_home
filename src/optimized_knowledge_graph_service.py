import json
import os
import re
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
from termcolor import colored

from base_knowledge_graph_service import BaseKnowledgeGraphService
from models import RelationshipType, ToolResult, ToolSummary
from tool_summary_prompts import TOOL_SUMMARY_PROMPT
from compressor_implementations import CompressorRegistry


class OptimizedKnowledgeGraphService(BaseKnowledgeGraphService):
    """Hierarchical memory management with age-based compression"""
    
    def __init__(self, workflow_id: str, api_key: Optional[str] = None, max_tokens: int = 100000):
        super().__init__(workflow_id, api_key, max_tokens)
        self.reserved_tokens = 20000
        self.available_tokens = max_tokens - self.reserved_tokens
        
        self.compressor_registry = CompressorRegistry(
            self.neo4j_service, 
            self.token_counter, 
            self.workflow_id,
            self.llm_service
        )

    

    
    def compress_tool_results_hierarchical(self, tool_ids: List[str], target_tokens: int) -> str:
        """Compress tools with progressive detail levels to fit target token count"""
        if not tool_ids:
            return ""
        
        for level_name in ["detailed", "brief", "ultra_compact"]:
            compressed_parts = []
            for tool_id in tool_ids:
                content = self._get_tool_content_at_level(tool_id, level_name)
                if content:
                    compressed_parts.append(content)
            
            if compressed_parts:
                formatted = "\n".join(compressed_parts)
                token_count = self.token_counter.count_tokens(formatted)
                
                if token_count <= target_tokens:
                    print(colored(f"Using {level_name} compression for {len(tool_ids)} tools", "green"))
                    return formatted
        
        # Fallback to ultra_compact
        compressed_parts = []
        for tool_id in tool_ids:
            content = self._get_tool_content_at_level(tool_id, "ultra_compact")
            if content:
                compressed_parts.append(content)
        
        return "\n".join(compressed_parts)
    
    
    def generate_summary(self, tool_id: str) -> Optional[ToolSummary]:
        """Generate all compression levels for a tool result"""
        try:
            tool_node = self.neo4j_service.get_node_by_metadata(
                self.workflow_id, 
                f"tool_result_{tool_id}"
            )
            
            if not tool_node:
                print(colored(f"Tool result {tool_id} not found", "red"))
                return None
                
            tool_content = json.loads(tool_node["content"])
            llm_summary, salient_data = self._generate_tool_summary(tool_content)
            
            # Pass pre-generated summary to avoid duplicate LLM calls
            pre_generated_summary = {
                "summary": llm_summary,
                "salient_data": salient_data
            }
            
            compression_results = self.compressor_registry.compress_tool(
                tool_id, tool_content, pre_generated_summary=pre_generated_summary
            )
            
            brief_content = self.compressor_registry.retrieve_tool(tool_id, "brief")
            if not brief_content:
                brief_content = f"[{tool_id}] {tool_content.get('action_type', 'unknown')}"
            
            summary = ToolSummary(
                tool_id=tool_id,
                summary_content=brief_content,
                salient_data=salient_data,
                token_count=self.token_counter.count_tokens(brief_content),
                timestamp=datetime.now().isoformat()
            )
            
            print(colored(f"Generated all compression levels for {tool_id}: {compression_results}", "green"))
            return summary
            
        except Exception as e:
            print(colored(f"Error generating summary for {tool_id}: {str(e)}", "red"))
            return None
    

    
    def _generate_tool_summary(self, tool_content: Dict[str, Any]) -> Tuple[str, Optional[Any]]:
        try:
            result = self.llm_service.generate_summary(tool_content, TOOL_SUMMARY_PROMPT)
            return result.get("summary", ""), result.get("salient_data")
        except Exception as e:
            print(colored(f"Error in LLM summary generation: {str(e)}", "red"))
            return f"Summary generation failed: {str(e)}", None
    

    
    def compress_tool_results(self, tool_ids: List[str]) -> bool:
        try:
            compressed_content = self.compress_tool_results_hierarchical(tool_ids, self.available_tokens)
            
            compression_id = f"compression_{'-'.join(tool_ids)}"
            compression_content = {
                "compressed_tools": tool_ids,
                "summary": compressed_content,
                "timestamp": datetime.now().isoformat()
            }
            
            self.neo4j_service.update_node(
                metadata=compression_id,
                summary=f"Compression of tools {', '.join(tool_ids)}",
                content=json.dumps(compression_content),
                workflow_id=self.workflow_id
            )
            
            for tool_id in tool_ids:
                self.neo4j_service.update_edge(
                    source_metadata=compression_id,
                    target_metadata=f"tool_result_{tool_id}",
                    relation_type=RelationshipType.COMPRESSES,
                    description=f"Compresses tool {tool_id}",
                    workflow_id=self.workflow_id
                )
                
            print(colored(f"Compressed tools {', '.join(tool_ids)} using hierarchical compression", "green"))
            return True
            
        except Exception as e:
            print(colored(f"Error compressing tools: {str(e)}", "red"))
            return False
    

    
    def generate_dashboard(self, compressed_tool_groups: Optional[Dict[str, Dict[str, Any]]] = None,
                          expanded_tools: Optional[Set[str]] = None) -> str:
                          
        tool_results = self.get_all_tool_results()
        
        if not tool_results:
            return "=== HIERARCHICAL MEMORY DASHBOARD (OPTIMIZED) ===\nNo tool results yet."
        compressed_tool_ids = set()
        if compressed_tool_groups:
            for group in compressed_tool_groups.values():
                compressed_tool_ids.update(group["tool_ids"])
        expanded_tool_ids = set()
        if expanded_tools:
            expanded_tool_ids.update(expanded_tools)
            
        sorted_tools = sorted(tool_results, key=lambda x: int(x.tool_id.split("-")[1]), reverse=True)
        recent_threshold = 2
        moderate_threshold = 5
        
        dashboard_parts = []
        current_tokens = 0
        
        for i, tool in enumerate(sorted_tools):
            tool_id = tool.tool_id

            if tool_id in compressed_tool_ids and tool_id not in expanded_tool_ids:
            
                
                if i < recent_threshold:
                    compression_level = "detailed"
                    content = self._get_tool_content_at_level(tool_id, compression_level)
                    section_header = f"[RECENT] {content}" if content else f"[RECENT] {tool_id}: {tool.action_type} - {tool.status.upper()}"
                        
                elif i < moderate_threshold:
                    compression_level = "brief"
                    content = self._get_tool_content_at_level(tool_id, compression_level)
                    section_header = f"[MODERATE] {content}" if content else f"[MODERATE] {tool_id}: {tool.action_type} - {tool.status.upper()}"
                        
                else:
                    compression_level = "ultra_compact"
                    content = self._get_tool_content_at_level(tool_id, compression_level)
                    section_header = f"[OLD] {content}" if content else f"[OLD] {tool_id}: {tool.action_type}"
                
                section_tokens = self.token_counter.count_tokens(section_header)
                if current_tokens + section_tokens > self.available_tokens:
                    remaining_count = len(sorted_tools) - i
                    dashboard_parts.append(f"... and {remaining_count} more tools (truncated due to token limit)")
                    break
                    
                dashboard_parts.append(section_header)
                current_tokens += section_tokens
            else:
                compression_level = "detailed"
                content = self._get_tool_content_at_level(tool_id, compression_level)
                section_header = f"[EXPANDED] {content}" if content else f"[EXPANDED] {tool_id}: {tool.action_type} - {tool.status.upper()}"
                dashboard_parts.append(section_header)
                current_tokens += section_tokens

        dashboard_content = "\n".join(dashboard_parts)
        total_original_tokens = sum(tool.token_count for tool in tool_results)
        dashboard_tokens = self.token_counter.count_tokens(dashboard_content)
    
        print(f"\n OPTIMIZED SERVICE STATS:")
        print(f"   Original: {total_original_tokens:,} tokens")
        print(f"   Dashboard: {dashboard_tokens:,} tokens")
        print(f"   Compression: {dashboard_tokens/total_original_tokens*100:.1f}%")
        print(f"   Saved: {total_original_tokens - dashboard_tokens:,} tokens")
        
        return dashboard_content
    
 
    
    def _get_tool_content_at_level(self, tool_id: str, level: str) -> Optional[str]:
        try:
            content = self.compressor_registry.retrieve_tool(tool_id, level)
            
            if content:
                return content
            
            tool_node = self.neo4j_service.get_node_by_metadata(
                self.workflow_id,
                f"tool_result_{tool_id}"
            )
            
            if tool_node:
                tool_content = json.loads(tool_node["content"])
                compressor = self.compressor_registry.get_compressor(level)
                if compressor:
                    success = compressor.compress_and_store(tool_id, tool_content, None)
                    if success:
                        return compressor.retrieve_compressed(tool_id)
            
            return None
            
        except Exception as e:
            print(colored(f"Error getting content for {tool_id} at level {level}: {str(e)}", "red"))
            return None 