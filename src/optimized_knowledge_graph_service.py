import json
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
from termcolor import colored

from base_knowledge_graph_service import BaseKnowledgeGraphService
from models import ToolSummary
from tool_summary_prompts import TOOL_SUMMARY_PROMPT
from compressor_implementations import CompressorRegistry
from compression_analyzer import CompressionAnalyzer
from token_budget_manager import TokenBudgetManager
from llm_service import LLMService


class OptimizedKnowledgeGraphService(BaseKnowledgeGraphService):
    """Hierarchical memory management with content-aware and dynamic compression"""
    
    def __init__(self, workflow_id: str, llm_service: LLMService, max_tokens: int = 100000):
        super().__init__(workflow_id, llm_service, max_tokens)
        
        self.compressor_registry = CompressorRegistry(
            self.neo4j_service, 
            self.token_counter, 
            self.workflow_id,
            self.llm_service
        )
        self.compression_analyzer = CompressionAnalyzer()
        self.token_budget_manager = TokenBudgetManager(max_tokens)

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
            
            summary_content = self.compressor_registry.retrieve_tool(tool_id, "summary")
            if not summary_content:
                summary_content = f"[{tool_id}] {tool_content.get('action_type', 'unknown')}"
            
            summary = ToolSummary(
                tool_id=tool_id,
                summary_content=summary_content,
                salient_data=salient_data,
                token_count=self.token_counter.count_tokens(summary_content),
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
        """
        Compress multiple tool results
        
        Args:
            tool_ids: List of tool IDs to compress
            
        Returns:
            bool: Success status
        """
        # TODO: Not really used, so not supporting it for now
        return True
    
    def generate_dashboard(self, compressed_tool_groups: Optional[Dict[str, Dict[str, Any]]] = None,
                          expanded_tools: Optional[Set[str]] = None) -> str:
                          
        tool_results = self.get_all_tool_results()
        if not tool_results:
            return "=== HIERARCHICAL MEMORY DASHBOARD (OPTIMIZED) ===\nNo tool results yet."

        total_original_tokens = sum(tool.token_count for tool in tool_results)
        
        # Generate dashboard with feedback loop for token optimization
        max_iterations = 2
        iteration = 0
        dashboard_content = ""
        
        while iteration < max_iterations:
            tools_str = self._generate_tool_strings_with_budget(
                compressed_tool_groups, expanded_tools, iteration > 0
            )
            
            lines = ["=== HIERARCHICAL MEMORY DASHBOARD (OPTIMIZED) ==="]
            lines.extend(tools_str)
            dashboard_content = "\n".join(lines)
            
            current_tokens = self.token_counter.count_tokens(dashboard_content)
            
            print(f"\n Iteration {iteration + 1}:")
            print(f"   Token usage: {current_tokens:,} / {self.max_tokens:,} ({current_tokens/self.max_tokens*100:.1f}%)")
            
            # Check if we're within acceptable limits
            if current_tokens < self.max_tokens * 0.95:  # 95% threshold
                break
                
            iteration += 1
        
        if iteration >= max_iterations:
            print(colored(f"Warning: Max iterations ({max_iterations}) reached in dashboard generation", "yellow"))
        
        dashboard_tokens = self.token_counter.count_tokens(dashboard_content)
    
        print(f"\n OPTIMIZED SERVICE STATS:")
        print(f"   Original: {total_original_tokens:,} tokens")
        print(f"   Dashboard: {dashboard_tokens:,} tokens")
        print(f"   Compression: {dashboard_tokens/total_original_tokens*100:.1f}%")
        print(f"   Saved: {total_original_tokens - dashboard_tokens:,} tokens")
        
        return dashboard_content
    
    def _generate_tool_strings_with_budget(self, compressed_tool_groups: Optional[Dict[str, Dict[str, Any]]] = None,
                          expanded_tools: Optional[Set[str]] = None, use_token_budget: bool = True) -> List[str]:
        """Generate tool strings with optional token budget awareness"""
        tool_results = self.get_all_tool_results()

        if not tool_results:
            return []
        
        # Calculate current token usage if using budget
        token_budget = None
        if use_token_budget:
            # Estimate current usage based on tool results
            estimated_tokens = sum(t.token_count for t in tool_results) // 2  # Rough estimate
            token_budget = self.token_budget_manager.calculate_budget(estimated_tokens)
        
        compressed_tool_ids = set()
        if compressed_tool_groups:
            for group in compressed_tool_groups.values():
                compressed_tool_ids.update(group["tool_ids"])
        expanded_tool_ids = set()
        if expanded_tools:
            expanded_tool_ids.update(expanded_tools)
        
        tools_str = []
        total_tools = len(tool_results)
        
        for i, tool in enumerate(tool_results):
            tool_id = tool.tool_id
            
            if tool_id in expanded_tool_ids:
                # Always show expanded view for explicitly expanded tools
                compression_level = "detailed"
                content = self.compressor_registry.retrieve_tool(tool_id, compression_level)
                section_header = f"{content}" if content else f"{tool_id}: {tool.action_type} - {tool.status.upper()}"
            elif tool_id in compressed_tool_ids:
                # Use content-aware compression with token budget for compressed tools
                tool_node = self.neo4j_service.get_node_by_metadata(
                    self.workflow_id, 
                    f"tool_result_{tool_id}"
                )
                
                if tool_node:
                    tool_content = json.loads(tool_node["content"])
                    content_score = self.compression_analyzer.analyze_tool_content(
                        tool_content, i, total_tools, token_budget
                    )
                    compression_level = content_score.compression_level
                else:
                    # fallback if node not found
                    compression_level = "summary"
                
                content = self.compressor_registry.retrieve_tool(tool_id, compression_level)
                assert content is not None, f"Content is None for tool {tool_id}"
                if compression_level == "detailed":
                    section_header = f"{content}" if content else f"{tool_id}: {tool.action_type} - {tool.status.upper()}"
                elif compression_level == "summary":
                    section_header = f"[COMPRESSED] {content}" if content else f"[COMPRESSED] {tool_id}: {tool.action_type} - {tool.status.upper()}"
                else:  # ultra_compact
                    section_header = f"[ULTRA-COMPRESSED] {content}" if content else f"[ULTRA-COMPRESSED] {tool_id}: {tool.action_type}"
            else:
                compression_level = "detailed"
                content = self.compressor_registry.retrieve_tool(tool_id, compression_level)
                assert content is not None, f"Content is None for tool {tool_id}"
                section_header = f"{content}" if content else f"{tool_id}: {tool.action_type} - {tool.status.upper()}"
            
            section_header += "\n"
            tools_str.append(section_header)

        return tools_str
        
    def generate_tool_strings_for_dashboard(self, compressed_tool_groups: Optional[Dict[str, Dict[str, Any]]] = None,
                          expanded_tools: Optional[Set[str]] = None) -> List[str]:
        """Legacy method for compatibility, now uses budget-aware method"""
        return self._generate_tool_strings_with_budget(compressed_tool_groups, expanded_tools, use_token_budget=True)