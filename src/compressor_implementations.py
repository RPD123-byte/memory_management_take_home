import json
import re
from typing import Dict, Any, Optional, Tuple

from compressor_interface import ToolCompressor
from llm_service import LLMService, Message


class UltraCompressor(ToolCompressor):
    """Ultra-compact compression using LLM"""
    
    def __init__(self, neo4j_service, token_counter, workflow_id: str, llm_service: LLMService):
        super().__init__(neo4j_service, token_counter, workflow_id)
        self.llm_service = llm_service
    
    def get_compression_type(self) -> str:
        return "ultra_compact"
    
    def compress_and_store(self, tool_id: str, tool_content: Dict[str, Any], 
                          pre_generated_summary: Optional[Dict[str, Any]] = None) -> bool:
        try:
            compressed_content = self._compress_to_ultra_compact(tool_id, tool_content)
            return self._store_compressed_content(tool_id, compressed_content)
        except Exception as e:
            print(f"Error in ultra compression for {tool_id}: {str(e)}")
            return False
    
    def retrieve_compressed(self, tool_id: str) -> Optional[str]:
        return self._retrieve_compressed_content(tool_id)
    
    def _compress_to_ultra_compact(self, tool_id: str, tool_content: Dict[str, Any]) -> str:
        try:
            from tool_summary_prompts import ULTRA_COMPACT_PROMPT

            messages = [
                Message(role="system", content=ULTRA_COMPACT_PROMPT),
                Message(role="user", content=json.dumps(tool_content, indent=2))
            ]
            
            response = self.llm_service.generate(messages)
            ultra_compact = response.strip()
            
            if not ultra_compact.startswith(f"[{tool_id}]"):
                ultra_compact = f"[{tool_id}] {ultra_compact}"
            
            return ultra_compact
            
        except Exception as e:
            print(f"Error generating ultra-compact summary with LLM: {str(e)}")
            action_type = tool_content.get("action_type", "unknown")
            status = tool_content.get("result", {}).get("status", "unknown")
            status_symbol = "✓" if status == "success" else "✗"
            return f"[{tool_id}] {action_type}{status_symbol}"


class SummaryCompressor(ToolCompressor):
    """Summary compression - stores ToolSummary output"""
    
    def __init__(self, neo4j_service, token_counter, workflow_id: str, llm_service: LLMService):
        super().__init__(neo4j_service, token_counter, workflow_id)
        self.llm_service = llm_service
    
    def get_compression_type(self) -> str:
        return "summary"
    
    def compress_and_store(self, tool_id: str, tool_content: Dict[str, Any], 
                          pre_generated_summary: Optional[Dict[str, Any]] = None) -> bool:
        try:
            compressed_content = self._compress_to_summary(tool_id, tool_content, pre_generated_summary)
            return self._store_compressed_content(tool_id, compressed_content)
        except Exception as e:
            print(f"Error in summary compression for {tool_id}: {str(e)}")
            return False
    
    def retrieve_compressed(self, tool_id: str) -> Optional[str]:
        return self._retrieve_compressed_content(tool_id)
    
    def _compress_to_summary(self, tool_id: str, tool_content: Dict[str, Any], 
                            pre_generated_summary: Optional[Dict[str, Any]] = None) -> str:
        try:
            if pre_generated_summary:
                # Use pre-generated summary to avoid duplicate LLM call
                summary = pre_generated_summary.get("summary", "")
                salient_data = pre_generated_summary.get("salient_data")
            else:
                # Generate summary if not provided
                from tool_summary_prompts import TOOL_SUMMARY_PROMPT
                result = self.llm_service.generate_summary(tool_content, TOOL_SUMMARY_PROMPT)
                summary = result.get("summary", "")
                salient_data = result.get("salient_data")
            
            lines = [f"[{tool_id}] Summary: {summary}"]
            
            if salient_data:
                if isinstance(salient_data, dict):
                    salient_str = json.dumps(salient_data, indent=2)
                else:
                    salient_str = str(salient_data)
                lines.append(f"  Salient Data: {salient_str}")
            
            return "\n".join(lines)
            
        except Exception as e:
            print(f"Error generating summary with LLM: {str(e)}")
            action_type = tool_content.get("action_type", "unknown")
            status = tool_content.get("result", {}).get("status", "unknown")
            return f"[{tool_id}] {action_type} - {status.upper()}"


class DetailedCompressor(ToolCompressor):
    """Detailed compression - stores raw ToolResult"""
    
    def get_compression_type(self) -> str:
        return "detailed"
    
    def compress_and_store(self, tool_id: str, tool_content: Dict[str, Any], 
                          pre_generated_summary: Optional[Dict[str, Any]] = None) -> bool:
        try:
            compressed_content = self._compress_to_detailed(tool_id, tool_content)
            return self._store_compressed_content(tool_id, compressed_content)
        except Exception as e:
            print(f"Error in detailed compression for {tool_id}: {str(e)}")
            return False
    
    def retrieve_compressed(self, tool_id: str) -> Optional[str]:
        return self._retrieve_compressed_content(tool_id)
    
    def _compress_to_detailed(self, tool_id: str, tool_content: Dict[str, Any]) -> str:
        lines = []
        status = tool_content.get("result", {}).get("status", "unknown")
        warning = " ⚠️" if status == "error" or tool_content.get("token_count", 0) > 5000 else ""
        
        lines.append(f"[{tool_id}] {tool_content.get('action_type', 'unknown')} - {status} ({tool_content.get('token_count', 0):,} tokens){warning}")
        lines.append(f"Input: {json.dumps(tool_content.get('action', {}))}")
        lines.append(f"Result: {status.lower()}")
        
        output = tool_content.get("result", {}).get("output", "")
        if output:
            lines.append(f"Output: {output}")
        
        error = tool_content.get("result", {}).get("error", "")
        if error:
            lines.append(f"Error: {error}")
        
        return "\n".join(lines)


class CompressorRegistry:
    """Registry for managing different compressor types"""
    
    def __init__(self, neo4j_service, token_counter, workflow_id: str, llm_service: LLMService):
        self.neo4j_service = neo4j_service
        self.token_counter = token_counter
        self.workflow_id = workflow_id
        
        self.compressors = {
            "ultra_compact": UltraCompressor(neo4j_service, token_counter, workflow_id, llm_service),
            "summary": SummaryCompressor(neo4j_service, token_counter, workflow_id, llm_service),
            "detailed": DetailedCompressor(neo4j_service, token_counter, workflow_id)
        }
    
    def get_compressor(self, compression_type: str) -> Optional[ToolCompressor]:
        return self.compressors.get(compression_type)
    
    def compress_tool(self, tool_id: str, tool_content: Dict[str, Any], 
                     compression_types: Optional[list] = None,
                     pre_generated_summary: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        if compression_types is None:
            compression_types = list(self.compressors.keys())
        
        results = {}
        for compression_type in compression_types:
            compressor = self.get_compressor(compression_type)
            if compressor:
                results[compression_type] = compressor.compress_and_store(tool_id, tool_content, pre_generated_summary)
            else:
                results[compression_type] = False
        
        return results
    
    def retrieve_tool(self, tool_id: str, compression_type: str) -> Optional[str]:
        compressor = self.get_compressor(compression_type)
        if compressor:
            return compressor.retrieve_compressed(tool_id)
        return None 