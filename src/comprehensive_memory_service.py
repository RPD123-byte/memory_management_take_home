"""
Comprehensive Memory Management Service with Hierarchical Memory.

This service provides true memory management inspired by LlamaIndex and LangChain:
- Hierarchical memory blocks (working, static, facts, vector)
- Intelligent memory consolidation
- Context-aware retrieval
- Memory compression and expansion
- Fact extraction from executions
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
from termcolor import colored

from neo4j_service import Neo4jService
from models import RelationshipType, ToolResult, ToolSummary
from token_counter import TokenCounter
from llm_service import LLMService, Message
from structured_prompts import STRUCTURED_TOOL_SUMMARY_PROMPT
from tool_fingerprint import ToolFingerprintService
from semantic_memory import SemanticMemoryService
from hierarchical_memory import (
    HierarchicalMemoryManager,
    StaticMemoryBlock,
    FactExtractionMemoryBlock,
    VectorMemoryBlock,
    WorkingMemoryBlock
)


class ComprehensiveMemoryService:
    """
    Comprehensive memory management service with hierarchical memory architecture.
    
    Features:
    - Fingerprint-based deduplication
    - Semantic similarity search
    - Hierarchical memory blocks (working, static, facts, vector)
    - Automatic fact extraction
    - Context-aware memory retrieval
    - Memory compression strategies
    """
    
    def __init__(self, workflow_id: str, api_key: str = None, 
                 enable_semantic_memory: bool = True,
                 enable_fact_extraction: bool = True):
        """
        Initialize comprehensive memory service.
        
        Args:
            workflow_id: Unique workflow identifier
            api_key: API key for LLM and embeddings
            enable_semantic_memory: Enable vector-based semantic search
            enable_fact_extraction: Enable automatic fact extraction
        """
        self.workflow_id = workflow_id
        self.neo4j_service = Neo4jService()
        self.token_counter = TokenCounter()
        self.llm_service = LLMService(api_key=api_key)
        
        # Deduplication service
        self.fingerprint_service = ToolFingerprintService()
        
        # Semantic memory
        self.enable_semantic_memory = enable_semantic_memory
        if enable_semantic_memory:
            try:
                self.semantic_memory = SemanticMemoryService(api_key=api_key)
            except Exception as e:
                print(colored(f"Warning: Semantic memory disabled: {e}", "yellow"))
                self.enable_semantic_memory = False
                self.semantic_memory = None
        else:
            self.semantic_memory = None
        
        # Hierarchical memory manager
        self.memory_manager = HierarchicalMemoryManager(
            llm_service=self.llm_service if enable_fact_extraction else None,
            semantic_service=self.semantic_memory if enable_semantic_memory else None
        )
        
        # Tool counter
        self.tool_counter = self._get_next_tool_counter()
        
        # Metrics
        self.metrics = {
            "total_tools": 0,
            "duplicates_detected": 0,
            "semantic_matches_found": 0,
            "facts_extracted": 0,
            "tokens_saved": 0,
            "memory_compressions": 0
        }
        
    def _get_next_tool_counter(self) -> int:
        """Get next tool counter"""
        try:
            nodes = self.neo4j_service.get_all_nodes(self.workflow_id)
            max_counter = 0
            for node in nodes:
                if node["id"].startswith("tool_result_TR-"):
                    tool_id = node["id"].replace("tool_result_", "")
                    counter = int(tool_id.split("-")[1])
                    max_counter = max(max_counter, counter)
            return max_counter
        except Exception as e:
            return 0
    
    def initialize_static_memory(self, workspace_path: str, 
                                detected_languages: Dict[str, Any],
                                configured_integrations: List[str]):
        """
        Initialize static memory with project context.
        
        Args:
            workspace_path: Path to workspace
            detected_languages: Detected IaC languages
            configured_integrations: Available integrations
        """
        static_block = self.memory_manager.blocks.get("static")
        if isinstance(static_block, StaticMemoryBlock):
            static_block.add_project_info(
                workspace_path, detected_languages, configured_integrations
            )
            print(colored("Static memory initialized with project context", "green"))
    
    def set_current_plan(self, goal: str, steps: List[Dict[str, Any]]):
        """
        Set current plan in working memory.
        
        Args:
            goal: Plan goal
            steps: List of plan steps
        """
        working_block = self.memory_manager.blocks.get("working")
        if isinstance(working_block, WorkingMemoryBlock):
            working_block.set_plan(goal, steps)
    
    def add_context_note(self, note: str):
        """Add a context note to working memory"""
        working_block = self.memory_manager.blocks.get("working")
        if isinstance(working_block, WorkingMemoryBlock):
            working_block.add_context_note(note)
    
    def check_for_duplicates(self, action_type: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for duplicate tool calls before execution.
        
        Returns:
            Dict with duplicate information and recommendations
        """
        result = {
            "is_exact_duplicate": False,
            "previous_tool_id": None,
            "similar_tools": [],
            "relevant_facts": [],
            "recommendation": "",
            "should_skip": False
        }
        
        # Check exact duplicate
        is_dup, prev_id = self.fingerprint_service.is_duplicate_call(action_type, action)
        
        if is_dup:
            result["is_exact_duplicate"] = True
            result["previous_tool_id"] = prev_id
            result["should_skip"] = True
            result["recommendation"] = (
                f"[WARNING] EXACT DUPLICATE: Already executed as {prev_id}. "
                f"Reuse that result instead of re-executing."
            )
            self.metrics["duplicates_detected"] += 1
            return result
        
        # Search vector memory block (uses semantic search)
        vector_block = self.memory_manager.blocks.get("vector")
        if vector_block and isinstance(vector_block, VectorMemoryBlock):
            # Create a search query from action
            search_text = f"{action_type}: {self._extract_brief_params(action)}"
            similar = vector_block.search(search_text, top_k=3)
            
            if similar:
                # Filter by similarity threshold
                filtered_similar = [s for s in similar if s.get("similarity_score", 0) >= 0.75]
                
                if filtered_similar:
                    result["similar_tools"] = filtered_similar
                    self.metrics["semantic_matches_found"] += 1
                    
                    top_match = filtered_similar[0]
                    if top_match.get("similarity_score", 0) > 0.9:
                        result["should_skip"] = True
                        result["recommendation"] = (
                            f"[WARNING] HIGHLY SIMILAR: {top_match['tool_id']} "
                            f"({top_match['similarity_score']:.1%} similar) may have your answer."
                        )
                    else:
                        result["recommendation"] = (
                            f"[INFO] Similar tools: {', '.join([t['tool_id'] for t in filtered_similar])}. "
                            f"Check these first."
                        )
        
        # Search facts
        fact_block = self.memory_manager.blocks.get("facts")
        if isinstance(fact_block, FactExtractionMemoryBlock):
            # Extract key terms for fact search
            search_terms = []
            if "command" in action:
                search_terms.extend(action["command"].split()[:3])
            elif "file_path" in action:
                search_terms.append(action["file_path"])
            elif "query" in action:
                search_terms.extend(action["query"].split()[:3])
            
            for term in search_terms:
                matching_facts = fact_block.search_facts(term)
                result["relevant_facts"].extend(matching_facts)
            
            if result["relevant_facts"]:
                if not result["recommendation"]:
                    result["recommendation"] = ""
                result["recommendation"] += f"\n   Found {len(result['relevant_facts'])} relevant facts in memory."
        
        return result
    
    def add_tool_result(self, knowledge_entry: Dict[str, Any],
                       check_duplicates: bool = True) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Add tool result with comprehensive memory management.
        
        Args:
            knowledge_entry: Tool execution entry
            check_duplicates: Check for duplicates
            
        Returns:
            Tuple of (tool_id, duplicate_info)
        """
        action_type = knowledge_entry.get("action_type", "unknown")
        action = knowledge_entry.get("action", {})
        result = knowledge_entry.get("result", {})
        
        # Check duplicates
        dup_info = None
        if check_duplicates:
            dup_info = self.check_for_duplicates(action_type, action)
            if dup_info["should_skip"]:
                print(colored(dup_info["recommendation"], "yellow"))
                if dup_info["is_exact_duplicate"]:
                    return dup_info["previous_tool_id"], dup_info
        
        # Increment counter
        self.tool_counter += 1
        tool_id = f"TR-{self.tool_counter}"
        
        # Store in Neo4j
        result_text = json.dumps(knowledge_entry, indent=2)
        token_count = self.token_counter.count_tokens(result_text)
        
        tool_result = ToolResult(
            tool_id=tool_id,
            action_type=action_type,
            action=action,
            result=result,
            timestamp=knowledge_entry.get("timestamp", datetime.now().isoformat()),
            token_count=token_count,
            status=result.get("status", "unknown")
        )
        
        self._store_tool_result(tool_result, result_text)
        
        # Register fingerprint
        fp = self.fingerprint_service.generate_fingerprint(action_type, action)
        self.fingerprint_service.register_fingerprint(fp.fingerprint, tool_id)
        
        # Update working memory
        self.memory_manager.add_to_memory("working", {
            "type": "active_tool",
            "tool_id": tool_id
        })
        
        # Update metrics
        self.metrics["total_tools"] += 1
        
        print(colored(f"Added tool result {tool_id} ({token_count} tokens)", "green"))
        
        return tool_id, dup_info
    
    def _store_tool_result(self, tool_result: ToolResult, content: str):
        """Store tool result in Neo4j"""
        metadata = f"tool_result_{tool_result.tool_id}"
        summary = f"{tool_result.action_type}: {self._extract_brief_params(tool_result.action)} - {tool_result.status.upper()}"
        
        self.neo4j_service.update_node(
            metadata=metadata,
            summary=summary,
            content=content,
            workflow_id=self.workflow_id
        )
    
    def _extract_brief_params(self, action: Dict[str, Any]) -> str:
        """Extract brief description"""
        if "command" in action:
            return action["command"][:50]
        elif "file_path" in action:
            return action["file_path"]
        elif "query" in action:
            return action["query"][:50]
        return str(action)[:50]
    
    def generate_structured_summary(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """
        Generate structured summary and update all memory blocks.
        
        Args:
            tool_id: Tool ID to summarize
            
        Returns:
            Structured summary dict
        """
        try:
            # Get tool from Neo4j
            tool_node = self.neo4j_service.get_node_by_metadata(
                self.workflow_id, f"tool_result_{tool_id}"
            )
            
            if not tool_node:
                return None
            
            tool_content = json.loads(tool_node["content"])
            
            # Generate structured summary
            summary_result = self._generate_structured_summary_llm(tool_content)
            if not summary_result:
                return None
            
            # Extract components
            summary_text = summary_result.get("summary", "")
            salient_data = summary_result.get("salient_data", {})
            
            # Store in Neo4j
            token_count = self.token_counter.count_tokens(json.dumps(summary_result))
            summary_content = {
                **summary_result,
                "token_count": token_count,
                "timestamp": datetime.now().isoformat(),
                "tool_id": tool_id
            }
            
            summary_metadata = f"structured_summary_{tool_id}"
            self.neo4j_service.update_node(
                metadata=summary_metadata,
                summary=f"Summary of {tool_id}",
                content=json.dumps(summary_content),
                workflow_id=self.workflow_id
            )
            
            self.neo4j_service.update_edge(
                source_metadata=f"tool_result_{tool_id}",
                target_metadata=summary_metadata,
                relation_type=RelationshipType.SUMMARIZES,
                description=f"Structured summary of {tool_id}",
                workflow_id=self.workflow_id
            )
            
            # Update memory blocks
            self._update_memory_blocks(tool_id, tool_content, summary_result)
            
            # Mark tool as complete in working memory
            self.memory_manager.add_to_memory("working", {
                "type": "complete_tool",
                "tool_id": tool_id
            })
            
            print(colored(f"Generated structured summary for {tool_id}", "green"))
            
            return summary_content
            
        except Exception as e:
            print(colored(f"Error generating summary for {tool_id}: {e}", "red"))
            return None
    
    def _generate_structured_summary_llm(self, tool_content: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate structured summary using LLM"""
        try:
            messages = [
                Message(role="system", content=STRUCTURED_TOOL_SUMMARY_PROMPT),
                Message(role="user", content=json.dumps(tool_content, indent=2))
            ]
            
            response = self.llm_service.generate(messages, json_mode=True)
            return json.loads(response)
        except Exception as e:
            print(colored(f"LLM summary generation failed: {e}", "red"))
            return None
    
    def _update_memory_blocks(self, tool_id: str, tool_content: Dict[str, Any],
                             summary_result: Dict[str, Any]):
        """Update hierarchical memory blocks with tool execution"""
        
        # Update vector memory
        self.memory_manager.add_to_memory("vector", {
            "tool_id": tool_id,
            "tool_name": tool_content.get("action_type", "unknown"),
            "parameters": tool_content.get("action", {}),
            "result": tool_content.get("result", {}),
            "summary": summary_result.get("summary", "")
        })
        
        # Update fact extraction memory
        if "facts" in self.memory_manager.blocks:
            self.memory_manager.add_to_memory("facts", {
                "tool_id": tool_id,
                "summary": summary_result.get("summary", ""),
                "result": tool_content.get("result", {})
            })
            
            # Count facts
            fact_block = self.memory_manager.blocks["facts"]
            if isinstance(fact_block, FactExtractionMemoryBlock):
                self.metrics["facts_extracted"] = len(fact_block.facts)
    
    def get_full_memory_context(self, max_tokens: int = 15000) -> str:
        """
        Get complete memory context for agent prompt.
        
        Args:
            max_tokens: Maximum tokens for context
            
        Returns:
            Formatted memory context string
        """
        return self.memory_manager.get_full_context(max_tokens=max_tokens)
    
    def search_memory(self, query: str) -> Dict[str, Any]:
        """
        Search across all memory blocks.
        
        Args:
            query: Natural language query
            
        Returns:
            Dict with results from each memory type
        """
        return self.memory_manager.search_memory(query, search_vectors=True, search_facts=True)
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of memory usage"""
        memory_summary = self.memory_manager.get_memory_summary()
        memory_summary["metrics"] = self.metrics
        return memory_summary
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.metrics,
            "duplicate_rate": (self.metrics["duplicates_detected"] / max(self.metrics["total_tools"], 1)) * 100,
            "semantic_hit_rate": (self.metrics["semantic_matches_found"] / max(self.metrics["total_tools"], 1)) * 100
        }
    
    def close(self):
        """Close connections"""
        self.neo4j_service.close()
        self.memory_manager.clear_all()
    
    def reset_workflow(self):
        """Reset workflow and all memory"""
        self.neo4j_service.reset_graph_by_workflow(self.workflow_id)
        self.tool_counter = 0
        self.fingerprint_service.clear_cache()
        self.memory_manager.clear_all()
        self.metrics = {
            "total_tools": 0,
            "duplicates_detected": 0,
            "semantic_matches_found": 0,
            "facts_extracted": 0,
            "tokens_saved": 0,
            "memory_compressions": 0
        }
        print(colored(f"Reset workflow {self.workflow_id}", "yellow"))
    
    # Backward compatibility methods
    def get_all_tool_results(self) -> List[ToolResult]:
        """Get all tool results"""
        tool_results = []
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
        
        tool_results.sort(key=lambda x: int(x.tool_id.split("-")[1]))
        return tool_results
