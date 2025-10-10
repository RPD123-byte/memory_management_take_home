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
from concurrent.futures import ThreadPoolExecutor, Future

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
                 enable_fact_extraction: bool = True,
                 auto_compress: bool = False,
                 auto_compress_threshold: int = 10):
        """
        Initialize comprehensive memory service.
        
        Args:
            workflow_id: Unique workflow identifier
            api_key: API key for LLM and embeddings
            enable_semantic_memory: Enable vector-based semantic search
            enable_fact_extraction: Enable automatic fact extraction
            auto_compress: Enable automatic compression when threshold reached
            auto_compress_threshold: Number of tools before auto-compression triggers
        """
        self.workflow_id = workflow_id
        self.neo4j_service = Neo4jService()
        self.token_counter = TokenCounter()
        self.llm_service = LLMService(api_key=api_key)
        
        # Auto-compression settings
        self.auto_compress = auto_compress
        self.auto_compress_threshold = auto_compress_threshold
        self.tools_since_last_compress = 0
        
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
        
        # Async fact extraction
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="fact_extractor")
        self.pending_extractions: List[Future] = []
        
        # Embedding cache for duplicates (avoids expensive re-embedding)
        self.embedding_cache: Dict[str, Any] = {}
        
        # Tool counter
        self.tool_counter = self._get_next_tool_counter()
        
        # Compression tracking
        self.compression_groups: Dict[str, Dict[str, Any]] = {}
        self.compressed_tools: Set[str] = set()
        
        # Metrics
        self.metrics = {
            "total_tools": 0,
            "duplicates_detected": 0,
            "semantic_matches_found": 0,
            "facts_extracted": 0,
            "tokens_saved": 0,
            "memory_compressions": 0,
            "compression_groups_created": 0
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
            # Skip semantic search for small datasets (optimization)
            similar = None
            if len(vector_block.recent_tools) < 10:
                pass  # Skip search when fewer than 10 tools stored
            else:
                # Create a search query from action
                search_text = f"{action_type}: {self._extract_brief_params(action)}"
                similar = vector_block.search(search_text, top_k=3)
            
            if similar:
                # Filter by similarity threshold (tuned to 0.67 to balance recall vs precision)
                filtered_similar = [s for s in similar if s.get("similarity_score", 0) >= 0.67]
                
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
        
        # Check for auto-compression
        if self.auto_compress:
            self.tools_since_last_compress += 1
            if self.tools_since_last_compress >= self.auto_compress_threshold:
                self._try_auto_compress()
        
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
        
        # Update vector memory (fast, no LLM call)
        self.memory_manager.add_to_memory("vector", {
            "tool_id": tool_id,
            "tool_name": tool_content.get("action_type", "unknown"),
            "parameters": tool_content.get("action", {}),
            "result": tool_content.get("result", {}),
            "summary": summary_result.get("summary", "")
        })
        
        # Update fact extraction memory ASYNCHRONOUSLY (slow, has LLM call)
        if "facts" in self.memory_manager.blocks:
            # Submit to background thread instead of blocking
            future = self.executor.submit(
                self._async_extract_facts,
                tool_id,
                summary_result.get("summary", ""),
                tool_content.get("result", {})
            )
            self.pending_extractions.append(future)
    
    def _async_extract_facts(self, tool_id: str, summary: str, result: Dict[str, Any]):
        """Background task for fact extraction"""
        try:
            self.memory_manager.add_to_memory("facts", {
                "tool_id": tool_id,
                "summary": summary,
                "result": result
            })
            
            # Update metrics
            fact_block = self.memory_manager.blocks["facts"]
            if isinstance(fact_block, FactExtractionMemoryBlock):
                self.metrics["facts_extracted"] = len(fact_block.facts)
        except Exception as e:
            print(colored(f"Warning: Async fact extraction failed for {tool_id}: {e}", "yellow"))
    
    def wait_for_pending_extractions(self, timeout: float = 30.0):
        """
        Wait for all pending fact extractions to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        if not self.pending_extractions:
            return
        
        print(colored(f"Waiting for {len(self.pending_extractions)} pending fact extractions...", "cyan"))
        
        for future in self.pending_extractions:
            try:
                future.result(timeout=timeout)
            except Exception as e:
                print(colored(f"Warning: Fact extraction failed: {e}", "yellow"))
        
        self.pending_extractions.clear()
        print(colored("All fact extractions completed", "green"))
    
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
    
    def compress_tool_results(self, tool_ids: List[str], group_name: Optional[str] = None) -> bool:
        """
        Compress multiple tool results into a summary group to save context space.
        This provides explicit agent control over memory compression.
        
        Args:
            tool_ids: List of tool IDs to compress (e.g., ["TR-1", "TR-2", "TR-3"])
            group_name: Optional name for the compression group
            
        Returns:
            bool: True if compression successful
            
        Example:
            # Agent realizes memory pressure is high
            service.compress_tool_results(["TR-1", "TR-2", "TR-3"], "aws_setup_tools")
        """
        if not tool_ids:
            return False
        
        try:
            # Generate group ID
            group_id = group_name or f"CG-{len(self.compression_groups) + 1}"
            
            # Fetch full tool results
            tool_results = []
            total_tokens_before = 0
            
            for tool_id in tool_ids:
                if tool_id in self.compressed_tools:
                    print(colored(f"Warning: {tool_id} already compressed, skipping", "yellow"))
                    continue
                
                # Get from Neo4j using get_node_by_metadata
                node = self.neo4j_service.get_node_by_metadata(self.workflow_id, f"tool_result_{tool_id}")
                if node:
                    content = json.loads(node["content"])
                    tool_results.append({
                        "tool_id": tool_id,
                        "content": content
                    })
                    total_tokens_before += self.token_counter.count_tokens(json.dumps(content))
            
            if not tool_results:
                return False
            
            # Generate combined summary using LLM
            combined_summary = self._generate_compression_summary(tool_results)
            summary_tokens = self.token_counter.count_tokens(combined_summary)
            
            # Create compression group
            compression_group = {
                "group_id": group_id,
                "tool_ids": [tr["tool_id"] for tr in tool_results],
                "combined_summary": combined_summary,
                "created_at": datetime.now().isoformat(),
                "tokens_before": total_tokens_before,
                "tokens_after": summary_tokens,
                "tokens_saved": total_tokens_before - summary_tokens
            }
            
            # Store in Neo4j using update_node
            self.neo4j_service.update_node(
                metadata=f"compression_group_{group_id}",
                summary=combined_summary,
                content=json.dumps(compression_group),
                workflow_id=self.workflow_id
            )
            
            # Update tracking
            self.compression_groups[group_id] = compression_group
            for tool_id in compression_group["tool_ids"]:
                self.compressed_tools.add(tool_id)
            
            # Update metrics
            self.metrics["memory_compressions"] += 1
            self.metrics["compression_groups_created"] += 1
            self.metrics["tokens_saved"] += compression_group["tokens_saved"]
            
            # Update working memory (remove full tools, add group reference)
            working_block = self.memory_manager.blocks.get("working")
            if isinstance(working_block, WorkingMemoryBlock):
                working_block.add_compression_group(group_id, combined_summary)
            
            print(colored(
                f"✓ Compressed {len(tool_results)} tools into {group_id} "
                f"(saved {compression_group['tokens_saved']} tokens)",
                "green"
            ))
            
            return True
            
        except Exception as e:
            print(colored(f"Compression failed: {e}", "red"))
            return False
    
    def _generate_compression_summary(self, tool_results: List[Dict[str, Any]]) -> str:
        """Generate combined summary for compression group"""
        
        # Build prompt
        tools_text = []
        for tr in tool_results:
            content = tr["content"]
            tools_text.append(
                f"Tool {tr['tool_id']}:\n"
                f"  Action: {content.get('action_type')}\n"
                f"  Input: {json.dumps(content.get('action', {}), indent=2)}\n"
                f"  Result: {json.dumps(content.get('result', {}), indent=2)}\n"
            )
        
        prompt = f"""You are compressing multiple related tool executions into a single summary.

Tools to compress:
{chr(10).join(tools_text)}

Generate a concise but comprehensive summary that:
1. Describes what these tools accomplished together
2. Preserves key information that might be needed later
3. Notes any important outputs or state changes
4. Is written in third person past tense

Summary:"""
        
        try:
            messages = [
                Message(role="system", content="You are a memory compression assistant."),
                Message(role="user", content=prompt)
            ]
            
            summary = self.llm_service.generate(messages)
            return summary.strip()
            
        except Exception as e:
            print(colored(f"LLM compression failed, using fallback: {e}", "yellow"))
            # Fallback: simple concatenation
            return f"Compressed group containing {len(tool_results)} tools: " + \
                   ", ".join([tr["tool_id"] for tr in tool_results])
    
    def _try_auto_compress(self):
        """
        Attempt automatic compression of old tools.
        Groups tools by type and compresses older groups.
        """
        try:
            # Get all tools
            all_tools = self.get_all_tool_results()
            
            # Filter out already compressed tools
            uncompressed = [
                t for t in all_tools 
                if t.tool_id not in self.compressed_tools
            ]
            
            if len(uncompressed) < self.auto_compress_threshold:
                return
            
            # Group tools by action type
            groups = {}
            for tool in uncompressed[:-5]:  # Keep 5 most recent uncompressed
                action_type = tool.action_type
                if action_type not in groups:
                    groups[action_type] = []
                groups[action_type].append(tool.tool_id)
            
            # Compress largest group
            if groups:
                largest_group = max(groups.items(), key=lambda x: len(x[1]))
                action_type, tool_ids = largest_group
                
                if len(tool_ids) >= 3:  # Only compress if 3+ tools
                    print(colored(
                        f"\n[AUTO-COMPRESS] Compressing {len(tool_ids)} {action_type} tools...",
                        "cyan"
                    ))
                    self.compress_tool_results(tool_ids, f"auto_{action_type}")
                    self.tools_since_last_compress = 0
                    
        except Exception as e:
            print(colored(f"Auto-compression failed: {e}", "yellow"))
    
    def expand_tool_result(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """
        Expand a compressed tool result to see full details.
        Agent calls this when it needs specific information from a compressed tool.
        
        Args:
            tool_id: Tool ID to expand (e.g., "TR-1")
            
        Returns:
            Dict with full tool details, or None if not found
            
        Example:
            # Agent needs specific info from compressed tool
            full_details = service.expand_tool_result("TR-1")
        """
        try:
            # Check if tool is compressed
            if tool_id not in self.compressed_tools:
                print(colored(f"{tool_id} is not compressed, fetching normally", "cyan"))
            
            # Fetch full details from Neo4j
            node = self.neo4j_service.get_node_by_metadata(self.workflow_id, f"tool_result_{tool_id}")
            if not node:
                print(colored(f"Tool {tool_id} not found", "red"))
                return None
            
            content = json.loads(node["content"])
            
            # Add back to working memory temporarily (with expansion flag)
            working_block = self.memory_manager.blocks.get("working")
            if isinstance(working_block, WorkingMemoryBlock):
                working_block.add_expanded_tool(tool_id, content)
            
            print(colored(f"✓ Expanded {tool_id} (full details restored)", "green"))
            
            return content
            
        except Exception as e:
            print(colored(f"Expansion failed: {e}", "red"))
            return None
    
    def get_compression_summary(self) -> Dict[str, Any]:
        """Get summary of compression state"""
        return {
            "total_groups": len(self.compression_groups),
            "compressed_tools": len(self.compressed_tools),
            "total_tokens_saved": self.metrics.get("tokens_saved", 0),
            "groups": {
                group_id: {
                    "tool_count": len(group["tool_ids"]),
                    "tokens_saved": group["tokens_saved"],
                    "created_at": group["created_at"]
                }
                for group_id, group in self.compression_groups.items()
            }
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
        self.compression_groups = {}
        self.compressed_tools = set()
        self.metrics = {
            "total_tools": 0,
            "duplicates_detected": 0,
            "semantic_matches_found": 0,
            "facts_extracted": 0,
            "tokens_saved": 0,
            "memory_compressions": 0,
            "compression_groups_created": 0
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
