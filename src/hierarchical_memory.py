"""
Hierarchical Memory Blocks for Agent Memory Management.

Inspired by LlamaIndex, this module provides different types of
memory blocks that work together to create a comprehensive memory system:

1. Static Memory Block - Non-changing information (project context, configurations)
2. Fact Extraction Memory Block - Extracted facts from tool executions
3. Vector Memory Block - Embedding-based retrieval for conversation history
4. Working Memory Block - Short-term, immediately relevant information
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import json
from termcolor import colored

from llm_service import LLMService, Message
from semantic_memory import SemanticMemoryService
from token_counter import TokenCounter


class BaseMemoryBlock(ABC):
    """Base class for all memory blocks"""
    
    def __init__(self, name: str, priority: int = 0, max_tokens: int = 2000):
        """
        Initialize memory block.
        
        Args:
            name: Name of the memory block
            priority: Priority for memory retrieval (lower = higher priority)
            max_tokens: Maximum tokens this block can use
        """
        self.name = name
        self.priority = priority
        self.max_tokens = max_tokens
        self.token_counter = TokenCounter()
        
    @abstractmethod
    def get(self) -> str:
        """Get formatted content from this memory block"""
        pass
    
    @abstractmethod
    def put(self, data: Dict[str, Any]):
        """Add data to this memory block"""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear this memory block"""
        pass
    
    def get_token_count(self) -> int:
        """Get current token usage"""
        content = self.get()
        return self.token_counter.count_tokens(content) if content else 0
    
    def is_within_limit(self) -> bool:
        """Check if within token limit"""
        return self.get_token_count() <= self.max_tokens


@dataclass
class StaticInfo:
    """Static information entry"""
    key: str
    value: str
    category: str
    timestamp: str


class StaticMemoryBlock(BaseMemoryBlock):
    """
    Memory block for static, non-changing information.
    
    Examples:
    - Project configuration (workspace path, languages, integrations)
    - User preferences
    - System constraints
    - Environment information
    """
    
    def __init__(self, name: str = "static_info", priority: int = 0, max_tokens: int = 3000):
        super().__init__(name, priority, max_tokens)
        self.static_info: Dict[str, StaticInfo] = {}
        
    def put(self, data: Dict[str, Any]):
        """
        Add static information.
        
        Args:
            data: Dict with 'key', 'value', 'category'
        """
        key = data.get("key")
        value = data.get("value")
        category = data.get("category", "general")
        
        if not key or not value:
            return
        
        self.static_info[key] = StaticInfo(
            key=key,
            value=value,
            category=category,
            timestamp=datetime.now().isoformat()
        )
        
    def get(self) -> str:
        """Get formatted static information"""
        if not self.static_info:
            return ""
        
        # Group by category
        by_category: Dict[str, List[StaticInfo]] = {}
        for info in self.static_info.values():
            if info.category not in by_category:
                by_category[info.category] = []
            by_category[info.category].append(info)
        
        # Format output
        lines = ["=== STATIC CONTEXT ==="]
        for category, infos in sorted(by_category.items()):
            lines.append(f"\n{category.upper()}:")
            for info in infos:
                lines.append(f"  • {info.key}: {info.value}")
        
        return "\n".join(lines)
    
    def clear(self):
        """Clear all static information"""
        self.static_info.clear()
    
    def add_project_info(self, workspace_path: str, detected_languages: Dict[str, Any],
                        configured_integrations: List[str]):
        """Convenience method to add project information"""
        self.put({"key": "workspace_path", "value": workspace_path, "category": "project"})
        
        for integration in configured_integrations:
            self.put({"key": f"integration_{integration.lower()}", 
                     "value": "available", "category": "integrations"})
        
        for folder, languages in detected_languages.items():
            for lang, version in languages.items():
                self.put({"key": f"{folder}_{lang}", 
                         "value": version, "category": "languages"})


@dataclass
class Fact:
    """Extracted fact"""
    fact_id: str
    content: str
    source_tool_id: str
    confidence: float
    category: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class FactExtractionMemoryBlock(BaseMemoryBlock):
    """
    Memory block that extracts and stores facts from tool executions.
    
    Uses LLM to extract key facts that should be remembered:
    - Resource identifiers (ARNs, IDs, file paths)
    - Configuration values
    - Status information
    - Relationships between entities
    """
    
    FACT_EXTRACTION_PROMPT = """
<role>
You are an expert at extracting key facts from tool execution results that an AI agent should remember.
</role>

<task>
Extract 3-5 discrete, memorable facts from this tool execution. Be LIBERAL with extraction - if something could be useful, extract it!

Focus on:
1. Resource identifiers (ARNs, IDs, file paths, URLs)
2. Configuration values and settings
3. Status information and states
4. Relationships between resources
5. Important discoveries or outcomes
6. Capabilities and features discovered
7. Error patterns and constraints

IMPORTANT: Extract multiple facts per tool execution. Aim for 3-5 facts minimum.
Even simple tool executions contain multiple learnable facts (e.g., "tool exists", "tool works", "tool has parameter X", "tool returned format Y").

Return a JSON array of facts. Each fact should be:
- Self-contained and understandable without context
- Actionable or referenceable
- Stable (won't change frequently)
</task>

<output_format>
{
    "facts": [
        {
            "content": "Clear, specific fact statement",
            "category": "resource|config|status|relationship|discovery",
            "confidence": 0.0-1.0,
            "metadata": {"key": "value"}
        }
    ]
}
</output_format>

<examples>
Input: Created file app/database.py with SQLAlchemy configuration
Output:
{
    "facts": [
        {
            "content": "Database configuration file exists at app/database.py",
            "category": "resource",
            "confidence": 1.0,
            "metadata": {"file_type": "python", "purpose": "database_config"}
        },
        {
            "content": "Project uses SQLAlchemy for database connections",
            "category": "config",
            "confidence": 1.0,
            "metadata": {"library": "sqlalchemy"}
        }
    ]
}

Input: Listed IAM groups for user 'admin', found group 'Administrators' with ARN arn:aws:iam::123:group/Administrators
Output:
{
    "facts": [
        {
            "content": "User 'admin' belongs to IAM group 'Administrators'",
            "category": "relationship",
            "confidence": 1.0,
            "metadata": {"user": "admin", "group": "Administrators"}
        },
        {
            "content": "IAM group 'Administrators' has ARN arn:aws:iam::123:group/Administrators",
            "category": "resource",
            "confidence": 1.0,
            "metadata": {"resource_type": "iam_group", "group_name": "Administrators"}
        }
    ]
}
</examples>

Extract facts that will be useful for future decision-making.
"""
    
    def __init__(self, name: str = "extracted_facts", priority: int = 1, 
                 max_facts: int = 100, llm_service: Optional[LLMService] = None):
        super().__init__(name, priority, max_tokens=5000)
        self.max_facts = max_facts
        self.facts: Dict[str, Fact] = {}
        self.fact_counter = 0
        self.llm_service = llm_service
        
    def put(self, data: Dict[str, Any]):
        """
        Extract and store facts from tool execution.
        
        Args:
            data: Dict with 'tool_id', 'summary', 'result'
        """
        if not self.llm_service:
            return
        
        tool_id = data.get("tool_id", "unknown")
        summary = data.get("summary", "")
        result = data.get("result", {})
        
        # Extract facts using LLM
        try:
            facts_data = self._extract_facts(summary, result)
            
            for fact_data in facts_data.get("facts", []):
                self._add_fact(
                    content=fact_data.get("content", ""),
                    category=fact_data.get("category", "general"),
                    confidence=fact_data.get("confidence", 0.5),
                    source_tool_id=tool_id,
                    metadata=fact_data.get("metadata", {})
                )
        except Exception as e:
            print(colored(f"Warning: Fact extraction failed: {e}", "yellow"))
    
    def _extract_facts(self, summary: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract facts using LLM"""
        context = f"Summary: {summary}\n\nResult: {json.dumps(result, indent=2)}"
        
        messages = [
            Message(role="system", content=self.FACT_EXTRACTION_PROMPT),
            Message(role="user", content=context)
        ]
        
        response = self.llm_service.generate(messages, json_mode=True)
        return json.loads(response)
    
    def _add_fact(self, content: str, category: str, confidence: float,
                  source_tool_id: str, metadata: Dict[str, Any]):
        """Add a fact to memory"""
        self.fact_counter += 1
        fact_id = f"FACT-{self.fact_counter}"
        
        fact = Fact(
            fact_id=fact_id,
            content=content,
            source_tool_id=source_tool_id,
            confidence=confidence,
            category=category,
            timestamp=datetime.now().isoformat(),
            metadata=metadata
        )
        
        self.facts[fact_id] = fact
        
        # Enforce max facts limit
        if len(self.facts) > self.max_facts:
            # Remove lowest confidence fact
            to_remove = min(self.facts.values(), key=lambda f: f.confidence)
            del self.facts[to_remove.fact_id]
    
    def get(self) -> str:
        """Get formatted facts"""
        if not self.facts:
            return ""
        
        # Group by category
        by_category: Dict[str, List[Fact]] = {}
        for fact in self.facts.values():
            if fact.category not in by_category:
                by_category[fact.category] = []
            by_category[fact.category].append(fact)
        
        # Format output
        lines = ["=== EXTRACTED FACTS ==="]
        for category, facts in sorted(by_category.items()):
            lines.append(f"\n{category.upper()}:")
            for fact in sorted(facts, key=lambda f: f.confidence, reverse=True)[:10]:
                lines.append(f"  • {fact.content} (confidence: {fact.confidence:.0%})")
        
        return "\n".join(lines)
    
    def clear(self):
        """Clear all facts"""
        self.facts.clear()
        self.fact_counter = 0
    
    def get_facts_by_category(self, category: str) -> List[Fact]:
        """Get facts by category"""
        return [f for f in self.facts.values() if f.category == category]
    
    def search_facts(self, query: str) -> List[Fact]:
        """Search facts by keyword"""
        query_lower = query.lower()
        return [f for f in self.facts.values() 
                if query_lower in f.content.lower()]


class VectorMemoryBlock(BaseMemoryBlock):
    """
    Memory block using vector embeddings for semantic search.
    
    Stores tool execution history and enables semantic retrieval of
    relevant past executions.
    """
    
    def __init__(self, name: str = "vector_memory", priority: int = 2,
                 max_tokens: int = 10000, semantic_service: Optional[SemanticMemoryService] = None):
        super().__init__(name, priority, max_tokens)
        self.semantic_service = semantic_service
        self.recent_tools: List[str] = []  # Track recent tool IDs
        self.max_recent = 50
        
    def put(self, data: Dict[str, Any]):
        """
        Add tool execution to vector memory.
        
        Args:
            data: Dict with 'tool_id', 'tool_name', 'parameters', 'result', 'summary'
        """
        if not self.semantic_service:
            return
        
        tool_id = data.get("tool_id")
        if not tool_id:
            return
        
        try:
            self.semantic_service.add_memory(
                tool_id=tool_id,
                tool_name=data.get("tool_name", "unknown"),
                parameters=data.get("parameters", {}),
                result=data.get("result", {}),
                summary=data.get("summary", "")
            )
            
            # Track recent tools
            self.recent_tools.append(tool_id)
            if len(self.recent_tools) > self.max_recent:
                self.recent_tools.pop(0)
                
        except Exception as e:
            print(colored(f"Warning: Vector memory add failed: {e}", "yellow"))
    
    def get(self) -> str:
        """Get summary of vector memory"""
        if not self.recent_tools:
            return ""
        
        lines = ["=== VECTOR MEMORY (Recent Tools) ==="]
        lines.append(f"Total tools in memory: {len(self.recent_tools)}")
        lines.append(f"Recent: {', '.join(self.recent_tools[-5:])}")
        
        return "\n".join(lines)
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search vector memory"""
        if not self.semantic_service:
            return []
        
        similar_results = self.semantic_service.search_by_natural_language(
            query, top_k=top_k, similarity_threshold=0.6
        )
        
        # Convert SimilarToolResult objects to dicts
        return [
            {
                "tool_id": result.tool_id,
                "tool_name": result.tool_name,
                "similarity_score": result.similarity_score,
                "summary": result.metadata.get("summary", "") if result.metadata else ""
            }
            for result in similar_results
        ]
    
    def clear(self):
        """Clear vector memory"""
        if self.semantic_service:
            self.semantic_service.clear_memories()
        self.recent_tools.clear()


class WorkingMemoryBlock(BaseMemoryBlock):
    """
    Short-term working memory for immediately relevant information.
    
    Stores:
    - Current plan steps
    - Active tool executions
    - Immediate context
    - Pending decisions
    """
    
    def __init__(self, name: str = "working_memory", priority: int = -1, max_tokens: int = 4000):
        super().__init__(name, priority, max_tokens)
        self.current_plan: Optional[Dict[str, Any]] = None
        self.active_tools: List[str] = []
        self.context_notes: List[str] = []
        self.max_context_notes = 10
        # Compression support
        self.compression_groups: Dict[str, str] = {}  # group_id -> summary
        self.expanded_tools: Dict[str, Dict[str, Any]] = {}  # tool_id -> full content
        
    def put(self, data: Dict[str, Any]):
        """
        Add to working memory.
        
        Args:
            data: Dict with 'type' and relevant data
        """
        data_type = data.get("type")
        
        if data_type == "plan":
            self.current_plan = data.get("plan")
        elif data_type == "active_tool":
            tool_id = data.get("tool_id")
            if tool_id and tool_id not in self.active_tools:
                self.active_tools.append(tool_id)
        elif data_type == "context_note":
            note = data.get("note")
            if note:
                self.context_notes.append(note)
                if len(self.context_notes) > self.max_context_notes:
                    self.context_notes.pop(0)
        elif data_type == "complete_tool":
            tool_id = data.get("tool_id")
            if tool_id in self.active_tools:
                self.active_tools.remove(tool_id)
    
    def get(self) -> str:
        """Get formatted working memory"""
        lines = ["=== WORKING MEMORY ==="]
        
        if self.current_plan:
            lines.append("\nCurrent Plan:")
            lines.append(f"  Goal: {self.current_plan.get('goal', 'N/A')}")
            steps = self.current_plan.get('steps', [])
            if steps:
                lines.append(f"  Steps: {len(steps)} total")
                for i, step in enumerate(steps[:3], 1):
                    lines.append(f"    {i}. {step.get('name', 'Unknown')}")
        
        if self.active_tools:
            lines.append(f"\nActive Tools: {', '.join(self.active_tools)}")
        
        if self.context_notes:
            lines.append("\nRecent Context:")
            for note in self.context_notes[-3:]:
                lines.append(f"  • {note}")
        
        # Add compression info
        compression_info = self.get_compression_info()
        if compression_info:
            lines.append(compression_info)
        
        return "\n".join(lines)
    
    def clear(self):
        """Clear working memory"""
        self.current_plan = None
        self.active_tools.clear()
        self.context_notes.clear()
    
    def set_plan(self, goal: str, steps: List[Dict[str, Any]]):
        """Set current plan"""
        self.current_plan = {"goal": goal, "steps": steps}
    
    def add_context_note(self, note: str):
        """Add a context note"""
        self.put({"type": "context_note", "note": note})
    
    def add_compression_group(self, group_id: str, summary: str):
        """Add a compression group reference to working memory"""
        self.compression_groups[group_id] = summary
    
    def add_expanded_tool(self, tool_id: str, content: Dict[str, Any]):
        """Temporarily add expanded tool to working memory"""
        self.expanded_tools[tool_id] = content
    
    def get_compression_info(self) -> str:
        """Get formatted compression info"""
        if not self.compression_groups and not self.expanded_tools:
            return ""
        
        lines = []
        if self.compression_groups:
            lines.append("\n=== COMPRESSED GROUPS ===")
            for group_id, summary in self.compression_groups.items():
                lines.append(f"[{group_id}] {summary}")
        
        if self.expanded_tools:
            lines.append("\n=== EXPANDED TOOLS (Full Details) ===")
            for tool_id in self.expanded_tools.keys():
                lines.append(f"[{tool_id}] Full details available")
        
        return "\n".join(lines)


class HierarchicalMemoryManager:
    """
    Manages multiple memory blocks in a hierarchical system.
    
    Coordinates between different memory types and provides unified access.
    """
    
    def __init__(self, llm_service: Optional[LLMService] = None,
                 semantic_service: Optional[SemanticMemoryService] = None):
        """Initialize memory manager with services"""
        self.llm_service = llm_service
        self.semantic_service = semantic_service
        
        # Initialize memory blocks
        self.blocks: Dict[str, BaseMemoryBlock] = {}
        
        # Working memory (highest priority)
        self.blocks["working"] = WorkingMemoryBlock(priority=-1)
        
        # Static memory
        self.blocks["static"] = StaticMemoryBlock(priority=0)
        
        # Fact extraction
        if llm_service:
            self.blocks["facts"] = FactExtractionMemoryBlock(
                priority=1, llm_service=llm_service
            )
        
        # Vector memory
        if semantic_service:
            self.blocks["vector"] = VectorMemoryBlock(
                priority=2, semantic_service=semantic_service
            )
    
    def add_to_memory(self, block_name: str, data: Dict[str, Any]):
        """Add data to a specific memory block"""
        if block_name in self.blocks:
            self.blocks[block_name].put(data)
    
    def get_full_context(self, max_tokens: Optional[int] = None) -> str:
        """
        Get full memory context, respecting priorities and token limits.
        
        Args:
            max_tokens: Maximum tokens for context (None = no limit)
            
        Returns:
            Formatted memory context
        """
        # Sort blocks by priority
        sorted_blocks = sorted(self.blocks.values(), key=lambda b: b.priority)
        
        sections = []
        total_tokens = 0
        
        for block in sorted_blocks:
            content = block.get()
            if not content:
                continue
            
            block_tokens = block.get_token_count()
            
            if max_tokens and total_tokens + block_tokens > max_tokens:
                # Skip if over limit
                continue
            
            sections.append(content)
            total_tokens += block_tokens
        
        return "\n\n".join(sections)
    
    def search_memory(self, query: str, search_vectors: bool = True,
                     search_facts: bool = True) -> Dict[str, Any]:
        """
        Search across memory blocks.
        
        Args:
            query: Search query
            search_vectors: Search vector memory
            search_facts: Search extracted facts
            
        Returns:
            Dict with search results from each block
        """
        results = {}
        
        # Search vector memory
        if search_vectors and "vector" in self.blocks:
            vector_block = self.blocks["vector"]
            if isinstance(vector_block, VectorMemoryBlock):
                results["vector_matches"] = vector_block.search(query, top_k=5)
        
        # Search facts
        if search_facts and "facts" in self.blocks:
            fact_block = self.blocks["facts"]
            if isinstance(fact_block, FactExtractionMemoryBlock):
                results["fact_matches"] = fact_block.search_facts(query)
        
        return results
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of all memory blocks"""
        summary = {}
        
        for name, block in self.blocks.items():
            summary[name] = {
                "priority": block.priority,
                "token_count": block.get_token_count(),
                "max_tokens": block.max_tokens,
                "within_limit": block.is_within_limit()
            }
            
            # Add block-specific info
            if isinstance(block, FactExtractionMemoryBlock):
                summary[name]["fact_count"] = len(block.facts)
            elif isinstance(block, StaticMemoryBlock):
                summary[name]["info_count"] = len(block.static_info)
            elif isinstance(block, VectorMemoryBlock):
                summary[name]["tool_count"] = len(block.recent_tools)
        
        return summary
    
    def clear_all(self):
        """Clear all memory blocks"""
        for block in self.blocks.values():
            block.clear()
    
    def clear_block(self, block_name: str):
        """Clear a specific memory block"""
        if block_name in self.blocks:
            self.blocks[block_name].clear()
