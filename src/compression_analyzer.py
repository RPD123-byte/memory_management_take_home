"""Content-aware compression analysis for tool results."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Any, Optional

from token_budget_manager import TokenBudget, PressureLevel


class ToolImportance(IntEnum):
    """Tool importance levels for compression decisions."""
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1


@dataclass
class ContentScore:
    """Compression analysis result for a tool."""
    importance: ToolImportance
    keep_detailed_until: int
    compression_level: str


class CompressionAnalyzer:
    """Analyzes tool content to determine optimal compression strategy."""
    
    CRITICAL_ACTIONS = {
        'delete_file', 'modify_code', 'create_file'
    }
    
    HIGH_PRIORITY_ACTIONS = {
        'execute_command', 'call_integration_method', 'run_file'
    }
    
    MEDIUM_PRIORITY_ACTIONS = {
        'read_file_contents', 'query_codebase', 'search_documentation',
        'search_internet', 'retrieve_integration_methods', 
        'ask_human_question', 'get_tool_result'
    }
    
    LOW_PRIORITY_ACTIONS = {
        'request_human_intervention', 'workflow_complete', 
        'compress_tool_results'
    }
    
    ERROR_STATES = {'error', 'failed', 'failure'}
    
    # Retention thresholds by importance
    BASE_RETENTION = {
        ToolImportance.CRITICAL: 20,
        ToolImportance.HIGH: 10,
        ToolImportance.MEDIUM: 5,
        ToolImportance.LOW: 2
    }
    
    def analyze_tool_content(
        self, 
        tool_content: Dict[str, Any], 
        position: int, 
        total_tools: int, 
        token_budget: Optional[TokenBudget] = None
    ) -> ContentScore:
        """
        Analyze tool content and determine compression strategy.
        
        Args:
            tool_content: Tool result data
            position: Tool position in sequence (0-based)
            total_tools: Total number of tools
            token_budget: Current token budget state
            
        Returns:
            ContentScore with compression recommendations
        """
        action_type = tool_content.get('action_type', '')
        status = tool_content.get('result', {}).get('status', '').lower()
        output_size = len(str(tool_content.get('result', {}).get('output', '')))
        token_count = tool_content.get('token_count', 0)
        
        importance = self._calculate_importance(
            action_type, status, output_size, token_count
        )
        
        keep_detailed = self._calculate_retention(
            importance, position, total_tools, token_budget
        )
        
        compression = self._determine_compression(
            importance, position, keep_detailed, token_budget
        )
        
        return ContentScore(
            importance=importance,
            keep_detailed_until=keep_detailed,
            compression_level=compression
        )
    
    def _calculate_importance(
        self, 
        action_type: str, 
        status: str, 
        output_size: int, 
        token_count: int
    ) -> ToolImportance:
        """Calculate tool importance based on action and result."""
        # Errors are always critical
        if status in self.ERROR_STATES:
            return ToolImportance.CRITICAL
            
        # Check action categories
        if action_type in self.CRITICAL_ACTIONS:
            return ToolImportance.CRITICAL
            
        if action_type in self.HIGH_PRIORITY_ACTIONS:
            # Large outputs might contain important logs
            if output_size > 1000 or token_count > 2000:
                return ToolImportance.HIGH
            return ToolImportance.MEDIUM
            
        if action_type in self.MEDIUM_PRIORITY_ACTIONS:
            # Size-based importance for info gathering
            if token_count > 3000:
                return ToolImportance.MEDIUM
            return ToolImportance.LOW
            
        if action_type in self.LOW_PRIORITY_ACTIONS:
            return ToolImportance.LOW
            
        # Unknown actions - use token count heuristic
        if token_count > 4000:
            return ToolImportance.MEDIUM
            
        return ToolImportance.LOW
    
    def _calculate_retention(
        self, 
        importance: ToolImportance, 
        position: int, 
        total_tools: int, 
        token_budget: Optional[TokenBudget] = None
    ) -> int:
        """Calculate how many newer tools before compression."""
        retention = self.BASE_RETENTION[importance]
        
        # Apply pressure adjustment
        if token_budget:
            pressure_factor = 1.0 - token_budget.compression_multiplier
            retention = max(1, int(retention * pressure_factor))
        
        # Boost recent tools (last 20%)
        if total_tools > 0:
            relative_pos = position / total_tools
            if relative_pos < 0.2:
                retention = min(retention * 2, 30)
                
        return retention
    
    def _determine_compression(
        self, 
        importance: ToolImportance, 
        position: int, 
        keep_detailed: int, 
        token_budget: Optional[TokenBudget] = None
    ) -> str:
        """Determine compression level based on analysis."""
        # Force compression under high pressure
        if token_budget and token_budget.pressure_level in [PressureLevel.HIGH, PressureLevel.CRITICAL]:
            if position > 5 and importance < ToolImportance.HIGH:
                return "ultra_compact"
        
        # Position-based compression
        if position < keep_detailed:
            return "detailed"
            
        # Importance-based compression
        if importance >= ToolImportance.HIGH:
            return "summary"
            
        return "ultra_compact" 